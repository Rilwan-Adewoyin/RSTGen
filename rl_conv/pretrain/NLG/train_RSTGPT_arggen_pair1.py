#Uses dyploc cmv dataset in sync to finetune
import os
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ['TOKENIZERS_PARALLELISM'] = "true"

import train_RSTGPT
from train_RSTGPT import RSTGPT2, RSTGPT2_Config, RSTTokenizer, RSTGPT2_TrainingModule

import string
import argparse
import copy
import glob
import inspect
import json
import pickle
import random
import traceback
import types
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import (Any, Dict, Iterator, List, Optional, TypeVar)
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import bisect
import torch.distributed as dist
import ujson
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.distributed import _get_rank
from torch.utils.data import Sampler 
from torch.utils.data.dataset import Dataset
from pytorch_lightning.callbacks.finetuning import BaseFinetuning

from transformers.optimization import Adafactor, AdafactorSchedule, AdamW
from transformers.tokenization_utils_base import AddedToken
import transformers

import utils_nlg_v3 as utils
from utils_nlg_v3 import mpatch_save_model
from seg_bot_segmenter import Segmenter, Lang, PointerNetworks

from torch.nn.modules.batchnorm import _BatchNorm


T_co = TypeVar('T_co', covariant=True)

mp1 = os.path.abspath(os.path.join('..'))
mp2 = "../DockerImages/feng_hirst_rst_parser"
mp3 = "../DockerImages/feng_hirst_rst_parser/src"
mp4 = "../DockerImages/feng_hirst_rst_parser/model"
modules_paths = [mp1, mp2, mp3, mp4]

import sys

for path_ in modules_paths:
    if path_ not in sys.path:
        sys.path.append(path_)
from torch.nn.utils.rnn import pad_sequence
from transformers.utils import logging
logger = logging.get_logger(__name__)

class RSTGPT2PairConfig(RSTGPT2_Config):
    
    def __init__(self, 
                     base_model_name='gpt2',
                 model_name="RSTGPT2Pair",
                 scale_grad_by_freq=True,
                 max_len_rst=36,
                 max_len_key_phrase=64,
                 max_len_utt=270,
                 rst_tree_aligned_attention=False,
                 rst_segment_method='None',
                 max_rst_pos=4094,
                max_len_title=40, 
                embd_index_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
                resid_pdrop=0.1,
                **kwargs):

        model_name= "RSTGPT2Pair"
        
        super().__init__(base_model_name=base_model_name,
                 model_name=model_name,
                 scale_grad_by_freq=scale_grad_by_freq,
                 max_len_rst=max_len_rst,
                 max_len_key_phrase=max_len_key_phrase,
                 max_len_utt=max_len_utt,
                 rst_tree_aligned_attention=rst_tree_aligned_attention,
                 rst_segment_method=rst_segment_method,
                 max_rst_pos=max_rst_pos,

                embd_index_pdrop=embd_index_pdrop,
                embd_pdrop=embd_pdrop,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                 **kwargs)

        self.extra_pair_tokens = 1
        self.vocab_size = self.vocab_size + self.extra_pair_tokens
        self.max_len_title = max_len_title

class RSTGPT2Pair(RSTGPT2):
    
    def __init__(self, config: RSTGPT2PairConfig):
        
        super().__init__(config)
        # #Freeze all weights except for prefix weight,
        # for name, param in self.model.named_parameters(): 
        #     param.requires_grad = False

        with torch.no_grad():
            # Need to ensure the wte's below are not linked to the output embedding
            self.wte_title = copy.deepcopy( self.transformer.wte )

            self.wpe_title = copy.deepcopy( self.transformer.wpe )
            
    def embed(
            self,
            rst_start_token_id,
            rst_rel,
            rst_ns,
            rst_pos,
            key_phrase_ids,
            li_kprstpos,
            input_ids_utt,
            position_ids_keyphrase,
            position_ids_utt,
            **kwargs
        ):
            # RST context embedding
            rst_start_token_embed = self.discrete_embedding_dropout( rst_start_token_id, self.transformer.wte, self.config.embd_index_pdrop )
            rst_rel_embed   = self.embed_rst_rels(rst_rel)
            rst_ns_embed    = self.embed_rst_ns(rst_ns)
            # rst_rel_embed   = self.discrete_embedding_dropout( rst_rel, self.embed_rst_rels, self.config.embd_index_pdrop )
            # rst_ns_embed    = self.discrete_embedding_dropout( rst_ns, self.embed_rst_ns, self.config.embd_index_pdrop )
            rst_pos_embed   = self.embed_rst_pos(rst_pos)

            rst_embed = (rst_rel_embed + rst_ns_embed + rst_pos_embed)

            # Key Phrase context embedding
            # keyphrase_phrase_embed = self.transformer.wte( key_phrase_ids)
            keyphrase_phrase_embed = self.discrete_embedding_dropout(key_phrase_ids, self.transformer.wte, self.config.embd_index_pdrop)
            keyphrase_rst_pos_embed = self.embed_rst_pos(li_kprstpos)
            keyphrase_embed = keyphrase_rst_pos_embed + keyphrase_phrase_embed

            # input_id embedding
            utt_inputs_embeds = self.discrete_embedding_dropout(input_ids_utt, self.transformer.wte, self.config.embd_index_pdrop)

            title_embed = self.discrete_embedding_dropout(kwargs.get('ids_title'), self.wte_title, self.config.embd_index_pdrop)

            inputs_embed = torch.cat([
                rst_start_token_embed,
                rst_embed,
                keyphrase_embed,
                title_embed,
                utt_inputs_embeds], axis=-2)

            # Position Embedding
            position_embed_kp = self.transformer.wpe(position_ids_keyphrase)
            position_embed_utt = self.transformer.wpe(position_ids_utt)
            position_embed_title = self.wpe_title( kwargs.get('position_ids_title') )

            _ = position_embed_kp.shape
            position_embed_rst = position_embed_kp.new_zeros( [_[0], 1+rst_rel_embed.shape[1], _[2]] )
            position_embed = torch.cat( [position_embed_rst,
                                            position_embed_kp, 
                                            position_embed_title, 
                                            position_embed_utt], axis=1)
            
            return inputs_embed, position_embed

    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--base_model_name', default='gpt2', required=False)
        parser.add_argument('--model_name', default='RSTGPT2Pair', required=False)
        parser.add_argument('--max_len_utt', type=int, default=270)
        parser.add_argument('--max_len_rst', type=int, default=36)
        parser.add_argument('--max_len_key_phrase', type=int, default=64)
        parser.add_argument('--max_len_title', type=int, default=30)
        
        parser.add_argument('--scale_grad_by_freq', type=lambda x: bool(int(x)), default=True,
                            help="Inverse the gradients to the emebdding layers based on the occurence of each index in the minibatch ")
        parser.add_argument('--rst_tree_aligned_attention',
                            type=lambda x: bool(int(x)), default=False)
        parser.add_argument('--rst_segment_method', type=str,
                            default='None', choices=['None', 'fenghirst', 'segbot'])
        
        #Regularization params
        parser.add_argument('--embd_pdrop',type=float, default=0.1, help="Normal dropout of on output of embeddings")
        parser.add_argument('--attn_pdrop', type=float, default=0.1)
        parser.add_argument('--resid_pdrop', type=float, default=0.1)

        parser.add_argument('--embd_index_pdrop',type=float, default=0.1, help="We drop specific indices from embedding")
        
        mparams = parser.parse_known_args()[0]
        return mparams    

    @classmethod
    def load_model_tokenizer(cls, model_name="RSTGPT2Pair", model_version=None, mparams_new={}, device="cuda:0"):

        if model_version != None:
            # load from a pretrained RSTGPT2
            checkpoint = RSTGPT2Pair_TrainingModule.get_ckpt_file(
                f'./models/{model_name}/version_{model_version}/checkpoints')

            mparams = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                'base_model_name', 'model_name', 'max_len_key_phrase',
                'max_len_rst', 'max_len_utt','max_len_title',
                'scale_grad_by_freq', 'rst_tree_aligned_attention']}

            # overriding with new keys
            for key, value in mparams_new.items():
                mparams[key] = value

            mconfig = RSTGPT2PairConfig.from_pretrained(
                mparams['base_model_name'], **mparams)

            model = RSTGPT2Pair(mconfig)

            # Loading Training Module
            training_module = RSTGPT2Pair_TrainingModule(
                mconfig, mode='inference', model=model)
            training_module.load_state_dict(checkpoint['state_dict'])

            model = training_module.model
            tok = training_module.tokenizer
            model.correct_attn_bias()

            # Deleting checkpoints to free up GPU space
            del checkpoint
            torch.cuda.empty_cache()

            # if torch.cuda.is_available():
            if device != 'cpu' and torch.cuda.is_available():
                model = model.to(device)

            return model, tok

        else:
            raise ValueError(
                "At least one of model_version or mconfig must not be None ")

    def copy_embedding_weights(self):
        with torch.no_grad():
            # Need to ensure the wte's below are not linked to the output embedding
            self.wte_title = copy.deepcopy( self.transformer.wte )
            self.wpe_title = copy.deepcopy( self.transformer.wpe )

class RSTTokenizerPair(RSTTokenizer):

    title_start_token = "<|tl|>"

    max_len_title = 30
    
    def __init__(self, *args, **kwargs):
        self.max_len_title = kwargs.get( 'max_len_title' , self.max_len_title)

        super().__init__(*args, **kwargs)
        self.pad_token =  self.eos_token
        
    def encode_input(self, rst_rel, rst_ns, rst_pos, li_kp, li_kprstpos, utterance=None, utterance_prompt=None, dict_pos_edu=None, max_len_rst=None, max_len_key_phrase=None, exclude_from_output=[], device=None, title='', max_len_title=None):
       
        encoded = super().encode_input(rst_rel, rst_ns, rst_pos, li_kp, li_kprstpos, utterance=utterance, utterance_prompt=utterance_prompt, dict_pos_edu=dict_pos_edu, max_len_rst=max_len_rst, max_len_key_phrase=max_len_key_phrase, exclude_from_output=exclude_from_output, device=device)

        # Encoding title
        if title != None and title!="":
            title = self.title_start_token+title.lstrip(string.punctuation+" ")
            title = ' '.join( title.split(' ')[:self.max_len_title] )

            ids_title = self.encode(title, add_special_tokens=False,
                return_attention_mask=False,
                padding= 'max_length' if max_len_title else 'do_not_pad',
                truncation=True,
                max_length=max_len_title if max_len_title else self.max_len_title,
                return_tensors='pt')[0]
        else:
            ids_title = [self.pad_token_id]*max_len_title if max_len_title else [self.pad_token_id]*self.max_len_title
            ids_title = torch.tensor(ids_title,dtype=torch.long)
        

        title_pad = (ids_title == self.pad_token_id).sum(dim=0)
        
        encoded['ids_title'] = ids_title

        title_len = ids_title.shape[0]

        # chaining positions
        positions_ids_title= torch.arange(0, title_len, dtype=torch.long)

        encoded['position_ids_title'] = positions_ids_title
        encoded['position_ids_utt'] = encoded['position_ids_utt']
        
        if title_len >0:
            #changing labels
            if encoded.get('labels') is not None:
                new_labels = positions_ids_title.new_full( [title_len] , -100 )
                utt_len = encoded['position_ids_utt'].shape[0]
                encoded['labels'] = torch.cat( [ encoded['labels'][:-utt_len] ,new_labels, encoded['labels'][-utt_len:] ] )
            else:
                utt_len = encoded['position_ids_utt'].shape[0]


            # changing attn
                # Plan Make new empty attention matrix template
                # Fill in the context attn span
                # Fill in the utterance attn span
                # New title has torch trill attn to itself
                # New title does not attend to any of the context
                # New title does not attentd to any of the utterance

            context_len = encoded['attention_mask'].shape[0]-utt_len
            _ = title_len + utt_len + context_len
            new_attn_mask = encoded['attention_mask'].new_zeros( (_,_) )

            # Context attn
            new_attn_mask[ :context_len, :context_len ] = encoded['attention_mask'][ :context_len, :context_len ]

            # utterance attn
            new_attn_mask[ -utt_len: , -utt_len: ] = encoded ['attention_mask'][ -utt_len: , -utt_len: ] 
            new_attn_mask[ -utt_len:, :context_len] = encoded['attention_mask'][ -utt_len:, :context_len ]
            new_attn_mask[ -utt_len:, context_len:context_len+title_len-title_pad] = new_attn_mask.new_ones( ( utt_len, title_len-title_pad)  )


            # title attn
            new_attn_mask[ context_len:context_len+title_len-title_pad , context_len:context_len+title_len-title_pad  ] = \
                torch.tril( new_attn_mask.new_ones( (title_len-title_pad , title_len-title_pad) ) )

            del encoded['attention_mask']
            encoded['attention_mask'] = new_attn_mask

        return encoded 

    @classmethod
    def from_pretrained(cls,
                        dir_tokenizer="./tokenizers/RSTGPT2Pair",
                        base_tokenizer_name="gpt2",
                        rst_params={},
                        **kwargs):  # max_len_rst, max_len_key_phrase, max_rst_depth, max_len_utt, max_rst_pos

        if os.path.exists(dir_tokenizer):
            tokenizer = super(RSTTokenizer, cls).from_pretrained(
                dir_tokenizer, local_files_only=True, **kwargs, **rst_params)

        else:
            additional_special_tokens = kwargs.pop(
                'additional_special_tokens', [])

            at_title_start = AddedToken(cls.title_start_token, lstrip=False, rstrip=False) if isinstance(
                cls.title_start_token, str) else cls.title_start_token
            additional_special_tokens = additional_special_tokens [
                at_title_start] 

            cls = super(RSTTokenizerPair, cls).from_pretrained(
                                                                dir_tokenizer=dir_tokenizer,
                                                                base_tokenizer_name="gpt2",
                                                                additional_special_tokens=additional_special_tokens)

            cls.save_pretrained(dir_tokenizer)
            tokenizer = cls

        tokenizer.title_start_token_id = torch.full( (1,),      50259 , dtype=torch.long )
        tokenizer.rst_start_token_id = torch.full( (1,),        50257 , dtype=torch.long )
        tokenizer.keyphrase_start_token_id = torch.full( (1,),  50258 , dtype=torch.long )        
        tokenizer.keyphrase_start_token_id_np = tokenizer.keyphrase_start_token_id.numpy()

        for k, v in kwargs.items():
            setattr(tokenizer, k, v)

        return tokenizer
    
class RSTGPT2Pair_TrainingModule(pl.LightningModule):

    def __init__(self,
                 mconfig,
                 batch_size=20,
                 dir_data=None,
                 accumulate_grad_batches=1,
                 max_epochs=100,
                 gpus=1,
                 learning_rate=1e-4,
                 warmup_proportion=0.1,
                 workers=0,
                 mode='finetune',
                 tag='',
                 batching_style='effecient',
                 model =None,
                 tokenizer = None,
                 **kwargs):
        
        super().__init__()

        self.batch_size = batch_size
        self.gpus = gpus
        self.mode = mode
        self.workers = workers
        self.batching_style = batching_style
        self.dir_checkpoints = kwargs.get('dir_checkpoints')

        if tokenizer  == None:
            self.tokenizer = RSTTokenizerPair.from_pretrained(f"./tokenizers/{mconfig.model_name}",
                                                         base_tokenizer_name=mconfig.base_model_name,
                                                         rst_params={name: getattr(mconfig, name) for name in ['max_len_rst',
                                                                                                               'max_len_key_phrase',
                                                                                                               'max_rst_depth',
                                                                                                               'max_len_utt', 
                                                                                                               'max_rst_pos',
                                                                                                               'max_rst_pos',
                                                                                                               'max_len_title',
                                                                                                               'rst_tree_aligned_attention'] if hasattr(mconfig, name)
                                                                     }
                                                         )
        else:
            self.tokenizer = tokenizer

        if model is not None:
            self.model = model
        else:
            raise Exception
                    
        self.pad_values = {'rst_start_token': mconfig.eos_token_id,
                           'rst_rel': self.model.embed_rst_rels.padding_idx,
                           'rst_ns': self.model.embed_rst_ns.padding_idx,
                           'rst_pos': self.model.embed_rst_pos.padding_idx,

                           'key_phrase_ids': mconfig.eos_token_id,
                           'li_kprstpos': self.model.embed_rst_pos.padding_idx,

                            'position_ids_keyphrase':mconfig.n_ctx-1,
                            'position_ids_utt':mconfig.n_ctx-1,

                           'position_ids_kp_utt': mconfig.n_ctx-1,
                           'position_ids_title':mconfig.n_ctx-1,


                           'input_ids_utt': mconfig.eos_token_id,
                            'ids_title':mconfig.eos_token_id,
                           'attention_mask': 0.0,

                           'labels': self.model.loss_fct.ignore_index,
                            'edu_rstpos': -1,
                            'context_rstpos': -1,

                           'context_rst_rstpos':-1,
                           'context_kp_rstpos':-1
                           }
        
        self.tokenizer.pad_values = self.pad_values

        self.pad_maxlens = {
            'rst_start_token': 1,
            'rst_rel': mconfig.max_len_rst-1,
            'rst_ns': mconfig.max_len_rst-1,
            'rst_pos': mconfig.max_len_rst-1,

            'key_phrase_ids': mconfig.max_len_key_phrase,
            'li_kprstpos': mconfig.max_len_key_phrase,

            'input_ids_utt': mconfig.max_len_utt,
            'ids_title': mconfig.max_len_title,


            'labels': mconfig.max_len_rst + mconfig.max_len_key_phrase + mconfig.max_len_utt + mconfig.max_len_title,

            'attention_mask': mconfig.max_len_rst + mconfig.max_len_key_phrase + mconfig.max_len_utt + mconfig.max_len_title,  # axis:max_length

            'position_ids_keyphrase':mconfig.max_len_key_phrase,
            'position_ids_utt':mconfig.max_len_utt ,
            'position_ids_title':mconfig.max_len_title,


            'edu_rstpos': mconfig.max_rst_pos // 2,
            'context_rstpos':mconfig.max_len_rst + mconfig.max_len_key_phrase,

            'context_rst_rstpos':mconfig.max_len_rst,
            'context_kp_rstpos':mconfig.max_len_key_phrase
                            }

        self.tokenizer.pad_maxlens = self.pad_maxlens
        
        self.model.tokenizer = self.tokenizer

        if self.mode in ['finetune', 'train_cont', 'test']:
            self.dir_data = utils.get_path(dir_data, _dir=True)
            self.accumulate_grad_batches = accumulate_grad_batches
            self.tag = tag

            self.dg = DataLoaderGenerator(self.dir_data,  self.batch_size, self.tokenizer,
                                 workers=self.workers, mode=self.mode, gpus=self.gpus,
                                 pad_maxlens=self.pad_maxlens, pad_values=self.pad_values,
                                 batching_style=self.batching_style)

            if self.mode == "test":
                self.create_data_loaders(['test'])
            else:
                self.create_data_loaders(['train', 'val'] )
                # self.create_data_loaders(['inference'] )
                # self.inference_samples = list(islice(self.inference_dl, 3))
                # del self.inference_dl

        if self.mode in ['finetune', 'train_cont']:
            self.max_epochs = max_epochs
            self.warmup_proportion = warmup_proportion
            self.learning_rate = learning_rate

            train_params_to_save = self.return_params()
            mparams_to_save = {param: getattr(mconfig, param) for param in list(filter(
                lambda p: p not in ['self','kwargs'], list(inspect.signature(RSTGPT2PairConfig.__init__).parameters.keys()) ))}

            self.hparams.update({**train_params_to_save, **mparams_to_save})
            pl.core.saving.save_hparams_to_yaml(os.path.join(os.path.dirname(
                kwargs['dir_checkpoints']), "hparams.yaml"), self.hparams)
            
        if self.mode in ['inference']:
            self.eval()
            self.freeze()
         
    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data', default="./dataset_cmv/dyploc_pair_rst",
                            help="Relative directory path for datafiles")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('--max_epochs', default=70, type=int)
        parser.add_argument('--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('--batch_size', default=20, type=int)
        parser.add_argument('--batching_style', default='effecient', type=str, choices=['effecient','standard'])
        parser.add_argument('--finetune_version', type=int, default=6 )
        parser.add_argument('--num_nodes',default=1, type=int )
        parser.add_argument('--learning_rate', default=4e-3, type=float)
        parser.add_argument('--workers', default=16, type=int)  # TODO: change to 6
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--mode', default='finetune', type=str,
                            choices=['finetune', 'train_cont', 'test', 'inference'])
        parser.add_argument('--version', default=None, required=False,
                            type=int, help="The Experimental Versioning for this run")
        parser.add_argument('--precision', default=16, required=False,
                            type=int, help="Precision to use", choices=[16, 32])
        parser.add_argument('--tag', default='', required=True, type=str)
        parser.add_argument('--override', default=False,
                            type=lambda x: bool(int(x)), choices=[0, 1])
        tparams = parser.parse_known_args()[0]

        return tparams

    @staticmethod
    def instatiate_training_module(tparams=None, mparams=None):
        """Create training module

        Args:
            tparams ([type]): [description]
        """

        if tparams['mode'] in ["finetune"]:            
            
            checkpoint = RSTGPT2_TrainingModule.get_ckpt_file(f"./models/RSTGPT2/version_{tparams['finetune_version']}/checkpoints")
            mparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                'base_model_name', 'scale_grad_by_freq','rst_tree_aligned_attention' ]})
            
            mconfig = RSTGPT2PairConfig.from_pretrained(mparams['base_model_name'], **mparams)
            mconfig.vocab_size = mconfig.vocab_size-1
            model = RSTGPT2Pair(mconfig)
            model.config.vocab_size += 1
            pytorch_state_dict = { k[k.find('.')+1:]:v for k,v in checkpoint['state_dict'].items() }
            model.load_state_dict( pytorch_state_dict,strict=False )
            model.correct_attn_bias()
                
            tokenizer = RSTTokenizerPair.from_pretrained(**mparams)
            model.resize_token_embeddings(model.config.vocab_size)
            model.copy_embedding_weights()

            training_module = RSTGPT2Pair_TrainingModule(model.config, **tparams, model=model, tokenizer=tokenizer)

        elif tparams['mode'] in ["train_cont", "inference"]:

            checkpoint = RSTGPT2Pair_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            # restore/update param files from the checkpoint
            if "hyper_parameters" in checkpoint.keys() and tparams['override'] == False:
                tparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                     'learning_rate', 'precision', 'splits', 'tag']})

                mparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name', 'model_name', 'max_len_utt','max_len_rst','max_len_key_phrase',
                    'max_len_title','scale_grad_by_freq','rst_tree_aligned_attention',
                    'embd_pdrop','attn_pdrop','resid_drop','embd_index_pdrop' ]})

            else:
                print("param files not found utilsing default or user entered params\n")

            mconfig = RSTGPT2PairConfig( **mparams)

            model = RSTGPT2Pair(mconfig)
            pytorch_state_dict = { k[k.find('.')+1:]:v for k,v in checkpoint['state_dict'].items() }
            model.load_state_dict( pytorch_state_dict )
            model.correct_attn_bias()
            tokenizer = RSTTokenizerPair.from_pretrained(**mparams)            
            training_module = RSTGPT2Pair_TrainingModule(mconfig, **tparams, model=model, tokenizer=tokenizer)


        elif tparams['mode'] in ["test"]:

            checkpoint = RSTGPT2Pair_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            # restore/update param files from the checkpoint
            try:
                tparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                    'learning_rate', 'precision']})

                mparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name', 'model_name', 'max_len_utt','max_len_rst',
                    'max_len_key_phrase','max_len_title']})
            except KeyError:
                pass

            # Restore/update Training Module   
            training_module = RSTGPT2Pair_TrainingModule(
                **tparams, mparams=mparams)
            training_module.load_state_dict(checkpoint['state_dict'])

        else:
            raise ValueError(
                "tparams['mode'] must be in range [finetune, train_cont, test, inference]")
            
        return training_module

    @staticmethod
    def instatiate_trainer(tparams, tb_logger, training_module):
        """[summary]

            Creates The Trainer and callbacks
        """
        dir_checkpoints = tparams['dir_checkpoints']

        # Creating Callbacks    
        callbacks = []
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                            save_top_k=2,
                                            save_last =True,
                                              mode='min', dirpath=dir_checkpoints,
                                              filename='{epoch:03d}_{val_loss:.5f}')

        checkpoint_callback._save_model = types.MethodType(
            mpatch_save_model(checkpoint_callback._save_model), checkpoint_callback) 

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience = 2,       
            verbose=False,
            mode='min'
        )

        callbacks.append(checkpoint_callback)
        callbacks.append(early_stop_callback)

        if tparams['gpus'] in [0, 1]:
            trainer_vars = {}
        else:

            trainer_vars = {    'accelerator': 'ddp',
                            # 'plugins': DeepSpeedPlugin(stage=1, 
                            #                             contiguous_gradients=True,
                            #                              ) 
                            'plugins' : DDPPlugin(find_unused_parameters=False),
                            'nodes':tparams['num_nodes']

                            }

        if tparams['mode'] in ["finetune"]:

            trainer = pl.Trainer.from_argparse_args( argparse.Namespace(**tparams),
                                                    default_root_dir=tparams['dir_checkpoints'],
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks,
                                                    replace_sampler_ddp=False,
                                                    num_sanity_val_steps=0,
                                                    **trainer_vars,
                                                    )
                                                

        elif tparams['mode'] in ["train_cont", "inference"]:

            # restoring checkpoint
            checkpoint = RSTGPT2Pair_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])
           

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks, 
                                                    num_sanity_val_steps=0,
                                                    replace_sampler_ddp=False,
                                                    **trainer_vars,
                                                    )

            # load callback states
            trainer.on_load_checkpoint(checkpoint)
            
            try:
                trainer.global_step = checkpoint['global_step']
                trainer.current_epoch = checkpoint['epoch']
            except Exception:
                trainer.fit_loop.global_step = checkpoint['global_step']
                trainer.fit_loop.current_epoch = checkpoint['epoch']

            del checkpoint
            torch.cuda.empty_cache()

        elif tparams['mode'] in ["test"]:

            # restoring checkpoint
            checkpoint = RSTGPT2Pair_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            training_module.load_state_dict(checkpoint['state_dict'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                                                    check_val_every_n_epoch=1,
                                                    checkpoint_callback=False,
                                                    logger=tb_logger,
                                                    log_every_n_steps=1,
                                                    precision=tparams['precision'])

            # load callback states
            trainer.on_load_checkpoint(checkpoint)

        return trainer, training_module

    @staticmethod
    def get_ckpt_file(_dir_checkpoint, mode='best'):
        if mode == 'best':
            checkpoint_yaml_file = os.path.join(
                _dir_checkpoint, "best_k_models.yaml")
            # key= ckptpath, value = val_loss
            scores_dict = yaml.load(open(checkpoint_yaml_file, "r"), Loader=yaml.FullLoader)
            best_ckpt_path = min(scores_dict, key=scores_dict.get)

            if os.path.exists(best_ckpt_path) == False:
                root_dir = Path(__file__).resolve().parents[4]
                best_ckpt_path = os.path.join(
                    root_dir._str, best_ckpt_path[best_ckpt_path.index('mastering-conversation'):])

            
            checkpoint = torch.load(best_ckpt_path, map_location='cpu')
        
        else:
            raise NotImplementedError

        return checkpoint

    @staticmethod
    def start(trainer, tparams, training_module, mparams):

        if tparams['mode'] in ['finetune', 'train_cont']:
            trainer.fit(training_module)

        if tparams['mode'] in ["test"]:

            checkpoint = RSTGPT2Pair_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])
            training_module.load_state_dict(checkpoint['state_dict'])

            training_module.eval()
            training_module.freeze() 

            dict_results = trainer.test(
                test_dataloaders=training_module.test_dl, model=training_module)

            # Saving test results for model to file
            _dir = os.path.join(tparams['model_dir'], mparams['model_name'])
            fn = os.path.join(_dir, "results.json")

            if os.path.isfile(fn) == False:
                existing_results = {}
            else:
                with open(fn, 'r') as outfile:
                    existing_results = json.load(outfile)

            existing_results[f"{mparams['model_name']}_{tparams['version']}"] = dict_results[0]['test_loss']

            with open(fn, 'w') as outfile:
                json.dump(existing_results, outfile)


    def forward(self, input_):
            return self.model(**input_, return_dict=True)

    
    def step(self, batch, step_name):

        output = {}
        #edit
        model_output = self.forward(batch)

        if step_name == 'train':
            
            loss = model_output.loss 

            output["loss"]  = loss

            self.log( "loss", output["loss"], sync_dist=False)

        else:
            loss_key = f"{step_name}_loss"
            self.log( loss_key, model_output.loss, sync_dist=False)
            output[loss_key]= model_output.loss

        return output
    

    def training_step(self, batch, batch_idx):
        output = self.step(batch, "train")
        return output

    def validation_step(self, batch, batch_idx):
        output = self.step(batch, "val")
        return output

    def test_step(self, batch, batch_idx):
        output = self.step(batch, "test")
        return output
    
    def training_epoch_end(self, outputs):
        self.epoch_end_log(outputs, "train")

    def validation_epoch_end(self, outputs: List[dict]):
        self.epoch_end_log(outputs, "val")

    def test_epoch_end(self, outputs: List[dict]):
        self.epoch_end_log(outputs, "test")
    
    def epoch_end_log(self, outputs, step_name):

        if step_name == "train":
            pass
        else:
            loss = torch.stack([x[f"{step_name}_loss"]for x in outputs]).mean()
            
            self.log(f"{step_name}_loss", loss, logger=True, prog_bar=True, sync_dist=True)
        
        if False and step_name == "val" and _get_rank() == 0:
            # Making directory if it doesnt exist
            dir_infer = os.path.join(self.trainer.log_dir, "inference")
            
            if not os.path.exists(dir_infer):
                os.makedirs(dir_infer, exist_ok=True)

            # Adding true values and making csv files if thy dont already exists
            for idx, encoded_input_ in enumerate(self.inference_samples):
                
                encoded_input = { k:v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k,v in encoded_input_.items()}
                                
                fp = os.path.join(dir_infer, f"example_{idx:03d}.csv")

                # If there file does not exists we add the true observed records
                if not os.path.exists(fp):

                    df = pd.DataFrame(columns=['epoch', 'rst_rels', 'rst_ns', 'rst_pos',

                                               'keyphrase', 'utterance',
                                                'dict_pos_edu', 'li_kprstpos',
                                                 'orig_title'])

                    rst_rels = encoded_input.pop('orig_rst_rels')
                    rst_ns = encoded_input.pop('orig_rst_ns')
                    rst_pos = encoded_input.pop('orig_rst_pos')

                    keyphrase = encoded_input.pop('orig_key_phrase')
                    utterance = encoded_input.pop('orig_utt')
                    dict_pos_edu = encoded_input.pop('orig_dict_pos_edu')

                    orig_li_kprstpos = encoded_input.pop('orig_li_kprstpos')
                    orig_title =  encoded_input.pop('orig_title', None)
                    

                    datum = {
                        'epoch': -1,

                        'rst_rels': ', '.join(rst_rels),
                        'rst_ns': ', '.join(rst_ns),
                        'rst_pos': rst_pos,

                        "keyphrase": ', '.join(keyphrase),
                        "utterance": utterance,
                        "dict_pos_edu": json.dumps(dict_pos_edu),

                        "li_kprstpos": json.dumps(orig_li_kprstpos),
                        "orig_title": json.dumps(orig_title)
                        
                    }

                    df = df.append(datum, ignore_index=True)
                    df.to_csv(fp, index=False)

                # creating predition andding to existing results
                encoded_input.pop('orig_rst_rels', None)
                encoded_input.pop('orig_rst_ns', None)
                encoded_input.pop('orig_rst_pos', None)

                encoded_input.pop('orig_key_phrase', None)
                encoded_input.pop('orig_utt', None)
                encoded_input.pop('orig_dict_pos_edu', None)
                encoded_input.pop('orig_li_kprstpos', None)
                encoded_input.pop('orig_title', None)
                
                # encoded_input.pop('labels', None)

                generation_params = copy.deepcopy(self.model.generation_params)
                generation_params['max_length'] = 60
                generation_params['max_time'] = 20
                decoded_text = self.model.generate_plus( encoded_input, generation_params )

                datum = {
                    'epoch': self.current_epoch,
                    'rst_rels': '',
                    'keyphrase': '',
                    'utterance': json.dumps(decoded_text),
                    'dict_pos_edu': '',
                    'li_kprstpos': '',
                    'rst_ns': '',
                    'rst_pos': '',
                    'orig_title':''
                }

                pd.DataFrame.from_records([datum]).to_csv(fp, index=False, mode='a', header=False)
                # Saving to file
        
        else:
            pass
        
    def create_data_loaders(self, modes ):
        if 'train' in modes:
            self.train_dl = self.dg.prepare_dataloader(
                split_name='train')
            self.train_dl_used = False

        if 'val' in modes:
            self.val_dl = self.dg.prepare_dataloader(
                split_name='val')
        if 'test' in modes:
            self.test_dl = self.dg.prepare_dataloader(
                split_name='test')
        if 'inference' in modes:
            self.inference_dl = self.dg.prepare_dataloader(
                split_name='inference')

    def train_dataloader(self):
        # return self.train_dl
        if self.train_dl_used == False:
            self.train_dl_used = True
            return self.train_dl
        else:
            self.train_dl = self.dg.prepare_dataloader(split_name='train')
            return self.train_dl

    def val_dataloader(self):

        return self.val_dl 

    def test_dataloader(self):
        return self.test_dl

    @lru_cache()
    def total_steps(self):

        ds_size = len(self.train_dl)
        steps = (ds_size * self.max_epochs) // (self.accumulate_grad_batches)
        return steps

    def configure_optimizers(self):

        self.freeze_specific_modules( [ self.model ] )
        self.freeze_specific_modules( self.model.transformer.h, freeze=False )
        self.freeze_specific_modules( [ self.model.wte_title, self.model.wpe_title], freeze=False )
        

        parameters = filter( lambda p: p.requires_grad, self.model.parameters() )
        
        optimizer = Adafactor(parameters, scale_parameter=False, 
                        relative_step=True, warmup_init=True, lr=None,
                        weight_decay=0.01
                        )

        lr_scheduler = AdafactorSchedule(optimizer)

        # optimizer = Adafactor(parameters,
        #                      scale_parameter=False, 
        #                      relative_step=False,
        #                      warmup_init=False,
        #                      lr=self.learning_rate )

        # lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
        #                                 num_warmup_steps=  int( (3.0*self.total_steps())/self.max_epochs) ,
        #                                 num_training_steps=self.total_steps(),
        #                                 last_epoch=-1 )

        if self.mode == "train_cont":
            # restore the optimizers
            checkpoint = self.get_ckpt_file(self.dir_checkpoints)
            optimizer_states = checkpoint['optimizer_states']
            optimizer.load_state_dict(optimizer_states[0])
   
            # restore the lr schedulers
            lr_scheduler_states = checkpoint['lr_schedulers']
            lr_scheduler.load_state_dict(lr_scheduler_states[0])
            
        return { 'optimizer':optimizer, "lr_scheduler": lr_scheduler, "interval": "step", "monitor": "val_loss"}

    def freeze_specific_modules(self, modules, train_bn=True, freeze=True):

        modules = BaseFinetuning.flatten_modules(modules)

        for mod in modules:
            if isinstance(mod, _BatchNorm) and train_bn:
                BaseFinetuning.make_trainable(mod)
            else:
                # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
                for param in mod.parameters(recurse=False):
                    param.requires_grad = not freeze
        

    def return_params(self):
        params = {}
        keys = ['batch_size', 'accumulate_grad_batches', 'learning_rate', 'max_epochs', 'dir_data','tag']

        params = {
            k: self.__dict__[k] for k in keys if k in self.__dict__.keys()}

        return params

class DataLoaderGenerator():
    """Handles the creation of dataloaders for a train, val and test set
    """

    def __init__(self, dir_data, batch_size,
                 tokenizer,
                 workers=0, mode='finetune',
                 gpus=1,
                 pad_values={},
                 pad_maxlens={},
                 **kwargs):

        self.dir_data = dir_data
        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.workers = workers
        self.mode = mode
        self.gpus = gpus
        self.pad_values = pad_values
        self.pad_maxlens = pad_maxlens

    def prepare_dataloader(self,
                           split_name='train',
                           **kwargs):
        """Prepares a dataloader given a directory of data for NLG language module
            # The current method takes a percentage of data from each subdirectory
            Args:
                dir_dset ([type]): [description]
        """
        def filter_fns(fns):
            fn = next( ( fn for fn in fns if ("dict_lens" not in fn) ) )
            return fn
        
        # defining starting line and total lines to use for dataset
        
        if split_name == 'train':

            fn = filter_fns( glob.glob(  os.path.join( self.dir_data,"train","*") ))
            shuffle = True
            inference = False
            bs = self.batch_size 
            sampler = True
            sample_kps = True

        elif split_name == 'val':
            fn = filter_fns(glob.glob(  os.path.join( self.dir_data,"val","*") ))
            shuffle = False
            inference = False
            bs = self.batch_size
            sampler = True
            sample_kps = False
            
            
        elif split_name == 'test':
            fn = filter_fns(glob.glob(  os.path.join( self.dir_data,"test","*") ))
            shuffle = False
            bs = self.batch_size
            inference = False
            sampler = True
            sample_kps = False
            raise NotImplementedError

            
        elif split_name == 'inference':
            fn = filter_fns(glob.glob(  os.path.join( self.dir_data,"test","*") ))
            shuffle = False
            bs = 1
            sampler = None
            inference = True
            sample_kps = False
            

        ds = SingleDataset(fn, copy.deepcopy(self.tokenizer), inference, sample_kps=sample_kps )
        
        if self.gpus <= 1 and split_name not in ['inference', 'test']:
            sampler = SizedOrderedBatchSampler(ds, bs, True, shuffle=True) if sampler else sampler
            bs = 1
        else:
            sampler = SizedOrderedDistributedBatchSampler(ds, bs, drop_last=True, shuffle=shuffle, gpus=self.gpus)
            bs = 1


        dataloader = torch.utils.data.DataLoader(ds, 
                                                 num_workers=self.workers//2 if sampler else 1, 
                                                 batch_size=bs,
                                                 batch_sampler = sampler,
                                                 pin_memory=True,
                                                 collate_fn=self.tokenizer.default_collate_pad,
                                                )

        if split_name == "train":

            return dataloader 

        else:
            return dataloader

class SingleDataset(Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

        create a custom index which sorts the entries by their length
    """
    def __init__(self, file_path, tokenizer, inference,**kwargs):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.inference = inference
        self.tokenizer.sample_kps = kwargs.get('sample_kps',False)
        self.data = pd.read_csv(self.fp, sep=',', header=0 )

        fp_cached_order = os.path.join(os.path.dirname(
            file_path), f"gpt2_dict_lens.pkl")

        # # # resetting the cached order files
        # if os.path.exists( fp_cached_order):
        #     os.remove(fp_cached_order)

    
        if os.path.exists(fp_cached_order):
            dict_cached_order = pickle.load(open(fp_cached_order, "rb"))
            self.np_textlens = dict_cached_order['np_textlens']
            self.np_rstlens = dict_cached_order['np_rstlens']
            self.np_keyphrase_lens = dict_cached_order['np_keyphrase_lens']
            self.np_title_lens = dict_cached_order['np_title_lens']

        else:
            # len of text
            self.np_textlens = np.stack(
                [self.tokenizer.encode(ujson.loads(txt), return_tensors='np', add_special_tokens=False,
                                    truncation=False, padding='do_not_pad').size for txt in self.data.txt_preproc.values.tolist()])
            # len of rst
            self.np_rstlens = np.array(
                [1 + len(json.loads(rst)) for rst in self.data.rst.values.tolist()])

            # len of keyphrase
            li_li_pos_kp = [ json.loads(li_pos_kp) for li_pos_kp  in self.data.li_pos_kp.values.tolist() ]
            li_li_kp = [ [ kp for pos,kp in li_pos_kp]  for li_pos_kp in li_li_pos_kp ]
            li_kp = [ '<|kp|> ' + '<|kp|> '.join(li_kp) for li_kp in li_li_kp  ] 
            
            self.np_keyphrase_lens = np.array( [ self.tokenizer.encode(kp, 
                                        add_special_tokens=False, 
                                        truncation=False,
                                        padding = 'do_not_pad',
                                        return_tensors=None).__len__() for kp in li_kp ] )
                        
            li_title = [ self.tokenizer.title_start_token+ujson.loads(title).lstrip(string.punctuation+' ') for title in self.data.prompt.tolist() ]
            self.np_title_lens = np.array( [self.tokenizer.encode(title,
                                            truncation=False,
                                            padding = 'do_not_pad',
                                            return_tensors=None).__len__() for title in li_title] )


            dict_cached_order = {'np_textlens': self.np_textlens,
                                'np_rstlens': self.np_rstlens,
                                'np_keyphrase_lens': self.np_keyphrase_lens,
                                'np_title_lens':self.np_title_lens,
                                }

            pickle.dump(dict_cached_order, open(fp_cached_order, "wb"))

        #v2 We initialize the rst/kp lengths as the actual length of each entry
        # In the Sampler, we change the max length to that of its pre-prescribed batch
        self.rst_len = copy.deepcopy( self.np_rstlens )
        self.key_phrase_len = copy.deepcopy( self.np_keyphrase_lens )
        self.title_len = copy.deepcopy(self.np_title_lens)
        self.data = self.data.to_dict('records')

    def __len__(self):
        return len( self.data )

    def __getitem__(self, index):

        rst_rels, rst_ns, rst_pos, li_kp, li_kprstpos, utterance, dict_pos_edu, title = self.getitem_extract_datum(
            index)

        if self.inference == True:

            # utterance_prompt = ' '.join(utterance.split(' '))
            utterance_prompt = ""

            encoded = self.tokenizer.encode_input(rst_rel=rst_rels, rst_ns=rst_ns, rst_pos=rst_pos,
                                                  li_kp=li_kp,
                                                  li_kprstpos=li_kprstpos,
                                                  utterance_prompt=utterance_prompt,
                                                  dict_pos_edu=dict_pos_edu,
                                                  max_len_rst= min( self.rst_len[index], self.tokenizer.max_len_rst ),
                                                  max_len_key_phrase= min( self.key_phrase_len[index], self.tokenizer.max_len_key_phrase),
                                                    title=title,
                                                  max_len_title=min( self.title_len[index], self.tokenizer.max_len_title),
                                                   )

            encoded['orig_rst_rels'] = rst_rels
            encoded['orig_rst_ns'] = rst_ns
            encoded['orig_rst_pos'] = rst_pos

            encoded['orig_utt'] = utterance
            encoded['orig_key_phrase'] = li_kp

            encoded['orig_dict_pos_edu'] = dict_pos_edu
            encoded['orig_li_kprstpos'] = li_kprstpos
            encoded['orig_title'] = title


        elif self.inference==False:
            encoded = self.tokenizer.encode_input(
                rst_rels, rst_ns, rst_pos,
                li_kp=li_kp,
                li_kprstpos=li_kprstpos,
                utterance=utterance,
                dict_pos_edu=dict_pos_edu,
                max_len_rst= min( self.rst_len[index], self.tokenizer.max_len_rst ),
                max_len_key_phrase= min( self.key_phrase_len[index], self.tokenizer.max_len_key_phrase),
                max_len_title=min( self.title_len[index], self.tokenizer.max_len_title),
                 )

        return encoded

    def getitem_extract_datum(self, index):

        datum = self.data[index]

        # region RST
        li_rst = json.loads(datum['rst'])
        # list of dictionaries
        rst_rels = [_dict['rel'] for _dict in li_rst]
        rst_ns = [_dict['ns'] for _dict in li_rst]
        rst_pos = [_dict['pos'] for _dict in li_rst]

        # sorting the order to be left to right in binary tree
        sorted_order = [i[0] for i in sorted(enumerate(rst_pos), key=lambda x: (
            RSTTokenizer.edukp_pos_sort_function(x[1]), x[1]), 
            ) ]

        rst_rels = [rst_rels[idx] for idx in sorted_order]
        rst_ns = [rst_ns[idx] for idx in sorted_order]
        rst_pos = [rst_pos[idx] for idx in sorted_order]
        # endregion

        # Key phrase scores
        li_pos_kp = json.loads(datum['li_pos_kp'] )
        if len(li_pos_kp)>0:
            li_pos_kp = sorted( li_pos_kp, key=lambda pos_kp: RSTTokenizer.edukp_pos_sort_function(int(pos_kp[0])) )
            li_kprstpos, li_kp = zip(*li_pos_kp)
            li_kprstpos = tuple(int(pos) for pos in li_kprstpos)
            #Spaces already included in this cmv dataset
            li_kp = [kp.lstrip(" ") for kp in li_kp]
        else:
            li_kp = []
            li_kprstpos = []

        # Utterance
        utterance = ujson.loads(datum['txt_preproc'])
        
        #title        
        title = ujson.loads(datum['prompt']).lstrip( string.punctuation )


        #pos and edus
        dict_pos_edu = json.loads(datum['dict_pos_edu'])   

        return rst_rels, rst_ns, rst_pos, li_kp, li_kprstpos, utterance, dict_pos_edu, title

class SizedOrderedBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

        Args:
            sampler (Sampler or Iterable): Base sampler. Can be any iterable object
            batch_size (int): Size of mini-batch.
            drop_last (bool): If ``True``, the sampler will drop the last batch if
                its size would be less than ``batch_size``

        Example:
            >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
            >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_title = hasattr( self.data_source , "np_title_lens")

        self.prepare_ds()
               
    
    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.li_chunked_idxs)


    def __len__(self) -> int:
        return len(self.li_chunked_idxs)
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if self.shuffle:
            self.prepare_ds()       
    
    def prepare_ds(self):
        np_txt_lens = self.data_source.np_textlens
        np_rst_lens = self.data_source.np_rstlens
        np_key_phrase_lens = self.data_source.np_keyphrase_lens
        if self.has_title:
            np_title_lens = self.data_source.np_title_lens

       
        # Sorting and (maybe) shuffling
        tuple_factors = (np_txt_lens,)
        if self.has_title:
            tuple_factors = (np_title_lens, ) + tuple_factors

        tuple_factors = (np_rst_lens, np_key_phrase_lens,  ) + tuple_factors

        if self.shuffle:
            random_idxs = np.random.random( np_txt_lens.size )
            tuple_factors = (random_idxs, )+ tuple_factors
        np_ordered_idxs = np.lexsort(tuple_factors)


        # Handing drop last
        if self.drop_last:
            rem_records = np_txt_lens.size % self.batch_size
            li_ordered_idxs = np_ordered_idxs.tolist()
            for idx in range(rem_records):
                li_ordered_idxs.pop( random.randint(0,len(li_ordered_idxs)-1) )
            np_ordered_idxs = np.array(li_ordered_idxs)

        # We Randomly re-arrange them in batches of batch size
        self.li_chunked_idxs = [np_ordered_idxs[idx:idx+self.batch_size]
                            for idx in range(0, np_ordered_idxs.size - self.batch_size, self.batch_size)]

        if self.shuffle:
            random.shuffle(self.li_chunked_idxs)

        # Getting max sizes for rst in each chunk
        self.li_chunk_rst_len = [
            np.take(np_rst_lens, idxs).max() for idxs in self.li_chunked_idxs]

        self.li_chunk_key_phrase_len = [
            np.take(np_key_phrase_lens, idxs).max() for idxs in self.li_chunked_idxs]
        
        if self.has_title:
            self.li_chunk_title_len = [
                np.take(np_title_lens, idxs).max() for idxs in self.li_chunked_idxs]



        # Updating Chunk sizes
        for chunk_idx, data_idxs in enumerate(self.li_chunked_idxs):
            for data_idx in data_idxs:
                
                self.data_source.rst_len[data_idx] = self.li_chunk_rst_len[chunk_idx]
                self.data_source.key_phrase_len[data_idx] = self.li_chunk_key_phrase_len[chunk_idx]
                
                if self.has_title:
                    self.data_source.title_len[data_idx] = self.li_chunk_title_len[chunk_idx]

class SizedOrderedDistributedBatchSampler(Sampler[List[int]]):
    r"""
        Adapted so that each process takes sequential indices as opposed to strides across indices
    """
    r"""Sampler that restricts data loading to a subset of the dataset.
        It is especially useful in conjunction with
        :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
        process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
        :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
        original dataset that is exclusive to it.
        .. note::
            Dataset is assumed to be of constant size.
        Args:
            dataset: Dataset used for sampling.
            num_replicas (int, optional): Number of processes participating in
                distributed training. By default, :attr:`world_size` is retrieved from the
                current distributed group.
            rank (int, optional): Rank of the current process within :attr:`num_replicas`.
                By default, :attr:`rank` is retrieved from the current distributed
                group.
            shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
                indices.
            seed (int, optional): random seed used to shuffle the sampler if
                :attr:`shuffle=True`. This number should be identical across all
                processes in the distributed group. Default: ``0``.
            drop_last (bool, optional): if ``True``, then the sampler will drop the
                tail of the data to make it evenly divisible across the number of
                replicas. If ``False``, the sampler will add extra indices to make
                the data evenly divisible across the replicas. Default: ``False``.
        .. warning::
            In distributed mode, calling the :meth:`set_epoch` method at
            the beginning of each epoch **before** creating the :class:`DataLoader` iterator
            is necessary to make shuffling work properly across multiple epochs. Otherwise,
            the same ordering will be always used.
        Example::
            >>> sampler = DistributedSampler(dataset) if is_distributed else None
            >>> loader = DataLoader(dataset, shuffle=(sampler is None),
            ...                     sampler=sampler)
            >>> for epoch in rDange(start_epoch, n_epochs):
            ...     if is_distributed:
            ...         sampler.set_epoch(epoch)
            ...     train(loader)
        """

    def __init__(self,
                data_source: Dataset, batch_size: int,
                 drop_last:bool,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 0,
                 shuffle: bool = False,
                 gpus: int = 2,
                 ) -> None:

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            # num_replicas = dist.get_world_size()
            num_replicas = gpus
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            #rank = dist.get_rank()
            rank = _get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        # normal code
        self.num_replicas = num_replicas
        self.rank = rank
        self.has_title = hasattr(data_source,"np_title_lens")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_source = data_source
        self.shuffle = shuffle

        self.seed = seed
        self.epoch = 0

        self.prepare_ds()
       
    def __iter__(self) -> Iterator[T_co]:
    
        return iter( self.li_li_chunked_idxs[self.rank] )
    
    def __len__(self) -> int:
        
        return len(self.li_li_chunked_idxs[self.rank] )

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        if self.shuffle:
            self.prepare_ds()        

    def prepare_ds(self):
        # new code
        np_txt_lens = self.data_source.np_textlens
        np_rst_lens = self.data_source.np_rstlens
        np_key_phrase_lens = self.data_source.np_keyphrase_lens
        if self.has_title:
            np_title_lens = self.data_source.np_title_lens


        # Sorting and (maybe) shuffling
        tuple_factors = (np_txt_lens,)
        if self.has_title:
            tuple_factors = (np_title_lens, ) + tuple_factors
        tuple_factors = (np_rst_lens, np_key_phrase_lens,  ) + tuple_factors

        if self.shuffle:
            random_idxs= list(range(np_txt_lens.size))
            np.array( random.Random(self.seed+self.epoch).shuffle(random_idxs) )
            tuple_factors = (random_idxs, )+ tuple_factors
        np_ordered_idxs = np.lexsort(tuple_factors)

       # Handing drop_last
        if self.drop_last:
            rem_records = np_txt_lens.size % (self.batch_size*self.num_replicas)
            li_ordered_idxs = np_ordered_idxs.tolist()
            for idx in range(rem_records):
                li_ordered_idxs.pop( random.randint(0,len(li_ordered_idxs)-1) )
            np_ordered_idxs = np.array(li_ordered_idxs)


        # We Randomly re-arrange them in batches of batch size
        li_chunked_idxs = [np_ordered_idxs[idx:idx+self.batch_size]
                           for idx in range(0, np_ordered_idxs.size-self.batch_size, self.batch_size)]

        # Divide into n sublists,
        # Each sublist at index i, contains the indices for process at rank i
        # Each sublist at index i, represents similar sized items in the dataset
        li_li_chunked_idxs = [
            [li_chunked_idxs[(self.num_replicas*idx)+_rank]
             for idx in range( len(li_chunked_idxs)//self.num_replicas ) ]
                for _rank in range(self.num_replicas)]

        # shuffle each processes subllist in the same order to optimize paralel training
        if self.shuffle:
            _ = list(zip(*li_li_chunked_idxs))
            random.Random(self.seed+self.epoch).shuffle(_)
            # unpacking into worker size length list
            li_li_chunked_idxs = list(zip(*_))

        self.li_li_chunked_idxs = li_li_chunked_idxs

        # Getting max sizes for rst and key_phrase in each chunk
        li_li_chunk_rst_len = [[np.take(np_rst_lens, idxs).max() for idxs in li_chunked_idxs]
                                    for li_chunked_idxs in li_li_chunked_idxs]
        li_li_chunk_key_phrase_len = [[
            np.take(np_key_phrase_lens, idxs).max()
            for idxs in li_chunked_idxs] for li_chunked_idxs in li_li_chunked_idxs]
        
        if self.has_title:
            li_li_chunk_title_len = [[
                np.take(np_title_lens, idxs).max()
                for idxs in li_chunked_idxs] for li_chunked_idxs in li_li_chunked_idxs]

        #Updating the max_len_rst and max_len_keyphrase 
        for (idx, (li_chunked_idxs, li_chunk_rst_len, li_chunk_key_phrase_len) ) in enumerate( zip(li_li_chunked_idxs, li_li_chunk_rst_len, li_li_chunk_key_phrase_len) ):
            # iterating through chunk_idx, data_idxs enumerate(self.li_chunked):

            for chunk_idx, data_idxs in enumerate(li_chunked_idxs):

                for data_idx in data_idxs:
                    self.data_source.rst_len[data_idx] = li_chunk_rst_len[chunk_idx]
                    self.data_source.key_phrase_len[data_idx] =  li_chunk_key_phrase_len[chunk_idx]
                    
                    if self.has_title:
                        self.data_source.title_len[data_idx] = li_li_chunk_title_len[idx][chunk_idx]   
     

def main(tparams={}, mparams={}):

    # Defining Logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.abspath(tparams['model_dir']),
        name=mparams['model_name'],
        version=tparams['version'])

    tparams['version'] = tb_logger.version

    tparams['dir_checkpoints'] = os.path.join(
        tparams['model_dir'], mparams['model_name'], f"version_{tparams['version']}", 'checkpoints')

    os.makedirs(tparams['dir_checkpoints'], exist_ok=True)

    # initiating training loop
    training_module = RSTGPT2Pair_TrainingModule.instatiate_training_module(
        tparams, mparams)
    trainer, training_module = RSTGPT2Pair_TrainingModule.instatiate_trainer(
        tparams, tb_logger, training_module)
    RSTGPT2Pair_TrainingModule.start(trainer, tparams, training_module, mparams)

if __name__ == '__main__':

    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)

    # add model specific args
    mparams = RSTGPT2Pair.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = RSTGPT2Pair_TrainingModule.parse_train_specific_args(parent_parser)

    if tparams.mode == "test":
        assert tparams.gpus in [0, 1]

    if tparams.gpus not in [0, 1]:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '65503'

    try:
        main(vars(tparams), vars(mparams))
    except Exception:
        print(traceback.format_exc())

