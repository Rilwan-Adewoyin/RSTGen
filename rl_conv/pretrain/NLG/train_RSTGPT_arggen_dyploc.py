import os

os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ['TOKENIZERS_PARALLELISM'] = "true"

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
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
import torch.distributed as dist
import ujson
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin
from pytorch_lightning.utilities.distributed import _get_rank
from torch.utils.data import Dataset
from torch.utils.data import Sampler 

from transformers.optimization import Adafactor, AdafactorSchedule, AdamW
from transformers.tokenization_utils_base import AddedToken
import bisect
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
from train_RSTGPT import RSTGPT2, RSTGPT2_Config, RSTTokenizer, RSTGPT2_TrainingModule
logger = logging.get_logger(__name__)

  
class RSTGPT2DyplocConfig(RSTGPT2_Config):
    
    def __init__(self, max_len_title=40, max_len_claim=40, **kwargs):
        kwargs['model_name'] = "RSTGPT2Dyploc"
        
        super().__init__(**kwargs)
        self.claim_title_tokens = 2
        self.vocab_size = self.vocab_size + self.claim_title_tokens
        self.max_len_title = max_len_title
        self.max_len_claim = max_len_claim
        
class RSTGPT2Dyploc(RSTGPT2):
    
    def __init__(self, config: RSTGPT2DyplocConfig):
        
        super().__init__(config)
               

    def embed(self, rst_start_token_id, rst_rel, rst_ns, rst_pos, key_phrase_ids, li_kprstpos, input_ids_utt,position_ids_keyphrase, position_ids_utt, **kwargs ):
        
        inputs_embed, position_embed = super().embed(rst_start_token_id, rst_rel, rst_ns, rst_pos, key_phrase_ids, li_kprstpos, input_ids_utt, position_ids_keyphrase, position_ids_utt)

        #appending claim and position token embed
        claim_embeds = self.transformer.wte(kwargs.get('ids_claim'))
        title_embeds = self.transformer.wte(kwargs.get('ids_title'))
        inputs_embed = torch.cat( [claim_embeds, title_embeds, inputs_embed], axis=-2 )
        
        #appending claim and position embedding
        position_embed_claim = self.transformer.wpe( kwargs.get('position_ids_claim') )
        position_embed_title = self.transformer.wpe( kwargs.get('position_ids_title') )
        position_embed = torch.cat( [position_embed_claim, position_embed_title, position_embed] , axis=1 )
        
        return inputs_embed, position_embed

    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--base_model_name',
                            default='gpt2', required=False)
        parser.add_argument('--model_name', default='RSTGPT2Dyploc', required=False)
        parser.add_argument('--max_len_utt', type=int, default=140)
        parser.add_argument('--max_len_rst', type=int, default=28)
        parser.add_argument('--max_len_key_phrase', type=int, default=40)
        parser.add_argument('--max_len_claim', type=int, default=30)
        parser.add_argument('--max_len_title', type=int, default=30)
        
        parser.add_argument('--scale_grad_by_freq', type=lambda x: bool(int(x)), default=False,
                            help="Inverse the gradients to the emebdding layers based on the occurence of each index in the minibatch ")
        parser.add_argument('--rst_tree_aligned_attention',
                            type=lambda x: bool(int(x)), default=False)
        parser.add_argument('--rst_segment_method', type=str, default='None', choices=['None','fenghirst','segbot'])

        mparams = parser.parse_known_args()[0]
        return mparams

    @classmethod
    def load_model_tokenizer(cls, model_name="RSTGPT2Dyploc", model_version=None, mparams_new={}, device="cuda:0"):

        if model_version != None:
            # load from a pretrained RSTGPT2
            checkpoint = RSTGPT2Dyploc_TrainingModule.get_ckpt_file(
                f'./models/{model_name}/version_{model_version}/checkpoints')

            mparams = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                'base_model_name', 'model_name', 'max_len_key_phrase',
                'max_len_rst', 'max_len_utt','max_len_title','max_len_claim',
                'scale_grad_by_freq', 'rst_tree_aligned_attention']}

            # overriding with new keys
            for key, value in mparams_new.items():
                mparams[key] = value

            mconfig = RSTGPT2DyplocConfig.from_pretrained(
                mparams['base_model_name'], **mparams)

            model = RSTGPT2Dyploc(mconfig)
            
            # Loading Training Module
            training_module = RSTGPT2Dyploc_TrainingModule(
                mconfig, mode='inference', model=model)
            training_module.load_state_dict(checkpoint['state_dict'])

            model = training_module.model
            tok = training_module.tokenizer

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

class RSTTokenizerDyploc(RSTTokenizer):

    claim_start_token = "<|cl|>"
    title_start_token = "<|tl|>"
    
    max_len_claim = 20
    max_len_title = 20
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.pad_token =  self.eos_token
        self.max_len_claim = kwargs.get( 'max_len_claim', self.max_len_claim )
        self.max_len_title = kwargs.get( 'max_len_title', self.max_len_title  )
        
    def encode_input(self, rst_rel, rst_ns, rst_pos, li_kp, li_kprstpos, utterance=None, utterance_prompt=None, dict_pos_edu=None, max_len_rst=None, max_len_key_phrase=None, exclude_from_output=[], device=None, claim='', title='', max_claim_len=None, max_title_len=None):
       
        encoded = super().encode_input(rst_rel, rst_ns, rst_pos, li_kp, li_kprstpos, utterance=utterance, utterance_prompt=utterance_prompt, dict_pos_edu=dict_pos_edu, max_len_rst=max_len_rst, max_len_key_phrase=max_len_key_phrase, exclude_from_output=exclude_from_output, device=device)
        
       #encoding claim
       #TODO: make it work without this over 50 restriction
        if len( claim.split(' ') ) >45:
           claim = ""  
           ids_claim = torch.tensor([],dtype=torch.long)
        else:
            claim = self.claim_start_token + claim
            
            ids_claim = self.encode(claim, add_special_tokens=False,
                    return_attention_mask=False,
                    padding= 'max_length' if max_claim_len else 'do_not_pad',
                    truncation=True,
                    max_length=max_claim_len if max_claim_len else self.max_len_claim,
                    return_tensors='pt')[0]
            
        # ids_claim = torch.full((max_claim_len,),100, dtype=torch.long)
        claim_pad = (ids_claim == self.pad_token_id).sum(dim=0)
        
        
        #encoding title
        if title != None:
            title = title.lstrip(string.punctuation+" ")
            title = self.title_start_token + title
            ids_title = self.encode(title, add_special_tokens=False,
                return_attention_mask=False,
                padding= 'max_length' if max_title_len else 'do_not_pad',
                truncation=True,
                max_length=max_title_len if max_title_len else self.max_len_title,
                return_tensors='pt')[0]
        else:
           ids_title = torch.tensor([],dtype=torch.long)
                 
            
        # ids_title = torch.full((max_title_len,),100, dtype=torch.long)
        title_pad = (ids_title == self.pad_token_id).sum(dim=0)
        
 
        encoded['ids_claim'] = ids_claim
        encoded['ids_title'] = ids_title
        
        # chaining positions
        positions_ids_claim = torch.arange(0, ids_claim.shape[0], dtype=torch.long)
        positions_ids_title= torch.arange(0, ids_title.shape[0], dtype=torch.long)
       
        encoded['position_ids_claim'] = positions_ids_claim
        encoded['position_ids_title'] = positions_ids_title
        
        claim_len =ids_claim.shape[0]
        title_len = ids_title.shape[0]
       
        #changing labels
        if encoded.get('labels') is not None:
            new_labels = positions_ids_claim.new_full( [claim_len + title_len] , -100 )
            encoded['labels'] = torch.cat( [new_labels, encoded['labels'] ] )
        
        claim_title_len = claim_len + title_len
        
        # changing attn
        encoded['attention_mask'] = torch.nn.functional.pad( encoded['attention_mask'],
                                                                (claim_title_len,0,claim_title_len,0), value=0)
            #causal over two new sections
        encoded['attention_mask'][ :claim_len , :claim_len ] = torch.tril( torch.ones_like(encoded['attention_mask'][ :claim_len , :claim_len ]) )
        encoded['attention_mask'][ claim_len:claim_len+title_len , claim_len:claim_len+title_len ] = torch.tril( torch.ones_like(encoded['attention_mask'][ claim_len:claim_len+title_len , claim_len:claim_len+title_len ]) )
            #allowing text to attend two claim and title
        encoded['attention_mask'][ -encoded['input_ids_utt'].shape[0]: , :claim_len+title_len] = 1
            
            #handling padding
        encoded['attention_mask'][ claim_len-claim_pad:claim_len, : ] = 0
        encoded['attention_mask'][ :, claim_len-claim_pad:claim_len ] = 0

        encoded['attention_mask'][ claim_title_len-title_pad:claim_title_len, : ] = 0
        encoded['attention_mask'][ :, claim_title_len-title_pad:claim_title_len ] = 0
        
        return encoded 

    @classmethod
    def from_pretrained(cls,
                        dir_tokenizer="./tokenizers/RSTGPT2Dyploc",
                        base_tokenizer_name="gpt2",
                        rst_params={},
                        **kwargs):  # max_len_rst, max_len_key_phrase, max_rst_depth, max_len_utt, max_rst_pos

        if os.path.exists(dir_tokenizer):
            tokenizer = super(RSTTokenizer, cls).from_pretrained(
                dir_tokenizer, local_files_only=True, **kwargs, **rst_params)

        else:

            at_claim_start = AddedToken(cls.claim_start_token, lstrip=False, rstrip=False) if isinstance(
                cls.claim_start_token, str) else cls.claim_start_token
            at_title_start = AddedToken(cls.title_start_token, lstrip=False, rstrip=False) if isinstance(
                cls.title_start_token, str) else cls.title_start_token
            
            additional_special_tokens = [at_claim_start, at_title_start]

            cls = super(RSTTokenizerDyploc, cls).from_pretrained(
                                                                dir_tokenizer=dir_tokenizer,
                                                                base_tokenizer_name="gpt2",
                                                                additional_special_tokens=additional_special_tokens)

            cls.save_pretrained(dir_tokenizer)
            tokenizer = cls
        
        tokenizer.claim_start_token_id = torch.full( (1,), 50259 , dtype=torch.long )
        tokenizer.title_start_token_id = torch.full( (1,), 50260 , dtype=torch.long )
        

        tokenizer.rst_start_token_id = torch.full( (1,), 50257 , dtype=torch.long )
        tokenizer.keyphrase_start_token_id = torch.full( (1,), 50258 , dtype=torch.long )        
        tokenizer.keyphrase_start_token_id_np = tokenizer.keyphrase_start_token_id.numpy()

        for k, v in kwargs.items():
            setattr(tokenizer, k, v)

        return tokenizer
    
class RSTGPT2Dyploc_TrainingModule(pl.LightningModule):

    def __init__(self,
                 mconfig,
                 batch_size=20,
                 dir_data=None,
                 accumulate_grad_batches=1,
                 max_epochs=5,
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
            self.tokenizer = RSTTokenizerDyploc.from_pretrained(f"./tokenizers/{mconfig.model_name}",
                                                         base_tokenizer_name=mconfig.base_model_name,
                                                         rst_params={name: getattr(mconfig, name) for name in ['max_len_rst',
                                                                                                               'max_len_key_phrase',
                                                                                                               'max_rst_depth',
                                                                                                               'max_len_utt', 
                                                                                                               'max_rst_pos',
                                                                                                               'max_rst_pos',
                                                                                                               'max_len_title',
                                                                                                               'max_len_claim',
                                                                                                               'rst_tree_aligned_attention'] if hasattr(mconfig, name)
                                                                     }
                                                         )
        else:
            self.tokenizer = tokenizer
 

        if model is not None:
            self.model = model
        else:
            mconfig.vocab_size = mconfig.vocab_size-4
            self.model = RSTGPT2Dyploc.from_pretrained(
                mconfig.base_model_name, config=mconfig)
            mconfig.vocab_size = mconfig.vocab_size+4
            self.model.config.vocab_size = mconfig.vocab_size
            self.model.resize_token_embeddings(self.model.config.vocab_size)
                            
        self.pad_values = {'rst_start_token': mconfig.eos_token_id,
                           'rst_rel': self.model.embed_rst_rels.padding_idx,
                           'rst_ns': self.model.embed_rst_ns.padding_idx,
                           'rst_pos': self.model.embed_rst_pos.padding_idx,

                           'key_phrase_ids': mconfig.eos_token_id,
                           'li_kprstpos': self.model.embed_rst_pos.padding_idx,

                           'position_ids_kp_utt': mconfig.n_ctx-1,
                            'position_ids_keyphrase':mconfig.n_ctx-1,
                            'position_ids_utt':mconfig.n_ctx-1,
                           'position_ids_claim':mconfig.n_ctx-1,
                           'position_ids_title':mconfig.n_ctx-1,

                           'input_ids_utt': mconfig.eos_token_id,
                            'ids_title':mconfig.eos_token_id,
                            'ids_claim':mconfig.eos_token_id,
                           'attention_mask': 0.0,

                           'labels': self.model.loss_fct.ignore_index,
                            'edu_rstpos': -1,
                            'context_rstpos': -1
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
            'ids_claim':mconfig.max_len_claim,
            'labels': mconfig.max_len_rst + mconfig.max_len_key_phrase + mconfig.max_len_utt+ mconfig.max_len_claim + mconfig.max_len_title,

            'attention_mask': mconfig.max_len_rst + mconfig.max_len_key_phrase + mconfig.max_len_utt + mconfig.max_len_claim + mconfig.max_len_title,  # axis:max_length

            'position_ids_keyphrase':mconfig.max_len_key_phrase,
            'position_ids_utt':mconfig.max_len_utt ,
            'position_ids_kp_utt': mconfig.max_len_key_phrase+mconfig.max_len_utt,
            'position_ids_claim':mconfig.max_len_claim,
            'position_ids_title':mconfig.max_len_title,
            
            'edu_rstpos': mconfig.max_rst_pos // 2,
            'context_rstpos':mconfig.max_len_rst + mconfig.max_len_key_phrase }

        self.tokenizer.pad_maxlens = self.pad_maxlens
        
        self.model.tokenizer = self.tokenizer

        if self.mode in ['finetune', 'train_cont', 'test']:
            self.dir_data = utils.get_path(dir_data)
            self.accumulate_grad_batches = accumulate_grad_batches
            self.tag = tag

            self.dg = DataLoaderGenerator(self.dir_data,  self.batch_size, self.tokenizer,
                                 workers=self.workers, mode=self.mode, gpus=self.gpus,
                                 pad_maxlens=self.pad_maxlens, pad_values=self.pad_values,
                                 batching_style=self.batching_style
                                 )

            if self.mode == "test":
                self.create_data_loaders(['test'])
            else:
                self.create_data_loaders(['train', 'val', 'inference'] )
                self.inference_samples = list(islice(self.inference_dl, 3))
                del self.inference_dl

        if self.mode in ['finetune', 'train_cont']:
            self.max_epochs = max_epochs
            self.warmup_proportion = warmup_proportion
            self.learning_rate = learning_rate

            train_params_to_save = self.return_params()
            mparams_to_save = {param: getattr(mconfig, param) for param in list(filter(
                lambda p: p not in ['self','kwargs'], list(inspect.signature(RSTGPT2_Config.__init__).parameters.keys()) ))}

            self.hparams.update({**train_params_to_save, **mparams_to_save})
            pl.core.saving.save_hparams_to_yaml(os.path.join(os.path.dirname(
                kwargs['dir_checkpoints']), "hparams.yaml"), self.hparams)
            
            #Freezing RST related paremeters
            
            
        if self.mode in ['inference']:
            self.eval()
            self.freeze()
         
    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data', default="./dataset_cmv/dyploc_rst",
                            help="Relative directory path for datafiles")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('--max_epochs', default=50, type=int)
        parser.add_argument('--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('--batch_size', default=20, type=int)
        parser.add_argument('--batching_style', default='effecient', type=str, choices=['effecient','standard'])
        parser.add_argument('--finetune_version', type=int, default=6 )
        parser.add_argument('--learning_rate', default=1e-4, type=float)
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
            
            mconfig = RSTGPT2DyplocConfig.from_pretrained(mparams['base_model_name'], **mparams)
            mconfig.vocab_size = mconfig.vocab_size-2 
            model = RSTGPT2Dyploc(mconfig)
            model.config.vocab_size += 2 
            pytorch_state_dict = { k[k.find('.')+1:]:v for k,v in checkpoint['state_dict'].items() }
            model.load_state_dict( pytorch_state_dict )
            
                
            tokenizer = RSTTokenizerDyploc.from_pretrained(**mparams)
            model.resize_token_embeddings(model.config.vocab_size)
            # set initiation value of new token to that of 
            # with torch.no_grad():
            #     model.transformer.wte.weight[ -2:, : ] = model.transformer.wte.weight[ -5:-4, : ]

            training_module = RSTGPT2Dyploc_TrainingModule(model.config, **tparams, model=model, tokenizer=tokenizer)

        elif tparams['mode'] in ["train_cont", "inference"]:

            checkpoint = RSTGPT2Dyploc_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            # restore/update param files from the checkpoint
            if "hyper_parameters" in checkpoint.keys() and tparams['override'] == False:
                tparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                     'learning_rate', 'precision', 'splits', 'tag']})

                mparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name', 'model_name', 'max_len_utt','max_len_rst','max_len_key_phrase',
                    'max_len_claim','max_len_title',
                    'scale_grad_by_freq','rst_tree_aligned_attention' ]})

            else:
                print("param files not found utilsing default or user entered params\n")

            mconfig = RSTGPT2DyplocConfig.from_pretrained(mparams['base_model_name'], **mparams)
            

            # Restore/update Training Module
            model = RSTGPT2Dyploc(mconfig)
            pytorch_state_dict = { k[k.find('.')+1:]:v for k,v in checkpoint['state_dict'].items() }
            model.load_state_dict( pytorch_state_dict )
            tokenizer = RSTTokenizerDyploc.from_pretrained(**mparams)            
            training_module = RSTGPT2Dyploc_TrainingModule(mconfig, **tparams, model=model, tokenizer=tokenizer)
            

        elif tparams['mode'] in ["test"]:

            checkpoint = RSTGPT2Dyploc_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            # restore/update param files from the checkpoint
            try:
                tparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                    'learning_rate', 'precision']})

                mparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name', 'model_name', 'max_len_utt','max_len_rst','max_len_key_phrase']})
            except KeyError:
                pass

            # Restore/update Training Module
            training_module = RSTGPT2Dyploc_TrainingModule(
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
                                              mode='min', dirpath=dir_checkpoints,
                                              filename='{epoch:03d}_{val_loss:.5f}')
        
        rst_freeze = RSTFreezingCallBack()

        checkpoint_callback._save_model = types.MethodType(
            mpatch_save_model(checkpoint_callback._save_model), checkpoint_callback) 

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience = 8,       
            verbose=False,
            mode='min'
        )

        callbacks.append(checkpoint_callback)
        callbacks.append(early_stop_callback)
        # callbacks.append(rst_freeze)

        if tparams['gpus'] in [0, 1]:
            trainer_vars = {}
        else:

            trainer_vars = {    'accelerator': 'ddp',
                            # 'plugins': DeepSpeedPlugin(stage=1, 
                            #                             contiguous_gradients=True,
                            #                              ) 
                            'plugins' : DDPPlugin(find_unused_parameters=True)
                            }

        if tparams['mode'] in ["finetune"]:

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    default_root_dir=tparams['dir_checkpoints'],
                                                    logger=tb_logger,
                                                    reload_dataloaders_every_n_epochs=1,
                                                    
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks,
                                                    replace_sampler_ddp=False,
                                                    num_sanity_val_steps=0,
                                                    val_check_interval=0.25,
                                                    **trainer_vars,
                                                    )

        elif tparams['mode'] in ["train_cont", "inference"]:

            # restoring checkpoint
            checkpoint = RSTGPT2Dyploc_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])
           
            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks, 
                                                    reload_dataloaders_every_n_epochs=1,
                                                    num_sanity_val_steps=0,
                                                    replace_sampler_ddp=False,
                                                    **trainer_vars,
                                                    val_check_interval=0.25
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
            checkpoint = RSTGPT2Dyploc_TrainingModule.get_ckpt_file(
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

            if torch.cuda.is_available():
                checkpoint = torch.load(best_ckpt_path, map_location='cpu')

            else:
                checkpoint = torch.load(best_ckpt_path, map_location='cpu')
        else:
            raise NotImplementedError

        return checkpoint

    @staticmethod
    def start(trainer, tparams, training_module, mparams):

        if tparams['mode'] in ['finetune', 'train_cont']:
            trainer.fit(training_module)

        if tparams['mode'] in ["test"]:

            checkpoint = RSTGPT2Dyploc_TrainingModule.get_ckpt_file(
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

        elif tparams['mode'] in ['infernece']:
            training_module.eval()
            training_module.freeze()
            raise NotImplementedError

    def forward(self, input_):
        with torch.cuda.amp.autocast(enabled=True):
            return self.model(**input_, return_dict=True)

    #region
    def step(self, batch, step_name):

        model_output = self.forward(batch)
        output = {}
        
        if step_name == 'train':
            output["loss"] = model_output.loss
            # self.log( "loss", model_output.loss, sync_dist=True)
            self.log( "loss", model_output.loss, sync_dist=False)


        else:
            loss_key = f"{step_name}_loss"
            # self.log( loss_key, model_output.loss, sync_dist=True)
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
    #endregion
    
    def epoch_end_log(self, outputs, step_name):

        if step_name == "train":
            pass
        else:
            loss = torch.stack([x[f"{step_name}_loss"]for x in outputs]).mean()
            
            self.log(f"{step_name}_loss", loss, logger=True, prog_bar=True)
        
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
                                                'orig_claim', 'orig_title'])

                    rst_rels = encoded_input.pop('orig_rst_rels')
                    rst_ns = encoded_input.pop('orig_rst_ns')
                    rst_pos = encoded_input.pop('orig_rst_pos')

                    keyphrase = encoded_input.pop('orig_key_phrase')
                    utterance = encoded_input.pop('orig_utt')
                    dict_pos_edu = encoded_input.pop('orig_dict_pos_edu')

                    orig_li_kprstpos = encoded_input.pop('orig_li_kprstpos')
                    orig_claim =  encoded_input.pop('orig_claim', None)
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
                        "orig_claim": json.dumps(orig_claim),
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
                encoded_input.pop('orig_claim', None)
                encoded_input.pop('orig_title', None)
                
                # encoded_input.pop('labels', None)

                generation_params = copy.deepcopy(self.model.generation_params)
                generation_params['max_time'] = 45
                bad_words = ["<|rst|>", "<|kp|>" ]
        
                bad_words_ids = [self.tokenizer.encode(
                    bad_word) for bad_word in bad_words]

                bad_words_ids = bad_words_ids 
                
                generation_params['bad_words_ids'] = bad_words_ids
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
                    'orig_claim':'',
                    'orig_title':'',

                }

                pd.DataFrame.from_records([datum]).to_csv(fp, index=False, mode='a', header=False)
                # Saving to file
        
        else:
            pass
        
    def create_data_loaders(self, modes ):
        if 'train' in modes:
            self.train_dl = self.dg.prepare_dataloader(
                split_name='train', custom_dset_class= SingleDataset)
            self.train_dl_used = False
        if 'val' in modes:
            self.val_dl = self.dg.prepare_dataloader(
                split_name='val', custom_dset_class= SingleDataset)
        if 'test' in modes:
            self.test_dl = self.dg.prepare_dataloader(
                split_name='test',  custom_dset_class= SingleDataset)
        if 'inference' in modes:
            self.inference_dl = self.dg.prepare_dataloader(
                split_name='inference',  custom_dset_class= SingleDataset)

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

        ds_size = len(self.train_dl) // self.gpus
        steps = (ds_size * self.max_epochs) // (self.accumulate_grad_batches)
        return steps

    def configure_optimizers(self):
        
        
        self.freeze_specific_modules( [ self.model.embed_rst_rels, self.model.embed_rst_ns, self.model.embed_rst_pos] )

        parameters = filter( lambda p: p.requires_grad, self.model.parameters() )
        
        optimizer = Adafactor(parameters, scale_parameter=True, 
                        relative_step=True, warmup_init=True, lr=None,
                        weight_decay=0.01)

        lr_scheduler = AdafactorSchedule(optimizer)

        if self.mode == "train_cont":
            # restore the optimizers
            checkpoint = self.get_ckpt_file(self.dir_checkpoints)
            optimizer_states = checkpoint['optimizer_states']
            optimizer.load_state_dict(optimizer_states[0])
   
            # restore the lr schedulers
            lr_scheduler_states = checkpoint['lr_schedulers']
            lr_scheduler.load_state_dict(lr_scheduler_states[0])

        return { 'optimizer':optimizer, "lr_scheduler": lr_scheduler, "interval": "step", "monitor": "val_loss"}
    
    def freeze_specific_modules(self, modules, train_bn=True):

        modules = BaseFinetuning.flatten_modules(modules)

        for mod in modules:
            if isinstance(mod, _BatchNorm) and train_bn:
                BaseFinetuning.make_trainable(mod)
            else:
                # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
                for param in mod.parameters(recurse=False):
                    param.requires_grad = False
        
    
    def return_params(self):
        params = {}
        keys = ['batch_size', 'accumulate_grad_batches', 'learning_rate', 'max_epochs', 'dir_data'
                'tag']

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

        elif split_name == 'val':
            fn = filter_fns(glob.glob(  os.path.join( self.dir_data,"val","*") ))
            shuffle = False
            inference = False
            bs = self.batch_size
            sampler = True

        elif split_name == 'test':
            fn = filter_fns(glob.glob(  os.path.join( self.dir_data,"test","*") ))
            shuffle = False
            bs = self.batch_size
            inference = False
            sampler = True

        elif split_name == 'inference':
            fn = filter_fns(glob.glob(  os.path.join( self.dir_data,"test","*") ))
            shuffle = False
            bs = 1
            sampler = None
            inference = True

        if 'custom_dset_class' in kwargs:
            ds = kwargs.get('custom_dset_class')(fn, copy.deepcopy( self.tokenizer) ,inference)
        else:
            ds = SingleDataset(fn, copy.deepcopy(self.tokenizer), inference )
            
        sampler = SizedOrdered_Sampler(ds, bs, shuffle) if sampler else sampler

        def collate_fn(
                batch): return self.tokenizer.default_collate_pad(batch)
                    
        dataloader = torch.utils.data.DataLoader(ds, 
                                                batch_size= bs ,
                                                 num_workers=self.workers, 
                                                 sampler = sampler,
                                                 pin_memory=True,
                                                #  collate_fn=self.tokenizer.default_collate_pad,
                                                 collate_fn=collate_fn)
                                                #  timeout=30
                                                

                                                 
        return dataloader

class SizedOrdered_Sampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.
    #TODO; add this to pytorch. Sampler to sort nlp datasets by size
    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, batch_size, shuffle) -> None:
        self.data_source = data_source
        self.batch_size = batch_size


        np_txt_lens = self.data_source.np_textlens
        np_rst_lens = self.data_source.np_rstlens
        np_key_phrase_lens = self.data_source.np_keyphrase_lens
        np_claim_lens = self.data_source.np_claim_lens
        np_title_lens = self.data_source.np_title_lens

        # Indices are sorted in order of 1.tokenized txt length, key_phrase_length then rst length
        random_idxs = np.random.random( np_txt_lens.size )
        
        np_ordered_lens = np.lexsort(
            (random_idxs, np_rst_lens, np_key_phrase_lens, np_claim_lens+np_title_lens, np_txt_lens))
        # We Randomly re-arrange them in batches of batch size

        li_chunked_lens = [np_ordered_lens[idx:idx+batch_size]
                            for idx in range(0, np_ordered_lens.size - batch_size, batch_size)]

        if shuffle:
            random.shuffle(li_chunked_lens)

        # Getting max sizes for rst in each chunk
        self.li_chunk_rst_len = [
            np.take(np_rst_lens, idxs).max() for idxs in li_chunked_lens]

        self.li_chunk_key_phrase_len = [
            np.take(np_key_phrase_lens, idxs).max() for idxs in li_chunked_lens]
        
        self.li_chunk_claim_len = [
            np.take(np_claim_lens, idxs).max() for idxs in li_chunked_lens]

        self.li_chunk_title_len = [
            np.take(np_title_lens, idxs).max() for idxs in li_chunked_lens]

        self.li_chunked_ordered_lens = np.concatenate(
                li_chunked_lens).tolist()

        # iterating through chunk_idx, data_idxs enumerate(self.li_chunked):
        for chunk_idx, data_idxs in enumerate(li_chunked_lens):
            
            rst_len = self.li_chunk_rst_len[chunk_idx]
            key_phrase_len = self.li_chunk_key_phrase_len[chunk_idx]
            claim_len = self.li_chunk_claim_len[chunk_idx]
            title_len = self.li_chunk_title_len[chunk_idx]
            
            for data_idx in data_idxs:
                self.data_source.rst_len[data_idx] = rst_len
                self.data_source.key_phrase_len[data_idx] = key_phrase_len
                self.data_source.claim_len[data_idx] = claim_len
                self.data_source.title_len[data_idx] = title_len
                
    def __iter__(self):
        return iter(self.li_chunked_ordered_lens)

    def __len__(self) -> int:
        return len(self.data_source)
    
class SizedOrdered_DistributedSampler(Sampler[T_co]):
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

    def __init__(self, dataset: Dataset, batch_size: int,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 0,
                 shuffle: bool = False,
                 gpus: int = 2) -> None:

        self.batch_size = batch_size

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            #num_replicas = dist.get_world_size()
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
        self.epoch = 0

        # self.num_samples
        #self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

        # new code
        #self.dataset = dataset
        self.data_source = dataset
        np_txt_lens = np.concatenate(
            [ds.np_textlens for ds in self.data_source.datasets]).flatten()
        np_rst_lens = np.concatenate(
            [ds.np_rstlens for ds in self.data_source.datasets]).flatten()
        np_key_phrase_lens = np.concatenate(
            [ds.np_keyphrase_lens for ds in self.data_source.datasets]).flatten()
        np_claim_lens = np.concatenate(
            [ds.np_claim_lens for ds in self.data_source.datasets]).flatten()
        np_title_lens = np.concatenate(
            [ds.np_title_lens for ds in self.data_source.datasets]).flatten()
        
        # Indices are sorted in order of the text lens of records in the datasets
        random_idxs = np.random.random( np_title_lens.size )
        np_ordered_lens = np.lexsort(
            (random_idxs, np_rst_lens, np_key_phrase_lens, np_claim_lens+np_title_lens , np_txt_lens))

        # We Randomly re-arrange them in batches of batch size
        li_chunked_lens = [np_ordered_lens[idx:idx+batch_size]
                           for idx in range(0, np_ordered_lens.size-batch_size, batch_size)]

        # Divide into n sublists,
        # Each sublist at index i, contains the indices for process at rank i
        # Each sublist at index i, is a list non flatten indices. Each index represents items in the dataset

        li_li_chunked_lens = [
            [li_chunked_lens[(self.num_replicas*idx)+_rank]
             for idx in range(len(li_chunked_lens)//self.num_replicas)]
            for _rank in range(self.num_replicas)]

        # shuffle each processes subllist in the same order to optimize paralel training
        _ = list(zip(*li_li_chunked_lens))

        if shuffle:
            random.shuffle(_)

        # unpacking into worker size length list
        li_li_chunked_lens = list(zip(*_))

        # Getting max sizes for rst and key_phrase in each chunk
        self.li_li_chunk_rst_len = [[np.take(np_rst_lens, idxs).max() for idxs in li_chunked_lens]
                                    for li_chunked_lens in li_li_chunked_lens]
        self.li_li_chunk_key_phrase_len = [[np.take(np_key_phrase_lens, idxs).max()
            for idxs in li_chunked_lens] for li_chunked_lens in li_li_chunked_lens]

        self.li_li_chunk_claim_len = [[np.take(np_claim_lens, idxs).max()
            for idxs in li_chunked_lens] for li_chunked_lens in li_li_chunked_lens]

        self.li_li_chunk_title_len = [[np.take(np_title_lens, idxs).max()
            for idxs in li_chunked_lens] for li_chunked_lens in li_li_chunked_lens]

        self.li_li_chunked_ordered_lens = [np.concatenate(
            li_chunked_lens).tolist() for li_chunked_lens in li_li_chunked_lens]
        

        for (li_chunked_lens, li_chunk_rst_len, li_chunk_key_phrase_len, li_chunk_claim_len, li_chunk_title_len) in zip(li_li_chunked_lens, self.li_li_chunk_rst_len, self.li_li_chunk_key_phrase_len, self.li_li_chunk_claim_len, self.li_li_chunk_title_len ):
            # iterating through chunk_idx, data_idxs enumerate(self.li_chunked):

            for chunk_idx, data_idxs in enumerate(li_chunked_lens):
                
                rst_len = li_chunk_rst_len[chunk_idx]
                key_phrase_len = li_chunk_key_phrase_len[chunk_idx]
                claim_len = self.li_chunk_claim_len[chunk_idx]
                title_len = self.li_chunk_title_len[chunk_idx]
            
                for data_idx in data_idxs:
                    dataset_idx = bisect.bisect_right(
                        self.data_source.cumulative_sizes, data_idx)

                    if dataset_idx == 0:
                        sample_idx = data_idx
                    else:
                        sample_idx = data_idx - \
                            self.data_source.cumulative_sizes[dataset_idx - 1]

                    self.data_source.datasets[dataset_idx].rst_len[sample_idx] = rst_len
                    self.data_source.datasets[dataset_idx].key_phrase_len[sample_idx] = key_phrase_len
                    self.data_source.datasets[dataset_idx].claim_len[sample_idx] = claim_len
                    self.data_source.datasets[dataset_idx].title_len[sample_idx] = title_len

    def __iter__(self) -> Iterator[T_co]:

        return iter(self.li_li_chunked_ordered_lens[self.rank])

    def __len__(self) -> int:
        if self.batch_size != -1:
            return len(self.data_source)
        else:
            return len(self.li_li_chunked_ordered_lens[0])

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
    
class SingleDataset(Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

        create a custom index which sorts the entries by their length
    """
    def __init__(self, file_path, tokenizer, inference,**kwargs):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.inference = inference
        self.data = pd.read_csv(self.fp, sep=',', header=0 )

        
        fp_cached_order = os.path.join(os.path.dirname(
            file_path), f"gpt2_dict_lens.pkl")

        # # # resetting the cached order files
        if os.path.exists( fp_cached_order):
            os.remove(fp_cached_order)

    
        if os.path.exists(fp_cached_order):
            dict_cached_order = pickle.load(open(fp_cached_order, "rb"))
            self.np_textlens = dict_cached_order['np_textlens']
            self.np_rstlens = dict_cached_order['np_rstlens']
            self.np_keyphrase_lens = dict_cached_order['np_keyphrase_lens']
            self.np_title_lens = dict_cached_order['np_title_lens']
            try:
                self.li_claim_lens = dict_cached_order['li_claim_lens']
            except KeyError as e:
                li_claims = list( map( ujson.loads , self.data.li_claim.tolist()) )
                li_claims = [ [ f"<|cl|>{claim}"  for claim in claims] if len(claims)>0 else []  for claims in li_claims ]
                
                self.li_claim_lens =  [ [ self.tokenizer.encode(claim,
                                                add_special_tokens=False, 
                                                truncation=False,
                                                padding = 'do_not_pad',
                                                return_tensors=None).__len__() for claim in claims ] for claims in li_claims ] 
                dict_cached_order['li_claim_lens'] = self.li_claim_lens
                pickle.dump(dict_cached_order, open(fp_cached_order, "wb"))

                
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
            #TODO: this was corrected so need to redo dict cache records
            li_kp = [ '<|kp|> ' + '<|kp|> '.join(li_kp) for li_kp in li_li_kp  ]
            
            self.np_keyphrase_lens = np.array( [ self.tokenizer.encode(kp, 
                                        add_special_tokens=False, 
                                        truncation=False,
                                        padding = 'do_not_pad',
                                        return_tensors=None).__len__() for kp in li_kp ] )
            

            li_title = [ f"<|tl|>{ujson.loads(title)}" for title in self.data.prompt.tolist() ]
            
            self.np_title_lens = np.array( [self.tokenizer.encode(title,
                                            truncation=False,
                                            padding = 'do_not_pad',
                                            return_tensors=None).__len__() for title in li_title] )

            #encoding length of all claims for each datum
            li_claims = list( map( ujson.loads , self.data.li_claim.tolist()) )
            li_claims = [ [ f"<|cl|>{claim}"  for claim in claims] if len(claims)>0 else []  for claims in li_claims ]
            
            self.li_claim_lens =  [ [ self.tokenizer.encode(claim,
                                            add_special_tokens=False, 
                                            truncation=False,
                                            padding = 'do_not_pad',
                                            return_tensors=None).__len__() for claim in claims ] for claims in li_claims ] 
            
            
            dict_cached_order = {'np_textlens': self.np_textlens,
                                'np_rstlens': self.np_rstlens,
                                'np_keyphrase_lens': self.np_keyphrase_lens,
                                'np_title_lens':self.np_title_lens,
                                'li_claim_lens':self.li_claim_lens }

            pickle.dump(dict_cached_order, open(fp_cached_order, "wb"))
        
        
        #randomly choosing a claim
        self.li_claim_idxs = [ random.randint(0,len(claim_lens)-1) if len(claim_lens)!=0 else -1 for claim_lens in self.li_claim_lens ] 
        self.np_claim_lens = np.array( [ self.li_claim_lens[datum_idx][claim_idx] if claim_idx>-1 else 0 for datum_idx, claim_idx in enumerate(self.li_claim_idxs) ] )

        #v2 We initialize the rst/kp lengths as the actual length of each entry
        # In the Sampler, we change the max length to that of its pre-prescribed batch
        self.rst_len = copy.deepcopy( self.np_rstlens )
        self.key_phrase_len = copy.deepcopy( self.np_keyphrase_lens )
        self.claim_len = copy.deepcopy(self.np_claim_lens)
        self.title_len = copy.deepcopy(self.np_title_lens)

        self.data = self.data.to_dict('records')

    def __len__(self):
        return len( self.data )

    def __getitem__(self, index):

        rst_rels, rst_ns, rst_pos, li_kp, li_kprstpos, utterance, dict_pos_edu, claim, title = self.getitem_extract_datum(
            index)

        if self.inference == True:

            utterance_prompt = ""

            # print(self.key_phrase_len[index], self.tokenizer.max_len_key_phrase)
            
            encoded = self.tokenizer.encode_input(rst_rel=rst_rels, rst_ns=rst_ns, rst_pos=rst_pos,
                                                  li_kp=li_kp,
                                                  li_kprstpos=li_kprstpos,
                                                  utterance_prompt=utterance_prompt,
                                                  dict_pos_edu=dict_pos_edu,
                                                  max_len_rst= min( self.rst_len[index], self.tokenizer.max_len_rst ),
                                                  max_len_key_phrase= min( self.key_phrase_len[index], self.tokenizer.max_len_key_phrase),
                                                    claim=claim, title=title,
                                                    max_claim_len=min( self.claim_len[index], self.tokenizer.max_len_claim),
                                                    max_title_len=min( self.title_len[index], self.tokenizer.max_len_title) )

            encoded['orig_rst_rels'] = rst_rels
            encoded['orig_rst_ns'] = rst_ns
            encoded['orig_rst_pos'] = rst_pos

            encoded['orig_utt'] = utterance
            encoded['orig_key_phrase'] = li_kp

            encoded['orig_dict_pos_edu'] = dict_pos_edu
            encoded['orig_li_kprstpos'] = li_kprstpos
            encoded['orig_claim'] = claim
            encoded['orig_title'] = title

        elif self.inference == False:
            encoded = self.tokenizer.encode_input(
                rst_rels, rst_ns, rst_pos,
                li_kp=li_kp,
                li_kprstpos=li_kprstpos,
                utterance=utterance,
                dict_pos_edu=dict_pos_edu,
                max_len_rst= min( self.rst_len[index], self.tokenizer.max_len_rst ),
                max_len_key_phrase= min( self.key_phrase_len[index], self.tokenizer.max_len_key_phrase),
                claim=claim, title=title,
                max_claim_len=min( self.claim_len[index], self.tokenizer.max_len_claim),
                max_title_len=min( self.title_len[index], self.tokenizer.max_len_title) )

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
            # top 3 important prhases from utterance
            li_pos_kp = sorted( li_pos_kp, key=lambda pos_kp: RSTTokenizer.edukp_pos_sort_function(int(pos_kp[0])) )

            li_kprstpos, li_kp = zip(*li_pos_kp)
            li_kprstpos = [ int(pos) for pos in li_kprstpos ]
        else:
            li_kp = []
            li_kprstpos = []

        # Utterance
        utterance = ujson.loads(datum['txt_preproc'])

        # claim
        claim = ujson.loads(datum['li_claim'])[self.li_claim_idxs[index]] if self.li_claim_idxs[index] != -1 else ""
        
        title = ujson.loads(datum['prompt']).lstrip( string.punctuation )
        
        #pos and edus
        dict_pos_edu = json.loads(datum['dict_pos_edu'])   

        return rst_rels, rst_ns, rst_pos, li_kp, li_kprstpos, utterance, dict_pos_edu, claim, title

class RSTFreezingCallBack(BaseFinetuning):
    def __init__(self):
        super().__init__()
    
    def freeze_before_training(self, pl_module: "pl.LightningModule"):
        
        # self.freeze( pl_module.model.transformer )
        self.freeze( pl_module.model.embed_rst_rels )
        self.freeze( pl_module.model.embed_rst_ns )
        self.freeze( pl_module.model.embed_rst_pos )
    

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
    training_module = RSTGPT2Dyploc_TrainingModule.instatiate_training_module(
        tparams, mparams)
    trainer, training_module = RSTGPT2Dyploc_TrainingModule.instatiate_trainer(
        tparams, tb_logger, training_module)
    RSTGPT2Dyploc_TrainingModule.start(trainer, tparams, training_module, mparams)

if __name__ == '__main__':

    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)

    # add model specific args
    mparams = RSTGPT2Dyploc.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = RSTGPT2Dyploc_TrainingModule.parse_train_specific_args(parent_parser)

    if tparams.mode == "test":
        assert tparams.gpus in [0, 1]

    if tparams.gpus not in [0, 1]:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '65502'

    try:
        main(vars(tparams), vars(mparams))
    except Exception:
        print(traceback.format_exc())

# dullduks server version 1 - No Freezing, Full RST

# CUDA_VISIBLE_DEVICES=0 python3 train_RSTGPT_arggen_dyploc.py --batch_size 26 --version 1 --precision 16 --mode finetune --workers 6 --scale_grad_by_freq 1 --max_epochs 50 --gpus 1 --tag RSTGPT2 --max_len_utt 180 --max_len_rst 28 --max_len_key_phrase 40 --tag RSTGPT2 --learning_rate 3e-4 --finetune_version 1 --max_len_claim 40 --max_len_key_phrase 40 --rst_aligned_attention 1 --rst_segment_method segbot

# CUDA_VISIBLE_DEVICES=1 python3 train_RSTGPT_arggen_dyploc.py --batch_size 12 --version 11 --workers 6 --scale_grad_by_freq 1 --tag "RSTGPT2 Dyploc with rst aligned attn" --max_len_utt 240 --max_len_rst 36 --max_len_key_phrase 64 --finetune_version 11 --max_len_claim 40 --max_len_key_phrase 40 --rst_aligned_attention 1 --rst_segment_method segbot
# CUDA_VISIBLE_DEVICES=1 python3 train_RSTGPT_arggen_dyploc.py --batch_size 12 --version 12 --workers 6 --scale_grad_by_freq 1 --tag "RSTGPT2 Dyploc with non rst aligned attn" --max_len_utt 240 --max_len_rst 36 --max_len_key_phrase 64 --finetune_version 12 --max_len_claim 40 --max_len_key_phrase 40 --rst_aligned_attention 0 --rst_segment_method segbot
