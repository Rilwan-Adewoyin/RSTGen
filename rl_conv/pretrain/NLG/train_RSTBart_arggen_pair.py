import os

os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ['TOKENIZERS_PARALLELISM'] = "true"

from collections import OrderedDict
import argparse
import bisect
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
from typing import (Any, Dict, Iterator, List, Optional, TypeVar, Tuple, Union)

import einops
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

import torch.distributed as dist
import transformers
from transformers import BartForConditionalGeneration, Adafactor, AdamW, get_cosine_schedule_with_warmup
import ujson
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin
from pytorch_lightning.utilities.distributed import _get_rank
from sklearn import preprocessing as sklp
from torch.utils.data import Sampler 
from torch.utils.data.dataset import Dataset
from transformers import (BartConfig, BartTokenizerFast)
from transformers.modeling_outputs import (BaseModelOutput, ModelOutput,
                                           Seq2SeqLMOutput, Seq2SeqModelOutput)
from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration, shift_tokens_right)
from transformers.optimization import AdafactorSchedule
from transformers.tokenization_utils_base import AddedToken
import string
import utils_nlg_v3 as utils
from utils_nlg_v3 import EmbeddingRstPos, mpatch_save_model
from train_RSTBart import RSTBart, RSTBart_Config, RSTTokenizer
from train_RSTGPT_arggen_pair import DataLoaderGenerator
import torch_optimizer as optim

T_co = TypeVar('T_co', covariant=True)


from torch.nn.utils.rnn import pad_sequence
from seg_bot_segmenter import Segmenter, Lang, PointerNetworks


class RSTBartPairConfig(RSTBart_Config):

    def __init__(self, max_len_title=22, **kwargs):
        kwargs['model_name'] = "RSTBartPair"
        
        super().__init__(**kwargs)
        self.title_tokens = 1
        self.vocab_size = self.vocab_size + self.title_tokens
        self.max_len_title = max_len_title

class RSTBartPair(RSTBart):
    
    def __init__(self, config: RSTBartPairConfig):
        
        super().__init__(config)
        # #Freeze all weights except for prefix weight,
        # for name, param in self.model.named_parameters(): 
        #     param.requires_grad = False
        
    def embed(self, rst_start_token_id, rst_rel, rst_ns, rst_pos, key_phrase_ids, li_kprstpos, position_ids_kp, **kwargs ):
        

        inputs_embed = super().embed(rst_start_token_id, rst_rel, rst_ns, rst_pos, key_phrase_ids, li_kprstpos, position_ids_kp)

        #appending position token embed
        embeds_title = self.model.encoder.embed_tokens( kwargs.get('ids_title') ) * self.model.encoder.embed_scale
                
        #appending title position embedding

        position_embeds_title = super(
                            type(self.model.encoder.embed_positions), 
                            self.model.encoder.embed_positions).forward(kwargs.get('position_ids_title') + self.model.encoder.embed_positions.offset)
        
        embeds_title = embeds_title + position_embeds_title
        
        inputs_embed = torch.cat([embeds_title, inputs_embed],axis=1)
        return inputs_embed

    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--base_model_name',
                            default='Bart', required=False)
        parser.add_argument('--model_name', default='RSTBartPair', required=False)
        parser.add_argument('--max_len_utt', type=int, default=140)
        parser.add_argument('--max_len_rst', type=int, default=30)
        parser.add_argument('--max_len_key_phrase', type=int, default=40)
        parser.add_argument('--max_len_title', type=int, default=30)
        
        parser.add_argument('--scale_grad_by_freq', type=lambda x: bool(int(x)), default=False,
                            help="Inverse the gradients to the emebdding layers based on the occurence of each index in the minibatch ")
        parser.add_argument('--rst_tree_aligned_attention',
                            type=lambda x: bool(int(x)), default=False)
        parser.add_argument('--rst_segment_method', type=str, default=None, choices=['None','fenghirst','segbot'])
        mparams = parser.parse_known_args()[0]
        return mparams

    @classmethod
    def load_model(cls, model_name="RSTBartPair", model_version=None, mparams_new={}, device="cuda:0"):

        if model_version != None:
            # load from a pretrained RSTBart
            checkpoint = RSTBartPair_TrainingModule.get_ckpt_file(
                f'./models/{model_name}/version_{model_version}/checkpoints')

            mparams = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                'base_model_name', 'model_name', 'max_len_key_phrase',
                'max_len_rst', 'max_len_utt','max_len_title',
                'scale_grad_by_freq', 'rst_tree_aligned_attention']}

            # overriding with new keys
            for key, value in mparams_new.items():
                mparams[key] = value

            mconfig = RSTBartPairConfig.from_pretrained(
                mparams['base_model_name'], **mparams)

            # Loading Training Module
            training_module = RSTBartPair_TrainingModule(
                mconfig, mode='inference')
            training_module.load_state_dict(checkpoint['state_dict'])

            model = training_module.model
            tok = training_module.RSTTokenizer

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

    def prepare_inputs_for_generation(self, decoder_input_ids, past=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, decoder_cross_attention_mask=None, use_cache=None, encoder_outputs=None, decoder_context_rstpos=None, decoder_edu_rstpos=None, **kwargs):
        outputs = super().prepare_inputs_for_generation(decoder_input_ids, past=past, attention_mask=attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, decoder_cross_attention_mask=decoder_cross_attention_mask, use_cache=use_cache, encoder_outputs=encoder_outputs, decoder_context_rstpos=decoder_context_rstpos, decoder_edu_rstpos=decoder_edu_rstpos, **kwargs)

        outputs['ids_title'] = kwargs.get('ids_title')
        outputs['position_ids_title'] = kwargs.get('position_ids_title')
        
        return outputs
    
class RSTTokenizerPair(RSTTokenizer):
    
    title_start_token = "<tl>"
    max_len_title = 20

    def __init__(self, *args,  **kwargs):
        
        super().__init__(*args, **kwargs)
        self.max_len_title = kwargs.get( 'max_len_title', self.max_len_title)
        
    def encode_input(self, rst_rel, rst_ns, rst_pos, li_kp, li_kprstpos,
                     utterance=None, utterance_prompt=None, dict_pos_edu=None,
                     max_len_rst=None,
                     exclude_from_output=[], device=None, title='', max_title_len=None,
                     **kwargs):

        encoded = super().encode_input(rst_rel, rst_ns, rst_pos, li_kp, li_kprstpos,
                     utterance=utterance, utterance_prompt=utterance_prompt, dict_pos_edu=dict_pos_edu,
                     max_len_rst=max_len_rst, exclude_from_output= exclude_from_output, device=device)
        
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
        title_pad = (ids_title == self.pad_token_id).sum(dim=0)
        
        encoded['ids_title'] = ids_title
        
        #changing positions
        positions_ids_title = torch.arange(0, ids_title.shape[0], dtype=torch.long)
        encoded['position_ids_title'] =  positions_ids_title
        title_len = ids_title.shape[0]
        
        # changing attention mask
        encoded['attention_mask'] = torch.nn.functional.pad( encoded['attention_mask'], (title_len,0, title_len,0), value=0)
        
            #causal over new title section
        encoded['attention_mask'][ :title_len-title_pad , :title_len-title_pad ] = torch.ones_like(encoded['attention_mask'][ :title_len-title_pad , :title_len-title_pad ])
        
        # #handling padding
        # encoded['attention_mask'][ title_len-title_pad:title_len, : ] = 0
        # encoded['attention_mask'][ :, title_len-title_pad:title_len ] = 0

        # Changing cross attention mask
        encoded['decoder_cross_attention_mask'] = torch.nn.functional.pad( encoded['decoder_cross_attention_mask'], (title_len, 0), value=1)
        encoded['decoder_cross_attention_mask'][ :, title_len-title_pad:title_len ] = 0

        return encoded

    @classmethod
    def from_pretrained(cls,
                        dir_tokenizer="./tokenizers/RSTBartPair",
                        base_tokenizer_name="facebook/bart-base",
                        rst_params={},
                        **kwargs):  # max_len_rst, max_len_key_phrase, max_rst_depth, max_len_utt, max_rst_pos

        if os.path.exists(dir_tokenizer):
            tokenizer = super(RSTTokenizer, cls).from_pretrained(
                dir_tokenizer, local_files_only=True, **kwargs)

        else:
            at_title_start = AddedToken(cls.title_start_token, lstrip=False, rstrip=False) if isinstance(
                cls.title_start_token, str) else cls.title_start_token
            
            # additional_special_tokens = [at_rst_start, at_topic_start, at_title_start]
            additional_special_tokens = [at_title_start]

            cls = super(RSTTokenizerPair, cls).from_pretrained(
                                                                dir_tokenizer=dir_tokenizer,
                                                                base_tokenizer_name="facebook/bart-base",
                                                                additional_special_tokens=additional_special_tokens)

            cls.save_pretrained(dir_tokenizer)
            tokenizer = cls
        
        tokenizer.title_start_token_id = torch.full( (1,), 50267 , dtype=torch.long )

        tokenizer.rst_start_token_id = torch.full( (1,), 50265 , dtype=torch.long )
        tokenizer.keyphrase_start_token_id = torch.full( (1,), 50266 , dtype=torch.long )        
        tokenizer.keyphrase_start_token_id_np = tokenizer.keyphrase_start_token_id.numpy()

        for k, v in kwargs.items():
            setattr(tokenizer, k, v)

        return tokenizer

    def prepare_cross_attention_mask(self, dict_pos_edu=None, rst_pos=None, li_kprstpos=None, utt_len=None,
                                    rt_len=None, prev_mask=None, curr_edu_pos=None, context_rst_pos=None, utterance_ids=None, training=True):
        
        new_cross_attention_mask = super().prepare_cross_attention_mask(dict_pos_edu=dict_pos_edu,
                                                                        rst_pos=rst_pos, li_kprstpos=li_kprstpos, utt_len=utt_len, 
                                                                        rt_len=rt_len, prev_mask=prev_mask, curr_edu_pos=curr_edu_pos,
                                                                        context_rst_pos=context_rst_pos, utterance_ids=utterance_ids, training=training)

        # When rst_tree_aligned - During training, During Generation with prev_mask present
        if self.rst_tree_aligned_attention and (prev_mask is not None) :
            claim_title_len = prev_mask.shape[-1] - context_rst_pos.shape[1]
                        
            new_cross_attention_mask = torch.nn.functional.pad(new_cross_attention_mask, (claim_title_len ,0), value=1 )
        
        return new_cross_attention_mask
    
    
class RSTBartPair_TrainingModule(pl.LightningModule):

    def __init__(self,
                 mconfig,
                 batch_size=20,
                 dir_data=None,
                 accumulate_grad_batches=1,
                 max_epochs=25,
                 gpus=1,
                 learning_rate=1e-4,
                 warmup_proportion=0.1,
                 workers=0,
                 mode='finetune',
                 tag='',
                 batching_style='effecient',
                 optimizer = 'adafactor',
                 model =None,
                 tokenizer = None,
                 **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.gpus = gpus
        self.mode = mode
        self.workers = workers
        self.batching_style = batching_style
        
        if tokenizer  == None:
            self.RSTTokenizer = RSTTokenizerPair.from_pretrained(f"./tokenizers/{mconfig.model_name}",
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
            self.RSTTokenizer = tokenizer

        if model is not None:
            self.model = model
        else:
            raise Exception

        self.optimizer = optimizer
        
        self.pad_values = {'rst_start_token': mconfig.pad_token_id,
                           'rst_rel': self.model.embed_rst_rels.padding_idx,
                           'rst_ns': self.model.embed_rst_ns.padding_idx,
                           'rst_pos': self.model.embed_rst_pos.padding_idx,

                           'key_phrase_ids': mconfig.pad_token_id,
                           'li_kprstpos': self.model.embed_rst_pos.padding_idx,

                           'position_ids_kp': mconfig.pad_token_id,
                           'position_ids_title': mconfig.pad_token_id,

                           'ids_title': mconfig.pad_token_id,

                           'attention_mask': 0.0,

                           'labels': self.model.loss_fct.ignore_index,
                           'decoder_input_ids': mconfig.pad_token_id,
                           'decoder_cross_attention_mask': 0.0,

                            'decoder_edu_rstpos': -1,
                                
                            'decoder_context_rstpos': -1
                           }
        
        self.RSTTokenizer.pad_values = self.pad_values

        self.pad_maxlens = {
            'rst_start_token': 1,
            'rst_rel': mconfig.max_len_rst-1,
            'rst_ns': mconfig.max_len_rst-1,
            'rst_pos': mconfig.max_len_rst-1,

            'key_phrase_ids': mconfig.max_len_key_phrase,
            'li_kprstpos': mconfig.max_len_key_phrase,

            'labels': mconfig.max_len_utt if mconfig.max_len_utt else self.config.max_position_embeddings,
            'decoder_input_ids': mconfig.max_len_utt if mconfig.max_len_utt else self.config.max_position_embeddings,

            'attention_mask': mconfig.max_len_title + mconfig.max_len_rst + mconfig.max_len_key_phrase,  # axis:max_length
            'decoder_cross_attention_mask': [ mconfig.max_len_utt,  mconfig.max_len_title + mconfig.max_len_rst + mconfig.max_len_key_phrase ] , #max_lens in both 2d dimensions

            'position_ids_kp': mconfig.max_len_key_phrase,
            'position_ids_title':+mconfig.max_len_title,
                        
            'decoder_edu_rstpos': mconfig.max_rst_pos // 2,
            'decoder_context_rstpos':mconfig.max_len_rst + mconfig.max_len_key_phrase

        }
        self.RSTTokenizer.pad_maxlens = self.pad_maxlens
        
        self.model.RSTTokenizer = self.RSTTokenizer

        if self.mode in ['finetune', 'train_cont', 'test']:
            self.dir_data = utils.get_path(dir_data)
            self.accumulate_grad_batches = accumulate_grad_batches
            self.tag = tag

            self.dg = DataLoaderGenerator(self.dir_data,  self.batch_size, self.RSTTokenizer,
                                 workers=self.workers, mode=self.mode, gpus=self.gpus,
                                 pad_maxlens=self.pad_maxlens, pad_values=self.pad_values,
                                 batching_style=self.batching_style,
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
                lambda p: p not in ['self','kwargs'], list(inspect.signature(RSTBartPairConfig.__init__).parameters.keys()) ))}

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
        parser.add_argument('--max_epochs', default=8, type=int)
        parser.add_argument('--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('--batch_size', default=20, type=int)
        parser.add_argument('--batching_style', default='effecient', type=str, choices=['effecient','standard'])
        parser.add_argument('--finetune_version', type=int, default=6 )
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--workers', default=16, type=int)  
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--mode', default='finetune', type=str, choices=['finetune', 'train_cont', 'test', 'inference'])
        parser.add_argument('--version', default=None, required=False,
                            type=int, help="The Experimental Versioning for this run")
        parser.add_argument('--precision', default=16, required=False,
                            type=int, help="Precision to use", choices=[16, 32])
        parser.add_argument('--optimizer', default='adafactor', choices=['adafactor','adamw','adafactor_lr'])
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
            
            checkpoint = RSTBartPair_TrainingModule.get_ckpt_file(f"./models/RSTBart/version_{tparams['finetune_version']}/checkpoints")
            mparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                'base_model_name', 'scale_grad_by_freq','rst_tree_aligned_attention' ]})
            
            mconfig = RSTBartPairConfig.from_pretrained(mparams['base_model_name'], **mparams)
            mconfig.vocab_size = mconfig.vocab_size-1
            model = RSTBartPair(mconfig)
            model.config.vocab_size += 1
            pytorch_state_dict = { k[k.find('.')+1:]:v for k,v in checkpoint['state_dict'].items() }
            model.load_state_dict( pytorch_state_dict )
            
            tokenizer = RSTTokenizerPair.from_pretrained(**mparams)
            model.resize_token_embeddings(model.config.vocab_size)
            
            # set initiation value of new token to that of 
            with torch.no_grad():
                model.model.encoder.embed_tokens.weight[ -1:, : ] = model.model.encoder.embed_tokens.weight[ -4:-3, : ]

            training_module = RSTBartPair_TrainingModule(model.config, **tparams, model=model, tokenizer=tokenizer)

        elif tparams['mode'] in ["train_cont", "inference"]:

            checkpoint = RSTBartPair_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            # restore/update param files from the checkpoint
            if "hyper_parameters" in checkpoint.keys() and tparams['override'] == False:
                tparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                     'learning_rate', 'precision', 'splits', 'tag','optimizer']})

                mparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name', 'model_name', 'max_len_utt','max_len_rst','max_len_key_phrase',
                    'scale_grad_by_freq','rst_tree_aligned_attention' ]})

                mparams = mparams

            else:
                print("param files not found utilsing default or user entered params\n")

            mconfig = RSTBart_Config.from_pretrained(mparams['base_model_name'], **mparams)

  
            model = RSTBartPair(mconfig)
            pytorch_state_dict = { k[k.find('.')+1:]:v for k,v in checkpoint['state_dict'].items() }
            model.load_state_dict( pytorch_state_dict )
            tokenizer = RSTTokenizerPair.from_pretrained(**mparams)            
            training_module = RSTBartPair_TrainingModule(mconfig, **tparams, model=model, tokenizer=tokenizer)


        elif tparams['mode'] in ["test"]:

            checkpoint = RSTBartPair_TrainingModule.get_ckpt_file(
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
            training_module = RSTBartPair_TrainingModule(
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


        if tparams['gpus'] in [0, 1]:
            trainer_vars = {}
        else:
            # raise NotImplementedError
            trainer_vars = {    'accelerator': 'ddp',
                            # 'plugins': DeepSpeedPlugin(stage=1, 
                            #                             contiguous_gradients=True) 
                            'plugins' : DDPPlugin(find_unused_parameters=True)
                            }

        # Creating Callbacks    
        callbacks = []
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                            save_top_k=2,
                                            mode='min', dirpath=dir_checkpoints,
                                            filename='{epoch:03d}_{val_loss:.5f}')

        checkpoint_callback._save_model = types.MethodType(
            mpatch_save_model(checkpoint_callback._save_model), checkpoint_callback)  #

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience = 6,       
            verbose=False,
            mode='min'
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(early_stop_callback)
        
        if tparams['mode'] in ["finetune"]:

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    default_root_dir=tparams['dir_checkpoints'],
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks,
                                                    reload_dataloaders_every_n_epochs=1,
                                                    # replace_sampler_ddp=False,
                                                    num_sanity_val_steps=0,
                                                    val_check_interval=0.5,
                                                    **trainer_vars,
                                                    )

        elif tparams['mode'] in ["train_cont", "inference"]:
            # restoring checkpoint
            checkpoint = RSTBartPair_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            #restoring callback state
            for idx in range(len(callbacks)):
                if type(callbacks[idx]) == EarlyStopping:
                    callbacks[idx].on_load_checkpoint( checkpoint['callbacks'][type(callbacks[idx])] )

                elif type(callbacks[idx]) == ModelCheckpoint:
                    callbacks[idx].on_load_checkpoint( None, None, checkpoint['callbacks'][type(callbacks[idx])] )
                

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks, 
                                                    reload_dataloaders_every_n_epochs=1,
                                                    num_sanity_val_steps=0,
                                                    val_check_interval=0.5,
                                                    replace_sampler_ddp=False,
                                                    **trainer_vars,
                                                    )

            # load callback states
            trainer.on_load_checkpoint(checkpoint)
            
            try:
                #debugging
                trainer.current_epoch = checkpoint['epoch']
                trainer.global_step = checkpoint['global_step']


            except Exception:
                trainer.fit_loop.current_epoch = checkpoint['epoch']
                trainer.fit_loop.global_step = checkpoint['global_step']

            # restore the optimizers
            optimizer_states = checkpoint['optimizer_states']
            for optimizer, opt_state in zip(trainer.optimizers, optimizer_states):
                optimizer.load_state_dict(opt_state)

                # move optimizer to GPU 1 weight at a time
                # avoids OOM
                if trainer.root_gpu is not None:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda(trainer.root_gpu)

            # restore the lr schedulers
            lr_schedulers = checkpoint['lr_schedulers']

            for scheduler, lrs_state in zip(trainer.lr_schedulers, lr_schedulers):
                scheduler['scheduler'].load_state_dict(lrs_state)

            del checkpoint
            torch.cuda.empty_cache()

        elif tparams['mode'] in ["test"]:

            # restoring checkpoint
            checkpoint = RSTBartPair_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            training_module.load_state_dict(checkpoint['state_dict'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                                                    check_val_every_n_epoch=1,
                                                    checkpoint_callback=False,
                                                    logger=tb_logger,
                                                    log_every_n_steps=1,
                                                    precision=tparams['precision'],
                                                    )

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
                    str(root_dir), best_ckpt_path[best_ckpt_path.index('mastering-conversation'):])

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

            checkpoint = RSTBartPair_TrainingModule.get_ckpt_file(
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

    def step(self, batch, step_name):

        input_ = batch
        output = self.forward(input_)
        loss = output.loss

        loss_key = f"{step_name}_loss"
        output = {}

        if step_name == 'train':
            output["loss"] = loss

        else:
            str_loss_key = loss_key
            output[str_loss_key] = loss

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
            
            self.log(f"{step_name}_loss", loss, logger=True, prog_bar=True, sync_dist=True, on_step=False)
        
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

                    df = pd.DataFrame(columns=['epoch', 'rst_rels', 'topics', 'utterance',
                                            'dict_pos_edu', 'li_kprstpos',

                                            'rst_ns',
                                            'rst_pos',
                                            ])

                    rst_rels = encoded_input.pop('orig_rst_rels')
                    rst_ns = encoded_input.pop('orig_rst_ns')
                    rst_pos = encoded_input.pop('orig_rst_pos')

                    topics = encoded_input.pop('orig_key_phrase')
                    utterance = encoded_input.pop('orig_utt')
                    dict_pos_edu = encoded_input.pop('orig_dict_pos_edu')

                    orig_li_kprstpos = encoded_input.pop('orig_li_kprstpos')
                    orig_title =  encoded_input.pop('orig_title', None)

                    datum = {
                        'epoch': -1,

                        'rst_rels': ', '.join(rst_rels),
                        'rst_ns': ', '.join(rst_ns),
                        'rst_pos': rst_pos,

                        "topics": ', '.join(topics),
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

                generation_params = copy.deepcopy( self.model.generation_params )
                
                bad_words = ["<rst>", "<kp>", "<pad>"]
        
                bad_words_ids = [self.RSTTokenizer.encode(
                    bad_word,) for bad_word in bad_words]
                bad_words_ids = bad_words_ids + \
                    [self.RSTTokenizer.encode(bad_word) for bad_word in bad_words]
                bad_words_ids = bad_words_ids 
                
                generation_params['bad_words_ids'] = bad_words_ids
        
                generation_params['max_time'] = 45
                # generation_params['max_length'] = 150
                
                decoded_text = self.model.generate_plus(
                    encoded_input, generation_params)

                datum = {
                    'epoch': self.current_epoch,
                    'rst_rels': '',
                    'topics': '',
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
            self.train_dl = self.dg.prepare_dataloader(split_name='train', custom_dset_class= SingleDataset )
        if 'val' in modes:
            self.val_dl = self.dg.prepare_dataloader(split_name='val', custom_dset_class= SingleDataset)
        if 'test' in modes:
            self.test_dl = self.dg.prepare_dataloader(split_name='test', custom_dset_class= SingleDataset )
        if 'inference' in modes:
            self.inference_dl = self.dg.prepare_dataloader(split_name='inference', custom_dset_class= SingleDataset)

    def train_dataloader(self):

        return self.dg.prepare_dataloader(
                split_name='train')

    def val_dataloader(self):
        # return self.dg.prepare_dataloader(
        #         split_name='val')
        return self.val_dl 

    def test_dataloader(self):
        return self.test_dl

    @lru_cache()
    def total_steps(self):

        ds_size = len(self.train_dl) // self.gpus
        steps = (ds_size * self.max_epochs) // (self.accumulate_grad_batches * self.batch_size)
        return steps

    def configure_optimizers(self):

        if self.optimizer == 'adafactor':
            optimizer = Adafactor(self.model.parameters(), scale_parameter=True,
                            relative_step=True, warmup_init=True, lr=None )


            lr_scheduler = AdafactorSchedule(optimizer)

        elif self.optimizer == 'adafactor_lr':
            optimizer = Adafactor(self.model.parameters(), scale_parameter=False,
                            relative_step=False, warmup_init=False, lr=self.learning_rate )


            lr_scheduler = AdafactorSchedule(optimizer)

        
        elif self.optimizer == 'adamw':

            optimizer = torch.optim.AdamW( self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)

            lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=0.10*self.total_steps(),
                                                            num_training_steps=self.total_steps(),
                                                            num_cycles=1.5
                                                           )
            lr_scheduler = None


        return { 'optimizer':optimizer, "lr_scheduler": lr_scheduler, "interval": "step", "monitor": "val_loss"}    
    
    def return_params(self):
        params = {}
        keys = ['batch_size', 'accumulate_grad_batches', 'learning_rate', 'max_epochs', 'dir_data'
                'tag']

        params = {
            k: self.__dict__[k] for k in keys if k in self.__dict__.keys()
        }

        return params


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
            file_path), f"bart_dict_lens.pkl")

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
            li_kp = ['<kp> ' + '<kp> '.join(li_kp) for li_kp in li_li_kp  ]
            
            self.np_keyphrase_lens = np.array( [ self.tokenizer.encode(kp, 
                                        add_special_tokens=False, 
                                        truncation=False,
                                        padding = 'do_not_pad',
                                        return_tensors=None).__len__() for kp in li_kp ] )
            

            li_title = [ f"<tl>{ujson.loads(title)}" for title in self.data.prompt.tolist() ]
            
            self.np_title_lens = np.array( [self.tokenizer.encode(title,
                                            truncation=False,
                                            padding = 'do_not_pad',
                                            return_tensors=None).__len__() for title in li_title] )

            dict_cached_order = {'np_textlens': self.np_textlens,
                                'np_rstlens': self.np_rstlens,
                                'np_keyphrase_lens': self.np_keyphrase_lens,
                                'np_title_lens':self.np_title_lens}

            pickle.dump(dict_cached_order, open(fp_cached_order, "wb"))
        # length of title
                                    
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

            utterance_prompt = ' '.join(utterance.split(' ')[:2])

            encoded = self.tokenizer.encode_input(rst_rel=rst_rels, rst_ns=rst_ns, rst_pos=rst_pos,
                                                  li_kp=li_kp,
                                                  li_kprstpos=li_kprstpos,
                                                  utterance_prompt=utterance_prompt,
                                                  dict_pos_edu=dict_pos_edu,
                                                  max_len_rst= min( self.rst_len[index], self.tokenizer.max_len_rst ),
                                                    max_title_len=min( self.title_len[index], self.tokenizer.max_len_title) )

            encoded['orig_rst_rels'] = rst_rels
            encoded['orig_rst_ns'] = rst_ns
            encoded['orig_rst_pos'] = rst_pos

            encoded['orig_utt'] = utterance
            encoded['orig_key_phrase'] = li_kp

            encoded['orig_dict_pos_edu'] = dict_pos_edu
            encoded['orig_li_kprstpos'] = li_kprstpos

        elif self.inference==False:

            encoded = self.tokenizer.encode_input(
                rst_rels, rst_ns, rst_pos,
                li_kp=li_kp,
                li_kprstpos=li_kprstpos,
                utterance=utterance,
                dict_pos_edu=dict_pos_edu,
                max_len_rst= min( self.rst_len[index], self.tokenizer.max_len_rst ),
                title=title,
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
            RSTTokenizer.edukp_pos_sort_function(x[1]), x[1]), ) ]

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
            li_kprstpos = tuple(int(pos) for pos in li_kprstpos)
        else:
            li_kp = []
            li_kprstpos = []

        # Utterance
        utterance = ujson.loads(datum['txt_preproc'])
        
        title = ujson.loads(datum['prompt']).lstrip( string.punctuation )
        
        #pos and edus
        dict_pos_edu = json.loads(datum['dict_pos_edu'])   

        return rst_rels, rst_ns, rst_pos, li_kp, li_kprstpos, utterance, dict_pos_edu, title


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
    training_module = RSTBartPair_TrainingModule.instatiate_training_module(
        tparams, mparams)
    trainer, training_module = RSTBartPair_TrainingModule.instatiate_trainer(
        tparams, tb_logger, training_module)
    RSTBartPair_TrainingModule.start(trainer, tparams, training_module, mparams)

if __name__ == '__main__':

    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)

    # add model specific args
    mparams = RSTBartPair.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = RSTBartPair_TrainingModule.parse_train_specific_args(parent_parser)

    if tparams.mode == "test":
        assert tparams.gpus in [0, 1]

    if tparams.gpus not in [0, 1]:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '65502'

    try:
        main(vars(tparams), vars(mparams))
    except Exception:
        print(traceback.format_exc())

# CUDA_VISIBLE_DEVICES=1 python3 train_RSTBart.py --batch_size 32 --version 1  --precision 16 --mode finetune --workers 6 --rst_tree_aligned_attention 0 --scale_grad_by_freq 1 --max_epochs 12 --gpus 1 --max_len_utt 190 --max_len_rst 28 --max_len_key_phrase 40 --tag "RSTBart with non attention"
# CUDA_VISIBLE_DEVICES=2 python3 train_RSTBart.py --batch_size 32 --version 6  --precision 16 --mode finetune --workers 6 --rst_tree_aligned_attention 1 --scale_grad_by_freq 1 --max_epochs 12 --gpus 1 --max_len_utt 190 --max_len_rst 28 --max_len_key_phrase 40 --tag "RSTBart with normal attention"
