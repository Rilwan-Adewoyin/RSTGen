import os
from typing import Optional
import transformers

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['NCCL_SOCKET_IFNAME'] =  'lo' 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing as mp
from pytorch_lightning.utilities.distributed import _get_rank
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import Sampler
from itertools import islice
import glob
import pandas as pd
import json
from functools import lru_cache
from typing import List
import ujson
import torch.distributed as dist
from pathlib import Path
from torch.utils.data._utils.collate import default_convert, default_collate
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import CrossEntropyLoss

from sklearn import preprocessing as sklp
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pickle

import argparse
from rst_frameworks import utils
import random 
from typing import TypeVar, Iterator
T_co = TypeVar('T_co', covariant=True)
from rst_frameworks.utils import EmbeddingRstPos, mpatch_save_model, SaveModelCallBack

from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset

from transformers.optimization import Adafactor, AdafactorSchedule
from transformers.generation_beam_search import BeamHypotheses

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.core.decorators import auto_move_data
import yaml
import types

from typing import Optional, Optional, List

import string
from transformers import CTRLTokenizer, CTRLLMHeadModel, CTRLConfig
from transformers.tokenization_utils_base import AddedToken
import copy
import inspect
# New Prompt : <tl>Title \n <cl>Claim \n <cp> Concepts <te> Target Entities

class CTRL_ArgConfig(CTRLConfig):

    def __init__(self, 
                 base_model_name='ctrl',
                 model_name="ctrl_arg",
                 max_len_utt=140,
                 max_len_title=20, max_len_claim=20, 
                    max_len_concepts=20, max_len_target_entities=20,**kwargs):
                
        super().__init__(**kwargs)
        self.model_name = model_name
        self.base_model_name=base_model_name
        self.arg_token_count = 4 # <cl>, <cp>, <te>, </s>
                #TODO: add eos_token_id at the end of generation

        self.vocab_size = self.vocab_size + self.arg_token_count
        self.max_len_utt = max_len_utt
        self.max_len_title=max_len_title
        self.max_len_claim=max_len_claim
        self.max_len_concepts=max_len_concepts
        self.max_len_target_entities=max_len_target_entities
        self.max_length = self.max_len_utt+self.max_len_title+self.max_len_claim+self.max_len_concepts+self.max_len_target_entities
        
        self.eos_token = "</s>"
        self.eos_token_id = self.vocab_size - 1 #246537
        

class CTRL_ArgTokenizer(CTRLTokenizer, utils.EffeciencyMixin):

    max_len_title = 40
    max_len_claim = 40
    max_len_concepts = 40
    max_len_target_entities = 5
    max_len_utt = 140

    special_token_count = 4

    # title_start_token = "<tl>"
    claim_start_token = "<cl>"
    concepts_start_token =  "<cp>"
    target_entities_start_token = "<te>"
    eos_token = "</s>"
    new_line_token = "\n"
    new_line_id_tensor = torch.tensor( [246533], dtype=torch.long )
    
    def __init__(self, *args, **kwargs):
        
        super().__init__( *args, **kwargs)
        
        self.max_len_title = kwargs.get('max_len_title', self.max_len_title)
        self.max_len_claim = kwargs.get('max_len_claim',self.max_len_claim)
        self.max_len_concepts = kwargs.get('max_len_concepts', self.max_len_concepts)
        self.max_len_target_entities = kwargs.get('max_len_target_entities', self.max_len_target_entities )
        self.max_len_utt = kwargs.get('max_len_utt', self.max_len_utt)
        
        self.comment_start_id_tensor = self.encode("Comment: ",return_tensors='pt')[0]
        
    def encode_input(self, title, claim, concepts, target_entities, 
                     utterance=None,
                     utterance_prompt=None,
                     exclude_from_output=[], 
                     device=None):
        """

        """
        # Preprocessing
        if utterance != None:
            utterance = utterance.lstrip(string.punctuation+" ")
        if title != None:
            title = title.lstrip(string.punctuation+" ")
            

        # Encoding rst, keyphrase and utterance info
        
        title_ids = self.encode_title( title ) #TODO: Add Title: before

        claim_ids = self.encode_claim(claim)
        
        concepts_ids = self.encode_concepts(concepts)
        
        target_entities_ids =  self.encode_target_entities(target_entities)
        
        input_ids_utt =  self.encode_utterance(utterance, utterance_prompt) #Remember to add eos token to end

        input_ids = torch.cat( [ title_ids, claim_ids, concepts_ids, target_entities_ids, self.new_line_id_tensor, self.comment_start_id_tensor , input_ids_utt ], dim=-1)
        
         
        context_len = title_ids.shape[-1] + claim_ids.shape[-1] + concepts_ids.shape[-1] + target_entities_ids.shape[-1]
        context_len += self.new_line_id_tensor.numel() #+1 for new line token before utterance
        context_len += self.comment_start_id_tensor.numel() #+2 for "Commemt:" prefix before utterance 
        
        labels = torch.cat([torch.full(size=(context_len,), fill_value=-100, dtype=torch.long), input_ids_utt])
        
        output = {
                  'input_ids': input_ids,
                  'labels': labels,
                  }

        # moving to devie
        if device != None:
            for key in output:
                if output[key] != None:
                    output[key] = output[key].to(device).unsqueeze(0)

        # excluding items from output
        for key in exclude_from_output:
            output.pop(key, None)

        return output

    def encode_title(self, title ):
        
        title = "Opinion Title: " + title
        
        title_ids = self.encode(title, add_special_tokens=False,
                                        truncation=True,
                                        padding='do_not_pad',
                                        return_tensors='pt',
                                        max_length=self.max_len_title,
                                        return_special_tokens_mask=False)[0]
        return title_ids 

    def encode_claim(self, claim ):
            
        if len(claim)!=0 and len( claim.split(' ') ) <=50:
            claim = self.claim_start_token + claim
            
            claim_ids = self.encode(claim, add_special_tokens=False,
                                            truncation=True,
                                            padding='do_not_pad',
                                            return_tensors='pt',
                                            max_length=self.max_len_claim,
                                            return_special_tokens_mask=False)[0]
        else:
            claim_ids = torch.full((0,),0, dtype=torch.long )
            
        return claim_ids 

    def encode_concepts(self, concepts ):
        
        concepts = self.concepts_start_token + ', '.join(concepts)
        
        concepts_ids = self.encode(concepts, add_special_tokens=False,
                                        truncation=True,
                                        padding='do_not_pad',
                                        return_tensors='pt',
                                        max_length=self.max_len_concepts,
                                        return_special_tokens_mask=False)[0]
        return concepts_ids 

    def encode_target_entities(self, target_entities ):
                
        target_entities = self.target_entities_start_token + ', '.join(target_entities)
        
        target_entities_ids = self.encode(target_entities, add_special_tokens=False,
                                        truncation=True,
                                        padding='do_not_pad',
                                        return_tensors='pt',
                                        max_length=self.max_len_target_entities,
                                        return_special_tokens_mask=False)[0]
        return target_entities_ids 
        
    def encode_utterance(self, utterance=None, utterance_prompt=None, ):
        
        
        # Creating labels/targets
        if utterance_prompt != None:
            utterance_prompt = utterance_prompt

            utt_ids = self.encode(
                utterance_prompt,
                add_special_tokens=False,
                return_attention_mask=False,
                padding='do_not_pad',
                truncation=True,
                max_length=self.max_len_utt-1,
                return_tensors='pt')[0]
                          
        if utterance != None:
            utt_ids = utterance + self.eos_token
            utt_ids = self.encode(
                utterance,
                add_special_tokens=False,
                padding='do_not_pad',
                truncation=True,
                max_length=self.max_len_utt-1,
                return_tensors='pt',
            )[0]

                        
        return utt_ids

    @classmethod
    def from_pretrained(cls,
                        dir_tokenizer="./tokenizers/ctrl_arg",
                        base_tokenizer_name="ctrl",
                        **kwargs):  

        if os.path.exists(dir_tokenizer):
            tokenizer = super(CTRL_ArgTokenizer, cls).from_pretrained(
                dir_tokenizer, local_files_only=True, **kwargs)

        else:

            additional_special_tokens = kwargs.pop('additional_special_tokens', [])
            at_claim_start = AddedToken(cls.claim_start_token, lstrip=False, rstrip=False) if isinstance(
                cls.claim_start_token, str) else cls.claim_start_token
            at_concepts_start_token = AddedToken(cls.concepts_start_token, lstrip=False, rstrip=False) if isinstance(
                cls.concepts_start_token, str) else cls.concepts_start_token
            at_target_entities_start = AddedToken(cls.target_entities_start_token, lstrip=False, rstrip=False) if isinstance(
                cls.target_entities_start_token, str) else cls.target_entities_start_token
            at_eos_token = AddedToken(cls.eos_token, lstrip=False, rstrip=False) if isinstance (
                cls.eos_token, str) else cls.eos_token
            additional_special_tokens = [
                at_claim_start, at_concepts_start_token, at_target_entities_start, at_eos_token ] + additional_special_tokens

            cls = super(CTRL_ArgTokenizer, cls).from_pretrained(base_tokenizer_name,
                                                           additional_special_tokens=additional_special_tokens, **kwargs)

            cls.save_pretrained(dir_tokenizer)
            tokenizer = cls

        return tokenizer

class TrainingModule(pl.LightningModule):

    def __init__(self, 
                 mconfig,
                    batch_size=20, 
                    dir_data="./dataset_cmv/dyploc_ctrl", 
                    accumulate_grad_batches=1,
                    max_epochs=25,
                    gpus=1, 
                    learning_rate=1e-4,
                    workers=0,
                    mode = 'train_new',
                    tag='ctrl fine tuned on argument generation',
                    model=None,
                    tokenizer =None,
                    **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.gpus =  gpus
        self.mode = mode
        self.workers = workers

        if tokenizer  == None:
            self.tokenizer = CTRL_ArgTokenizer.from_pretrained(f"./tokenizers/{mconfig.model_name}",
                                                         base_tokenizer_name=mconfig.base_model_name,
                                                          **kwargs)
        else:
            self.tokenizer = tokenizer

        if model is not None:
            self.model = model
        else:
            self.model = TrainingModule.init_model_and_change_embedding_size(mconfig)
        
        # Adding padding information for training
        self.tokenizer.pad_values = { 'input_ids':self.model.config.eos_token_id , 'labels':-100 }
        pad_max_len = self.model.config.max_len_title + self.model.config.max_len_claim + self.model.config.max_len_concepts + self.model.config.max_len_target_entities + self.model.config.max_len_utt
        self.tokenizer.pad_maxlens = { 'input_ids': pad_max_len  , 'labels':pad_max_len }
        
                                         
        if self.mode in ['train_new','train_cont','test']:
            self.dir_data = utils.get_path(dir_data)
            self.accumulate_grad_batches = accumulate_grad_batches
            self.dg = DataLoaderGenerator(self.dir_data,  self.batch_size, self.tokenizer,
                                 workers=self.workers, mode=self.mode, gpus=self.gpus,
                                 )
            self.tag = tag
            
            if self.mode == "test":
                self.create_data_loaders(['test'])
            else:
                self.create_data_loaders(['train', 'val', 'inference'] )
                self.inference_samples = list(islice(self.inference_dl, 3))
                del self.inference_dl

        if self.mode in ['train_new','train_cont']:
            self.max_epochs = max_epochs
            self.learning_rate = learning_rate
            train_params_to_save = self.return_params()
            mparams_to_save = {param: getattr(mconfig, param) for param in list(filter(
                lambda p: p not in ['self','kwargs'], list(inspect.signature(CTRL_ArgConfig.__init__).parameters.keys()) ))}

            self.hparams.update({**train_params_to_save, **mparams_to_save})
            pl.core.saving.save_hparams_to_yaml(os.path.join(os.path.dirname(
                kwargs['dir_checkpoints']), "hparams.yaml"), self.hparams)

            #Debugging:Freeze embedding layers
            #Consider only finetuning the new embedding layers
            # with torch.no_grad():
                # self.model.transformer.embedding.requires_grad = False
                # self.model.lm_head.transformer.requries_grad = False

        if self.mode in ['inference']:
            self.eval() 
            self.freeze() 

    @staticmethod
    def init_model_and_change_embedding_size(mconfig):
        #This assumes
        mconfig.vocab_size = mconfig.vocab_size- mconfig.arg_token_count
        model = CTRLLMHeadModel.from_pretrained(mconfig.base_model_name, config=mconfig)
        mconfig.vocab_size = mconfig.vocab_size + mconfig.arg_token_count
        model.config.vocab_size = mconfig.vocab_size
        model.resize_token_embeddings(model.config.vocab_size)
        return model

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data', default="./dataset_cmv/dyploc_ctrl", help="Relative directory path for datafiles")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('--max_epochs', default=6, type=int)
        parser.add_argument('--accumulate_grad_batches', default=10, type=int)
        parser.add_argument('-b','--batch_size', default=1, type=int)
        parser.add_argument('--learning_rate', default=2e-4, type=float, required=False)
        parser.add_argument('--workers', default=6, type=int) #TODO: change to 6
        parser.add_argument('--gpus', default=2, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','train_cont','test','inference'])
        parser.add_argument('--version', default=1,required=False, type=int, help="The Experimental Versioning for this run" )
        parser.add_argument('--precision', default=16, required=False, type=int, help="Precision to use", choices=[16,32] )
        parser.add_argument('--override', default=False, required=False, type= lambda val: bool(int(val)), choices=[True,False] )
        parser.add_argument('--tag',default='ctrl',required=False, type=str)
        tparams = parser.parse_known_args()[0]

        return tparams
    
    @staticmethod
    def instatiate_training_module( tparams=None, mparams=None ):
        """Create training module

        Args:
            tparams ([type]): [description]
        """
        #pl.seed_everything(10)

        if tparams['mode'] in ["train_new"]:
            
            mconfig = CTRL_ArgConfig.from_pretrained(mparams['base_model_name'], **mparams)
            model = TrainingModule.init_model_and_change_embedding_size(mconfig)
            tokenizer = CTRL_ArgTokenizer.from_pretrained(f"./tokenizers/{mconfig.model_name}",
                                                         base_tokenizer_name=mconfig.base_model_name,
                                                         **mparams)
            training_module = TrainingModule( mconfig, model=model, tokenizer=tokenizer, **tparams )
            
        elif tparams['mode'] in ["train_cont", "inference"]:
            
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            if "hyper_parameters" in checkpoint.keys() and tparams['override'] == False:
                tparams.update ( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'batch_size', 'precision','tag']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'model_name','max_len_utt','max_len_title','max_len_claim','max_len_concepts','max_len_target_entities']} )
                
            else:
                print("param files not found utilsing default or user entered params\n")
                
            #Restore/update Training Module
            mconfig = CTRL_ArgConfig.from_pretrained(mparams['base_model_name'], **mparams)
            model = CTRLLMHeadModel(mconfig)
            pytorch_state_dict = { k[k.find('.')+1:]:v for k,v in checkpoint['state_dict'].items() }
            model.load_state_dict( pytorch_state_dict )
            tokenizer = CTRL_ArgTokenizer.from_pretrained(f"./tokenizers/{mconfig.model_name}",base_tokenizer_name=mconfig.base_model_name)  
            training_module = TrainingModule( mconfig, model=model, tokenizer=tokenizer, **tparams )

        elif tparams['mode'] in ["test"]:
            
            raise NotImplementedError
        
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            #restore/update param files from the checkpoint
            try:
                tparams.update ( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                     'precision','optimizer_type']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name','model_name','max_len_utt']} )
            except KeyError:
                pass
            
            
            #Restore/update Training Module
            training_module = TrainingModule(**tparams, **mparams)
            training_module.load_state_dict(checkpoint['state_dict'])

        else:
            raise ValueError("tparams['mode'] must be in range [train_new, train_cont, test, inference]")

        return training_module

    @staticmethod
    def instatiate_trainer( tparams, tb_logger, training_module):
        """[summary]

            Creates The Trainer and callbacks
        """
        dir_checkpoints = tparams['dir_checkpoints']
        
        if tparams['gpus'] in [0, 1]:
            trainer_vars = {}
        else:

            trainer_vars = {'accelerator': 'ddp',
                            # 'plugins': DeepSpeedPlugin(stage=1,
                            #                             contiguous_gradients=True,
                            #                              )
                            'plugins': DDPPlugin(find_unused_parameters=False)
                            }
        
        # Creating Callbacks
        callbacks = []        
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, 
                                                mode='min',
                                                dirpath=dir_checkpoints, 
                                                filename='{epoch:03d}_{val_loss:.5f}')
        
        checkpoint_callback._save_model  = types.MethodType(utils.monkey_save_model,checkpoint_callback)
        
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
            patience=15,
            verbose=False,
            mode='min'
        )
        save_model_callback = SaveModelCallBack()
        callbacks.append(checkpoint_callback)
        callbacks.append(early_stop_callback)
        # callbacks.append(save_model_callback)
        
        
        if tparams['mode'] in ["train_new"]:
                        
            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    default_root_dir=tparams['dir_checkpoints'],
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks,
                                                    val_check_interval=0.1,
                                                    reload_dataloaders_every_n_epochs=1,
                                                    num_sanity_val_steps=2,
                                                    replace_sampler_ddp=False,
                                                    **trainer_vars)

        elif tparams['mode'] in ["train_cont","inference"]:

            # restoring checkpoint
            checkpoint = TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            # restoring callback state
            for idx in range(len(callbacks)):
                if type(callbacks[idx]) == EarlyStopping:
                    callbacks[idx].on_load_checkpoint(
                        checkpoint['callbacks'][type(callbacks[idx])])

                elif type(callbacks[idx]) == ModelCheckpoint:
                    callbacks[idx].on_load_checkpoint(
                        None, None, checkpoint['callbacks'][type(callbacks[idx])])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks,
                                                    val_check_interval=0.05,
                                                    limit_val_batches=0.25,
                                                    reload_dataloaders_every_n_epochs=1,
                                                    num_sanity_val_steps=2,
                                                    replace_sampler_ddp=False,
                                                    **trainer_vars,
                                                    )
            trainer.scaler.load_state_dict(checkpoint['native_amp_scaling_state'])

            # load callback states
            trainer.on_load_checkpoint(checkpoint)

            try:
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
            raise NotImplementedError
        
            #restoring checkpoint             
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            training_module.load_state_dict(checkpoint['state_dict'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace( **tparams),
                    progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                    check_val_every_n_epoch=1,
                    checkpoint_callback=False,
                    logger=tb_logger,
                    log_every_n_steps=1,   
                    precision=tparams['precision'],
                    track_grad_norm=2,
                    )

            # load callback states
            trainer.on_load_checkpoint(checkpoint)
           

        return trainer,training_module
    
    @staticmethod
    def get_ckpt_file(_dir_checkpoint,mode='best'):
        if mode=='best':
            checkpoint_yaml_file = os.path.join( _dir_checkpoint,"best_k_models.yaml" )
            scores_dict = yaml.load(open(checkpoint_yaml_file, "r"), Loader=yaml.FullLoader)
            best_ckpt_path = min(scores_dict, key=scores_dict.get)
            
            if os.path.exists(best_ckpt_path) == False:
                root_dir = Path(__file__).resolve().parents[4]
                best_ckpt_path = os.path.join( str(root_dir), best_ckpt_path[ best_ckpt_path.index('mastering-conversation'): ] )

            checkpoint = torch.load(best_ckpt_path, map_location='cpu' )  
          
        else:
            raise NotImplementedError
        
        return checkpoint
        
    @staticmethod
    def start(trainer, tparams,training_module, mparams ):
        
        if tparams['mode'] in ['train_new','train_cont']:    
            trainer.fit(training_module )
        
        if tparams['mode'] in ["test"]:
            
            raise NotImplementedError
        
            #trainer.on_load_checkpoint(checkpoint)
            training_module.eval()
            training_module.freeze()
            
            dict_results = trainer.test(test_dataloaders=training_module.test_dl, model = training_module)

            #Saving test results for model to file
            _dir = os.path.join(tparams['model_dir'], mparams['model_name'])
            fn = os.path.join(_dir,"results.json")

            if os.path.isfile(fn) == False:
                existing_results = {}
            else:
                with open( fn, 'r' ) as outfile:
                    existing_results = json.load( outfile )

            existing_results[ f"{mparams['model_name']}_{tparams['version']}" ] = dict_results[0]['test_loss']
            
            with open( fn, 'w' ) as outfile:
                json.dump( existing_results, outfile)
                       
    @staticmethod
    def load_model_tokenizer(cls, model_name="ctrl", model_version=0, mparams_new={}, device="cuda:0"):
        
        checkpoint = TrainingModule.get_ckpt_file(f'./models/{model_name}/version_{model_version}/checkpoints')

        mparams = {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [ 'base_model_name',
            'model_name','max_len_utt','max_len_title','max_len_claim','max_len_concepts','max_len_target_entities']}
        for key, value in mparams_new.items():
            mparams[key] = value

        mconfig = CTRL_ArgConfig.from_pretrained(
            mparams['base_model_name'], **mparams)
            
        # Loading Training Module
        training_module = TrainingModule(mconfig, mode='inference')
        training_module.load_state_dict(checkpoint['state_dict'] )
        
        model = training_module.model
        tokenizer = training_module.tokenizer

        # Deleting checkpoints to free up GPU space
        del checkpoint
        torch.cuda.empty_cache()
          
        #if torch.cuda.is_available():
        if device != 'cpu' and torch.cuda.is_available():
            model = model.to(device)
        
        tokenizer = CTRL_ArgTokenizer.from_pretrained("./tokenizers/ctrl_arg", **mparams)
        return model, tokenizer

    def forward(self, batch):
        with torch.cuda.amp.autocast(enabled=True): 
            return  self.model(**batch)

    def step(self, batch, step_name):
        
        ctrl_output = self.forward(batch) #(lm_logits and loss)
       
        loss_key = f"{step_name}_loss"

        output = {}
        if step_name == 'train':
            output["loss"] = ctrl_output.loss
            self.log( "loss", ctrl_output.loss)
        else:   
            self.log( loss_key, ctrl_output.loss)
            output[loss_key]=ctrl_output.loss

        return  output
           
    def training_step(self, batch, batch_idx):
        output = self.step(batch,"train")
        return output

    def validation_step(self, batch, batch_idx):
        output = self.step(batch, "val")
        return output

    def test_step(self, batch, batch_idx):
        output = self.step(batch, "test")
        return output

    def training_epoch_end(self, outputs):
        self.epoch_end_log(outputs,"train")

    def validation_epoch_end(self, outputs: List[dict]):
        self.epoch_end_log(outputs, "val")
         
    def test_epoch_end(self, outputs: List[dict]):
        self.epoch_end_log(outputs, "test")

    def epoch_end_log(self, outputs, step_name):

        if step_name == "train":
            pass
        else:
            loss = torch.stack([x[f"{step_name}_loss"]for x in outputs]).mean()
            
            self.log(f"{step_name}_loss", 
                            loss, 
                            logger=True,
                            prog_bar=True,
                            sync_dist=True )
        
        if step_name == "val" and _get_rank() == 0:
            
            # Making directory if it doesnt exist
            dir_infer = os.path.join(self.trainer.log_dir, "inference")
            
            if not os.path.exists(dir_infer):
                os.makedirs(dir_infer, exist_ok=True)
            
            # Generation Params
            generation_params = {'max_new_tokens':200,
            'no_repeat_ngram_size': 3,
                                 }
            bad_words = ["<cl>", "<cp>", "<te>","Comment:"]
            generation_params['bad_words_ids'] = [self.tokenizer.encode(
                bad_word) for bad_word in bad_words] + [ self.tokenizer.new_line_id_tensor.numpy().tolist() ]

            
            # Adding true values and making csv files if thy dont already exists
            for idx, encoded_input_ in enumerate(self.inference_samples):
                
                encoded_input = { k:v.detach().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k,v in encoded_input_.items()}
                                
                fp = os.path.join(dir_infer, f"example_{idx:03d}.csv")

                # If there file does not exists we add the true observed records
                if not os.path.exists(fp):

                    df = pd.DataFrame(columns=['epoch', 
                                                'title',
                                                'claim',
                                                'concepts',
                                                'target_entities',
                                                'utterance'])

                    title = encoded_input.pop('orig_title')
                    claim = encoded_input.pop('orig_claim')
                    concepts = encoded_input.pop('orig_concepts')
                    target_entities = encoded_input.pop('orig_target_entities')
                    utterance = encoded_input.pop('orig_utterance')
                  
                    datum = {
                        'epoch': -1,

                        'title': json.dumps(title),
                        'claim': json.dumps(claim),
                        'concepts': json.dumps(concepts),

                        "target_entities": json.dumps(target_entities),
                        "utterance": json.dumps(utterance)
                    }

                    df = df.append(datum, ignore_index=True)
                    df.to_csv(fp, index=False)

                # creating predition andding to existing results
                encoded_input.pop('orig_title', None)
                encoded_input.pop('orig_claim', None)
                encoded_input.pop('orig_concepts', None)
                encoded_input.pop('orig_target_entities', None)
                encoded_input.pop('orig_utterance', None)                

                output_ids = self.model.generate( encoded_input['input_ids'].to(self.device), **generation_params, use_cache=True, )
                decoded_text = self.tokenizer.decode(output_ids[0]).split('Comment:')[-1]
                
                datum = {
                    'epoch': self.current_epoch,
                    'title': '',
                    'claim': '',
                    'concepts':'',
                    'target_entities':'',
                    'utterance':json.dumps(decoded_text)}

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
                    
        # optimial settings for T5
        optimizer = Adafactor(self.model.parameters(), scale_parameter=True, 
                               relative_step=True, warmup_init=True, lr=None )
        
        lr_scheduler = AdafactorSchedule(optimizer)

        return [optimizer], [{ "scheduler":lr_scheduler ,"interval": "step", "monitor":"val_loss"}]
    
    def return_params(self):
        params = {}
        keys = ['batch_size','accumulate_grad_batches','learning_rate','max_epochs','dir_data','tag',]
        
        params = {
            k:self.__dict__[k] for k in keys if k in self.__dict__.keys()
        }

        return params

class DataLoaderGenerator():
    """Handles the creation of dataloaders for a train, val and test set
    """
    def __init__(self, dir_data, batch_size,
                 tokenizer,
                 workers=0, mode='train_new',
                 gpus=1,
                 **kwargs):

        self.dir_data = dir_data
        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.workers = workers
        self.mode = mode
        self.gpus = gpus
        self.pad_values = self.tokenizer.pad_values
        self.pad_maxlens = self.tokenizer.pad_maxlens

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
            ds = kwargs.get('custom_dset_class')(fn, self.tokenizer,inference)
        else:
            ds = SingleDataset(fn, self.tokenizer, inference )
            
        sampler = SizedOrdered_Sampler(ds, bs, shuffle) if sampler else sampler

        def collate_fn(
                batch): return self.tokenizer.default_collate_pad(batch)
                    
        dataloader = torch.utils.data.DataLoader(ds, 
                                                batch_size= bs ,
                                                 num_workers=self.workers, 
                                                 sampler = sampler,
                                                 pin_memory=False,
                                                 collate_fn=collate_fn,
                                                 multiprocessing_context = kwargs.get('multiprocessing_context', None) )                                                
        return dataloader

class SizedOrdered_Sampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.
    #TODO; add this to pytorch. Sampler to sort nlp datasets by size
    Args:
        data_source (Dataset): dataset to sample from
    """
    
    def __init__(self, data_source, batch_size, shuffle) -> None:
        self.data_source = data_source
        
        np_title_lens = self.data_source.np_title_lens
        np_claim_lens = self.data_source.np_claim_lens
        np_target_entities_lens = self.data_source.np_target_entities_lens
        np_concepts_lens = self.data_source.np_concepts_lens

        #Indices are sorted in order of the text lens of records in the datasets. This randomizes the elements of the batches
        random_idxs = np.random.random( np_title_lens.size )
        np_ordered_lens = np.lexsort([ random_idxs, np_title_lens + np_claim_lens + np_target_entities_lens + np_concepts_lens ])
        li_chunked_lens = [np_ordered_lens[idx:idx+batch_size]
                            for idx in range(0, np_ordered_lens.size - batch_size, batch_size)]  

        if shuffle:
            random.shuffle(li_chunked_lens)
        
        self.li_chunked_ordered_lens = np.concatenate(li_chunked_lens).tolist()
        
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
            >>> for epoch in range(start_epoch, n_epochs):
            ...     if is_distributed:
            ...         sampler.set_epoch(epoch)
            ...     train(loader)
        """

    def __init__(self, dataset: Dataset, batch_size: int, agb,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, 
                 seed: int = 0,
                 shuffle: bool = False,
                 gpus: int = 2) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            #num_replicas = dist.get_world_size()
            num_replicas = gpus
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
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
        
        self.seed = seed
        np_title_lens = self.data_source.np_title_lens
        np_claim_lens = self.data_source.np_claim_lens
        np_target_entities_lens = self.data_source.np_target_entities_lens
        np_concepts_lens = self.data_source.np_content_lens

        
        #Indices are sorted in order of the text lens of records in the datasets. This randomizes the elements of the batches
        random_idxs = np.random.random( np_title_lens.size )
        np_ordered_lens = np.lexsort([ random_idxs, np_title_lens + np_claim_lens + np_target_entities_lens + np_concepts_lens ])
        li_chunked_lens = [np_ordered_lens[idx:idx+batch_size]
                            for idx in range(0, np_ordered_lens.size - batch_size, batch_size)]    

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

        self.li_li_chunked_ordered_lens = [np.concatenate(
            li_chunked_lens).tolist() for li_chunked_lens in li_li_chunked_lens]
        
        
    def __iter__(self) -> Iterator[T_co]:

        return iter( self.li_li_chunked_ordered_lens[ self.rank] )

    def __len__(self) -> int:
        return self.data_source

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

class SingleDataset(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

        create a custom index which sorts the entries by their length
    """
    def __init__(self, file_path, tokenizer, inference, **kwargs):
        self.fp = file_path
        self.tokenizer = tokenizer     
        self.inference = inference   
        self.data = pd.read_csv(self.fp , sep="," , header=0 )

        fp_cached_order = os.path.join(os.path.dirname(
            file_path), f"ctrl_dict_lens.pkl")

        # # # resetting the cached order files
        # if os.path.exists( fp_cached_order):
        #     os.remove(fp_cached_order)
        
        if os.path.exists(fp_cached_order):
            dict_cached_order = pickle.load(open(fp_cached_order, "rb"))
            self.np_title_lens = dict_cached_order['np_title_lens']
            self.li_claim_lens = dict_cached_order['li_claim_lens']
            self.np_concepts_lens = dict_cached_order['np_concepts_lens']
            self.np_target_entities_lens = dict_cached_order['np_target_entities_lens']
            self.np_textlens = dict_cached_order['np_textlens']

        else:
            #We do not these lengths for truncation just for sorting
            self.np_title_lens = np.stack(
                [self.tokenizer.encode(ujson.loads(txt), return_tensors='np', add_special_tokens=False,
                                    truncation=False, padding='do_not_pad').size for txt in self.data.prompt.values.tolist()])

            li_claims = list( map( ujson.loads , self.data.li_claim.tolist()) )
            li_claims = [ [self.tokenizer.claim_start_token + claim  for claim in claims] if len(claims)>0 else []  for claims in li_claims ]
            
            self.li_claim_lens =  [ [ self.tokenizer.encode(claim,
                                            add_special_tokens=False, 
                                            truncation=False,
                                            padding = 'do_not_pad',
                                            return_tensors=None).__len__() for claim in claims ] for claims in li_claims ] 
            self.np_concepts_lens = np.stack(
                [self.tokenizer.encode( self.tokenizer.concepts_start_token + ', '.join(ujson.loads(li_concepts)), return_tensors='np', add_special_tokens=False,
                                    truncation=False, padding='do_not_pad').size for li_concepts in self.data.li_concepts.values.tolist()])
            self.np_target_entities_lens = np.stack(
                [self.tokenizer.encode( self.tokenizer.target_entities_start_token + ', '.join(ujson.loads( li_target_entity)), return_tensors='np', add_special_tokens=False,
                                    truncation=False, padding='do_not_pad').size for li_target_entity in self.data.li_target_entity.values.tolist()])
            
            self.np_textlens = np.stack(
                [self.tokenizer.encode(ujson.loads(txt), return_tensors='np', add_special_tokens=False,
                                    truncation=False, padding='do_not_pad').size for txt in self.data.txt_preproc.values.tolist()])
            
            
            dict_cached_order = {
                'np_title_lens':self.np_title_lens,
                'li_claim_lens':self.li_claim_lens,
                'np_concepts_lens':self.np_concepts_lens,
                'np_target_entities_lens':self.np_target_entities_lens,
                'np_textlens':self.np_textlens
            }

            pickle.dump(dict_cached_order, open(fp_cached_order, "wb"))
      
        #randomly choosing a claim
        self.li_claim_idxs = [ random.randint(0,len(claim_lens)-1) if len(claim_lens)!=0 else -1 for claim_lens in self.li_claim_lens ] 
        self.np_claim_lens = np.array( [ self.li_claim_lens[datum_idx][claim_idx] if claim_idx>-1 else 0 for datum_idx, claim_idx in enumerate(self.li_claim_idxs) ] )

        self.data = self.data.to_dict('records')

        # Clipping the values in the array to our max_lens
        self.np_title_lens = np.clip( self.np_title_lens, 0, self.tokenizer.max_len_title )
        self.np_claim_lens = np.clip( self.np_claim_lens, 0, self.tokenizer.max_len_claim )
        self.np_target_entities_lens = np.clip( self.np_target_entities_lens, 0, self.tokenizer.max_len_target_entities )
        self.np_concepts_lens = np.clip( self.np_concepts_lens, 0, self.tokenizer.max_len_concepts )
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        title, claim, concepts, target_entities, utterance = self.getitem_extract_datum(index)
        
        if self.inference == True:
            # utterance_prompt = ' '.join(utterance.split(' ')[:2])
            utterance_prompt = ''
            orig_utterance = copy.deepcopy(utterance)
            utterance =None 
        else:
            utterance_prompt=None
            utterance = utterance
                
        encoded = self.tokenizer.encode_input( title=title, claim=claim,
                                                concepts=concepts,
                                                target_entities=target_entities,
                                                utterance=utterance,
                                                utterance_prompt=utterance_prompt )
        
        if self.inference == True:
            encoded['orig_title'] = title
            encoded['orig_claim'] = claim
            encoded['orig_concepts'] = concepts
            encoded['orig_target_entities'] = target_entities
            encoded['orig_utterance'] = orig_utterance
     
        return encoded

    def getitem_extract_datum(self, index):
        
        datum = self.data[index]

        #region RST

        title =  ujson.loads(datum['prompt'])
        
        claim = ujson.loads(datum['li_claim'])[self.li_claim_idxs[index]] if self.li_claim_idxs[index] != -1 else ""
        
        concepts = ujson.loads(datum['li_concepts'])   
        # if len(concepts)> self.tokenizer.max_len_concepts:
        random.shuffle(concepts)
        
        target_entities = ujson.loads(datum['li_target_entity'])
        
        # if len(target_entities)> self.tokenizer.max_len_target_entities:
        random.shuffle(target_entities)
        
        utterance = ujson.loads(datum['txt_preproc'])
        
        return title, claim, concepts, target_entities, utterance




def main(tparams={}, mparams={}):
   
    
    # Defining Logger
    tb_logger = pl_loggers.TensorBoardLogger( 
                    save_dir = os.path.abspath(tparams['model_dir']),
                    name = mparams['model_name'],
                    version = tparams['version'] )
    tparams['version'] =  tb_logger.version
    
    tparams['dir_checkpoints'] = os.path.join(tparams['model_dir'],mparams['model_name'],f"version_{tparams['version']}",'checkpoints' )
    
    os.makedirs(tparams['dir_checkpoints'],exist_ok=True)

    # initiating training loop
    training_module = TrainingModule.instatiate_training_module( tparams, mparams)
    trainer, training_module = TrainingModule.instatiate_trainer( tparams, tb_logger, training_module)
    TrainingModule.start(trainer, tparams, training_module, mparams)

def parse_model_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
                                    
    parser.add_argument('--max_len_utt', type=int, default=140)
    parser.add_argument('--max_len_claim', type=int, default=20)
    parser.add_argument('--max_len_title', type=int, default=20)
    parser.add_argument('--max_len_concepts', type=int, default=20)
    parser.add_argument('--max_len_key_phrase', type=int, default=30)
    parser.add_argument('--max_len_target_entities', type=int, default=20)
    parser.add_argument('--base_model_name', type=str, default='ctrl',  choices=['ctrl','sshleifer/tiny-ctrl'] )    
    
    parser.add_argument('--model_name', type=str, default='ctrl')    
    mparams = parser.parse_known_args( )[0]
    
    return mparams


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False) 
    
    # add model specific args
    mparams = parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = TrainingModule.parse_train_specific_args(parent_parser)

    if tparams.mode == "test":
        assert tparams.gpus in [0,1]

    if tparams.gpus not in [0,1]:
        os.environ['MASTER_ADDR'] = 'localhost' #'127.0.0.1'
        os.environ['MASTER_PORT'] = '65302'

    main(vars(tparams), vars(mparams))


