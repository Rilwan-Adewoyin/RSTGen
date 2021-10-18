import os
from typing import Optional, Tuple

from torch._C import Value
from torch.nn.modules import loss
import transformers
from transformers.utils.dummy_pt_objects import MBartPreTrainedModel

#os.environ["NCCL_DEBUG"]="INFO"
#os.environ["NCCL_DEBUG_SUBSYS"]="ALL"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["NCCL_P2P_LEVEL"] = "3"
#os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_SOCKET_IFNAME'] =  'lo' 
#os.environ['NCCL_SOCKET_IFNAME'] =  'enp3s0'
#os.environ['CUDA_LAUNCH_BLOCKING']="1"
import fairscale
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing as mp
from pytorch_lightning.utilities.distributed import _get_rank
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import Sampler
import glob
import pandas as pd
import json
from functools import lru_cache
from typing import List
import ujson
import torch.distributed as dist
from pathlib import Path
from itertools import cycle, islice
from torch.utils.data._utils.collate import default_convert, default_collate
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import CrossEntropyLoss

from sklearn import preprocessing as sklp
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import deepspeed

import argparse
import utils_nlg_v3 as utils
import random 
from typing import TypeVar, Iterator
T_co = TypeVar('T_co', covariant=True)

from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers.generation_beam_search import BeamHypotheses

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.core.decorators import auto_move_data
import yaml
import types

import copy

from itertools import permutations, combinations, combinations_with_replacement
from typing import Optional, Callable, Union, Optional, List, Iterable


from transformers import CTRLTokenizer, CTRLLMHeadModel

class TrainingModule(pl.LightningModule):

    def __init__(self, max_input_len=1024, batch_size=20, 
                    dir_data=None, 
                    accumulate_grad_batches=1,
                    max_epochs=25,
                    gpus=1, 
                    learning_rate=1e-4,
                    workers=0,
                    mode = 'train_new',
                    tag='ctrl fine tuned on argument generation',
                    model_name='ctrl',
                    *args,
                    **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.gpus =  gpus
        
        self.mode = mode
        self.workers = workers
        self.max_input_len = max_input_len
        self.model = CTRLLMHeadModel.from_pretrained(model_name)
        self.model.tokenizer = CTRLTokenizer.from_pretrained(model_name)
        self.model.tokenizer.default_collate_pad = types.MethodType(utils.EffeciencyMixin.default_collate_pad,self.model.tokenizer)                                        
        self.model.tokenizer.pad_values  = { 'input_ids': 246532 , 'labels':-100 }                                   
        self.model.tokenizer.pad_maxlens = { 'input_ids':self.max_input_len, 'labels':self.max_input_len }
        self.model.tokenizer.model_max_length = self.max_input_len

        if self.mode in ['train_new','train_cont','test','finetune']:
            self.dir_data = utils.get_path(dir_data)
            self.accumulate_grad_batches = accumulate_grad_batches
            self.create_data_loaders( )
            self.tag = tag

        if self.mode in ['train_new','train_cont','finetune']:
            self.max_epochs = max_epochs
            self.learning_rate = learning_rate

        if self.mode in ['inference']:
            self.eval() 
            self.freeze() 

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data', default="./dataset_cmv/seq2seq", help="Relative directory path for datafiles")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('--max_epochs', default=6, type=int)
        parser.add_argument('--accumulate_grad_batches', default=30, type=int)
        parser.add_argument('-b','--batch_size', default=1, type=int)
        parser.add_argument('--learning_rate', default=2e-4, type=float)
        parser.add_argument('--workers', default=6, type=int) #TODO: change to 6
        parser.add_argument('--gpus', default=2, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','train_cont','test','inference','finetune'])
        parser.add_argument('--version', default=1,required=False, type=int, help="The Experimental Versioning for this run" )
        parser.add_argument('--precision', default=16, required=False, type=int, help="Precision to use", choices=[16,32] )
        parser.add_argument('--tag',default='ctrl',required=False, type=str)
            #TODO: check --version of required type None actually works
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
            training_module = TrainingModule(**tparams, **mparams  )
            
        elif tparams['mode'] in ["train_cont", "inference"]:
            
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            #restore/update param files from the checkpoint

            if "hyper_parameters" in checkpoint.keys() and tparams['override'] == False:
                tparams.update ( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'batch_size', 'precision','tag']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'model_name','max_input_len']} )
                
                        
            else:
                print("param files not found utilsing default or user entered params\n")
                
            #Restore/update Training Module
            training_module = TrainingModule(**tparams, **mparams)
            training_module.load_state_dict(checkpoint['state_dict'])

        elif tparams['mode'] == "finetune":
            training_module = TrainingModule(**tparams, **mparams)
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])            
            state = checkpoint['state_dict']
            
            state_dict = training_module.state_dict()

            for k, v in state.items():
                state_dict.update({k: v})

            training_module.load_state_dict(state_dict)


        elif tparams['mode'] in ["test"]:
            
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            #restore/update param files from the checkpoint
            try:
                tparams.update ( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                     'precision','optimizer_type']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name','model_name','max_input_len']} )
            except KeyError:
                pass
            
            #Restore/update Training Module
            training_module = TrainingModule(**tparams, **mparams)
            training_module.load_state_dict(checkpoint['state_dict'])

        else:
            raise ValueError("tparams['mode'] must be in range [train_new, train_cont, test, inference]")

        return training_module

    @staticmethod
    def instatiate_trainer( tparams, model_name, tb_logger, training_module):
        """[summary]

            Creates The Trainer and callbacks
        """
        dir_checkpoints = tparams['dir_checkpoints']
        
        # Creating Callbacks
        callbacks = []        
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=3, 
            mode='min', dirpath=dir_checkpoints, 
            filename='{epoch:03d}_{val_loss:.5f}')
        
        checkpoint_callback._save_model  = types.MethodType(utils.monkey_save_model,checkpoint_callback) #monkey patch

        if model_name == "ctrl":
            val_check_interval = 0.1
            patience = 30
        elif model_name == "sshleifer/tiny-ctrl":
            val_check_interval = 0.5
            patience = 3

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=patience,
            verbose=False,
            mode='min'
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(early_stop_callback)

       
        if tparams['gpus'] in [0,1]:
            
            trainer_vars = { 'accelerator':None }

        else:
            if model_name == "ctrl":
                trainer_vars = {
                   #'plugins': DeepSpeedPlugin(stage=2 ),     
                    #   'plugins':DDPPlugin(find_unused_parameters=False),
                    "plugins" : 'ddp_sharded',
                      'accelerator':'ddp',
                                
                            'amp_level': 'O1'
                                }
            else:
                trainer_vars = {
                    'accelerator':'ddp',
                    'plugins':DDPPlugin(find_unused_parameters=False)
                }
        
        if tparams['mode'] in ["train_new", "finetune"]:
            
            trainer = pl.Trainer.from_argparse_args(argparse.Namespace( **tparams),
                        progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                        default_root_dir=tparams['dir_checkpoints'],
                        logger=tb_logger,
                        precision=tparams['precision'], 
                        callbacks=callbacks,
                        #limit_train_batches = 5,
                        #overfit_batches = 10,
                        #accelerator=accelerator,
                        #plugins=DDPPlugin(find_unused_parameters=False),
                        val_check_interval=val_check_interval,
                        num_sanity_val_steps=0, 
                       replace_sampler_ddp=False,
                    reload_dataloaders_every_n_epochs=1,

                        #track_grad_norm=2 ,
                        #gradient_clip_val = 1 ,
                        **trainer_vars)

        elif tparams['mode'] in ["train_cont","inference"]:
            #restoring checkpoint             
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])


            trainer = pl.Trainer.from_argparse_args(argparse.Namespace( **tparams),
                    progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                     logger=tb_logger,
                      
                    precision=tparams['precision'],
                    callbacks=callbacks,
                        #accelerator=accelerator,
                        val_check_interval=val_check_interval,
                        num_sanity_val_steps=0, 
                        replace_sampler_ddp=False,
                        reload_dataloaders_every_n_epochs=1
                        #plugins=DDPPlugin(find_unused_parameters=False),
                        #track_grad_norm=2
                        
                    )

            # load callback states
            #trainer.on_load_checkpoint(checkpoint)
            trainer.global_step = checkpoint['global_step']
            trainer.current_epoch = checkpoint['epoch']
            
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
            scores_dict = yaml.load( open(checkpoint_yaml_file,"r") ) #key= ckptpath, value = val_loss
            best_ckpt_path = min(scores_dict, key=scores_dict.get)
            
            if os.path.exists(best_ckpt_path) == False:
                root_dir = Path(__file__).resolve().parents[4]
                best_ckpt_path = os.path.join( root_dir, best_ckpt_path[ best_ckpt_path.index('mastering-conversation'): ] )

            checkpoint = torch.load(best_ckpt_path, map_location='cpu' )  
          
        else:
            raise NotImplementedError
        
        return checkpoint
        
    @staticmethod
    def start(trainer, tparams,training_module, mparams ):
        
        if tparams['mode'] in ['train_new','train_cont','finetune']:    
            trainer.fit(training_module )
        
        if tparams['mode'] in ["test"]:
            
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
                       
        elif tparams['mode'] in ['infernece']: 
            training_module.eval() 
            training_module.freeze() 
            raise NotImplementedError   
    
    @staticmethod
    def load_ctrlmodel(model_name="ctrl", model_version=0, device="cuda:0"):
        # Loading in NLG model
        checkpoint = TrainingModule.get_ckpt_file(f'./models/{model_name}/version_{model_version}/checkpoints')
            
        # Loading Training Module
        training_module = TrainingModule(mode='inference',model_name=model_name)
        training_module.load_state_dict(checkpoint['state_dict'])
        model = training_module.model

        # Deleting checkpoints to free up GPU space
        del checkpoint
        torch.cuda.empty_cache()
          
        #if torch.cuda.is_available():
        if device != 'cpu' and torch.cuda.is_available():
            model = model.to(device)
        
        return model

    #@auto_move_data
    def forward(self, input_ids):
        with torch.cuda.amp.autocast(enabled=True): 
            return  self.model(input_ids)

    def step(self, batch, step_name):
        
        dict_m_output = self.forward(batch['input_ids']) #(lm_logits and loss)
        dict_m_output = self.loss(dict_m_output, labels=batch['labels']  )
        loss_key = f"{step_name}_loss"
        
        output = {}

        if step_name == 'train':
            output["loss"] = dict_m_output.loss
            self.log( loss_key, dict_m_output.loss)

        else:
                   
            self.log( loss_key, dict_m_output.loss)
            
            output[loss_key]=dict_m_output.loss


        return  output 

    def loss(self, dict_m_output, labels):
        
        
    
        # Shift so that tokens < n predict n
        
        shift_logits = dict_m_output.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
                
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        dict_m_output.loss = loss
        return dict_m_output
        
        
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
            loss = torch.stack([x[f"{step_name}_loss"] for x in outputs]).mean()
            self.log(f"{step_name}_loss", loss, logger=True, prog_bar=True)
                     
    def create_data_loaders(self, shuffle=False, **kwargs):
       
        dg = DataLoaderGenerator(self.dir_data,  self.batch_size, self.model.tokenizer, 
                workers=self.workers, mode=self.mode, gpus=self.gpus, agb=self.accumulate_grad_batches
                 )

        _dict_dl = dg()
        self.train_dl = _dict_dl['train_dl']
        self.val_dl = _dict_dl['val_dl']
        self.test_dl = _dict_dl['test_dl']

    def train_dataloader(self):
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
        optimizer = Adafactor(self.model.parameters(), scale_parameter=False, 
                               relative_step=False, warmup_init=False, lr=self.learning_rate )
        
        #optimizer = deepspeed.ops.adam.FusedAdam(self.model.parameters(), lr=self.learning_rate  )
        
        #optimizer = deepspeed.runtime.fp16.onebit.adam.OnebitAdam(self.model.parameters(), lr=self.learning_rate  )

        lr_scheduler = transformers.get_constant_schedule_with_warmup(optimizer,
                            num_warmup_steps= 0.10*self.total_steps(),
                            )
    

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
                    agb=1,
                    **kwargs):
        
        self.dir_data = dir_data
        self.tokenizer = tokenizer
        self.gpus = gpus
        self.bs = batch_size
        self.workers  = workers
        self.mode = mode
        self.accumulate_grad_batches = agb


    def prepare_dataloaders(self):
        """prepares a train, validation and test set

        Returns:
            [type]: [description]
        """
                
        if self.mode in [ 'train_new', 'train_cont','finetune']:
            train_dl = self.prepare_dataloader(self.dir_data, shuffle=True, split_name='train' )
            val_dl = self.prepare_dataloader(self.dir_data, shuffle=False,split_name='val'  )
            test_dl = self.prepare_dataloader(self.dir_data, shuffle=False,split_name='test'  )
        
        elif self.mode in ['test']:
            train_dl= None
            val_dl = None
            test_dl = self.prepare_dataloader(self.dir_data, shuffle=False ,split_name='test' )

                    
        dict_dl = {'train_dl':train_dl,
                    'val_dl':val_dl,
                    'test_dl':test_dl,
                    }

        return dict_dl 

    def prepare_dataloader(self, dir_data, shuffle=False, 
        split_name='train'):

        """Prepares a dataloader given a directory of data for NLG language module
            # The current method takes a percentage of data from each subdirectory
            Args:
                dir_dset ([type]): [description]
        """

        #defining starting line and total lines to use for dataset
        if split_name == 'train':

            fn = glob.glob(  os.path.join( dir_data,"train","*") )[0]
            shuffle = True
            bs = self.bs
            unique_features = True

        elif split_name == 'val':
            fn = glob.glob(  os.path.join( dir_data,"val","*") )[0]
            shuffle = False
            bs = self.bs
            unique_features = False

        elif split_name == 'test':
            fn = glob.glob(  os.path.join( dir_data,"test","*") )[0]
            shuffle = False
            bs = self.bs
            unique_features = False

        dset = SingleDataset(fn, self.tokenizer, unique_features=unique_features) 

        if self.gpus <= 1 or split_name not in ['inference','test'] :
            sampler = SizedOrdered_Sampler(dset, bs, shuffle=shuffle, agb=self.accumulate_grad_batches )
        else:
            sampler = SizedOrdered_DistributedSampler( dset, bs, shuffle=shuffle, gpus=self.gpus, agb=self.accumulate_grad_batches )


        if split_name in ['train','val','test']:
            dataloader = torch.utils.data.DataLoader(dset, batch_size=bs,
                shuffle=False, num_workers=self.workers,
                collate_fn=lambda batch: self.tokenizer.default_collate_pad(batch),
                #pin_memory=True,
                pin_memory=False,
                sampler=sampler,
                multiprocessing_context=torch.multiprocessing.get_context('spawn') )
        else:
            dataloader = torch.utils.data.DataLoader(dset, batch_size=bs,
                shuffle=shuffle, num_workers=self.workers,
                )

        return dataloader

    def __call__(self):
        dict_dl = self.prepare_dataloaders()
        return dict_dl
    
class SingleDataset(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

        create a custom index which sorts the entries by their length
    """
    def __init__(self, file_path, tokenizer, unique_features=False ):
        self.fp = file_path
        self.tokenizer = tokenizer        
        self.data = pd.read_csv(self.fp , header=0 )

        if unique_features:
            self.data = self.data.sample(frac=1).drop_duplicates(subset=['prompt']).reset_index(drop=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        utterance, prompt = self.getitem_extract_datum(index)
    
        encoded = self.getitem_tokenize( utterance,
                                            prompt )

        return encoded

    def getitem_extract_datum(self, index):
        
        datum = self.data[index:index+1]

        #region RST

        prompt = f"{json.loads(datum['prompt'].values[0])}"
        
        utterance = json.loads(datum['reference'].values[0])
        
        
        return utterance, prompt

    def getitem_tokenize(self, utterance, prompt):

        
        prompt_tok = torch.cat( [ self.tokenizer.encode(f"Argument Title: {prompt}", return_tensors="pt", truncation=True)[0],
                                torch.tensor( [246533], dtype=torch.long ) 
                               #,self.tokenizer.encode("Comment:", return_tensors="pt", truncation=True)[0] 
                               ],
                                axis=-1
                                )

        utterance_tok = torch.cat( [self.tokenizer.encode(utterance, return_tensors="pt", truncation=True)[0] , 
                                    torch.tensor( [246533], dtype=torch.long ) ]
                                  ,axis=-1)
        
        #TODO: we use the /n token to indicate end of sentence, 
        # In the reddit data this indicates another person is speaking
        # so it aligns well to our interpretation of eos for argumentation
        
        input_ids = torch.cat([prompt_tok, utterance_tok], axis=-1)

        # labels = torch.cat(
        #     [ input_ids.new_full( [ 1 ], -100 ),
        #         input_ids[ 1: ]
        #     ], axis=-1
        # )
        labels = input_ids.detach().clone()
        labels[:len(prompt_tok)] = -100

                        
        encoded = {
            'input_ids':input_ids,
            'labels':labels
        }
        return encoded

class SizedOrdered_Sampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.
    #TODO; add this to pytorch. Sampler to sort nlp datasets by size
    Args:
        data_source (Dataset): dataset to sample from
    """
    
    def __init__(self, data_source, batch_size, agb, shuffle) -> None:
        self.data_source = data_source
        
        li_records = self.data_source.data.to_dict('records')
        inp_lens = [ len( ujson.loads(record['prompt']).split(' ') )+len(ujson.loads(record['reference']).split(' ')) for record in li_records ]
        np_txt_lens = np.array(inp_lens)

        #Indices are sorted in order of the text lens of records in the datasets
        np_ordered_lens = np_txt_lens.argsort()
        
        ebs = int( batch_size*agb )
        # We Randomly re-arrange them in batches of batch size
        li_chunked_lens = [ np_ordered_lens[idx:idx+ebs] for idx in range(0, np_ordered_lens.size - ebs, ebs) ]
        
        if shuffle:
            random.shuffle( li_chunked_lens )
        
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
        data_source = dataset
        li_records = data_source.data.to_dict('records')
        inp_lens = [ len( ujson.loads(record['prompt']).split(' ') )+len(ujson.loads(record['reference']).split(' ')) for record in li_records ]
        np_txt_lens = np.array(inp_lens)

        #Indices are sorted in order of the text lens of records in the datasets
        np_ordered_lens = np_txt_lens.argsort()
        
        ebs = int( batch_size*agb )
            # We Randomly re-arrange them in batches of batch size
        li_chunked_lens = [ np_ordered_lens[idx:idx+ebs] for idx in range(0, np_ordered_lens.size-ebs, ebs) ]

            # Divide into n sublists,
            # Each sublist at index i, contains the indices for process at rank i
            # Each sublist at index i, is a list non flatten indices. Each index represents items in the dataset
        li_li_chunked_lens = [ 
                              [ li_chunked_lens[ (self.num_replicas*idx)+_rank ] for idx in range(len(li_chunked_lens)//self.num_replicas)  ] 
                                
                                for _rank in range(self.num_replicas)]
        
        # shuffle each processes subllist in the same order to optimize paralel training
        _ = list( zip( *li_li_chunked_lens ))
        
        if shuffle:
            random.shuffle( _ )
        
        li_li_chunked_lens = list(zip(*_))
        
        self.li_li_sizeorderedidx = [ np.concatenate(li_chunked_lens).tolist() for li_chunked_lens in li_li_chunked_lens ]
        self.num_samples = len(self.li_li_sizeorderedidx[0])
        self.total_size = self.num_samples * self.num_replicas
        
        
    def __iter__(self) -> Iterator[T_co]:

        return iter( self.li_li_sizeorderedidx[ self.rank] )

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


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
    trainer, training_module = TrainingModule.instatiate_trainer( tparams, mparams['model_name']  ,tb_logger, training_module)
    TrainingModule.start(trainer, tparams, training_module, mparams)

def parse_model_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
                                    
    parser.add_argument('--max_input_len', type=int, default=200)
    
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


