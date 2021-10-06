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
from typing import (Any, Dict, Iterator, List, Optional, TypeVar)
import functools
import operator
import einops
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

import torch.distributed as dist
import ujson
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin
from pytorch_lightning.utilities.distributed import _get_rank
from sklearn import preprocessing as sklp
from torch.nn import CrossEntropyLoss
from torch.utils.data import Sampler 
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

from transformers import GPT2Config, GPT2TokenizerFast
from transformers.modeling_outputs import ( ModelOutput,
                                            BaseModelOutputWithPastAndCrossAttentions,
                                            CausalLMOutputWithCrossAttentions
                                           )
from transformers import GPT2LMHeadModel
from transformers.optimization import Adafactor, AdafactorSchedule, AdamW
from transformers.tokenization_utils_base import AddedToken

import utils_nlg_v3 as utils
from utils_nlg_v3 import EmbeddingRstPos, mpatch_save_model

from train_RSTGPT import RSTGPT2, RSTGPT2_Config, RSTTokenizer, RSTGPT2_TrainingModule

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

from DockerImages.feng_hirst_rst_parser.src.parse2 import DiscourseParser
from DockerImages.feng_hirst_rst_parser.src import parser_wrapper3
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.lr_schedules import WarmupLR

from transformers.utils import logging
logger = logging.get_logger(__name__)
from pytorch_lightning.callbacks import Callback
from sentence_transformers import SentenceTransformer

class SaveModelCallBack(Callback):

    def on_train_end(self, trainer, pl_module):

        # Saving Model using the pytorch method.
        # This allows relaoding using from_pretrained
        os.makedirs(f"./models_pt/{pl_module.model_name}/version_{trainer.logger.version}/", exist_ok=True)
        
        pl_module.model.save_pretrained(
            f"./models_pt/{pl_module.model_name}/version_{trainer.logger.version}/")
        
        if pl_module.model.rst_tree_aligned_attention:
            pl_module.model.self.rst_parser.unload()    


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype ):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    ndim = mask.ndim
    if ndim == 2:
        bsz, src_len = mask.shape


        attention_mask = mask[:, None, None, :].to(dtype)

    elif ndim == 3:
        
        attention_mask = mask[:, None, :, :].to(dtype)

    else:
        raise ValueError("Encoder Attention mask should have three dimensions Decoder Attention mask should have two dimensions")
        
    attention_mask = (1.0 - attention_mask) * -10000.0

    return attention_mask

class RSTGPT2_TrainingModule_dyploc(pl.LightningModule):

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
            self.RSTTokenizer = RSTTokenizer.from_pretrained(f"./tokenizers/{mconfig.model_name}",
                                                         base_tokenizer_name=mconfig.base_model_name,
                                                         rst_params={name: getattr(mconfig, name) for name in ['max_len_rst',
                                                                                                               'max_len_key_phrase',
                                                                                                               'max_rst_depth',
                                                                                                               'max_len_utt', 
                                                                                                               'max_rst_pos',
                                                                                                               'max_rst_pos',
                                                                                                               'rst_tree_aligned_attention'] if hasattr(mconfig, name)
                                                                     }
                                                         )
        else:
            self.RSTTokenizer = tokenizer

        if model==None:
            mconfig.vocab_size = mconfig.vocab_size-2 
            self.model = RSTGPT2.from_pretrained( mconfig.base_model_name, config=mconfig )
            mconfig.vocab_size = mconfig.vocab_size+2
            self.model.config.vocab_size = mconfig.vocab_size
            self.model.resize_token_embeddings(self.model.config.vocab_size)
        else:
            self.model = model

        
        self.pad_values = {'rst_start_token': mconfig.eos_token_id,
                           'rst_rel': self.model.embed_rst_rels.padding_idx,
                           'rst_ns': self.model.embed_rst_ns.padding_idx,
                           'rst_pos': self.model.embed_rst_pos.padding_idx,

                           'key_phrase_ids': mconfig.eos_token_id,
                           'li_kprstpos': self.model.embed_rst_pos.padding_idx,

                           'position_ids_kp_utt': mconfig.n_ctx-1,

                           'input_ids_utt': mconfig.eos_token_id,
                           'attention_mask': 0.0,

                           'labels': self.model.loss_fct.ignore_index,

                            'edu_rstpos': -1,
                            'context_rstpos': -1
                           }
        
        self.RSTTokenizer.pad_values = self.pad_values

        self.pad_maxlens = {
            'rst_start_token': 1,
            'rst_rel': mconfig.max_len_rst-1,
            'rst_ns': mconfig.max_len_rst-1,
            'rst_pos': mconfig.max_len_rst-1,

            'key_phrase_ids': mconfig.max_len_key_phrase,
            'li_kprstpos': mconfig.max_len_key_phrase,

            'input_ids_utt': mconfig.max_len_utt,
            'labels': mconfig.max_len_rst + mconfig.max_len_key_phrase + mconfig.max_len_utt,

            'attention_mask': mconfig.max_len_rst + mconfig.max_len_key_phrase + mconfig.max_len_utt,  # axis:max_length

            'position_ids_kp_utt': mconfig.max_len_key_phrase+mconfig.max_len_utt,
            
            'edu_rstpos': mconfig.max_rst_pos // 2,
            'context_rstpos':mconfig.max_len_rst + mconfig.max_len_key_phrase }

        self.RSTTokenizer.pad_maxlens = self.pad_maxlens
        
        self.model.RSTTokenizer = self.RSTTokenizer

        if self.mode in ['finetune', 'train_cont', 'test']:
            self.dir_data = utils.get_path(dir_data)
            self.accumulate_grad_batches = accumulate_grad_batches
            self.tag = tag

            self.dg = DataLoaderGenerator(self.dir_data,  self.batch_size, self.RSTTokenizer,
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
            
            # self.trainer.logger.add_hparams( vars(mconfig) )

        if self.mode in ['inference']:
            self.eval()
            self.freeze()

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data', default="./dataset_v3_1",
                            help="Relative directory path for datafiles")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('--max_epochs', default=8, type=int)
        parser.add_argument('--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('--batch_size', default=20, type=int)
        parser.add_argument('--batching_style', default='effecient', type=str, choices=['effecient','standard'])

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
            mconfig = RSTGPT2_Config.from_pretrained(mparams['base_model_name'], **mparams)

            model, tokenizer = RSTGPT2_TrainingModule.load_model( None, model_version=mparams['finetune_version'] )
            training_module = RSTGPT2_TrainingModule_dyploc(mconfig, **tparams, model=model, tokenizer=tokenizer)


        elif tparams['mode'] in ["train_cont", "inference"]:

            checkpoint = RSTGPT2_TrainingModule_dyploc.get_ckpt_file(
                tparams['dir_checkpoints'])

            # restore/update param files from the checkpoint
            if "hyper_parameters" in checkpoint.keys() and tparams['override'] == False:
                tparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                     'learning_rate', 'precision', 'splits', 'tag']})

                mparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name', 'model_name', 'max_len_utt','max_len_rst','max_len_key_phrase',
                    'scale_grad_by_freq','rst_tree_aligned_attention' ]})

                mparams = mparams

            else:
                print("param files not found utilsing default or user entered params\n")

            mconfig = RSTGPT2_Config( **mparams)

            # Restore/update Training Module
            training_module = RSTGPT2_TrainingModule_dyploc(mconfig, **tparams)

            training_module.load_state_dict(checkpoint['state_dict'])

        elif tparams['mode'] in ["test"]:

            checkpoint = RSTGPT2_TrainingModule_dyploc.get_ckpt_file(
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
            training_module = RSTGPT2_TrainingModule_dyploc(
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

        checkpoint_callback._save_model = types.MethodType(
            mpatch_save_model(checkpoint_callback._save_model), checkpoint_callback)  #

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience = 15,       
            verbose=False,
            mode='min'
        )

        save_model_callback = SaveModelCallBack()
        callbacks.append(checkpoint_callback)
        callbacks.append(early_stop_callback)
        callbacks.append(save_model_callback)

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
                                                    #progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                                                    default_root_dir=tparams['dir_checkpoints'],
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks,
                                                    val_check_interval=0.5,
                                                    reload_dataloaders_every_n_epochs=1,
                                                    num_sanity_val_steps=2,
                                                    replace_sampler_ddp=False,

                                                    **trainer_vars,
                                                    )

        elif tparams['mode'] in ["train_cont", "inference"]:

            # restoring checkpoint
            checkpoint = RSTGPT2_TrainingModule_dyploc.get_ckpt_file(
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
                                                    val_check_interval=0.5,
                                                    reload_dataloaders_every_n_epochs=1,
                                                    num_sanity_val_steps=2,
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
            checkpoint = RSTGPT2_TrainingModule_dyploc.get_ckpt_file(
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

            checkpoint = RSTGPT2_TrainingModule_dyploc.get_ckpt_file(
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
    
    def step_end(self, output, step_name):
        
        if step_name == "train":
            loss_key = "loss"
            loss = output[loss_key].mean()
            on_step = True
            on_epoch = False
        else:
            loss_key =  f"{step_name}_loss"
            loss = output[loss_key].mean()
            on_step = False
            on_epoch = True

        self.log(loss_key, loss, logger=True, on_step=on_step, on_epoch=on_epoch, sync_dist=True )

    def training_step(self, batch, batch_idx):
        output = self.step(batch, "train")
        return output

    def validation_step(self, batch, batch_idx):
        output = self.step(batch, "val")
        return output

    def test_step(self, batch, batch_idx):
        output = self.step(batch, "test")
        return output
    
    def training_step_end(self, output ):
        return self.step_end(output, "train")
    
    def validation_step_end(self, output):
        return self.step_end(output, "val")

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
        
        if step_name == "val" and _get_rank() == 0:
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

                    df = pd.DataFrame(columns=['epoch', 'rst_rels', 'keyphrase', 'utterance',
                                            'dict_pos_edu', 'li_kprstpos',

                                            'rst_ns',
                                            'rst_pos',
                                            ])

                    rst_rels = encoded_input.pop('orig_rst_rels')
                    rst_ns = encoded_input.pop('orig_rst_ns')
                    rst_pos = encoded_input.pop('orig_rst_pos')

                    keyphrase = encoded_input.pop('orig_key_phrase')
                    utterance = encoded_input.pop('orig_utt')
                    dict_pos_edu = encoded_input.pop('orig_dict_pos_edu')

                    orig_li_kprstpos = encoded_input.pop('orig_li_kprstpos')

                    datum = {
                        'epoch': -1,

                        'rst_rels': ', '.join(rst_rels),
                        'rst_ns': ', '.join(rst_ns),
                        'rst_pos': rst_pos,

                        "keyphrase": ', '.join(keyphrase),
                        "utterance": utterance,
                        "dict_pos_edu": json.dumps(dict_pos_edu),

                        "li_kprstpos": json.dumps(orig_li_kprstpos),
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
                # encoded_input.pop('labels', None)

                generation_params = copy.deepcopy(self.model.generation_params)
                generation_params['max_length'] = 60
                generation_params['time'] = 20
                decoded_text = self.model.generate_plus( encoded_input, generation_params )

                datum = {
                    'epoch': self.current_epoch,
                    'rst_rels': '',
                    'keyphrase': '',
                    'utterance': json.dumps(decoded_text),
                    'dict_pos_edu': '',
                    'li_kprstpos': '',
                    'rst_ns': '',
                    'rst_pos': ''
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
        if self.train_dl_used == False:
            self.train_dl_used = True
            return self.train_dl
        else:
            self.train_dl =  self.dg.prepare_dataloader(
                split_name='train')
            return self.train_dl

    def val_dataloader(self):
        # return self.dg.prepare_dataloader(
        #         split_name='val')
        return self.val_dl 

    def test_dataloader(self):
        return self.test_dl

    @lru_cache()
    def total_steps(self):

        ds_size = len(self.train_dl) // self.gpus
        steps = (ds_size * self.max_epochs) // (self.accumulate_grad_batches)
        return steps

    def configure_optimizers(self):


        optimizer = Adafactor(self.model.parameters(), scale_parameter=True, 
                        relative_step=True, warmup_init=True, lr=None )


        lr_scheduler = AdafactorSchedule(optimizer)

        return { 'optimizer':optimizer, "lr_scheduler": lr_scheduler, "interval": "step", "monitor": "val_loss"}
    
    def return_params(self):
        params = {}
        keys = ['batch_size', 'accumulate_grad_batches', 'learning_rate', 'max_epochs', 'dir_data'
                'tag']

        params = {
            k: self.__dict__[k] for k in keys if k in self.__dict__.keys()
        }

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
                 batching_style='effecient',
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
                           split_name='train'):
        """Prepares a dataloader given a directory of data for NLG language module
            # The current method takes a percentage of data from each subdirectory
            Args:
                dir_dset ([type]): [description]
        """
        dir_data = self.dir_data

        # getting all files from all different subreddits/types of conversation
        #debugging
        fns = glob.glob(os.path.join(utils.get_path(dir_data), "*", "*"))
        fns = [fn for fn in fns if os.path.split(
            fn)[-1] != "lock" and "dict_len" not in fn]#[:10 ]
                
        # getting number of utterances records in each file
        files_sizes = [int(fn[-10:]) for fn in fns]

        # defining starting line and total lines to use for dataset
        if split_name == 'train':

            fn = glob.glob(  os.path.join( dir_data,"train","*") )[0]
            shuffle = True
            inference = False
            bs = self.batch_size

        elif split_name == 'val':
            fn = glob.glob(  os.path.join( dir_data,"val","*") )[0]
            shuffle = False
            inference = False
            bs = self.batch_size

        elif split_name == 'test':
            fn = glob.glob(  os.path.join( dir_data,"test","*") )[0]
            shuffle = False
            bs = self.batch_size
            inference = False

        elif split_name == 'inference':
            fn = glob.glob(  os.path.join( dir_data,"test","*") )[0]
            shuffle = True
            bs = 1
            sampler = None
            inference = True

        ds =SingleDataset(fn, self.tokenizer,inference )


        dataloader = torch.utils.data.DataLoader(ds, 
                                                batch_size= bs ,
                                                 num_workers=self.workers,                                                
                                                 sampler=sampler,
                                                 pin_memory=True,
                                                 collate_fn=self.tokenizer.default_collate_pad,
                                                )

                                                 
        return dataloader

# concat basically makes all entries one long list of sequential indexes
# sampler creates a randomised index list to sample from list above
# In smapler, we can access concat dataset and each individual dataset
# In each dataset add a list of the rst_len_pad

class SingleDataset(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

        create a custom index which sorts the entries by their length
    """
    def __init__(self, file_path, tokenizer, line_start, line_end, inference,**kwargs):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.inference = inference

        self.data = pd.read_csv(self.fp, sep=',', header=0 )

        fp_cached_order = os.path.join(os.path.dirname(
            file_path), f"gpt2_dict_lens.pkl")

        # # resetting the cached order files
        # if os.path.exists( fp_cached_order) and "australia" in self.fp:
        #     os.remove(fp_cached_order)

    
        if os.path.exists(fp_cached_order):
            dict_cached_order = pickle.load(open(fp_cached_order, "rb"))
            self.np_textlens = dict_cached_order['np_textlens']
            self.np_rstlens = dict_cached_order['np_rstlens']
            self.np_keyphrase_lens = dict_cached_order['np_keyphrase_lens']

        else:
            # len of text

            self.np_textlens = np.stack(
                [self.tokenizer.encode(ujson.loads(txt), return_tensors='np', add_special_tokens=False,
                                    truncation=False, padding='do_not_pad').size for txt in self.data.txt_preproc.values.tolist()]
            )

            # len of rst
            self.np_rstlens = np.array(
                [1 + len(json.loads(rst)) for rst in self.data.rst.values.tolist()])

            # len of keyphrase

            li_li_pos_kp = [ json.loads(li_pos_kp) for li_pos_kp  in self.data.li_pos_kp.values.tolist() ]

            li_li_kp = [ [ kp for pos,kp in li_pos_kp]  for li_pos_kp in li_li_pos_kp ]

            li_kp = [ ''.join(['<|kp|> '+ kp for kp in li_kp]) for li_kp in li_li_kp  ]

            self.np_keyphrase_lens = np.array( [ self.tokenizer.encode(kp, 
                                        add_special_tokens=False, 
                                        truncation=False,
                                        padding = 'do_not_pad',
                                        return_tensors=None).__len__() for kp in li_kp ] )


            dict_cached_order = {'np_textlens': self.np_textlens,
                                'np_rstlens': self.np_rstlens,
                                'np_keyphrase_lens': self.np_keyphrase_lens}

            pickle.dump(dict_cached_order, open(fp_cached_order, "wb"))

        # # These ones below are used during the RSTTokenizers encode_input function to pad rst and kp
        # #v1 We initialize the rst/kp  lengths as the max lengths of our training set
        # # In the Sampler, we change the max length to that of its pre-prescribed batch
        # self.rst_len = [tokenizer.max_len_rst]*self.__len__()
        # self.key_phrase_len = [tokenizer.max_len_key_phrase]*self.__len__()

        #v2 We initialize the rst/kp lengths as the actual length of each entry
        # In the Sampler, we change the max length to that of its pre-prescribed batch
        self.rst_len = self.np_rstlens
        self.key_phrase_len = self.np_keyphrase_lens

        self.data = self.data.to_dict('records')

    def __len__(self):
        return len( self.data )

    def __getitem__(self, index):

        rst_rels, rst_ns, rst_pos, li_kp, li_kprstpos, utterance, dict_pos_edu, claim = self.getitem_extract_datum(
            index)

        if self.inference == True:

            utterance_prompt = ' '.join(utterance.split(' ')[:2])
            encoded = self.tokenizer.encode_input(rst_rel=rst_rels, rst_ns=rst_ns, rst_pos=rst_pos,
                                                  li_kp=li_kp,
                                                  li_kprstpos=li_kprstpos,
                                                  utterance_prompt=utterance_prompt,
                                                  dict_pos_edu=dict_pos_edu,
                                                  rst_len=self.rst_len[index],
                                                  key_phrase_len=self.key_phrase_len[index]
                                                  )

            encoded['orig_rst_rels'] = rst_rels
            encoded['orig_rst_ns'] = rst_ns
            encoded['orig_rst_pos'] = rst_pos

            encoded['orig_utt'] = utterance
            encoded['orig_key_phrase'] = li_kp

            encoded['orig_dict_pos_edu'] = dict_pos_edu
            encoded['orig_li_kprstpos'] = li_kprstpos
            encoded['orig_claim'] = 'claim'


        elif self.inference==False:
            encoded = self.tokenizer.encode_input(
                rst_rels, rst_ns, rst_pos,
                li_kp=li_kp,
                li_kprstpos=li_kprstpos,
                utterance=utterance,
                dict_pos_edu=dict_pos_edu,
                rst_len=self.rst_len[index],
                key_phrase_len=self.key_phrase_len[index]
            )

        encoded['claim'] = 'claim'
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
            li_kprstpos, li_kp = zip(*li_pos_kp)
            li_kprstpos = tuple(int(pos) for pos in li_kprstpos)
        else:
            li_kp = []
            li_kprstpos = []

        # Utterance
        utterance = ujson.loads(datum['txt_preproc'])

        # claim
        li_claim = ujson.loads(datum['li_claim'])[0]
        
        #pos and edus
        dict_pos_edu = json.loads(datum['dict_pos_edu'])   


        return rst_rels, rst_ns, rst_pos, li_kp, li_kprstpos, utterance, dict_pos_edu, li_claim

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
    training_module = RSTGPT2_TrainingModule_dyploc.instatiate_training_module(
        tparams, mparams)
    trainer, training_module = RSTGPT2_TrainingModule_dyploc.instatiate_trainer(
        tparams, tb_logger, training_module)
    RSTGPT2_TrainingModule_dyploc.start(trainer, tparams, training_module, mparams)

if __name__ == '__main__':

    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)

    # add model specific args
    mparams = RSTGPT2.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = RSTGPT2_TrainingModule_dyploc.parse_train_specific_args(parent_parser)

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
# CUDA_VISIBLE_DEVICES=0 python3 train_RSTGPT2.py --batch_size 60 --version 2 --precision 16 --mode finetune --workers 13 --scale_grad_by_freq 1 --max_epochs 50 --gpus 1 --tag RSTGPT2 --max_len_utt 180 --max_len_rst 20 --max_len_key_phrase 38 --tag RSTGPT2 --learning_rate 3e-4 

# CUDA_VISIBLE_DEVICES=0 python3 train_RSTGPT2.py --batch_size 32 --version 12  --precision 16 --mode finetune --workers 14 --scale_grad_by_freq 1 --max_epochs 50 --gpus 1 --max_len_utt 190 --max_len_rst 28 --max_len_key_phrase 40 --tag "RSTGPT2 with normal attention scheme"
# CUDA_VISIBLE_DEVICES=1 python3 train_RSTGPT2.py --batch_size 32 --version 13  --precision 16 --mode finetune --workers 14 --rst_tree_aligned_attention 1 --scale_grad_by_freq 1 --max_epochs 50 --gpus 1 --max_len_utt 190 --max_len_rst 28 --max_len_key_phrase 40 --tag "RSTGPT2 with aligned attention scheme"
