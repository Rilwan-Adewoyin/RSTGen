import os

os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ['TOKENIZERS_PARALLELISM'] = "true"
import string
import argparse
import copy
import glob
import inspect
import json
from train_RSTGPT_arggen_pair4 import SingleDataset, SizedOrderedBatchSampler, RSTGPT2_Config, SizedOrderedDistributedBatchSampler, RSTTokenizerPair, RSTGPT2Pair
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
from torch.utils.data.sampler import Sampler
from pytorch_lightning.callbacks.finetuning import BaseFinetuning

from transformers.optimization import Adafactor, AdafactorSchedule, AdamW
from transformers.tokenization_utils_base import AddedToken

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
from train_RSTGPT import RSTGPT2, RSTGPT2_Config, RSTTokenizer, RSTGPT2_TrainingModule

from transformers.utils import logging
logger = logging.get_logger(__name__)
        
class RSTGPT2Pair_TrainingModule(pl.LightningModule):

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
            'labels': mconfig.max_len_rst + mconfig.max_len_key_phrase + mconfig.max_len_utt + mconfig.max_len_title,

            'attention_mask': mconfig.max_len_rst + mconfig.max_len_key_phrase + mconfig.max_len_utt + mconfig.max_len_title,  # axis:max_length

            'position_ids_kp_utt': mconfig.max_len_key_phrase+mconfig.max_len_utt,

            'position_ids_keyphrase':mconfig.max_len_key_phrase,
            'position_ids_utt':mconfig.max_len_utt ,

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
                                 batching_style=self.batching_style)

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
            model.load_state_dict( pytorch_state_dict )
            model.correct_attn_bias()

            
                
            tokenizer = RSTTokenizerPair.from_pretrained(**mparams)
            model.resize_token_embeddings(model.config.vocab_size)
            # set initiation value of new token to that of 
            # with torch.no_grad():
            #     model.transformer.wte.weight[ -1:, : ] = model.transformer.wte.weight[ -4:-3, : ]

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
                    'max_len_title','scale_grad_by_freq','rst_tree_aligned_attention' ]})

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
                    'base_model_name', 'model_name', 'max_len_utt','max_len_rst','max_len_key_phrase']})
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
                                              mode='min', dirpath=dir_checkpoints,
                                              filename='{epoch:03d}_{val_loss:.5f}')

        checkpoint_callback._save_model = types.MethodType(
            mpatch_save_model(checkpoint_callback._save_model), checkpoint_callback) 

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience = 16,       
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
                            'plugins' : DDPPlugin(find_unused_parameters=True)
                            }

        if tparams['mode'] in ["finetune"]:

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    default_root_dir=tparams['dir_checkpoints'],
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks,
                                                    replace_sampler_ddp=False,
                                                    num_sanity_val_steps=0,
                                                    val_check_interval=0.9999,
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
                                                    reload_dataloaders_every_n_epochs=1,
                                                    num_sanity_val_steps=0,
                                                    val_check_interval=0.9999,

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

        elif tparams['mode'] in ['infernece']:
            training_module.eval()
            training_module.freeze()
            raise NotImplementedError

    def forward(self, input_):
        with torch.cuda.amp.autocast(enabled=True):
            return self.model(**input_, return_dict=True)

    
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
            

        elif split_name == 'inference':
            fn = filter_fns(glob.glob(  os.path.join( self.dir_data,"test","*") ))
            shuffle = False
            bs = 1
            sampler = None
            inference = True
            sample_kps = False
            
        ds = SingleDataset(fn, copy.deepcopy( self.tokenizer), inference, sample_kps=sample_kps )

        if self.gpus <= 1 and split_name not in ['inference', 'test']:
            sampler = SizedOrderedBatchSampler(ds, bs, True, shuffle=True) if sampler else sampler
            bs = None
        
        else:
            sampler = SizedOrderedDistributedBatchSampler(ds, bs, drop_last=True, shuffle=shuffle, gpus=self.gpus)
            bs = None


        dataloader = torch.utils.data.DataLoader(ds, 
                                                 batch_size=bs,
                                                 num_workers=self.workers//2 if sampler else 1, 
                                                 batch_sampler = sampler,
                                                 pin_memory=True,
                                                 collate_fn=self.tokenizer.default_collate_pad,
                                                )

                                                 
        return dataloader

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
        os.environ['MASTER_PORT'] = '65502'

    try:
        main(vars(tparams), vars(mparams))
    except Exception:
        print(traceback.format_exc())

# CUDA_VISIBLE_DEVICES=1 python3 train_RSTGPT_arggen_pair.py --batch_size 12 --version 11 --workers 6 --scale_grad_by_freq 1 --tag "RSTGPT2 Dyploc with rst aligned attn" --max_len_utt 240 --max_len_rst 36 --max_len_key_phrase 64 --finetune_version 11 --max_len_key_phrase 40 --rst_aligned_attention 1 --rst_segment_method segbot
# CUDA_VISIBLE_DEVICES=1 python3 train_RSTGPT_arggen_pair.py --batch_size 12 --version 12 --workers 6 --scale_grad_by_freq 1 --tag "RSTGPT2 Dyploc with non rst aligned attn" --max_len_utt 240 --max_len_rst 36 --max_len_key_phrase 64 --finetune_version 12 --max_len_key_phrase 40 --rst_aligned_attention 0 --rst_segment_method segbot
