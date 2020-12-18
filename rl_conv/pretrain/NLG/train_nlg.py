import numpy as np
import warnings
import sklearn


import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import glob
import pandas as pd
import json
from functools import lru_cache

from typing import List

import pickle


from itertools import cycle, islice
from torch.utils.data._utils.collate import default_convert, default_collate

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

import pytorch_lightning as pl

from sklearn import preprocessing as sklp

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import utils_nlg as utils
import random 

from pytorch_lightning import loggers as pl_loggers

from collections import OrderedDict
import yaml

#Monkey Patching 
#TODO: suggest this change on github pytorch lightning 
def monkey_save_model(self, filepath: str, trainer, pl_module):
    # in debugging, track when we save checkpoints
    trainer.dev_debugger.track_checkpointing_history(filepath)

    # make paths
    if trainer.is_global_zero:
        self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)

    # delegate the saving to the trainer
    if self.save_function is not None:
        self.save_function(filepath, self.save_weights_only)
    
    self.to_yaml()

ModelCheckpoint._save_model = monkey_save_model


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        return input_ * torch.tanh(F.softplus(input_))


class NLG(nn.Module):
    """NLG unit
    """

    def __init__(self, 
        base_model_name= 'distilgpt2', model_name="NLG",
        reset_base_transformer=True, loss_type="CrossEntropy",
        **kwargs):
            #base model uses same code as 'microsoft/DialoGPT-small'
        super(NLG, self).__init__()
        
        self.base_model_name = base_model_name   
        self.model_name = model_name
        self.reset_base_transformer = reset_base_transformer

        # Retreive/Instantiate base transformer
        self.transformer = utils.load_pretrained_transformer(self.base_model_name, transformer=True)['transformer']    
        
        # Optional Reset the Transformer model
        if self.reset_base_transformer:
            for param in self.transformer.parameters():
                param.reset_parameters() # re-initializing weights

        self.nlg_tokenizer = utils.load_pretrained_tokenizer_local( model_name = model_name)
        if self.nlg_tokenizer == False:
            self.nlg_tokenizer = NLG_tokenizer(base_model_name)
            self.tokenizer.save("./models/roberta/tokenizer.json") 
        
        
        self.transformer.resize_token_embeddings( len(self.nlg_tokenizer.e2m_tokenizer) )

        # Embedding Layers
        self.embd_outp_dim = self.transformer.config.n_embd
        self.embedding_das = torch.nn.conv1d(1, self.embd_outp_dim, kernel=20)
        #self.embedding_das_pos = torch.nn.Embedding( 6, self.embd_outp_dim )
        self.embedding_rst_rels = torch.nn.conv1d( 1, self.embd_outp_dim, kernel=16)
        self.embedding_rst_ns = torch.nn.conv1d( 1, self.embd_outp_dim, kernel=3)
        self.embedding_rst_pos = torch.nn.Embedding( 6, self.embd_outp_dim )
        self.embedding_general = torch.nn.Embedding( len(self.nlg_tokenizer.e2m_tokenizer) , self.embd_outp_dim, sparse=True, scale_grad_by_freq=True  )
            #TODO: consider initializing embedding from a pre-existing GPT-2 embedding layer 
        self.embedding_topics_score = torch.nn.conv1d(1, self.embd_outp_dim, kernel=1)
        #Remove the Transformer models embedding layers


        
        
    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
                
        parser.add_argument('--base_model_name', default='distilgpt2', required=False)
        parser.add_argument('--reset_base_transformer', default=True, required=False, type=bool)
        #parser.add_argument('--model_name', default='NLG', required=False)
        parser.add_argument('--loss_type', default='CrossEntropy', required=False, 
            choices=['CrossEntropy','UtteranceSimilarity']) 
        
        mparams = parser.parse_known_args( )[0]
        if mparams.config_file != None:
            mparams = json.load(open(utils.get_path(path)),"r" )
        
        return mparams

    def forward(self, input_):
        """[summary]

        Args:
            input_ (torch.tensor): dict of inputs

        Returns:
            [type]: [description]
        """
        # Creating embedded inputs and attention mask
        input_embed = layer_embed( input_ )

        # Feed input to distilgpt2
        if self.loss_type == "CrossEntropy":      
            outputs = self.transformer( inputs_embeds=input_embed,
                                        attention_mask = input_['attn_mask'],
                                        labels= input_['labels'] ,
                                        position_ids=None, #check pos_ids are being removed
                                        return_dict=True )
        
        elif self.loss_type == "UtteranceSimilarity":
            raise NotImplementedError
        

        return outputs #dictionary
    
    def layer_embedding(self, input_):
        """Creates embedding for input

        Args:
            input_ ([type]): [description]

        Returns:
            input_embedded [type]: [description]
            

        """
        #cls_token = self.embedding_general( input_[cls_token])
        da_start_token = self.embedding_general( input_['da_start_token'] )
        das_word = self.embedding_das(input_['tnsr_das'])
        #das_pos = self.embedding_das_pos(input_['tnsr_da_pos'])
        das = das_word
        
        rst_start_token = self.embedding_general( input_['rst_start_token'] )        
        rst_rels = self.embedding_rst_rels( input_['tnsr_rst_rels'] )
        rst_ns = self.embedding_rst_rels( input_['tnsr_rst_ns'] )
        rst_pos = self.embedding_rst_pos( input_['tnsr_rst_pos'] )
        rst = rst_rels + rst_ns + rst_pos

        topics_start_token = self.embedding_general( input_['topics_start_token'])
        topics_phrase = self.embedding_general( input_['tnsr_topics_phrase']) #TODO: Handle Case where a phrase is predicted (flatten)
        topics_score = self.embedding_topics_score( input_['tnsr_topics_score'])
        topics = topics_phrase + topics_score

        eodrt_token = self.embedding_general( input_['eodrt_token'])
        embed_utt = self.embedding_general( input_['tknzd_utt'] )
        eos_token  =self.embedding_general( input_['eos_token'] )
        padding  = self.embedding_general( input_['padding'] ).repeat( 1, input_['padding_count'], 1 )

        
        input_embedded = torch.concat(
            [da_start_token, das,
             rst_start_token, rst,
             topics_start_token, topics,
             eodrt_token,
             embed_utt,
             eos_token,
             padding], axis = 1
        ) #dim [bs, 1024, dim1]
        

        return input_embedded

    def return_params(self):
        params = {}

        params['base_model_name'] = self.batch_size
        params['reset_base_transformer'] = self.accumulate_grad_batches
        params['loss_type'] = self.loss_type 

        return params

    def get_predicted_utterance(self):
        """Given an input sequence, outputs predictions up until an <EOS> is emmitted

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """

class NLG_tokenizer():
    """Rough Implmentation of the tokenizer for the NLG model

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """

    def __init__(self,
                 e2m_base_model_name='distilgpt2'):

        # RST utilities
        self.rst_rel_li = ['Attribution',
            'Background','Cause','Comparing','Condition',
            'Contrast','Elaboration','Enablement','Evaluation',
            'Explanation','Joint','Manner-Means','Topic-Comment',
            'Summary','Temporal','Topic Change']

        self.rst_rel_binarizer = sklp.MultiLabelBinarizer()
        self.rst_rel_binarizer.fit( [ rst_rel_li ] )

        self.rst_ns_li = ['NN','NS','SN',None]
        self.rst_ns_binarizer = sklp.MultiLabelBinarizer()
        self.rst_ns_binarizer.fit( [ rst_ns_li ] )

        # DA utilities

        # Entities 2 Mention Utilities
            # Will use BERTS's encoding for words
        self.e2m_base_model_name = e2m_base_model_name
        self.e2m_tokenizer = utils.load_pretrained_transformer(self.e2m_base_model_name , transformer=False, tokenizer=True)['tokenizer']

        # Other
        self.max_input_len = 1024
            # change this to make new tokenizer if it doesnt already exist
        special_tokens_dict = {'dialogue_act': '[DA]'
                                'rhetorical_structure_theory':'[RST]',
                                'topic': '[TA]',
                                'end_of_drt':'[EODRT]'}

        num_added_toks = self.e2m_tokenizer.add_special_tokens( special_tokens_dict )

    
    def encode( self, rst_rels, rst_ns, rst_pos, das,
        topics, topics_score, utterance ,prev_das=None, prev_rst=None,
        labels=True):

        """Return 

            attn_mask : Bidirectional up to EODRT token, Causal Up till EOS, 0s till end of padding

        Note this method returns integer encodings for tokens that will be processed by BERT embedding layer
            and possibly one-hot encoded vectors that will not be encoded by same pert embedding layer
        """
        
        #Getting Vectors
        tnsr_das, tnsr_da_pos = self.encode_da( das ) #dims (n2, 20), (n2,) # Use an upscaling convolution layer  and an embedding layer
        tnsr_rst_rels, tnsr_rst_ns, tnsr_rst_pos = self.encode_rst(rst_rels, rst_ns, rst_pos)   # dims (n1, 13) (n1, 3) (n1,) # Use an upscaling convolution layer, upscaling convolution layer, embedding layer  
        tnsr_topics_phrase, tnsr_topics_score = self.encode_topic( topics, topics_score) #dims (n3, ) (n3, )  # Use an embedding layer E, upscaling convolutional layer
        tknzd_utt = self.encode_utterance(utterance)
        
            #This will already have a EOS token
            #TODO: make sure to remove CLS token

        #Getting Special Tokens
        #cls_token = self.e2m_tokenizer.token_to_id("[CLS]")     # Use same embedding layer E
        da_start_token = self.e2m_tokenizer.token_to_id("[DA]") # Use same embedding layer E        
        rst_start_token = self.e2m_tokenizer.token_to_id("[RST]") # Use same embedding layer E
        topics_start_token = self.e2m_tokenizer.token_to_id("[TA]") # Use same embedding layer E
        eodrt_token = self.e2m_tokenizer.token_to_id("[EODRT]") #(end of da, rst, topic ) # Use same embedding layer E 
        eos_token = self.e2m_tokenizer.token_to_id("<|endoftext|>")   # Use same embedding layer E
        
        
        # adding padding
        seq_len_nopad = rst_vectors.shape[0] + da_vectors.shape[0] + 
                            topics_vectors.shape[0] + tknzed_utt.shape[0] + 5
        
        padding_req = self.max_input_len - seq_len_nopad
        padding =  self.e2m_tokenizer.token_to_id("[PAD]") # tnsr of padding tokens to make 
        
        # making attn_mask section by section
        preutt_dim = tnsr_das.size[0] + 
                           tnsr_rst_rels.shape[1]+
                           tnsr_topics_phrase.shape[1]+
                           4 # da,rst,topics,eordt tokens
        utt_dim = tknzd_utt.size + 1 # +1 for EOS token
        posteos_dim  = padding_req
        
        mask = torch.tril( torch.ones([self.max_input_len,self.max_input_len]))
        mask[ :preutt_dim , :preutt_dim ] = 1 #pre_utt masking
        mask[ preutt_dim:preutt_dim+utt_dim,
                preutt_dim:preutt_dim+utt_dim] = torch.tril( 
                    torch.ones([ utt_dim,utt_dim]) ) #utt_dim
        mask[ -posteos_dim: , -posteos_dim: ] = 0 #posteos dim

        #Creating labels/targets for GPT Language Model Head
        start_idx = preutt_dim
        end_idx = preutt_dim + utt_dim
        labels = -100* torch.ones( [ 1024, 1] )
        labels[start_idx:end_idx-1] = tknzd_utt
        labels[start_idx:end_idx-1] = eos_token

        return { #'cls_token': cls_token,
                 'da_start_token':da_start_token, 'tnsr_das':tnsr_das, 'tnsr_da_pos':tnsr_da_pos, 
                 'rst_start_token':rst_start_token, 'tnsr_rst_rels':tnsr_rst_rels,'tnsr_rst_ns':tnsr_rst_ns,'tnsr_rst_pos':tnsr_rst_pos,
                 'topics_start_token':topics_start_token, 'tnsr_topics_phrase':tnsr_topics_phrase, 'tnsr_topics_score', tnsr_topics_score,
                 'eodrt_token': eodrt_token, 'tknzd_utt':tknzd_utt, 'eos_token':eos_token
                        #TODO: possibly text generated up until step i here
                        'padding_token':padding, 'padding_count':padding_count,
                        'attn_mask':attn_mask,
                        'labels':labels}

    
    def encode_rst(self, rst_rels, rst_ns, rst_pos):
        """Converts the three lists into a seeries of vectors

        Args:
            rst_rels ([type]): [description]
            rst_ns ([type]): [description]
            rst_pos ([type]): [description]
        """
        rst_rels_encoded = self.rst_rel_binarizer.transform( rst_rels )
        rst_ns_encoded = self.rst_ns_binarizer.transform( rst_ns )
        rst_pos = rst_pos

        tnsr_rels = torch.from_numpy( rst_rels_encoded ) #dim [ n, encode_dim1]
        tnsr_ns = torch.from_numpy( rst_ns_encoded )    #dim [ n, encode_dim2]
        tnsr_pos = torch.from_numpy( rst_pos )          #dim [ n, encode_dim3]
        
        #tnsr_rst = torch.cat( [tnsr_rels, tnsr_ns, tnsr_pos ] , axis=1  )

        return tnsr_rels, tnsr_ns, tnsr_pos
    
    def encode_da(self, das):
        """[summary]

        Args:
            das ([type]): [list of da probabilites]
        """
        #TODO: add some normalization of da probabilities here
        tnsr_das = torch.FloatTensor( das) #dim [n, encode_dim1]
        tnsr_pos = torch.unsqueeze( torch.range(start=1, end=tnsr_das.shape[0], dtype=torch.float32 ), dim=1 )

        #tnsr_daspos = torch.cat( [tnsr_das, tnsr_pos], axis=1)
        
        return tnsr_das, tnsr_pos

    def encode_topic(self, topics, topics_score):
        """[summary]

        Args:
            topics ([type]): [list of topics (phrases or words)]
            topics_score ([type]): [list of float scores for each topic relevancy]

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """

        tnsr_topics_phrase = self.e2m_tokenizer( topics, add_special_tokens=False,
                                                padding=None, truncation=3, 
                                                return_tensors='pt',
                                                return_token_type_ids=False) # shape () topic_count, )
        
        tnsr_score = torch.unsqueeze( torch.FloatTensor( topics_score ) , dim=-1 ) # shape (topic_count, )

        # tnr_topic = torch.cat( [tnsr_topics_phrase, tnsr_score], axis=1 )

        return tnsr_topics_phrase, tnsr_score

    def encode_utterance(self, utterance):
        tknzd_utt = self.e2m_tokenizer(utterance,add_special_tokens=False,
                                                padding=None, truncation=300, 
                                                return_tensors='pt',
                                                return_token_type_ids=False)
        
        return tknzd_utt

class TrainingModule(pl.LightningModule):

    def __init__(self, model_params, batch_size=20, 
                    dir_data=None, 
                    accumulate_grad_batches=1,
                    max_epochs=100,
                    gpus=1, 
                    learning_rate=1e-3,
                    warmup_proportion=0.1,
                    workers=1,
                    lr_schedule='LROnPlateau',
                    mode = 'train_new',
                    *args,
                    **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.gpus =  gpus
        self.model = NLG( **model_params )
        self.mode = mode
        self.workers = workers
        
        if self.mode in ['train_new','train_cont','test']:
            self.create_data_loaders(self.workers)
            self.dir_data = utils.get_path(dir_data)
            self.accumulate_grad_batches = accumulate_grad_batches

        if self.mode in ['train_new','train_cont']:
            self.max_epochs = max_epochs
            self.warmup_proportion = warmup_proportion
            self.lr_schedule = lr_schedule
            self.learning_rate = learning_rate
        

        if self.mode in ['train_new']:
            train_params_to_save = self.return_params()
            model_params_to_save = self.model.return_params()
            self.save_hyperparameters( train_params_to_save )
            self.save_hyperparameters( model_params_to_save )

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
        parser.add_argument('--dir_data', default="./dataset/reddit_small_mc", help="Relative directory path for datafiles")
        parse.add_argument('--dir_model', defailt="./model/")
        parser.add_argument('--max_epochs', default=150, type=int)
        parser.add_argument('--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('--batch_size', default=20, type=int)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--warmup_proportion', default=0.05)
        parser.add_argument('--workers', default=0, type=int)
        parser.add_argument('--gpus', default=0, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','test','train_cont','inference'])
        parser.add_argument('--lr_schedule', default='LROnPlateau', required=False, choices =['LROnPlateau','hard_restarts'])
        parser.add_argument('--splits', default='{"train":0.6, "val":0. 2, "test":0.2}',required=False, type=json.loads )
        parser.add_argument('--version', default=None,required=False, type=int, help="The Experimental Versioning for this run" )
        
            #TODO: check --version of required type None actually works

        tparams = parser.parse_known_args()[0]
        if tparams.config_file != None:
            tparams = json.load(open(utils.get_path(tparams.config_file)),"r" )

        return tparams
    
    @staticmethod
    def instatiate_training_module( tparams=None, mparams=None, model_dir=None ):
        """Create training module

        Args:
            tparams ([type]): [description]
        """
        
        if tparams['mode'] in ["train"]:
            training_module = TrainingModule(**vars(tparams), model_params=mparams  )

        elif tparams['mode'] in ["test", "train_cont", "inference"]:
            #Retreiving best checkpoint from specific model-version
            checkpoint_dir = tparams['checkpoint_dir'] if tparams['checkpoint_dir']!=None \
                                else f"{model_dir}/{mparams['model_name']}/version_{tparams['version']:03d}/"
            
            checkpoint_callback = 
            checkpoint_path  = checkpoint_callback.best_model_path

            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

            #restore/update param files from the logs yaml
            tparams = {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                'batch_size', 'accumulate_grad_batches', 'lr_schedule', 'learning_rate',
                'max_epochs','warmup_proportion']} 

            mparams = {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                'base_model_name','reset_base_transformer','loss_type']} 

            os.makedirs(checkpoint_dir, exist_ok=True)

            training_module = TrainingModule(**vars(tparams), mparams=mparams)
            training_module.load_state_dict(checkpoint['state_dict'])
        
        return training_module

    @staticmethod
    def instatiate_trainer( tparams, tb_logger, dir_model) ):
        """[summary]

            Creates The Trainer and callbacks
        """
        # Creating Callbacks
        callbacks = []
        dir_checkpoints = os.path.join(dir_model,'checkpooints',mparams['model_name'],f"version_{tparams['version']:03d}" )
        os.makedirs(dir_checkpoints, exist_ok=True)
        
        if tparams.mode in ["train_new"]:
            
            checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=2, 
                mode='min', dirpath=dir_checkpoint, 
                filename='{epoch:03d}_{val_loss:.5f}')
            
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=0.00,
                patience=8,
                verbose=False,
                mode='auto'

            )
            callbacks.append(checkpoint_callback)
            callbacks.append(early_stop_callback)

            trainer = pl.Trainer.from_argparse_args(tparams, progress_bar_refresh_rate=1,
                        check_val_every_n_epoch=1, logger=tb_logger,
                        log_every_n_steps=5,
                        precision=16, callbacks=callbacks,
                        #track_grad_norm = True,
                        #overfit_batches=5
                        #,fast_dev_run=2, 
                        #log_gpu_memory=True
                        )

        if tparams.mode in ["train_cont","test","inference"]:
             
            checkpoint_yaml_file = os.path.join( dir_checkpoints,"best_k_models.yaml" )
            scores_dict = yaml.load( open(checkpoint_yaml_file,"r") ) #key= ckptpath, value = val_loss
            best_ckpt_path = min(scores_dict, key=scores_dict.get)

            trainer = pl.Trainer.from_argparse_args(tparams, progress_bar_refresh_rate=1,
                    resume_from_checkpoint = best_ckpt_path
                    check_val_every_n_epoch=1, logger=tb_logger,
                    log_every_n_steps=5,
                    precision=16
                    #track_grad_norm = True,
                    #overfit_batches=5
                    #,fast_dev_run=2, 
                    #log_gpu_memory=True
                    )
            
            return trainer
        
    @staticmethod
    def start(trainer, tparams, training_module ):
        
        if tparams.mode in ["test"]:
            training_module.eval() 
            training_module.freeze() 
            trainer.test(test_dataloaders=training_module.test_dl, model=training_module, ckpt_path='best')
            
            
        elif tparams.mode in ['train_new','train_cont']:    
            trainer.fit(training_module)
            
            trainer.checkpoint_callback.to_yaml()

            trainer.test(test_dataloaders=training_module.test_dl, ckpt_path='best')
        
        elif tparams.mode in ['infernece']: 
            raise NotImplementedError   

           
    def step(self, batch, step_name):

        #target = batch.pop('tknzed_target')
       
        input_= batch
        output = self.forward(input_)

        #TODO: Make version where loss is based on meaning of the true utterance and my predicted utterance
        loss = output['loss']

        #TODO: removing losses on utterances that had no valid rst relation tag
            # requires loss to not already be batch reduced
        keep_mask = batch['tnsr_rst'] != None
        loss = loss[keep_mask]

        loss_key = f"{step_name}_loss"
        
        output =  output.to('cpu')


        if step_name == 'train':
            str_loss_key = "loss"
            on_step = True
            on_epoch = False
            prog_bar = True
            logger = False

        else:
            str_loss_key = loss_key
            on_step = False
            on_epoch = True
            prog_bar = False
            logger = True
        
            self.log( str_loss_key, loss)#, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)
        
        _dict = { str_loss_key: loss }
                
        #self.log(f'{step_name}_l', self.dict_acc[step_name].compute(), on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)
        
        return  _dict 
        
    #@auto_move_data
    def forward(self, input_, *args):
        
        return self.model(input_)

    def get_predicted_utterance(self, output):
        """Converts list of logit scores for Dialogue Acts
            to a list of OrderedDictionaries where
            each dict contains the DA names and probabilities 


        Args:
            preds ([type]): [description]

        Returns:
            [type]: [description]
        """
        predicted_utterance = self.model.get_predicted_utterance( output )
        
        return predicted_utterance

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
            workers=self.workers, mode=self.mode )
        
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
        train_files = [name for name in glob.glob(os.path.join(self.dir_data,"train","*")) ]

        utterance_count = sum( [ sum(1 for line in open(f,'r')) for f in train_files ] )

        conv_count = len(train_files)

        train_size = utterance_count - conv_count*self.context_history_len

        return ( train_size // self.batch_size*self.accumulate_grad_batches ) * self.max_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        total_steps = self.total_steps()
        warmup_steps = int( self.total_steps() * self.warmup_proportion )

        if self.lr_schedule == "hard_restarts" :
            lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles = 3
                )
        elif self.lr_schedule == "LROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4,)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "monitor":"val_loss"}]

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def return_params():
        params = {}

        params['batch_size'] = self.batch_size
        params['accumulate_grad_batches'] = self.accumulate_grad_batches
        params['lr_schedule'] = self.lr_schedule 
        params['learning_rate'] = self.learning_rate
        params['max_epochs'] = self.max_epochs
        params['warmup_proportion'] = self.warmup_proportion 

        return params

class DataLoaderGenerator():
    """Handles the creation of dataloaders for a train, val and test set
    """
    def __init__(self, dir_data, batch_size,
                    tokenizer, 
                    workers=0, mode='train_new',
                    **kwargs):
        
        self.dir_data = dir_data
        self.tokenizer = tokenizer
        label_mapping = json.load(open(utils_nlg.get_path("../DialogueAct/label_mapping.json"),"r"))     

        self.bs = batch_size
        self.workers  = workers
        self.mode = mode

    def prepare_datasets(self):
        """prepares a train, validation and test set

        Returns:
            [type]: [description]
        """
                
        if self.mode in [ 'train_new', 'train_cont']:
            train_dl, = self.prepare_dataset(self.dir_data, shuffle=True, split_name='train' )
            val_dl = self.prepare_dataset(self.dir_data, shuffle=False,split_name='val'  )
            test_dl = self.prepare_dataset(self.dir_data, shuffle=False,split_name='test'  )
        
        elif self.mode in ['test']:
            train_dl = None
            val_dl = None
            test_dl = self.prepare_dataset(self.dir_data, shuffle=False ,split_name='test' )

        
        elif self.mode in ['inference']:
            train_dl = None
            val_dl = None
            test_dl = None

        
        dict_dl = {'train_dl':train_dl,
                    'val_dl':val_dl,
                    'test_dl':test_dl}

        return dict_dl 

    def prepare_dataset(self, dir_data, shuffle=False, 
        split_name='train'):

        """Prepares a dataloader given a directory of data for NLG language module
            # The current method takes a percentage of data from each subdirectory
        Args:
            dir_dset ([type]): [description]
        
        """
        #getting all files from all different subreddits/types of conversation
        files = glob.glob(  os.path.join( utils.get_path(dir_data),"*[0-9999999999]") )
        
        #getting number of utterances records in each file
        files_sizes = [ int(_f[-10:]) for _f in files]

        #defining starting line and total lines to use for dataset
        if split_name == 'train':
            line_starts = [0]*len(files)
            line_ends = [ ls+int(fs*self.splits['train']) for ls,fs in zip(line_start,files_sizes)  ]
        
        elif split_name == 'val':
            line_starts = [ int(fs*self.splits['train']) for fs in files_sizes  ]
            line_ends = [ ls+int(fs*self.splits['val']) for ls,fs in zip(line_start,files_sizes)  ]
        
        elif split_name == 'test':
            line_starts = [ int(fs*(1-self.splits['test']) ) for fs in files_sizes  ]
            line_ends = files_sizes


        li_dsets = [ SingleDataset(_f, self.tokenizer, line_start, line_end) 
                        for _f in zip(files, line_starts, line_ends) ]

        concat_dset = torch.utils.data.ConcatDataset(li_dsets)
        dataloader = torch.utils.data.DataLoader(concat_dset, batch_size=self.bs,
            shuffle=shuffle, num_workers=self.workers, collate_fn=default_collate)
        
        return dataloader

    def __call__(self):
        
        dict_dl = self.prepare_datasets()
        return dict_dl
    
class SingleDataset(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

    """
    def __init__(self, file_path, tokenizer, line_start, line_end  ):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.line_start = line_start
        self.line_end = line_end

        skiprows = linestart if line_start!=0 else None
        with open(self.fp, 'r') as f:
            self.data = pd.read_csv(file_path, sep=',', header=0, skiprows =skiprows, nrows=(self.line_end-self.line_start) )
                    
    def __len__(self):
        return (self.line_end - self.line_start)
    
    def __getitem__(self, index):
        datum = self.data[index]

        #Dialogue Act
        das = json.loads(datum['li_da'])
        
        #RST
        rst = json.loads(datum['rst']) 
        rst_rels = rst['rels']
        rst_ns = rst['ns']
        rst_pos = rst['pos']
        
        #Topic scores
        #topics_rake = json.loads(datum['topic_rake'])
        topics_textrank = json.loads(datum['topic_textrank'])
        rst_topics, rst_topics_score = zip( *topics_textrank ) #top 3 important words from utterance
        
        #Utterance
        utterance = json.loads(datum['txt_preproc'])
        
        # encoding inputs
        encoded_input = self.tokenizer.encode(das, rst_rels, rst_ns, rst_pos, topics, topics_score, utterance, prev_das=None, prev_rst=None )
            #( da_start_token, tnsr_das, tnsr_da_pos, rst_start_token, tnsr_rst_rels, tnsr_rst_ns, tnsr_rst_pos,
                #topics_start_token, tnsr_topics_phrase, tnsr_topics_score, eodrt_token, tknzd_utt ,padding_token, padding_count)      
        
        
        map_datum = {**encoded_input, 'tknzed_utt': tknzed_target }
        
        return map_datum


def main(tparams={}, mparams={}):
    gc.collect()
    torch.cuda.empty_cache()
        
    model_dir = utils.get_path(f'./models/')
       
    # Defining Logger
    tb_logger = pl_loggers.TensorBoardLogger( 
                    save_dir = os.path.join(model_dir,"logs"),
                    name = mparams['model_name']
                    version = tparams['version'] )
    tparams['version'] =  tb_logger.version()

 
    training_module = NLG.instatiate_training_module( tparams, mparams, 
                        model_dir=model_dir  )

    trainer = NLG.instatiate_trainer( tb_logger)
    NLG.start(trainer, tparams, training_module)
                
 

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False) 
    #parent_parser2 = argparse.ArgumentParser(add_help=False)    
    
    #parser_program = parent_parser.add_argument_group("program")

        
    # add model specific args
    mparams = NLG.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = TrainingModule.parse_train_specific_args(parent_parser)

    main(tparams, mparams)


