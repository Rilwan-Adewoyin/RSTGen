#TODO: adding caching
import numpy as np
import warnings
import sklearn
import gc

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

from sklearn import preprocessing as sklp

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import utils_nlg as utils
import random 

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig

from pytorch_lightning import loggers as pl_loggers
from collections import OrderedDict
import yaml
import ast
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

        self.nlg_tokenizer = NLG_tokenizer(base_model_name,
                                os.path.join( ("./models"), f"{model_name}_tokenizer"))
        
        self.transformer.resize_token_embeddings( len(self.nlg_tokenizer.e2m_tokenizer) )

        # Embedding Layers
        self.embd_outp_dim = self.transformer.config.n_embd
        self.embedding_das = torch.nn.Conv1d(1, self.embd_outp_dim, kernel_size=20)
        self.embedding_rst_rels = torch.nn.Conv1d( 1, self.embd_outp_dim, kernel_size=18)
        self.embedding_topics_score = torch.nn.Conv1d(1, self.embd_outp_dim, kernel_size=1)
        self.token_type_embeddings = torch.nn.Embedding( 2 + self.nlg_tokenizer.context_len['topics']//2, self.embd_outp_dim)

        self.loss_type = loss_type 

    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
                
        parser.add_argument('--base_model_name', default='distilgpt2', required=False)
        parser.add_argument('--reset_base_transformer', default=False, required=False, type=bool)
        parser.add_argument('--model_name', default='NLG', required=False)
        parser.add_argument('--loss_type', default='CrossEntropy', required=False, 
            choices=['CrossEntropy','UtteranceSimilarity']) 
        
        mparams = parser.parse_known_args( )[0]
       
        return mparams

    def forward(self, input_):
        """[summary]

        Args:
            input_ (torch.tensor): dict of inputs

        Returns:
            [type]: [description]
        """
        
        # Creating embedded inputs and attention mask
        input_embed = self.layer_embedding( input_ )

        token_type_embedding = self.token_type_embeddings( input_['token_type_ids'])
        
        input_embed = input_embed + token_type_embedding

        # Feed input to distilgpt2
        if self.loss_type == "CrossEntropy":      
            outputs = self.transformer( inputs_embeds=input_embed,
                                        attention_mask = input_['attn_mask'],
                                        labels= input_['labels'] ,
                                        position_ids=input_['position_ids'], #check pos_ids are being removed
                                        token_type_ids = None, #token type embedding new (This gpt implementation incorrectly uses same embedding layer as for input)
                                        return_dict=True )
        
        elif self.loss_type == "UtteranceSimilarity":
            raise NotImplementedError
        

        return outputs #dictionary
    
    def layer_embedding(self, input_):
        """Performs the input embedding and token type embedding calcs

        Args:
            input_ ([type]): [description]

        Returns:
            input_embedded [type]: [description]
            
            Note: logic for token_type_ids embedding is usually performed in transformer, but has been moved here
                since gpt-2 code indicates that 
        """
        # token embedding
        da_start_embed = self.transformer.transformer.wte( input_['da_start_token'] )
        das_embed = self.embedding_das(input_['tnsr_das'])

        rst_start_embed = self.transformer.transformer.wte( input_['rst_start_token'] )
        rst_embed = self.embedding_rst_rels( input_['tnsr_rst_rels'] )

        topics_phrase_embed = self.transformer.transformer.wte( input_['tnsr_topics_phrase']) #TODO: Add positional encoding to each sub-phrase
        topics_score_embed = self.embedding_topics_score( input_['tnsr_topics_score'])
        topics_embed = topics_phrase_embed + topics_score_embed

        utt_embed = self.transformer.transformer.wte(input_['tknzd_utt'] ) #TODO: Add positional encoding for each word too

        input_embeds = torch.cat(
            [da_start_embed, das_embed,
             rst_start_embed, rst_embed,
             topics_embed,
             utt_embed], axis = -1
            ) #dim [bs, 1024, dim1]
            
        return input_embeds

    def return_params(self):
        params = {}

        params['base_model_name'] = self.base_model_name
        params['reset_base_transformer'] = self.reset_base_transformer
        params['loss_type'] = self.loss_type 

        return params

    def get_predicted_utterance(self, output):
        """Given an input sequence, outputs predictions up until an <EOS> is emmitted

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """
        raise NotImplementedError

        return None

class NLG_tokenizer():
    """Rough Implmentation of the tokenizer for the NLG model

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """

    def __init__(self,
                 e2m_base_model_name='distilgpt2',
                 dir_tokenizer='./models/NLG_tokenizer'):

        self.e2m_base_model_name = e2m_base_model_name

        # RST utilities
            #TODO: add case when rel == 'n' when it could not be classified
        self.rst_rel_li = ['Attribution',
            'Background','Cause','Comparing','Condition',
            'Contrast','Elaboration','Enablement','Evaluation',
            'Explanation','Joint','Manner-Means','Topic-Comment',
            'Summary','Temporal','Topic-Change','n','same'] #Add this to savable config


        self.rst_rel_binarizer = sklp.MultiLabelBinarizer()
        self.rst_rel_binarizer.fit( [ self.rst_rel_li ] )

        self.rst_ns_li = ['NN','NS','SN','a'] #TODO: add this to config
        self.rst_ns_binarizer = sklp.MultiLabelBinarizer()
        self.rst_ns_binarizer.fit( [ self.rst_ns_li ] )

        self.context_len = { 'da':2, 'rst':8, 'topics':16, 'utt':208 } #add this to config
        
        # Initalising tokenzier
        if os.path.isdir(dir_tokenizer):
            self.e2m_tokenizer = AutoTokenizer.from_pretrained(dir_tokenizer)

        # retreiving base tokenizer from online or from local distillgpt2
        else:
            dir_transformer = os.path.join("./models",e2m_base_model_name)
            exists = os.path.isdir(dir_transformer)            

            if exists==True:
                self.e2m_tokenizer = AutoTokenizer.from_pretrained(dir_transformer)
                config = AutoConfig.from_pretrained(dir_transformer)

            elif exists==False:
                self.e2m_tokenizer = AutoTokenizer.from_pretrained(self.e2m_base_model_name)
                config = AutoConfig.from_pretrained(self.e2m_base_model_name)
            
            special_tokens_dict = {'additional_special_tokens':
                    ['<|da|>','<|rst|>','<|ta|>']}
            
            if str(special_tokens_dict['additional_special_tokens']) != \
                    self.e2m_tokenizer.special_tokens_map.get('additional_special_tokens',''):
                num_added_toks = self.e2m_tokenizer.add_special_tokens(special_tokens_dict)
                os.makedirs(dir_tokenizer)
                self.e2m_tokenizer.save_pretrained(dir_tokenizer)
                config.save_pretrained(dir_tokenizer)

        self.max_input_len = self.e2m_tokenizer.max_len
    
    def encode_v1( self, das ,rst_rels, rst_ns, rst_pos,
        topics, topics_score, utterance ,prev_das=None, prev_rst=None):

        """Return 

            attn_mask : Bidirectional up to bos token, Causal Up till EOS, 0s till end of padding

        Note this method returns integer encodings for tokens that will be processed by BERT embedding layer
            and possibly one-hot encoded vectors that will not be encoded by same pert embedding layer
        """
        
        #Getting Vectors
        tnsr_das = self.encode_da( das ) #dims (n2, 20), (n2,) # Use an upscaling convolution layer  and an embedding layer
        tnsr_rst_rels, tnsr_rst_ns, tnsr_rst_pos = self.encode_rst(rst_rels, rst_ns, rst_pos)   # dims (n1, 13) (n1, 3) (n1,) # Use an upscaling convolution layer, upscaling convolution layer, embedding layer  
        tnsr_topics_phrase, tnsr_topics_score = self.encode_topic( topics, topics_score) #dims (n3, ) (n3, )  # Use an embedding layer E, upscaling convolutional layer
        tknzd_utt = self.encode_utterance(utterance)
            #This will already have a EOS token
            #TODO: make sure to remove CLS token

        #Getting Special Tokens
        da_start_token = self.e2m_tokenizer.encode('<|da|>')
        rst_start_token = self.e2m_tokenizer.encode("<|rst|>") 
        padding_token =  self.e2m_tokenizer.encode("<|endoftext|>") # 
        
                
        # Building Attention Mask
            #da rst topics dimension length
        drt_dim = tnsr_das.shape[0] + \
                      tnsr_rst_rels.shape[0]+ \
                      tnsr_topics_phrase.shape[1]+ \
                      2 # da, rst, ta, tokens
            #utterance dimension len
        utt_dim = tknzd_utt.shape[1]

            # padding dimension
        nopad_dim = drt_dim + utt_dim
        padding_dim = self.max_input_len - nopad_dim
        
            # creating mask
        attn_mask = torch.tril( torch.ones([self.max_input_len,self.max_input_len]))
                #pre_utt masking
        attn_mask[ :drt_dim , :drt_dim ] = 1 
        
                #padding masking
        attn_mask[ -padding_dim: , : ] = 0 
                #utt_dim
        attn_mask[ drt_dim:drt_dim+utt_dim, drt_dim:drt_dim+utt_dim] \
                    = torch.tril(torch.ones([ utt_dim,utt_dim])) 


        #Creating labels/targets for GPT Language Model Head
        labels = -100* torch.ones( size=[1, 1024], dtype = torch.long  ) 
        labels[0][drt_dim:nopad_dim] = tknzd_utt[0]

        #Method 1a: RST: -> pad each rst subtype to the first 4/8 elements
        #Method 1b: tnsr_topic and tnsr score -> pad each one to 10 elements
        #Method 1c: tknzd_utt -> padd to 100 length 
        #Method 1d: correct the attn_mask

        # Full information Version
        return { 'da_start_token':da_start_token, 'tnsr_das':tnsr_das,
                 'rst_start_token':rst_start_token, 'tnsr_rst_rels':tnsr_rst_rels,'tnsr_rst_ns':tnsr_rst_ns,'tnsr_rst_pos':tnsr_rst_pos,
                 'tnsr_topics_phrase':tnsr_topics_phrase, 'tnsr_topics_score': tnsr_topics_score,
                 'tknzd_utt':tknzd_utt,
                 'padding_token':padding_token, 'padding_count':padding_dim,
                 'attn_mask':attn_mask,
                 'labels':labels}

    
    def encode_v2( self, das ,rst_rels,topics, topics_score, 
                    utterance ,prev_das=None, prev_rst=None):

        """
            This version is a smaller output space than v1, by dropping rst_pos and rst_ns
            Return 
            
            dictionary
            attn_mask : Bidirectional up to bos token, Causal Up till EOS, 0s till end of padding

        Note this method returns integer encodings for tokens that will be processed by BERT embedding layer
            and possibly one-hot encoded vectors that will not be encoded by same pert embedding layer
        """

        #Getting Special Tokens
        da_start_token = self.e2m_tokenizer.encode("<|da|>")
        rst_start_token = self.e2m_tokenizer.encode("<|rst|>") 
        padding_token =  self.e2m_tokenizer.encode("<|endoftext|>") 

        #Defining max len for subsections
        da_len = self.context_len['da']
        rst_len = self.context_len['rst']
        topics_len = self.context_len['topics'] # This means at least topics_len/2 topics included
        utt_len = self.context_len['utt']

        #Getting Vectors
        tnsr_das = self.encode_da( das ) #dims (1, 20), 
        tnsr_rst_rels, rst_pad_count = self.encode_rst_v2(rst_rels, max_padding=rst_len-1)   # dims (max_padding, 13) 
        tnsr_topics_phrase, tnsr_topics_score, topics_pad_count, ta_tokens_pos  = self.encode_topic_v2( topics, topics_score, max_padding=topics_len, padding_token=padding_token) # dims (max_padding, 13) 
        tknzd_utt, utt_pad_count = self.encode_utterance_v2(utterance, max_padding=utt_len, padding_token=padding_token)
                            
        # Building Attention Mask
            # calc the ending cumulative dim for da rst topics utt segments
        d_dim = da_len
        dr_dim = da_len + rst_len       # d_dim + tnsr_rst_rels.shape[0]+ 1
        drt_dim = dr_dim +topics_len    # dr_dim + tnsr_topics_phrase.shape[1]
        utt_dim = drt_dim + utt_len        
            # padding dimensions (after utterance end)
        padding_len = self.max_input_len - utt_dim
        tknzd_utt = torch.cat( [tknzd_utt, torch.LongTensor(padding_token).unsqueeze(0).repeat(1,padding_len)], axis=-1)
        
            # creating mask
        attn_mask = torch.tril( torch.ones([self.max_input_len,self.max_input_len]))
                    #pre_utterance general masking
        attn_mask[ :drt_dim , :drt_dim ] = 1 

                    #correcting for padding in rst section
        attn_mask[ dr_dim-rst_pad_count:dr_dim ,:] = 0
        attn_mask[ :, dr_dim-rst_pad_count:dr_dim] = 0

                    #correcting for padding in topic section
        attn_mask[ drt_dim-topics_pad_count: drt_dim, :  ] = 0
        attn_mask[ :, drt_dim-topics_pad_count: drt_dim ] = 0
                
                #correcting for padding in and after utterance section
        attn_mask[ utt_dim-utt_pad_count: , : ] = 0
#        attn_mask[ : , utt_dim-utt_pad_count: ] = 0

        #Creating labels/targets for GPT Language Model Head
        labels = -100* torch.ones( size=[1, 1024], dtype = torch.long  ) 
        labels[0][drt_dim:utt_dim-utt_pad_count] = tknzd_utt[0][:-(utt_pad_count+padding_len)]

        # Creating Positional Emebeddings
            # ALL words in drt get a positional encoding of 0 -> No positional meaning
            # utterance has normal positional encoding        
        position_ids_drt = torch.zeros([1,drt_dim], dtype=torch.long) 
        position_ids_utt =  torch.arange( 1, utt_dim-drt_dim + 1  , dtype=torch.long).unsqueeze(0).view(-1, utt_dim-drt_dim )
        position_ids = torch.cat([position_ids_drt,position_ids_utt], axis=-1)
        
        # Creating Token Type Ids
            # 0:da, 1:rst, 
            # n for each word in a topic phrase including leading <ta> where 3>=n>=3+topics_len/2
            # 2:utterance part
        token_type_ids_d = torch.zeros( [1, da_len ] , dtype=torch.long)
        token_type_ids_r = torch.ones( [1, rst_len], dtype=torch.long) 
        token_type_ids_utt = torch.ones( [1, utt_len], dtype=torch.long ) + 1
        _ = torch.zeros( [1, topics_len ], dtype=torch.long)
        _[ :, ta_tokens_pos ] = 1
        token_type_ids_t =  _.cumsum(axis=-1) + 2

        token_type_ids = torch.cat( [token_type_ids_d, token_type_ids_r,\
                            token_type_ids_t, token_type_ids_utt], axis=-1 )


        return { 'da_start_token':da_start_token, 'tnsr_das':tnsr_das,
                 'rst_start_token':rst_start_token, 'tnsr_rst_rels':tnsr_rst_rels,
                 'tnsr_topics_phrase':tnsr_topics_phrase, 'tnsr_topics_score': tnsr_topics_score,
                 'tknzd_utt':tknzd_utt,
                 'attn_mask':attn_mask,
                 'labels':labels,
                 'position_ids':position_ids,
                 'token_type_ids':token_type_ids}

    def encode_rst(self, rst_rels, rst_ns, rst_pos):
        """Converts the three lists into a seeries of vectors

        Args:
            rst_rels ([type]): [description]
            rst_ns ([type]): [description]
            rst_pos ([type]): [description]
        """
        
        rst_rel_encoded = self.rst_rel_binarizer.transform(rst_rels)
        rst_ns_encoded = self.rst_ns_binarizer.transform( rst_ns )
        rst_pos = rst_pos

        tnsr_rels = torch.FloatTensor( rst_rel_encoded )
        tnsr_ns = torch.from_numpy( rst_ns_encoded )    #dim [ n, encode_dim2]
        tnsr_pos = torch.FloatTensor( rst_pos )          #dim [ n, encode_dim3]

        return tnsr_rels, tnsr_ns, tnsr_pos

    def encode_rst_v2(self,rst_rels, max_padding=8):
        """Converts rst_rels in a series of vectors

            Args:
                rst_rels ([type]): [description]
                max_padding ([type]): padding amount
                rst_pos ([type]): [description]
        """
        rst_rel_encoded = self.rst_rel_binarizer.transform(rst_rels)
        tnsr_rels = torch.FloatTensor( rst_rel_encoded )
        
        #Padding out to max_padding length
        _len = tnsr_rels.shape[0]
        diff = (max_padding - _len)
        if diff > 0:
            tnsr_rels = torch.cat([tnsr_rels, torch.zeros([diff, len(self.rst_rel_binarizer.classes_)], dtype=torch.int64)] , axis=0 )
        else:
            tnsr_rels = tnsr_rels[:max_padding,:]
            diff = 0

        return tnsr_rels, diff

    def encode_da(self, das):
        """[summary]

        Args:
            das ([type]): [list of da probabilites]
        """
        #TODO: add some normalization of da probabilities here
        tnsr_das = torch.unsqueeze( torch.FloatTensor( das), axis=0 ) #dim [1, encode_dim1]
        
        return tnsr_das

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
        str_topics = ''.join([ '<|ta|>'+topic  for topic in topics ])
        dict_encoding = self.e2m_tokenizer(str_topics, add_special_tokens=False,
                                                 return_attention_mask = False, 
                                                 truncation = True,
                                                 padding='do_not_pad', 
                                                 return_tensors='np',
                                                 return_token_type_ids=None,
                                                 return_special_tokens_mask=False,
                                                 return_length=True) # shape () topic_count, )
        tnsr_topic_phrases = dict_encoding['input_ids']
        
        #Repeating each score in the case where the score is allocated to a phrase topic which is broken down into constituent words
                # e.g. topics - ["fast car", "motorbike", "long rail road"], scores = [0.9, 0.4, 0.2] -> scores = [0.9, 0.9, 0.9, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2]
                # have to do it after tokenization due to bytepair encoding 
            # get index of where <|ta|> idxs occur
        ta_idxs = np.where( tnsr_topic_phrases[0]==self.e2m_tokenizer('<|ta|>',return_attention_mask=False)['input_ids'] )[0]
            #get difference in index position between <|ta|> tag n and <|ta|> tag n+1 ( for final tag use difference between tag and end of list)
        ta_phrase_lens = np.diff( ta_idxs, append=dict_encoding['length'] ) 
            # copies each score phrase_len time and handles case where there is 
        topics_score = [ [score]*phrase_len for score, phrase_len in zip(topics_score, ta_phrase_lens) ]
        topics_score = sum(topics_score,[]) #flattening list

        tnsr_score = torch.unsqueeze( torch.FloatTensor( topics_score ) , dim=0 ) # shape (topic_count, )
        # tnr_topic = torch.cat( [tnsr_topics_phrase, tnsr_score], axis=1 )
        return torch.FloatTensor(tnsr_topic_phrases), tnsr_score

    def encode_topic_v2(self, topics, topics_score, max_padding=16, padding_token="<|endoftext|>"):
        """[summary]

            Args:
                topics ([type]): [list of topics (phrases or words)]
                topics_score ([type]): [list of float scores for each topic relevancy]

            Raises:
                Exception: [description]

            Returns:
                [type]: [description]
        """
        str_topics = ''.join([ '<|ta|>'+topic  for topic in topics ])
        dict_encoding = self.e2m_tokenizer(str_topics, add_special_tokens=False,
                                            return_attention_mask = False, 
                                            truncation = True,
                                            padding='do_not_pad', 
                                            return_tensors='np',
                                            return_token_type_ids=None,
                                            return_special_tokens_mask=False,
                                            return_length=True)
        topic_phrases = dict_encoding['input_ids']
        
        #Repeating each score in the case where the score is allocated to a phrase topic which is broken down into constituent words
                # e.g. topics - ["fast car", "motorbike", "long rail road"], scores = [0.9, 0.4, 0.2] -> scores = [0.9, 0.9, 0.9, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2]
                # have to do it after tokenization due to bytepair encoding 
            # get index of where <|ta|> tokens occur
        ta_idxs = np.where( topic_phrases[0]==self.e2m_tokenizer('<|ta|>',return_attention_mask=False)['input_ids'] )[0]
            #get difference in index position between <|ta|> tag n and <|ta|> tag n+1 ( for final tag use difference between tag and end of list)
        ta_phrase_lens = np.diff( ta_idxs, append=dict_encoding['length'] ) 
            # copies each score phrase_len time and handles case where there is 
        topics_score = [ [score]*phrase_len for score, phrase_len in zip(topics_score, ta_phrase_lens) ]
        topics_score = sum(topics_score,[]) #flattening list
        tnsr_score = torch.unsqueeze( torch.FloatTensor( topics_score ) , dim=0 ) # shape (topic_count, )
        topic_phrases = torch.FloatTensor(topic_phrases)
        
        #Padding out to max_padding
        _len = dict_encoding['length']
        diff = (max_padding - _len)[0]
        if diff>0:
            topic_phrases = torch.cat( [ topic_phrases, torch.ones([1,diff], dtype=torch.int64 )*padding_token[0]] , axis=-1 )
            tnsr_score = torch.cat( [tnsr_score, torch.zeros( [1,diff]) ], axis=-1 )
        else:
            topic_phrases = topic_phrases[:, :max_padding]
            tnsr_score = tnsr_score[:, :max_padding]
            diff = 0

        return topic_phrases , tnsr_score, diff, ta_idxs


    def encode_utterance(self, utterance, max_padding=208):
        tknzd_utt = self.e2m_tokenizer( '<|endoftext|>' +utterance + '<|endoftext|>' ,add_special_tokens=True,
                                        return_attention_mask = False, 
                                        padding='do_not_pad', truncation=True, 
                                        return_tensors='pt',
                                        return_token_type_ids=None)['input_ids']
        
        return tknzd_utt

    def encode_utterance_v2(self, utterance, max_padding=208, padding_token='<|endoftext|>'):
        
        utterance ='<|endoftext|>' + utterance + '<|endoftext|>'
        encoded = self.e2m_tokenizer( utterance, add_special_tokens=False,
                                        return_attention_mask = False, 
                                        padding='do_not_pad',
                                        truncation=True, 
                                        return_tensors='pt',
                                        return_length=True,
                                        return_token_type_ids=None)
        
        tknzd_utt = encoded['input_ids']
        _len = encoded['length']
        diff = (max_padding - _len)[0]
        
        if diff>0:
            tknzd_utt = torch.cat( [ tknzd_utt, torch.ones([1,diff], dtype=torch.int64)*padding_token[0]] , axis=-1 )
        else:
            tknzd_utt = tknzd_utt[:, :max_padding]
            diff = 0

        return tknzd_utt, diff

class TrainingModule(pl.LightningModule):

    def __init__(self, model_params, batch_size=20, 
                    dir_data=None, 
                    accumulate_grad_batches=1,
                    max_epochs=100,
                    gpus=1, 
                    learning_rate=1e-4,
                    warmup_proportion=0.1,
                    workers=0,
                    lr_schedule='hard_restarts',
                    mode = 'train_new',
                    data_splits = {'train':0.6,'val':0.2,'test':0.2},
                    *args,
                    **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.gpus =  gpus
        self.model = NLG( **model_params )
        self.mode = mode
        self.workers = workers
        self.data_splits = data_splits
        
        if self.mode in ['train_new','train_cont','test']:
            self.dir_data = utils.get_path(dir_data)
            self.create_data_loaders(self.workers)
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
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('--max_epochs', default=80, type=int)
        parser.add_argument('--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('-bs','--batch_size', default=20, type=int)
        parser.add_argument('--learning_rate', default=2e-3, type=float)
        parser.add_argument('--warmup_proportion', default=0.15)
        parser.add_argument('--workers', default=0, type=int) #TODO: change to 6
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','train_cont','test','inference'])
        parser.add_argument('--lr_schedule', default='hard_restarts', required=False, choices =['LROnPlateau','hard_restarts'])
        parser.add_argument('--splits', default={'train':0.6,'val':0.2,'test':0.2}, required=False, type=str )
        parser.add_argument('--version', default=None,required=False, type=int, help="The Experimental Versioning for this run" )
        
            #TODO: check --version of required type None actually works
        tparams = parser.parse_known_args()[0]
        #tparams.splits = json.loads(tparams.splits)

        return tparams
    
    @staticmethod
    def instatiate_training_module( tparams=None, mparams=None ):
        """Create training module

        Args:
            tparams ([type]): [description]
        """
        
        if tparams['mode'] in ["train_new"]:
            training_module = TrainingModule(**tparams, model_params=mparams  )
            
        elif tparams['mode'] in ["test", "train_cont", "inference"]:
            
            #Retreiving best checkpoint from specific model-version
            checkpoint_yaml_file = os.path.join( tparams['dir_checkpoints'],"best_k_models.yaml" )
            scores_dict = yaml.load( open(checkpoint_yaml_file,"r") ) #key= ckptpath, value = val_loss
            best_ckpt_path = min(scores_dict, key=scores_dict.get)

            if torch.cuda.is_available():
                checkpoint = torch.load(best_ckpt_path)
            else:
                checkpoint = torch.load(best_ckpt_path, map_location=torch.device('cpu'))

            #restore/update param files from the checkpoint
            tparams.update ( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                'batch_size', 'lr_schedule', 'learning_rate']} )

            mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                'base_model_name','reset_base_transformer','loss_type','model_name']} )

            #Restore/update Training Module
            training_module = TrainingModule(**vars(tparams), mparams=mparams)
            training_module.load_state_dict(checkpoint['state_dict'])
        
        else:
            raise ValueError("tparams['mode'] must be in range [train_new, train_cont, test, inference]")
        return training_module

    @staticmethod
    def instatiate_trainer( tparams, tb_logger):
        """[summary]

            Creates The Trainer and callbacks
        """
        dir_checkpoints = tparams['dir_checkpoints']
        # Creating Callbacks
        callbacks = []        
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=2, 
            mode='min', dirpath=dir_checkpoints, 
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

        if tparams['mode'] in ["train_new"]:
            
            trainer = pl.Trainer.from_argparse_args(argparse.Namespace( **tparams),
                        progress_bar_refresh_rate=50,
                        default_root_dir=tparams['dir_checkpoints'],
                        check_val_every_n_epoch=1, logger=tb_logger,
                        log_every_n_steps=5,
                        precision=16, callbacks=callbacks,
                        #track_grad_norm = True,
                        #overfit_batches=5,
                        #fast_dev_run=2, 
                        #log_gpu_memory=True
                        )

        if tparams['mode'] in ["train_cont","test","inference"]:
            #restoring checkpoint             
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            trainer = pl.Trainer.from_argparse_args(tparams, progress_bar_refresh_rate=1,
                    check_val_every_n_epoch=1, logger=tb_logger,
                    precision=16
                    #track_grad_norm = True,
                    #overfit_batches=5
                    #,fast_dev_run=2, 
                    #log_gpu_memory=True
                    )

            # load callback states
            trainer.on_load_checkpoint(checkpoint)
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

        return trainer
    
    @staticmethod
    def get_ckpt_file(_dir_checkpoint,mode='best'):
        if mode=='best':
            checkpoint_yaml_file = os.path.join( dir_checkpoints,"best_k_models.yaml" )
            scores_dict = yaml.load( open(checkpoint_yaml_file,"r") ) #key= ckptpath, value = val_loss
            best_ckpt_path = min(scores_dict, key=scores_dict.get)

            if torch.cuda.is_available():
                checkpoint = torch.load(best_ckpt_path)
            else:
                checkpoint = torch.load(best_ckpt_path, map_location=torch.device('cpu'))            
        else:
            raise NotImplementedError
        
        return checkpoint
        
    @staticmethod
    def start(trainer, tparams, training_module ):
        
        if tparams['mode'] in ["test"]:
            training_module.eval() 
            training_module.freeze() 
            trainer.test(test_dataloaders=training_module.test_dl, model=training_module, ckpt_path='best')
              
        elif tparams['mode'] in ['train_new','train_cont']:    
            trainer.fit(training_module )
            #trainer.checkpoint_callback.to_yaml()
            trainer.test(test_dataloaders=training_module.test_dl, ckpt_path='best')
        
        elif tparams['mode'] in ['infernece']: 
            raise NotImplementedError   


    def step(self, batch, step_name):

        #target = batch.pop('tknzed_target')
        input_= batch
        output = self.forward(input_)
        loss = output['loss']
        
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
       
        dg = DataLoaderGenerator(self.dir_data,  self.batch_size, self.model.nlg_tokenizer, 
            workers=self.workers, mode=self.mode, split=self.data_splits )
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

        ds_size = self.train_dl.__len__() // self.gpus
        steps = (ds_size * self.max_epochs) // (self.accumulate_grad_batches)
        return steps

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        total_steps = self.total_steps()
        warmup_steps = int( total_steps * self.warmup_proportion )

        if self.lr_schedule == "hard_restarts" :
            lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles = 3)

        elif self.lr_schedule == "LROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch", "monitor":"val_loss"}]

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def return_params(self):
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
                    splits={'train':0.6,'val':0.2,'test':0.2},
                    **kwargs):
        
        self.dir_data = dir_data
        self.tokenizer = tokenizer
        self.splits = splits
        #label_mapping = json.load(open(utils.get_path("../DialogueAct/label_mapping.json"),"r"))     

        self.bs = batch_size
        self.workers  = workers
        self.mode = mode

    def prepare_dataloaders(self):
        """prepares a train, validation and test set

        Returns:
            [type]: [description]
        """
                
        if self.mode in [ 'train_new', 'train_cont']:
            train_dl = self.prepare_dataloader(self.dir_data, shuffle=True, split_name='train' )
            val_dl = self.prepare_dataloader(self.dir_data, shuffle=False,split_name='val'  )
            test_dl = self.prepare_dataloader(self.dir_data, shuffle=False,split_name='test'  )
        
        elif self.mode in ['test']:
            train_dl= None
            val_dl = None
            test_dl = self.prepare_dataloader(self.dir_data, shuffle=False ,split_name='test' )

        
        elif self.mode in ['inference']:
            train_dl = None
            val_dl = None
            test_dl = None

        
        dict_dl = {'train_dl':train_dl,
                    'val_dl':val_dl,
                    'test_dl':test_dl}

        return dict_dl 

    def prepare_dataloader(self, dir_data, shuffle=False, 
        split_name='train'):

        """Prepares a dataloader given a directory of data for NLG language module
            # The current method takes a percentage of data from each subdirectory
            Args:
                dir_dset ([type]): [description]
        """
        #getting all files from all different subreddits/types of conversation
        fns = glob.glob(  os.path.join( utils.get_path(dir_data),"*","*") )
        
        #getting number of utterances records in each file
        files_sizes = [ int(fn[-10:]) for fn in fns]

        #defining starting line and total lines to use for dataset
        if split_name == 'train':
            line_starts = [0]*len(files_sizes)
            line_ends = [ ls+int(fs*self.splits['train']) for ls,fs in zip(line_starts, files_sizes)  ]
            shuffle = True
        
        elif split_name == 'val':
            line_starts = [ int(fs*self.splits['train']) for fs in files_sizes  ]
            line_ends = [ ls+int(fs*self.splits['val']) for ls,fs in zip(line_starts, files_sizes)  ]
            shuffle = False

        elif split_name == 'test':
            line_starts = [ int(fs*(1-self.splits['test']) ) for fs in files_sizes  ]
            line_ends = files_sizes
            shuffle = False

        li_dsets = [ SingleDataset(_f, self.tokenizer, line_start, line_end) 
                        for _f, line_start, line_end in zip(fns, line_starts, line_ends) ]

        concat_dset = torch.utils.data.ConcatDataset(li_dsets)
        dataloader = torch.utils.data.DataLoader(concat_dset, batch_size=self.bs,
            shuffle=shuffle, num_workers=self.workers, collate_fn=default_collate)
        
        return dataloader

    def __call__(self):
        
        dict_dl = self.prepare_dataloaders()
        return dict_dl
    
class SingleDataset(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

    """
    def __init__(self, file_path, tokenizer, line_start, line_end  ):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.line_start = line_start
        self.line_end = line_end

        skiprows = self.line_start if self.line_start!=0 else None
        with open(self.fp, 'r') as f:
            self.data = pd.read_csv(file_path, sep=',', header=0, skiprows =skiprows, nrows=(self.line_end-self.line_start) )
                    
    def __len__(self):
        return (self.line_end - self.line_start)
    
    def __getitem__(self, index):
        datum = self.data[index:index+1]

        #Dialogue Act
        das = json.loads(datum['li_da'].values[0])
        
        #RST
        try:
            li_rst = json.loads(datum['rst'].values[0])  #list of dictionaries 
        except json.decoder.JSONDecodeError as e:
            li_rst = ast.literal_eval(datum['rst'].values[0])
        
        rst_rels = [ [_dict['rel']] for _dict in li_rst ]
        rst_ns = [ [_dict['ns']] for _dict in li_rst ]
        rst_pos = [ _dict['pos'] for _dict in li_rst ]

        
        #Topic scores
        try:
            topics_textrank = json.loads(datum['topic_textrank'].values[0])
        except json.decoder.JSONDecodeError as e:
            topics_textrank = ast.literal_eval(datum['topic_textrank'].values[0])

        if len(topics_textrank)!=0: #TODO: remove later
            topics, topics_score = zip( *topics_textrank ) #top 3 important words from utterance
        else:
            topics = [""]
            topics_score = [0.0]
        
        #Utterance
        utterance = datum['txt_preproc'].values[0].strip('\"')
        
        #encoding inputs
        #encoded = self.tokenizer.encode(das, rst_rels, rst_ns, rst_pos, topics, topics_score, utterance, prev_das=None, prev_rst=None )
            #( da_start_token, tnsr_das,    
             #rst_start_token, tnsr_rst_rels, tnsr_rst_ns, tnsr_rst_pos,
             #tnsr_topics_phrase, tnsr_topics_score, 
             # tknzd_utt,
             # padding_token, padding_count,
             # attn_mask
             # labels)
        encoded = self.tokenizer.encode_v2(das, rst_rels, topics, topics_score, utterance, prev_das=None, prev_rst=None )      
            #( da_start_token, tnsr_das,    
             #rst_start_token, tnsr_rst_rels,
             #tnsr_topics_phrase, tnsr_topics_score, 
             # tknzd_utt,
             # attn_mask
             # labels)
        return encoded
 
class SingleIterDataset(torch.utils.data.IterableDataset):
    """creates a dataloader given a directory of text files each containing a conversation

    """
    def __init__(self, file_path, tokenizer, line_start, line_end  ):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.line_start = line_start
        self.line_end = line_end

        skiprows = self.line_start if self.line_start!=0 else None

        with open(self.fp, 'r') as f:
            self.data = pd.read_csv(file_path, sep=',', header=0, skiprows=skiprows, nrows=(self.line_end-self.line_start) )
                    
    def __len__(self):
        return (self.line_end - self.line_start)
    
    def __getitem__(self, index):
        datum = self.data[index:index+1]

        #Dialogue Act
        das = json.loads(datum['li_da'].values[0])
        
        #RST
        try:
            li_rst = json.loads(datum['rst'].values[0])  #list of dictionaries 
        except json.decoder.JSONDecodeError as e:
            li_rst = ast.literal_eval(datum['rst'].values[0])
        
        
        rst_rels = [ _dict['rel'] for _dict in li_rst ]
        rst_ns = [ _dict['ns'] for _dict in li_rst ]
        rst_pos = [ _dict['pos'] for _dict in li_rst ]
        
        #Topic scores
        #topics_rake = json.loads(datum['topic_rake'])
        try:
            topics_textrank = json.loads(datum['topic_textrank'].values[0])
        except json.decoder.JSONDecodeError as e:
            topics_textrank = ast.literal_eval(datum['topics_textrank'].values[0])

        topics, topics_score = zip( *topics_textrank ) #top 3 important words from utterance
        
        #Utterance
        utterance = json.loads(datum['txt_preproc'])
        
        # encoding inputs
        encoded_input = self.tokenizer.encode(das, rst_rels, rst_ns, rst_pos, topics, topics_score, utterance, prev_das=None, prev_rst=None )
            #( da_start_token, tnsr_das, tnsr_da_pos, rst_start_token, tnsr_rst_rels, tnsr_rst_ns, tnsr_rst_pos,
                #topics_start_token, tnsr_topics_phrase, tnsr_topics_score, bos_token, tknzd_utt ,padding_token, padding_count)      
        
        map_datum = {**encoded_input }
        
        return map_datum

def main(tparams={}, mparams={}):
    gc.collect()
    torch.cuda.empty_cache()
     
           
    # Defining Logger
    tb_logger = pl_loggers.TensorBoardLogger( 
                    save_dir = os.path.abspath(tparams['model_dir']),
                    name = mparams['model_name'],
                    version = tparams['version'] )
    tparams['version'] =  tb_logger.version
    tparams['dir_checkpoints'] = os.path.join(tparams['model_dir'],mparams['model_name'],f"version_{tparams['version']:02d}",'checkpoints' )
    

    # initiating training loop
    training_module = TrainingModule.instatiate_training_module( tparams, mparams)
    trainer = TrainingModule.instatiate_trainer( tparams,  tb_logger)
    TrainingModule.start(trainer, tparams, training_module)
                
if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False) 

    # add model specific args
    mparams = NLG.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = TrainingModule.parse_train_specific_args(parent_parser)

    main(vars(tparams), vars(mparams))


