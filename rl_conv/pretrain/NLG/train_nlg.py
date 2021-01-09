import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["NCCL_DEBUG"]="INFO"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ['NCCL_P2P_DISABLE'] = '1'
#TODO: adding caching
import numpy as np
import warnings
import sklearn
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F


import glob
import pandas as pd
import json
from functools import lru_cache
from typing import List
import pickle

from itertools import cycle, islice
from torch.utils.data._utils.collate import default_convert, default_collate
from torch.nn import CrossEntropyLoss

from sklearn import preprocessing as sklp

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import utils_nlg as utils
import random 

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from transformers import Adafactor

from pytorch_lightning import loggers as pl_loggers
from collections import OrderedDict
import yaml
import ast
import types

#Monkey Patching the save module
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

    # torch.save({
    #     'epoch'
    # })

ModelCheckpoint._save_model = monkey_save_model

#Monkey patching the forward on distill bert
def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])


    if past_key_values is None:
        past_length = 0
        past_key_values = [None] * len(self.h)
    else:
        past_length = past_key_values[0][0].size(-2)
    
    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # Attention mask.
    if attention_mask is not None:

        #assert batch_size > 0, "batch_size has to be defined and > 0"
        # attention_mask = attention_mask.view(batch_size, -1)
        # # We create a 3D attention mask from a 2D tensor mask.
        # # Sizes are [batch_size, 1, 1, to_seq_length]
        # # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # # this attention mask is more simple than the triangular masking of causal attention
        # # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        # attention_mask = attention_mask[:, None, None, :]

        # # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # # masked positions, this operation will create a tensor which is 0.0 for
        # # positions we want to attend and -10000.0 for masked positions.
        # # Since we are adding it to the raw scores before the softmax, this is
        # # effectively the same as removing these entirely.
        # attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        # attention_mask = (1.0 - attention_mask) * -10000.0

        #--PATCH-- 
        # Since we need a 3D attn, an  implementation similar to the origin al GPT model is used
        # This allows different masking per batch
        attention_mask = attention_mask[:, None, :, :]
        attention_mask = attention_mask.to(dtype=self.dtype) # fp16 compatibility
        attention_mask = (1.0 - attention_mask) *-10000.0

    # If a 2D ou 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.add_cross_attention and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = input_shape + (hidden_states.size(-1),)

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if getattr(self.config, "gradient_checkpointing", False):

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # checkpointing only works with tuple returns, not with lists
                    return tuple(output for output in module(*inputs, use_cache, output_attentions))

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                layer_past,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states, present = outputs[:2]
        if use_cache is True:
            presents = presents + (present,)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (outputs[3],)

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(*output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
    else:
        raise NotImplementedError


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
        self.transformer.forward = types.MethodType(forward,self.transformer) #monkey patch

        # Embedding Layers
        self.embd_outp_dim = self.transformer.config.n_embd
        self.embedding_das = torch.nn.Conv1d( 12, self.embd_outp_dim, kernel_size=1 )
        self.embedding_rst_rels = torch.nn.Conv1d( 19, self.embd_outp_dim, kernel_size=1 )
        self.embedding_topics_score = torch.nn.Conv1d( 1, self.embd_outp_dim, kernel_size=1)
        self.token_type_embeddings = torch.nn.Embedding( 4 + self.nlg_tokenizer.context_len['topics']//2, self.embd_outp_dim) #The maximum value this can take is based on the different types of input
                                            #1 for each of da, rst, utterance and + 1 for each topic phrase (note that each topic phrase includes a <topic> token.
                                            #      therefore the largest number of different topics is topic_ctx//2 if every topic only has one word)
        
        self.lm_head = nn.Linear( self.embd_outp_dim, self.transformer.config.vocab_size, bias=False  )
        self.lm_head.weight.data.normal_(mean=0.0, std=0.02)
        
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
        input_ = self.layer_embedding( input_ )
        
        
        # Feed input to distilgpt2
        if self.loss_type == "CrossEntropy":      
            outputs = self.transformer( inputs_embeds=input_['input_embeds'],
                                        attention_mask = input_['attn_mask'],
                                        position_ids=input_['position_ids'], #check pos_ids are being removed
                                        token_type_ids = None, #token type embedding new (This gpt implementation incorrectly uses same embedding layer as for input)
                                        return_dict=False )
        
        # Add LM head to model
            hidden_states = outputs[0]
            lm_logits = self.lm_head( hidden_states )

            # must output loss 
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = input_['labels'][..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))



        elif self.loss_type == "UtteranceSimilarity":
            raise NotImplementedError
        
        return loss #dictionary
    
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
        da_start_embed = self.transformer.wte( input_['da_start_token'] ).unsqueeze(1)
        das_embed = self.embedding_das(input_['tnsr_das']).permute(0,2,1)

        rst_start_embed = self.transformer.wte( input_['rst_start_token'] ).unsqueeze(1)
        rst_embed = self.embedding_rst_rels( input_['tnsr_rst_rels'] ).permute(0,2,1) # (bs, channels, seq_len)

        topics_phrase_embed = self.transformer.wte(input_['tnsr_topics_phrase']  )  #TODO: Add positional encoding to each sub-phrase
        topics_score_embed = self.embedding_topics_score( input_['tnsr_topics_score']).permute(0,2,1)

        topics_embed = topics_phrase_embed + topics_score_embed

        utt_embed = self.transformer.wte(input_['tknzd_utt'] ) #TODO: Add positional encoding for each word too

        input_embeds = torch.cat(
            [da_start_embed, das_embed,
             rst_start_embed, rst_embed,
             topics_embed,
             utt_embed], axis = 1
            ) #dim [bs, 1024, dim1]
        
        # token type embeddings
        token_type_embedding = self.token_type_embeddings( input_['token_type_ids'])

        input_embeds += token_type_embedding
        input_['input_embeds'] = input_embeds

        # padding/shrinking input_embeds based on the max seq_len in batch
        _ = input_embeds.shape
        max_seq_len = torch.max( input_['seq_len'] )

        #By default NLG tokenizer pads it to 1024, This gives the option to reduce it all down        
        # if truncate :

        # if max_seq_len>_[1]:
        #     padding =  torch.zeros([_[0],max_seq_len-_[1],_[2]],dtype=torch.int32).type_as(max_seq_len) #This model does not have a padding token. uses attention to mask it out.
            
        #     input_embeds = torch.cat( [input_embeds, padding ], axis=1 )
        #     input_['input_embeds'] = input_embeds

        #     #padding position_ids
        #     padding_posids =  torch.zeros([_[0],max_seq_len-_[1],],dtype=torch.int32).type_as(max_seq_len) #This model does not have a padding token. uses attention to mask it out.
        #     input_['position_ids'] = torch.cat( [input_['position_ids'], padding_posids ], axis=1 )

        # elif max_seq_len == _[1]:
            
            

        # else:
        #     input_embeds = input_embeds[:, :max_seq_len, :]
        #     input_['attn_mask'] = input_['attn_mask'][:, :max_seq_len, :max_seq_len ]

        #     #TODO: make sure to slice the tensors that are above as well 
        #     raise NotImplementedError
        
        return input_

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
            'Background','Cause','Comparison','Condition',
            'Contrast','Elaboration','Enablement','Evaluation',
            'Explanation','Joint','Manner-Means','Topic-Comment',
            'Summary','Temporal','Topic-Change','n','same-unit','textual-organization'] #Add this to savable config


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

        self.max_input_len = 216 #512 #self.e2m_tokenizer.max_len
    
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
        labels = -100* torch.ones( size=[1, self.max_input_len], dtype = torch.long  ) 
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
                    utterance, prev_das=None, prev_rst=None):

        """
            This version is a smaller output space than v1, by dropping rst_pos and rst_ns
            Return 
            
            dictionary
            attn_mask : Bidirectional up to bos token, Causal Up till EOS, 0s till end of padding

        Note this method returns integer encodings for tokens that will be processed by BERT embedding layer
            and possibly one-hot encoded vectors that will not be encoded by same pert embedding layer
        """
        #effect max_sequence length

        #Getting Special Tokens
        da_start_token = self.e2m_tokenizer.encode("<|da|>")[0]
        rst_start_token = self.e2m_tokenizer.encode("<|rst|>")[0] 
        padding_token =  self.e2m_tokenizer.encode("<|endoftext|>") 

        #Defining max len for subsections
        da_len = self.context_len['da']
        rst_len = self.context_len['rst']
        topics_len = self.context_len['topics'] # This means at least topics_len/2 topics included
        utt_len = self.context_len['utt']

        #Getting Vectors
        tnsr_das = self.encode_da( das ) #dims (1, 20), 
        tnsr_rst_rels, rst_pad_count = self.encode_rst_v2(rst_rels, max_padding=rst_len - 1)   # dims (max_padding, n) #not we do rt_len-1 since we add the rst start token outside this method
        tnsr_topics_phrase, tnsr_topics_score, topics_pad_count, ta_tokens_pos  = self.encode_topic_v2( topics, topics_score, max_padding=topics_len, padding_token=padding_token) # dims (max_padding, 13) 
        tknzd_utt, utt_pad_count = self.encode_utterance_v2(utterance, max_padding=utt_len, padding_token=padding_token)
                            
        # Building Attention Mask

            # calc the ending cumulative dim for da, rst, topics, utt, segments,
        d_dim = da_len
        dr_dim = da_len + rst_len       # d_dim + tnsr_rst_rels.shape[0]+ 1
        drt_dim = dr_dim +topics_len    # dr_dim + tnsr_topics_phrase.shape[1]
        utt_dim = drt_dim + utt_len  

            # padding dimensions (after utterance end)
        seq_len =  utt_dim 
        padding_len = self.max_input_len - utt_dim 

        if padding_len>0:
            tknzd_utt = torch.cat( [tknzd_utt, torch.LongTensor(padding_token).repeat(padding_len)] )
        else:
            tknzd_utt  = tknzd_utt[:padding_len]

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
        #attn_mask[ : , utt_dim-utt_pad_count: ] = 0

        #Creating labels/targets for GPT Language Model Head
        labels = -100* torch.ones( size=[1, self.max_input_len], dtype = torch.long  ) 
        
        if padding_len>0:
            labels[0][drt_dim:utt_dim-utt_pad_count] = tknzd_utt[:-(utt_pad_count+padding_len)]
        else:
            labels[0][drt_dim:utt_dim] = tknzd_utt[ : utt_dim- drt_dim ]

        # Creating Positional Emebeddings
            # ALL words in drt get a positional encoding of 0 -> No positional meaning
            # utterance has normal positional encoding        
        position_ids_drt = torch.zeros([drt_dim], dtype=torch.long) 
        position_ids_utt =  torch.arange( 1, utt_dim-drt_dim + 1  , dtype=torch.long)
        

        if padding_len>0:
            position_ids_pad = torch.zeros( [padding_len], dtype=torch.long)
            position_ids = torch.cat([position_ids_drt,position_ids_utt, position_ids_pad], axis=-1)
        else:
            position_ids = torch.cat([position_ids_drt,position_ids_utt[:padding_len]], axis=-1)

        # Creating Token Type Ids
            # 1:da, 
            # 2:rst, 
            # n<m for each word in each topic phrase i of length m_i including leading <ta> where 4>=n>=4+topics_len//2
            # 3:utterance part

        token_type_ids_d = torch.zeros( [da_len ] , dtype=torch.long) + 1
        token_type_ids_r = torch.zeros( [rst_len], dtype=torch.long) + 2
        token_type_ids_utt = torch.zeros( [utt_len], dtype=torch.long ) + 3
        
        _ = torch.zeros( [topics_len ], dtype=torch.long)
        _[ ta_tokens_pos ] = 1
        token_type_ids_t =  _.cumsum(axis=0) + 3

        if padding_len>0:
            token_type_ids_pad = torch.ones( [padding_len], dtype=torch.long)
            token_type_ids = torch.cat( [token_type_ids_d, token_type_ids_r,\
                                token_type_ids_t, token_type_ids_utt, token_type_ids_pad] ) 

        else:
            token_type_ids = torch.cat( [token_type_ids_d, token_type_ids_r,\
                                token_type_ids_t, token_type_ids_utt[:padding_len] ] ) 


        return { 'da_start_token':da_start_token, 'tnsr_das':tnsr_das,
                 'rst_start_token':rst_start_token, 'tnsr_rst_rels':tnsr_rst_rels,
                 'tnsr_topics_phrase':tnsr_topics_phrase, 'tnsr_topics_score': tnsr_topics_score,
                 'tknzd_utt':tknzd_utt,
                 'attn_mask':attn_mask,
                 'labels':labels,
                 'position_ids':position_ids,
                 'token_type_ids':token_type_ids,
                 'seq_len':torch.IntTensor( [seq_len] )
                 }

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
        rst_rel_encoded = self.rst_rel_binarizer.transform(rst_rels).reshape( [ -1, 1] )
        tnsr_rels = torch.FloatTensor( rst_rel_encoded )
        
        #Padding out to max_padding length
        _len = tnsr_rels.shape[-1]
        diff = (max_padding - _len)
        if diff > 0:
            tnsr_rels = torch.cat([tnsr_rels, torch.zeros( [len(self.rst_rel_binarizer.classes_), diff], dtype=torch.int64)] , axis=-1 ) #backwards due to convolution embedding
        else:
            tnsr_rels = tnsr_rels[:, :max_padding]
            diff = 0

        return tnsr_rels, diff

    def encode_da(self, das):
        """[summary]

        Args:
            das ([type]): [list of da probabilites]
        """
        #TODO: add some normalization of da probabilities here
        tnsr_das = torch.unsqueeze( torch.FloatTensor( das), axis=-1 ) #dim [encode_dim1, 1]
        
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
        topic_phrases = dict_encoding['input_ids'][0]
        
        #Repeating each score in the case where the score is allocated to a phrase topic which is broken down into constituent words
                # e.g. topics - ["fast car", "motorbike", "long rail road"], scores = [0.9, 0.4, 0.2] -> scores = [0.9, 0.9, 0.9, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2]
                # have to do it after tokenization due to bytepair encoding 
            # get index of where <|ta|> tokens occur
        ta_idxs = np.where( topic_phrases==self.e2m_tokenizer('<|ta|>',return_attention_mask=False)['input_ids'] )[0]
        #filtering out idxs if index is larger than padding value
        ta_idxs = ta_idxs[ta_idxs<max_padding]

            #get difference in index position between <|ta|> tag n and <|ta|> tag n+1 ( for final tag use difference between tag and end of list)
        ta_phrase_lens = np.diff( ta_idxs, append=dict_encoding['length'] ) 
        
            # copies each score phrase_len times to cover that phrase and handles case where there is no phrase
        topics_score = [ [score]*phrase_len for score, phrase_len in zip(topics_score, ta_phrase_lens) ]
        topics_score = sum(topics_score,[]) #flattening list
        tnsr_score = torch.unsqueeze( torch.FloatTensor( topics_score ) , dim=0 ) # shape (1, topic_count) #pytorch has convolution dims opposite to tf
        topic_phrases = torch.LongTensor(topic_phrases)
        
        #Padding out to max_padding
        _len = dict_encoding['length']
        diff = (max_padding - _len)[0]
        if diff>0:
            topic_phrases = torch.cat( [ topic_phrases, torch.ones([diff], dtype=torch.int64 )*padding_token[0]] , axis=-1 )
            tnsr_score = torch.cat( [tnsr_score, torch.zeros( [1, diff] ) ], axis=-1 )
        else:
            topic_phrases = topic_phrases[:max_padding]
            tnsr_score = tnsr_score[:, :max_padding ]
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
        
        tknzd_utt = encoded['input_ids'][0]
        _len = encoded['length']
        diff = (max_padding - _len)[0]
        
        if diff>0:
            tknzd_utt = torch.cat( [ tknzd_utt, torch.ones([diff], dtype=torch.int64)*padding_token[0]] )
        else:
            tknzd_utt = tknzd_utt[:max_padding]
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
        parser.add_argument('-agb','--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('-bs','--batch_size', default=5, type=int)
        parser.add_argument('--learning_rate', default=2e-4, type=float)
        parser.add_argument('--warmup_proportion', default=0.15)
        parser.add_argument('--workers', default=0, type=int) #TODO: change to 6
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','train_cont','test','inference'])
        parser.add_argument('--lr_schedule', default='hard_restarts', required=False, choices =['LROnPlateau','hard_restarts'])
        parser.add_argument('--splits', default={'train':0.6,'val':0.2,'test':0.2}, required=False, type=str )
        parser.add_argument('--version', default=None,required=False, type=int, help="The Experimental Versioning for this run" )
        parser.add_argument('--precision', default=16,required=False, type=int, help="Precision to use", choices=[16,32] )
        
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
                        progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                        default_root_dir=tparams['dir_checkpoints'],
                        check_val_every_n_epoch=1, logger=tb_logger,
                        log_every_n_steps=1,
                        precision=tparams['precision'], callbacks=callbacks,
                        accelerator='ddp',
                        limit_train_batches = 0.4,
                        #track_grad_norm = True,
                        #overfit_batches=5,
                        #fast_dev_run=2, 
                        #log_gpu_memory=True
                        )

        if tparams['mode'] in ["train_cont","test","inference"]:
            #restoring checkpoint             
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            trainer = pl.Trainer.from_argparse_args(tparams,
                    check_val_every_n_epoch=1, logger=tb_logger,
                    progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                    precision=tparams['precision'],
                    accelerator='ddp',
                    limit_train_batches = 0.4,
                    log_every_n_steps=1,                    
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
            checkpoint_yaml_file = os.path.join( _dir_checkpoint,"best_k_models.yaml" )
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
      
        input_= batch
        loss = self.forward(input_)
        loss_key = f"{step_name}_loss"
        
        if step_name == 'train':
            str_loss_key = "loss"
        else:
            str_loss_key = loss_key       
            self.log( str_loss_key, loss)
        
        _dict = { str_loss_key: loss }   

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
        #optimizer = Adafactor(self.model.parameters(), lr=self.learning_rate, scale_parameter=False, relative_step=False)

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
            if self.line_start == 0:
            
                self.data = pd.read_csv(file_path, sep=',', header=0, 
                    skiprows =skiprows, nrows=(self.line_end-self.line_start) )
            
            else:    
                self.data = pd.read_csv(file_path, sep=',', 
                    names = ['text', 'subreddit', 'txt_preproc', 'rst', 'li_da', 'topic_textrank'],
                    skiprows =skiprows, nrows=(self.line_end-self.line_start) )
                    
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
        
        rst_rels = [ [ _dict['rel'] for _dict in li_rst ] ]
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
            
            #v2 does not include rst_ns and rst_pos
        encoded = self.tokenizer.encode_v2(das, rst_rels, topics, topics_score, utterance, prev_das=None, prev_rst=None )      
            #( da_start_token, tnsr_das,    
             #rst_start_token, tnsr_rst_rels,
             #tnsr_topics_phrase, tnsr_topics_score, 
             # tknzd_utt,
             # attn_mask
             # labels)
        return encoded

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
    
    os.makedirs(tparams['dir_checkpoints'],exist_ok=True)

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

    if tparams.gpus not in [0,1]:
        os.environ['MASTER_ADDR'] = 'localhost' #'127.0.0.1'
        os.environ['MASTER_PORT'] = '65302'

    main(vars(tparams), vars(mparams))

# CUDA_VISIBLE_DEVICES=0,1,2 python3 train_nlg.py -bs 24 -agb 3 --gpus 3 
