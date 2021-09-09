import os
from typing import Optional, Tuple
from pytorch_lightning.utilities import distributed

from torch._C import Value

#os.environ["NCCL_DEBUG"]="INFO"
#os.environ["NCCL_DEBUG_SUBSYS"]="ALL"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["NCCL_P2P_LEVEL"] = "3"
#os.environ['NCCL_P2P_DISABLE'] = '1'
#os.environ['NCCL_SOCKET_IFNAME'] =  'lo' 
#os.environ['NCCL_SOCKET_IFNAME'] =  'enp3s0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
from pathlib import Path
from itertools import cycle, islice
from torch.utils.data._utils.collate import default_convert, default_collate
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import CrossEntropyLoss

from sklearn import preprocessing as sklp

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import utils_nlg_v3 as utils
import random 

from transformers.optimization import Adafactor, AdafactorSchedule
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig

from transformers.generation_beam_search import BeamHypotheses

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.core.decorators import auto_move_data
import yaml
import types

from itertools import permutations, combinations, combinations_with_replacement
from typing import Optional, Callable, Union, Optional, List, Iterable

from transformers.generation_logits_process import (
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from utils_nlg_v3 import EmbeddingRstPos

from typing import TypeVar, Iterator

from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset

import torch.distributed as dist

T_co = TypeVar('T_co', covariant=True)

#Monkey patching the forward on gpt

class NLG(nn.Module, utils.GenerationMixin42_gpt):
    """NLG unit
    """

    def __init__(self, 
        base_model_name= 'gpt2-medium', model_name="NLG",
        max_input_len=264, freeze_pretrained=0,
        scale_grad_by_freq=False, **kwargs ):
            #base model uses same code as 'microsoft/DialoGPT-small'
        super(NLG, self).__init__()
        
        self.base_model_name = base_model_name   
        self.model_name = model_name
        self.freeze_pretrained = freeze_pretrained
        self.scale_grad_by_freq = scale_grad_by_freq

        # Retreive/Instantiate base transformer
        self.transformer = utils.load_pretrained_transformer(self.base_model_name, transformer=True)['transformer']    
        # self._use_cache = False


        self.nlg_tokenizer = NLG_tokenizer(base_model_name,
                                os.path.join( ("./models"), f"{model_name}_tokenizer"),
                                 max_input_len=max_input_len, nlg_model=self,
                                 **kwargs)
        
        self.transformer.resize_token_embeddings( len(self.nlg_tokenizer.e2m_tokenizer) )
        self.transformer.transformer.forward = types.MethodType(utils.forward_gpt,self.transformer.transformer) #monkey patch
        
        self.config= self.transformer.config

        # region Embedding Layers
        self.embd_outp_dim = self.transformer.config.n_embd
                
        self.embedding_rst_rels = torch.nn.Embedding( len(self.nlg_tokenizer.rst_rel_li )+1, self.embd_outp_dim, padding_idx=len(self.nlg_tokenizer.rst_rel_li ), scale_grad_by_freq=self.scale_grad_by_freq )
        #self.embedding_rst_rels.weight.data.normal_(mean=0.0, std=0.0005)

        self.embedding_rst_ns = torch.nn.Embedding( len(self.nlg_tokenizer.rst_ns_li )+1, self.embd_outp_dim, padding_idx=len(self.nlg_tokenizer.rst_ns_li ),scale_grad_by_freq=self.scale_grad_by_freq )
        #self.embedding_rst_ns.weight.data.normal_(mean=0.0, std=0.0005)

        self.embedding_rst_pos = EmbeddingRstPos(   max_rst_index=self.nlg_tokenizer.rst_pos_maxidx,
                                                    max_rst_level = NLG_tokenizer.node_level(self.nlg_tokenizer.rst_pos_maxidx),
                                                    rst_encoding_ndim=self.embd_outp_dim,
                                                    #init_val=0.0005
                                                    )
              
        self.token_type_embeddings = torch.nn.Embedding( self.nlg_tokenizer.special_token_count -1 + 1 , self.embd_outp_dim, #-1 since <|pad|> is a special token. +1 for padding token
                                                        padding_idx=(self.nlg_tokenizer.special_token_count -1 + 1 )-1,
                                                        scale_grad_by_freq=self.scale_grad_by_freq) #The maximum value this can take is based on the different types of input
        #self.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.0005)
            # 1 for each of da, rst and + 1 for each topic phrase (note that each topic phrase includes a <topic> token.
            #      therefore the largest number of different topics is topic_ctx//2 if every topic only has one word)
        #endregion
        
        if self.freeze_pretrained == 1:
            #Freeze all weights except for: New embedding layers, language model head, wte
            for name, param in self.transformer.transformer.named_parameters(): 
                if 'wte' not in name:
                    param.requires_grad = False
            
        with torch.no_grad():
                # initialising new special tokens to to eos token value
            #self.transformer.transformer.wte.weight[-self.nlg_tokenizer.special_token_count:-1,:] = self.transformer.transformer.wte.weight[-self.nlg_tokenizer.special_token_count-1:-self.nlg_tokenizer.special_token_count,:] 
                # initialising a pad token
            self.transformer.transformer.wte.weight[  -1 ].fill_(0)
        self.transformer.tie_weights()
        
        self.loss_fct = CrossEntropyLoss()
        
        self.nlg_tokenizer.init_pad_values()

    def get_output_embeddings(self):
        return self.transformer.lm_head

    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
                
        parser.add_argument('--base_model_name', default='distilgpt2', required=False)
        parser.add_argument('--reset_base_transformer', default=False, required=False, type=bool)
        parser.add_argument('--freeze_pretrained', default=0, required=False, type=int )

        parser.add_argument('--model_name', default='NLG_v3', required=False)
        parser.add_argument('--loss_type', default='CrossEntropy', required=False, 
            choices=['CrossEntropy','UtteranceSimilarity']) 
        
        parser.add_argument('--context_len', type= lambda x: eval(x), default={'rst':14, 'topics':18 } )
                       
        parser.add_argument('--max_input_len', type=int, default=200)
        
        parser.add_argument('--scale_grad_by_freq', type=lambda x: bool(int(x)) , default=False, 
                help="Inverse the gradients to the emebdding layers based on the occurence of each index in the minibatch ")
        mparams = parser.parse_known_args( )[0]
       
        return mparams


    def forward(self, input_, skip_embed1=False, output_attentions=False):
        """[summary]

            Args:
                input_ (torch.tensor): dict of inputs

            Returns:
                [type]: [description]
        """       

        # Handles embedding of our new non word features
        if skip_embed1 == False:
            input_ = self.forward_embedding(input_)

        transformer_outputs = self.transformer.transformer.forward( 
                                    output_attentions=output_attentions,
                                    return_dict=False,
                                    **input_,
                                    )
                                    
        hidden_states = transformer_outputs[0]

        lm_logits = self.transformer.lm_head( hidden_states )

        if 'labels' in input_:
            
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = input_['labels'][..., 1:].contiguous() 
            
            loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # if self.freeze_pretrained:
            #     # removing gradient from wte. Only the new special tokens get fine_tuned
            #     if self.transformer.transformer.wte.weight.grad != None:
            #         self.transformer.transformer.wte.weight.grad[ :-self.nlg_tokenizer.special_token_count ] = 0
                
            return (lm_logits, loss) 

        else:
            output = (lm_logits,) + transformer_outputs[1:]

            return output

    def forward_embedding(self, input_):
        """Performs the input embedding and token type embedding calcs

            Args:
                input_ ([type]): [description]

            Returns:
                input_embedded [type]: [description]
                
                Note: logic for token_type_ids embedding is usually performed in transformer, but has been moved here
                    since gpt-2 code indicates that 
        """
        new_input = {}
        attention_mask = input_['attention_mask']
        labels = input_['labels']

        # rst token emebedding
        rst_start_embed = self.transformer.transformer.wte( input_['rst_start_token'] ) 
        
        rst_rel_embed = self.embedding_rst_rels( input_['tnsr_rst_rels'] ) 
        rst_ns_embed = self.embedding_rst_ns( input_['tnsr_rst_ns'] ) 
        rst_pos_embed = self.embedding_rst_pos( input_['tnsr_rst_pos'] ) 
        rst_embed = rst_rel_embed + rst_ns_embed + rst_pos_embed

        
        # topic embedding
        topics_pos_embed =  self.embedding_rst_pos( input_['tnsr_topics_pos'])
        topics_phrase_embed = self.transformer.transformer.wte(input_['tnsr_topics_phrase']  )  #this is contiguous from the tokenizer
        topics_embed = topics_phrase_embed + topics_pos_embed
        
        # utterance embedding
        utt_embed = self.transformer.transformer.wte( input_['tknzd_utt'] ) #this is contiguous from the tokenizer

        input_embeds = torch.cat(
            [rst_start_embed, rst_embed,
             topics_embed, utt_embed
             ], axis = 1
            ) #dim [bs, 1024, dim1]
        
        # Token type embedding is only added to the context section
            # We do not bother making a context type embedding for the utterance
        token_type_embedding = self.token_type_embeddings( input_['token_type_ids'] )
        input_embeds[ :, :token_type_embedding.shape[1], :] += token_type_embedding

        # position embedding
        utt_position_embeds = self.transformer.transformer.wpe(input_['position_ids']) 
        _ = utt_position_embeds.shape
        ctx_position_embeds = utt_position_embeds.new_zeros(  [_[0], rst_start_embed.shape[1]+rst_embed.shape[1]+topics_embed.shape[1] ,_[2]]  ) #ctx has no position embedding 
        position_embeds = torch.cat( [ctx_position_embeds, utt_position_embeds], axis=1 )
        
        # Padding will have been inserted in: 
            # rst section in input_embeds
            # topics section in input_embeds
            # tknzed utterance section in input_embeds
            
            #corresponding atte_mask sections
        li_input_ids = [ input_['rst_start_token'], input_['tnsr_rst_rels'], input_['tnsr_topics_phrase'], input_['tknzd_utt'] ]
        li_pad_token_ids = [ self.nlg_tokenizer.e2m_tokenizer.added_tokens_encoder['<|pad|>'], 
                            self.embedding_rst_rels.padding_idx, 
                            self.nlg_tokenizer.e2m_tokenizer.added_tokens_encoder['<|pad|>'],
                            self.nlg_tokenizer.e2m_tokenizer.added_tokens_encoder['<|pad|>'] ]
        
        input_embeds,position_embeds = self.nlg_tokenizer.compress_padding( li_input_ids , li_pad_token_ids,
                                                    input_embeds,
                                                    (position_embeds, 1)
                                                    ) 
        
        new_input['input_embeds'] = input_embeds
        new_input['attention_mask'] = attention_mask
        new_input['position_embeds'] = position_embeds
        new_input['labels'] = labels
        
        return new_input

    def return_params(self):
        keys = ['base_model_name','freeze_pretrained','max_input_len',
                        'scale_grad_by_freq']

        json_keys = ['context_len']
        
        params = {
            k:self.__dict__[k] for k in keys if k in self.__dict__.keys()
        }

        json_params = {
            k:json.dumps(self.__dict__[k]) for k in json_keys if k in self.__dict__.keys()
        }

        json_params = {
            k:json.dumps(self.nlg_tokenizer.__dict__[k]) for k in json_keys if k in self.nlg_tokenizer.__dict__.keys()
        }

        params_ = {**params, **json_params}
        

        return params_

    def get_predicted_utterance(self, rst_rels, rst_ns,
            rst_pos, topics, topics_pos, 
            prompt, generation_params={}):
        """Given an encoded input, outputs predictions up until an <EOS> is emmitted
                The input will have been encoded using nlg_tokenizer
                
            Raises:
                Exception: [description]

            Returns:
                [type]: [description]
        """



        #type checks
        if type(topics) == list:
            topics = tuple(topics)
        
        if type(topics_pos) == list:
            topics_score = tuple(topics_pos)       

        # default generation params
            #TODO: add these to config so they are automatically done
        if 'bad_words_ids' not in generation_params:
            bad_words = ['"',"<|rst|>","<|ta|>", "<|pad|>",'\n', "\s"," \s", ". \s", "|", '\\n', "\\", "\\t", "#|",r'\""']
            bad_words_ids = [self.nlg_tokenizer.e2m_tokenizer.encode(bad_word, add_prefix_space=False) for bad_word in bad_words]
            bad_words_ids = [self.nlg_tokenizer.e2m_tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in bad_words]
            bad_words_ids = bad_words_ids + [[526], [55],[8172], [3467], [59], [6852], [7479],[7879],[13426],[17405],[91],[8614],[930],[10],[9],[12],[1303],[2],[4242], [2235],[46424]]
            generation_params['bad_words_ids'] = bad_words_ids
        else:
            bad_words_ids = None

        default_generation_params = {'num_beams':1, 'temperature':1.2, 'repitition_penalty':2.0, 
                            'early_stopping':True, 'do_sample':False, 'no_repeat_ngram_size':3, 'bad_words_ids':bad_words_ids,'max_length':300 } #,'min_length':4

        for key,value in default_generation_params.items():
            if key not in generation_params:
                generation_params[key] = value


        encoded_input = self.nlg_tokenizer.encode(rst_rels, rst_ns, rst_pos ,
                                                    topics, topics_pos, prompt,
                                                    pad_utterance=False, generate_mode=True)

            # Add batch dimension to data and moving to GPU
        device = next(self.parameters()).device
        for key in ['tnsr_rst_rels', 'tnsr_rst_ns', 'tnsr_rst_pos',
                    'tnsr_topics_phrase','tnsr_topics_pos','tknzd_utt',
                    'position_ids','token_type_ids',
                        'attention_mask','rst_start_token']:
            encoded_input[key] = torch.unsqueeze( encoded_input[key],axis=0).to(device)

            # Generating Text
        output = self.generate(encoded_input, **generation_params)
        gen_text = self.nlg_tokenizer.e2m_tokenizer.decode(output[0],skip_special_tokens=True)

        return gen_text

class NLG_tokenizer(utils.EffeciencyMixin, utils.RstTokenizerMixin):
    """Rough Implmentation of the tokenizer for the NLG model

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """

    def __init__(self,
                 e2m_base_model_name='distilgpt2',
                 dir_tokenizer='./models/NLG_tknzr_v3',
                 max_input_len = 216,  
                 nlg_model = None,               
                 **kwargs
                 ):

        self.e2m_base_model_name = e2m_base_model_name
        self.nlg_model = nlg_model
                

        assert max_input_len < 1025
        self.max_input_len = max_input_len  #self.e2m_tokenizer.max_len
        
        # Setting up RST utilities
        self.rst_rel_li = ['Attribution',
            'Background','Cause','Comparison','Condition',
            'Contrast','Elaboration','Enablement','Evaluation',
            'Explanation','Joint','Manner-Means','Topic-Comment',
            'Summary','Temporal','Topic-Change','same-unit','textual-organization'] #Add this to savable config

        self.rst_rel_labeler = sklp.LabelEncoder()
        self.rst_rel_labeler.fit(  self.rst_rel_li )

        self.rst_ns_li = ['NN','NS','SN'] 
        self.rst_ns_labeler = sklp.LabelEncoder()
        self.rst_ns_labeler.fit( self.rst_ns_li  )

        self.rst_pos_maxidx = 4094 + 1 #final one is padding


        # Setting up context lengths
        self.context_len = kwargs.get( 'context_len', { 'rst':12, 'topics':24 } )
        
        self.context_len_pre_utterance =  sum(self.context_len.values())

        self.context_len['utt'] = self.max_input_len - self.context_len_pre_utterance

        # Initalising tokenzier
        if os.path.isdir(dir_tokenizer):
            self.e2m_tokenizer = AutoTokenizer.from_pretrained(dir_tokenizer,use_fast=False)

            self.special_token_count = 2

        # retreiving   base tokenizer from online or from local distillgpt2
        else:
            dir_transformer = os.path.join("./models",e2m_base_model_name)
            exists = os.path.isdir(dir_transformer)            

            if exists==True:
                self.e2m_tokenizer = AutoTokenizer.from_pretrained(dir_transformer, use_fast=False)
                config = AutoConfig.from_pretrained(dir_transformer)

            elif exists==False:
                self.e2m_tokenizer = AutoTokenizer.from_pretrained(self.e2m_base_model_name,use_fast=False)
                config = AutoConfig.from_pretrained(self.e2m_base_model_name)


            # Adding special tokens
            special_tokens_dict = {'additional_special_tokens':
                    [ '<|rst|>', '<|ta|>', '<|pad|>']}
            
            self.special_token_count = len(special_tokens_dict['additional_special_tokens'])

            if str(special_tokens_dict['additional_special_tokens']) != \
                    self.e2m_tokenizer.special_tokens_map.get('additional_special_tokens',''):
                
                num_added_toks = self.e2m_tokenizer.add_special_tokens(special_tokens_dict)
                os.makedirs(dir_tokenizer)
                
                self.e2m_tokenizer.init_kwargs['name_or_path'] = dir_tokenizer
                self.e2m_tokenizer.init_kwargs['special_tokens_map_file'] = os.path.join(dir_tokenizer,"special_tokens_map.json")
                
                self.e2m_tokenizer.save_pretrained(dir_tokenizer)
                config.save_pretrained(dir_tokenizer)

        self.pad_maxlens = {
                        'rst_start_token':1, 
                        'tnsr_rst_rels': self.context_len['rst']-1, 
                        'tnsr_rst_ns': self.context_len['rst']-1,
                        'tnsr_rst_pos': self.context_len['rst']-1,
                            
                        'tnsr_topics_phrase': self.context_len['topics'] ,
                        'tnsr_topics_pos': self.context_len['topics'],
                        
                        'tknzd_utt': self.context_len['utt'],

                        'attention_mask':sum(self.context_len.values()),
                        'labels':sum(self.context_len.values()),
                        
                        'position_ids':sum(self.context_len.values()) ,
                        'token_type_ids':self.context_len['rst']+self.context_len['topics']
            }
    
    def init_pad_values(self):
        if self.nlg_model !=  None:
            self.pad_values = {'rst_start_token':self.e2m_tokenizer.added_tokens_encoder['<|pad|>'] , 
                        'tnsr_rst_rels': self.nlg_model.embedding_rst_rels.padding_idx , 
                        'tnsr_rst_ns': self.nlg_model.embedding_rst_ns.padding_idx,
                        'tnsr_rst_pos': self.nlg_model.embedding_rst_pos.padding_idx,
                            
                        'tnsr_topics_phrase': self.e2m_tokenizer.added_tokens_encoder['<|pad|>'] ,
                        'tnsr_topics_pos': self.nlg_model.embedding_rst_pos.padding_idx,
                        
                        'tknzd_utt':self.e2m_tokenizer.added_tokens_encoder['<|pad|>'] ,
                        'attention_mask':0.0,
                        
                        'labels':self.nlg_model.loss_fct.ignore_index,
                        
                        'position_ids':self.nlg_model.config.n_ctx-1 ,
                        'token_type_ids':self.nlg_model.token_type_embeddings.padding_idx
                        
                        }
        else:
            self.pad_values = {'rst_start_token':self.e2m_tokenizer.added_tokens_encoder['<|pad|>'] , 
                        'tnsr_rst_rels':len(self.rst_rel_li )-1,
                        'tnsr_rst_ns': len(self.rst_ns_li )-1,
                        'tnsr_rst_pos': self.rst_pos_maxidx-1,
                            
                        'tnsr_topics_phrase': self.e2m_tokenizer.added_tokens_encoder['<|pad|>'] ,
                        'tnsr_topics_pos': self.rst_pos_maxidx-1,
                        
                        'tknzd_utt':self.e2m_tokenizer.added_tokens_encoder['<|pad|>'] ,
                        'attention_mask':0.0,
                        
                        'labels':-100,
                        
                        'position_ids': 1023 ,
                        'token_type_ids':(self.nlg_tokenizer.special_token_count -1 + 1 )-1
                        
                        }

    def rst_vectors(self, version="combinations", relations="all", **kwargs):
            """
                Allows the user to select partiuclar rst_vectors in order to control their output

                version: rule to decide how to compose relations
                relations: A list of the relations to utilise
             """
            count = kwargs.get('count',1)
            assert ( count>0 and count<7 )

            #selecting sub rst relations to evaluate
            rst_rel_li = [ rel for rel in self.rst_rel_li if ( rel in relations) or relations=="all" ]

            if version == "independent":
                rst_names = [[rel] for rel in rst_rel_li]

                rst_rel_encoded = [ self.rst_rel_labeler.transform(rel) for rel in rst_names]
            
            
            if version=="combinations":
                
                combination_count = kwargs.get('combinatoric_count',3)
                iter_rst_comb = combinations( rst_rel_li, combination_count )
                li_rst_comb =  list(iter_rst_comb)
                random.shuffle(li_rst_comb)
                li_rst_comb = li_rst_comb[: kwargs.get("return_count",10) ]           
                
                rst_names = li_rst_comb
                rst_rel_encoded = [  self.rst_rel_labeler.transform(rst_comb) for rst_comb in li_rst_comb] #list of list of each relation
            
            elif version=="permutations":
                
                combination_count = kwargs.get('combinatoric_count',3)
                iter_rst_perm = permutations( rst_rel_li, combination_count )
                li_rst_perm =  iter_rst_perm.tolist()
                random.shuffle(li_rst_perm)

                li_rst_perm = li_rst_perm [: kwargs.get("return_count",10) ]   
                
                rst_names = li_rst_perm
                rst_rel_encoded = [  self.rst_rel_labeler.transform(rst_perm) for rst_perm in li_rst_perm] #list of list of each relation

            elif version=="combinations_with_replacement":
                
                combination_count = kwargs.get('combinatoric_count',3)
                iter_rst_combwr = combinations_with_replacement( rst_rel_li, combination_count )
                li_rst_combwr =  list(iter_rst_combwr)
                random.shuffle(li_rst_combwr)

                li_rst_combwr = li_rst_combwr[: kwargs.get("return_count",10) ]           
                rst_names = li_rst_combwr

                rst_rel_encoded = [  self.rst_rel_labeler.transform(rst_combwr) for rst_combwr in li_rst_combwr] #list of list of each relation

            return rst_rel_encoded, rst_names

    def encode( self, rst_rels, rst_ns , rst_pos, topics, topics_pos, 
        utterance, pad_utterance, generate_mode):

        """
            This version is a smaller output space than v1, by dropping rst_pos and rst_ns
            Return 
            w/o da
            dictionary
            attn_mask : Bidirectional up to bos token, Causal Up till EOS, 0s till end of padding

        Note this method returns integer encodings for tokens that will be processed by BERT embedding layer
            and possibly one-hot encoded vectors that will not be encoded by same pert embedding layer
        """
        
        #Getting Special Tokens
        rst_start_token = self.e2m_tokenizer.encode("<|rst|>",return_tensors="pt")[0] 
        
        #tnsr_rst_rels, rst_pad_count, tnsr_rst_ns, tnsr_rst_pos = self.encode_rst(rst_rels, rst_ns, rst_pos, max_len=self.context_len['rst'] - 1)   # dims (max_padding, n) #not we do rt_len-1 since we add the rst start token outside this method
        #tnsr_topics_phrase, tnsr_topics_pos, topics_pad_count, ta_tokens_pos, ta_phrase_lens  = self.encode_topic( topics, topics_pos, max_len=self.context_len['topics'], padding_token=padding_token) # dims (max_padding, 13) 
        #tknzd_utt, utt_pad_count = self.encode_utterance(utterance, pad_utterance, generate_mode, padding_token)
        tnsr_rst_rels, tnsr_rst_ns, tnsr_rst_pos = self.encode_rst(rst_rels, rst_ns, rst_pos)   # dims (max_padding, n) #not we do rt_len-1 since we add the rst start token outside this method
        tnsr_topics_phrase, tnsr_topics_pos, ta_tokens_pos, ta_phrase_lens  = self.encode_topic( topics, topics_pos ) # dims (max_padding, 13) 
        tknzd_utt = self.encode_utterance(utterance, pad_utterance, generate_mode ) 
        
            # calc the ending cumulative dim for, rst, topics, utt, segments,


        r_dim = rst_start_token.shape[-1] + tnsr_rst_rels.shape[-1]
        rt_dim = r_dim+ tnsr_topics_phrase.shape[-1]
        utt_dim = rt_dim + tknzd_utt.shape[-1]

        # Building Attention Mask
        attn_mask = torch.tril( torch.ones([utt_dim, utt_dim]), diagonal=rt_dim )
            #correcting for padding in rst section


            # Implementing causal masking for each topic subphrase 
                # First, each topic subphrase only attends to other words within that topic subphrase (including ta token) and not the other topcics
                # So set the template attn to 0 
        attn_mask[ r_dim:rt_dim, r_dim:rt_dim ] = 0
                #Second each topic phrase has causal masking on the tokens within the topic phrase
                # use ta_tokens_pos
                # essentially for idx i, i+1
                # add a tril att attn_mask[ i:i+1 , i:i+1 ] so that in each topic phrase each word only attmeds to the previous words
        for ta_idx, phrase_len in zip( ta_tokens_pos, ta_phrase_lens):
            attn_mask[ r_dim+ta_idx:r_dim+ta_idx+phrase_len, r_dim+ta_idx:r_dim+ta_idx+phrase_len ] = torch.tril( attn_mask.new_ones( [phrase_len,phrase_len]  )  )

                #correcting for padding in and after utterance section
                    #when the utterance padding starts then we mask
        #attn_mask[ utt_dim-utt_pad_count: , : ] = 0
        attn_mask[ utt_dim: , : ] = 0

        #Creating labels/targets
        try:
            labels = -100* torch.ones( size=[ utt_dim], dtype = torch.long  ) 

            #first eos token should be excluded from loss calc. so rt_dim + 1 used
            #labels[rt_dim+1:utt_dim-utt_pad_count] =  tknzd_utt[ 1: utt_dim-utt_pad_count-rt_dim ]
            labels[rt_dim+1:utt_dim] =  tknzd_utt[ 1: utt_dim-rt_dim ]

        except Exception:
            labels = None
        
        # Creating Positional Emebeddings
            # ALL words in rt get a positional encoding of 0 -> No positional meaning
            # utterance has normal positional encoding        
        position_ids_utt =  torch.arange( 0, utt_dim-rt_dim, dtype=torch.long)
        position_ids = position_ids_utt

        # Creating Token Type Ids
            #     0:rst, 
            #     1: topics

        token_type_ids_r = torch.full( [r_dim], 0 ,dtype=torch.long)
        token_type_ids_t = torch.full( [rt_dim-r_dim], 1 ,dtype=torch.long)


        token_type_ids = torch.cat( [token_type_ids_r, token_type_ids_t] ) 


        return { 'rst_start_token':rst_start_token,
                'tnsr_rst_rels':tnsr_rst_rels,'tnsr_rst_ns':tnsr_rst_ns, 'tnsr_rst_pos':tnsr_rst_pos,
                'tnsr_topics_phrase':tnsr_topics_phrase.contiguous(), 'tnsr_topics_pos': tnsr_topics_pos.contiguous(),
                 'tknzd_utt':tknzd_utt.contiguous(),
                 'attention_mask':attn_mask.contiguous(),
                 'labels':labels,
                 'position_ids':position_ids.contiguous(),
                 'token_type_ids':token_type_ids.contiguous()
                 }

    def encode_rst(self,rst_rels, rst_ns, rst_pos):
        """Converts rst_rels in a series of vectors

            Args:
                rst_rels ([type]): [description]
                max_padding ([type]): padding amount
                rst_pos ([type]): [description]
                rst_ns
            Also includes an encoding for rst_ns and rst_pos
        """

        #tnsr_rels, diff = self.encode_rst_rels(rst_rels, max_len=max_len)
        tnsr_rels = self.encode_rst_rels(rst_rels )


        # Encoding the rst ns 
        rst_ns_encoded = self.rst_ns_labeler.transform( rst_ns ) #.reshape( [1,-1] )  
        tnsr_ns = torch.LongTensor(rst_ns_encoded)
    
        tnsr_pos = torch.LongTensor( self.clamp_values( np.array(rst_pos), utils.MAX_LONG_VALUE ) ) #.reshape([1,-1])
            

        # padding ns and pos
            # The ns and pos embedding layer uses the index value 0 as a padding index
            # For this index the vector is initialized to zer0 and as such never updates
        
        max_len = self.context_len['rst'] - 1
        len_ =  tnsr_rels.shape[0]
        if len_ > max_len:
            tnsr_rels = tnsr_rels[:max_len ] 
            tnsr_ns = tnsr_ns[:max_len]
            tnsr_pos = tnsr_pos[:max_len]

        #return tnsr_rels, diff, tnsr_ns, tnsr_pos
        
        return tnsr_rels, tnsr_ns, tnsr_pos

    def encode_rst_rels(self,rst_rels):
        """Converts rst_rels in a series of vectors

            Args:
                rst_rels ([type]): [description]
                max_padding ([type]): padding amount
                rst_pos ([type]): [description]
        """
        rst_rel_encoded = self.rst_rel_labeler.transform(rst_rels) #.reshape( [1 , -1] )
        tnsr_rels = torch.LongTensor( rst_rel_encoded )
        
        # #Padding out to max_len length
        # _len = tnsr_rels.shape[0]
        # diff = (max_len - _len)
        # if diff > 0:
        #     tnsr_rels = torch.cat([tnsr_rels, torch.full( [diff], len(self.rst_rel_li ), dtype=torch.long)] , axis=-1 ) 
        # elif diff == 0:
        #     pass
        # else:
        #     tnsr_rels = tnsr_rels[ :max_len]
        #     diff = 0

        return tnsr_rels#, diff
    

    def encode_topic(self, topics, topics_pos):
        """[summary]

            Args:
                topics ([type]): [list of topics (phrases or words)]
                topics_score ([type]): [list of float scores for each topic relevancy]

            Raises:
                Exception: [description]

            Returns:
                [type]: [description]
        """
        max_len =self.context_len['topics']
        
        str_topics = ''.join([ '<|ta|>'+topic  for topic in topics ])
        dict_encoding = self.e2m_tokenizer(str_topics, add_special_tokens=False,
                                            return_attention_mask = False, 
                                            truncation = True,
                                            padding='do_not_pad', 
                                            return_tensors='np',
                                            max_length = max_len,
                                            return_token_type_ids=None,
                                            return_special_tokens_mask=False,
                                            return_length=True )
        topic_phrases = dict_encoding['input_ids'][0]
        
        #Repeating each score in the case where the score is allocated to a phrase topic which is broken down into constituent words
                # e.g. topics - ["fast car", "motorbike", "long rail road"], scores = [0.9, 0.4, 0.2] -> scores = [0.9, 0.9, 0.9, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2]
                # have to do it after tokenization due to bytepair encoding 
        # get index of where <|ta|> tokens occur
        ta_idxs = np.where( topic_phrases==self.e2m_tokenizer('<|ta|>',return_attention_mask=False)['input_ids'] )[0]
        topic_phrases = torch.LongTensor(topic_phrases)

        #filtering out idxs if index is larger than padding value
        ta_idxs = ta_idxs[ta_idxs<max_len]

        #get difference in index position between <|ta|> tag n and <|ta|> tag n+1 ( for final tag use difference between tag and end of list)
        ta_phrase_lens = np.diff( ta_idxs, append=dict_encoding['length'] ) 
        
            # copies each score phrase_len times to cover that phrase and handles case where there is no phrase
        topics_pos = [ [score]*phrase_len for score, phrase_len in zip(topics_pos, ta_phrase_lens) ]
        topics_pos = sum(topics_pos,[]) #flattening list
        tnsr_pos = torch.LongTensor( self.clamp_values( np.array(topics_pos), utils.MAX_LONG_VALUE )  )
        
        
        #Padding out to max_len
        # diff = (max_len - dict_encoding['length'])[0]
        # if diff>0:
        #     topic_phrases = torch.cat( [ topic_phrases, torch.ones([diff], dtype=torch.int64 )*padding_token[0]] , axis=-1 )
            
        #     tnsr_pos = torch.cat( [tnsr_pos, torch.zeros( [1, diff] ) ], axis=-1 )
        # else:
        #     topic_phrases = topic_phrases[:max_len]
        #     tnsr_pos = tnsr_pos[:, :max_len ]
        #     diff = 0

        #return topic_phrases , tnsr_pos, diff, ta_idxs, ta_phrase_lens
        return topic_phrases , tnsr_pos, ta_idxs, ta_phrase_lens


    def encode_utterance(self, utterance, pad=True, generate_mode=False ):
        #pad: 
        #   set to True during training to ensure all batches have the same length
        #   set to False in the case of Generation in order to work with huggingface .generate()
        
        
        if generate_mode == False:
            utterance ='<|endoftext|>' + utterance + '<|endoftext|>'
            
        else:
            utterance ='<|endoftext|>' + utterance
            
            
        # if pad == True:
        #     encoded = self.e2m_tokenizer( utterance, add_special_tokens=False,
        #                                 return_attention_mask = False, 
                                        
        #                                 padding='do_not_pad',
        #                                 truncation=True, 
        #                                 max_length= self.context_len['utt'],
                                                                                
        #                                 return_tensors='pt',
        #                                 return_length=True,
        #                                 return_token_type_ids=None,
        #                                 add_prefix_space = False
        #                                 )
            
        #     #tokenizer length usually returns the padded length as opposed the original length.
        #     #So using 'do_not_pad' and then padding manually
        #     tknzd_utt_no_pad_len = encoded['length'][0]
        #     pad_count = self.context_len['utt'] - tknzd_utt_no_pad_len            
        #     tknzd_utt = torch.cat( [ encoded['input_ids'], torch.LongTensor(1,pad_count).fill_(padding_token.item() ) ],axis=-1 )[0]
                                           
        
        # elif pad == False:
                        
            # encoded = self.e2m_tokenizer( utterance, add_special_tokens=False,
            #                             return_attention_mask = False, 
            #                             padding='do_not_pad',
            #                             truncation=True, 
            #                             max_length= self.context_len['utt'],
            #                             return_tensors='pt',
            #                             return_length=True,
            #                             return_token_type_ids=None,
            #                             add_prefix_space=False)
            # tknzd_utt_no_pad_len = encoded['length'][0]
            # pad_count = 0

        
        # tknzd_utt = encoded['input_ids'][0]

        
        encoded = self.e2m_tokenizer( utterance, add_special_tokens=False,
                                    return_attention_mask = False, 
                                    padding='do_not_pad',
                                    truncation=True, 
                                    max_length= self.context_len['utt'],
                                    return_tensors='pt',
                                    return_length=True,
                                    return_token_type_ids=None,
                                    add_prefix_space=False)
            
        tknzd_utt = encoded['input_ids'][0]
        
        #return tknzd_utt, pad_count
        return tknzd_utt

class TrainingModule(pl.LightningModule):

    def __init__(self, model_params, batch_size=20, 
                    dir_data=None, 
                    accumulate_grad_batches=1,
                    max_epochs=25,
                    gpus=1, 
                    learning_rate=1e-4,
                    warmup_proportion=0.1,
                    workers=0,
                    lr_schedule='hard_restarts',
                    mode = 'train_new',
                    data_splits = {'train':0.6,'val':0.2,'test':0.2},
                    inference_context_utt = None, #amount of words from utterance to use as context
                    optimizer_type="AdamW",
                    tag='',
                    *args,
                    **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.gpus =  gpus
        self.model = NLG( **model_params )
        self.mode = mode
        self.workers = workers
        self.data_splits = data_splits
        self.optimizer_type = optimizer_type
        
        
        if self.mode in ['train_new','train_cont','test']:
            self.dir_data = utils.get_path(dir_data)
            self.inference_context_utt = inference_context_utt
            #self.create_data_loaders( )
            self.accumulate_grad_batches = accumulate_grad_batches
            self.tag = tag

        if self.mode in ['train_new','train_cont']:
            self.max_epochs = max_epochs
            self.warmup_proportion = warmup_proportion
            self.lr_schedule = lr_schedule
            self.learning_rate = learning_rate
        
            train_params_to_save = self.return_params()
            model_params_to_save = self.model.return_params()


            self.hparams.update({ **train_params_to_save, **model_params_to_save})
            pl.core.saving.save_hparams_to_yaml( os.path.join( os.path.dirname(kwargs['dir_checkpoints']), "hparams.yaml"), self.hparams )
                
            
            bad_words = ["<|rst|>","<|ta|>","<|pad|>",r"\n" ] 
            bad_words_ids = [self.model.nlg_tokenizer.e2m_tokenizer.encode(bad_word, add_prefix_space=False) for bad_word in bad_words]
            bad_words_ids = bad_words_ids + [self.model.nlg_tokenizer.e2m_tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in bad_words]
            bad_words_ids = bad_words_ids + [[526], [55],[8172]]
            

            generation_params = {'num_beams':1, 'temperature':1.1, 'repitition_penalty':1.2, 
                                'top_k': 50, 'top_p':0.85,
                                'length_penalty':1.5, 'early_stopping':True,
                                'do_sample':True, 'bad_words_ids':bad_words_ids, 'no_repeat_ngram_size':3,
                                'min_length':5, 'max_length':80  } 
                                
                                # 'max_length':self.model.nlg_tokenizer.max_input_len  } 
            
            self.inference_generation_params = generation_params
            self.init_data()
            

        if self.mode in ['inference']:
            self.eval() 
            self.freeze() 
    
    def init_data(self):
        if self.mode in ['train_new','train_cont']:
            self.create_data_loaders( ['train','val','inference'] )
            self.inference_samples = list( islice( self.inference_dl, 10 ) )
            del self.inference_dl
            
        elif self.mode in ['test']:
            self.create_data_loaders( ['test'] )
        
    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data', default="./dataset_v3", help="Relative directory path for datafiles")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('--max_epochs', default=8, type=int)
        parser.add_argument('--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('-b','--batch_size', default=5, type=int)
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--warmup_proportion', default=0.35)
        parser.add_argument('--workers', default=16, type=int) #TODO: change to 6
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','train_cont','test','inference'])
        parser.add_argument('--lr_schedule', default='cosine_warmup', required=False, choices =['cosine_warmup','LROnPlateau','hard_restarts','constant'])
        parser.add_argument('--splits', default={'train':0.6,'val':0.2,'test':0.2}, required=False, type=str )
        parser.add_argument('--version', default=None,required=False, type=int, help="The Experimental Versioning for this run" )
        parser.add_argument('--precision', default=16,required=False, type=int, help="Precision to use", choices=[16,32] )
        parser.add_argument('--optimizer_type', default="Adafactor",required=False, type=str, help="Optimizer to use", choices=["AdamW","Adafactor"] )
        parser.add_argument('--tag',default='',required=True, type=str)
        parser.add_argument('--override',default=False, type = lambda x: bool(int(x)), choices=[0,1] )
        parser.add_argument('--inference_context_utt', default=4, type=int)
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
        #pl.seed_everything(10)

        if tparams['mode'] in ["train_new"]:
            training_module = TrainingModule(**tparams, model_params=mparams  )
            
        elif tparams['mode'] in ["train_cont", "inference"]:
            
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            #restore/update param files from the checkpoint
            if "hyper_parameters" in checkpoint.keys() and tparams['override'] == False:
                tparams.update ( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'batch_size', 'lr_schedule', 'learning_rate','precision','splits','optimizer_type','tag']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name','loss_type','model_name','max_input_len',
                    'frst_version','scale_grad_by_freq','freeze_pretrained']} )
                
                mparams_json = {k:json.loads(v) for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'context_len'] }
        
                mparams =  {**mparams, **mparams_json}
            
            else:
                print("param files not found utilsing default or user entered params\n")
                
            #Restore/update Training Module
            training_module = TrainingModule(**tparams, model_params=mparams)
            training_module.load_state_dict(checkpoint['state_dict'])

        elif tparams['mode'] in ["test"]:
            
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            #restore/update param files from the checkpoint
            try:
                tparams.update ( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'lr_schedule', 'learning_rate','precision','splits','optimizer_type']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name','loss_type','model_name','max_input_len']} )
            except KeyError:
                pass
            
            #Restore/update Training Module
            training_module = TrainingModule(**tparams, model_params=mparams)
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
        
        # Creating Callbacks
        callbacks = []        
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, 
            mode='min', dirpath=dir_checkpoints, 
            filename='{epoch:03d}_{val_loss:.5f}')
        
        checkpoint_callback._save_model  = types.MethodType(utils.monkey_save_model,checkpoint_callback) #monkey patch

        val_check_interval = 0.1
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=max(6, int(1/val_check_interval)/2 ),
            verbose=False,
            mode='min'
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(early_stop_callback)

       
        if tparams['gpus'] in [0,1]:
            #accelerator=None
            trainer_vars = {}
        else:

            trainer_vars = { 'accelerator':'ddp',
                    'plugins':DeepSpeedPlugin( stage=2 )
                        }

        
        if tparams['mode'] in ["train_new"]:
            
            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                        progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                        default_root_dir=tparams['dir_checkpoints'],
                        logger=tb_logger,
                        #log_every_n_steps=20,
                        precision=tparams['precision'], 
                        callbacks=callbacks,
                        #accelerator=accelerator,
                        val_check_interval=val_check_interval,
                        num_sanity_val_steps=0, 
                        replace_sampler_ddp=False,
                        #track_grad_norm = True,
                        #overfit_batches=25,
                        #fast_dev_run=2, 
                        #log_gpu_memory=True
                        **trainer_vars,

                        )
            #training_module.init_data()

        elif tparams['mode'] in ["train_cont","inference"]:
            #restoring checkpoint             
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            #training_module.load_state_dict(checkpoint['state_dict'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace( **tparams),
                    progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                     logger=tb_logger,
                      
                    precision=tparams['precision'],
                    callbacks=callbacks,accelerator=accelerator,
                         val_check_interval=val_check_interval,
                        num_sanity_val_steps=0, 
                        replace_sampler_ddp=False,
                        #track_grad_norm = True,
                        #overfit_batches=25,
                        #fast_dev_run=2, 
                        #log_gpu_memory=True
                        **trianer_vars,
                    )
            #training_module.init_data()
            
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
                    )

            # load callback states
            trainer.on_load_checkpoint(checkpoint)
            #training_module.init_data()

        return trainer,training_module
    
    @staticmethod
    def get_ckpt_file(_dir_checkpoint,mode='best'):
        if mode=='best':
            checkpoint_yaml_file = os.path.join( _dir_checkpoint,"best_k_models.yaml" )
            scores_dict = yaml.load( open(checkpoint_yaml_file,"r") ) #key= ckptpath, value = val_loss
            best_ckpt_path = min(scores_dict, key=scores_dict.get)
            
            if os.path.exists(best_ckpt_path) == False:
                root_dir = Path(__file__).resolve().parents[4]
                best_ckpt_path = os.path.join( root_dir._str, best_ckpt_path[ best_ckpt_path.index('mastering-conversation'): ] )

            if torch.cuda.is_available():
                checkpoint = torch.load(best_ckpt_path, map_location='cpu' )  

            else:
                checkpoint = torch.load(best_ckpt_path, map_location='cpu')            
        else:
            raise NotImplementedError
        
        return checkpoint
        
    @staticmethod
    def start(trainer, tparams,training_module, mparams ):
        
        if tparams['mode'] in ['train_new','train_cont']:    
            trainer.fit(training_module )
        
        if tparams['mode'] in ["test"]:
            
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])
            training_module.load_state_dict(checkpoint['state_dict'])

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
    def load_nlgmodel(model_name="NLG_rt", model_version=11,max_input_len=None, device="cuda:0"):
        # Loading in NLG model
        checkpoint = TrainingModule.get_ckpt_file(f'./models/{model_name}/version_{model_version}/checkpoints')

        # Getting tparams
        tparams = {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
            'batch_size', 'lr_schedule', 'learning_rate','precision','splits','optimizer_type',
            'tag']}

        tparams['mode'] = 'inference'

        mparams =  {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
            'base_model_name','loss_type','model_name','max_input_len',
            'freeze_pretrained','frst_version','scale_grad_by_freq']}
        
        if model_version in [14,15,16]:
            mparams_json = {'context_len': {'rst':16, 'topics':30} }
        else:
            mparams_json = {k:json.loads(v) for k,v in checkpoint['hyper_parameters'].items() if k in [
            'context_len'] }

        mparams =  {**mparams, **mparams_json}
        
        if max_input_len != None:
            mparams['max_input_len'] = max_input_len
            
        # Loading Training Module
        training_module = TrainingModule(**tparams, model_params=mparams )
        training_module.load_state_dict(checkpoint['state_dict'])
        nlg_model = training_module.model

        # Deleting checkpoints to free up GPU space
        del checkpoint
        torch.cuda.empty_cache()
          
        #if torch.cuda.is_available():
        if device != 'cpu' and torch.cuda.is_available():
            nlg_model = nlg_model.to(device)
        
        return nlg_model

    @auto_move_data
    def forward(self, input_):
        return self.model(input_)

    def step(self, batch, step_name):
        
        input_= batch
        _, loss = self.forward(input_) #(lm_logits and loss)
        loss_key = f"{step_name}_loss"
        
        output = {}

        if step_name == 'train':
            output["loss"] = loss

        else:
            str_loss_key = loss_key       
            self.log( str_loss_key, loss)
            
            output[str_loss_key]=loss


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
            loss = torch.stack([x[f"{step_name}_loss"] for x in outputs]).mean()
            self.log(f"{step_name}_loss", loss, logger=True, prog_bar=True)
        
        # if step_name == "val" and _get_rank() == 0 :
             
        #     # Making directory if it doesnt exist
        #     dir_infer = os.path.join(self.trainer.log_dir,"inference")
        #     if not os.path.exists(dir_infer):
        #         os.makedirs(dir_infer,exist_ok=True)

        #     # Adding true values and making csv files if thy dont already exists
        #     for idx, encoded_input in enumerate(self.inference_samples):
        #         fp =  os.path.join( dir_infer,f"example_{idx:03d}.csv")

        #         # If there file does not exists we add the true observed records
        #         if not os.path.exists(fp):
                    
        #             df = pd.DataFrame(columns=[ 'epoch' ,'rst_rels','topics','utterance'])
        #             rst_rels = encoded_input.pop('orig_rst_rels')
        #             topics = encoded_input.pop('orig_topics')
        #             utterance = encoded_input.pop('orig_utt')
                    
        #             datum = { 'val_round': -1,
        #                         'rst_rels': ', '.join( sum( rst_rels, ())),
        #                         "topics": ', '.join(sum( topics, () )),
        #                         "utterance":utterance[0] }
                
        #             df = df.append(datum, ignore_index=True)
        #             df.to_csv( fp, index=False)
                
        #         # Loading in dataframe of previous predictions
        #         df = pd.read_csv(fp)    

        #         # creating predition andding to existing results
        #         encoded_input.pop('orig_rst_rels', None)
        #         encoded_input.pop('orig_topics', None)
        #         encoded_input.pop('orig_utt', None)

        #         for k in encoded_input.keys():
        #             encoded_input[k] = encoded_input[k].to(torch.device('cuda:0') )

        #         output = self.model.generate(encoded_input, **self.inference_generation_params)
        #         output = output[0].detach().to('cpu')
        #         decoded_text = self.model.nlg_tokenizer.e2m_tokenizer.decode(output,
        #                             skip_special_tokens=True)
        #         datum = {
        #             'val_round':df['val_round'].max()+1,
        #             'rst_rels': '',
        #             'topics':'',
        #             'utterance':json.dumps(decoded_text) }
        #         df = df.append(datum, ignore_index=True)
        #         df.to_csv( fp, index=False)
        #         # Saving to file
                   
    def create_data_loaders(self, modes):
       
        dg = DataLoaderGenerator(self.dir_data,  self.batch_size, self.model.nlg_tokenizer, 
                workers=self.workers, mode=self.mode, split=self.data_splits,
                inference_context_utt=self.inference_context_utt, gpus=self.gpus )

        if 'train' in modes:
            self.train_dl =  dg.prepare_dataloader(shuffle=True, split_name = 'train')
        if 'val' in modes:
            self.val_dl = dg.prepare_dataloader(shuffle=False, split_name = 'val')
        if 'test' in modes:
            self.test_dl =  dg.prepare_dataloader(shuffle=False, split_name = 'test')
        if 'inference' in modes:
            self.inference_dl =  dg.prepare_dataloader(shuffle=True, split_name = 'inference')

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
        
        if self.optimizer_type == "AdamW":
            
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            
            warmup_steps = int( self.warmup_proportion*self.total_steps() )

            lr_schedule = get_cosine_schedule_with_warmup(optimizer, 
                            warmup_steps, self.total_steps(), 0.5 )

            return [optimizer], [{ "scheduler":lr_schedule ,"interval": "step", "monitor":"val_loss"}]
        

        elif self.optimizer_type == "Adafactor":
            optimizer = Adafactor(self.model.parameters(), scale_parameter=False, 
                                relative_step=False, warmup_init=False, lr=self.learning_rate )
            
            lr_scheduler = get_constant_schedule_with_warmup(optimizer,
                                num_warmup_steps= 0.10*self.total_steps(),
                                )
        

            return [optimizer], [{ "scheduler":lr_scheduler ,"interval": "step", "monitor":"val_loss"}]


    def return_params(self):
        params = {}
        keys = ['batch_size','accumulate_grad_batches','lr_schedule','learning_rate','max_epochs','dir_data'
            'warmup_proportion','optimizer_type','tag','inference_context_type']
        
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
                    splits={'train':0.6,'val':0.2,'test':0.2},
                    inference_context_utt=0,
                    gpus = 1,
                    pad_values = {},
                    **kwargs):
        
        self.dir_data = dir_data
        self.tokenizer = tokenizer
        self.splits = splits

        self.bs = batch_size
        self.workers  = workers
        self.mode = mode
        self.gpus = gpus
        self.inference_context_utt = inference_context_utt
        self.pad_values = tokenizer.pad_values
        self.pad_maxlens = tokenizer.pad_maxlens

    def prepare_dataloaders(self):
        """prepares a train, validation and test set

        Returns:
            [type]: [description]
        """
                
        if self.mode in [ 'train_new', 'train_cont']:
            train_dl = self.prepare_dataloader(self.dir_data, shuffle=True, split_name='train' )
            val_dl = self.prepare_dataloader(self.dir_data, shuffle=False,split_name='val'  )
            test_dl = self.prepare_dataloader(self.dir_data, shuffle=False,split_name='test'  )
            inference_dl = self.prepare_dataloader(self.dir_data, shuffle=True, split_name="inference")
        
        elif self.mode in ['test']:
            train_dl= None
            val_dl = None
            test_dl = self.prepare_dataloader(self.dir_data, shuffle=False ,split_name='test' )
            inference_dl = None

                    
        dict_dl = {'train_dl':train_dl,
                    'val_dl':val_dl,
                    'test_dl':test_dl,
                    'inference_dl':inference_dl}

        return dict_dl 

    def prepare_dataloader(self, shuffle=False, 
        split_name='train'):

        """Prepares a dataloader given a directory of data for NLG language module
            # The current method takes a percentage of data from each subdirectory
            Args:
                dir_dset ([type]): [description]
        """
        dir_data = self.dir_data
        
        #getting all files from all different subreddits/types of conversation
        fns = glob.glob(  os.path.join( utils.get_path(dir_data),"*", "*") )
        fns = [fn for fn in fns if os.path.split(fn)[-1]!="lock" and "dict_len" not in fn]
        #getting number of utterances records in each file
        files_sizes = [ int(fn[-10:]) for fn in fns]

        #defining starting line and total lines to use for dataset
        if split_name == 'train':
            line_starts = [0]*len(files_sizes)
            line_ends = [ ls+int(fs*self.splits['train']) for ls,fs in zip(line_starts, files_sizes)  ]
            #line_ends = [ 100 for ls,fs in zip(line_starts, files_sizes)  ]
            ifc = 0
            bs = self.bs
            shuffle = True
        
        elif split_name == 'val':
            line_starts = [ int(fs*self.splits['train']) for fs in files_sizes  ]
            line_ends = [ ls+int(fs*self.splits['val']) for ls,fs in zip(line_starts, files_sizes)  ]
            #line_ends = [ ls+40 for ls,fs in zip(line_starts, files_sizes)  ]
            shuffle = False
            ifc = 0
            bs = self.bs
            
        elif split_name == 'test':
            line_starts = [ int(fs*(1-self.splits['test']) ) for fs in files_sizes  ]
            line_ends = files_sizes
            #line_ends = [ ls+40 for ls,fs in zip(line_starts, files_sizes)  ]
            ifc = 0
            bs = self.bs
            shuffle = False

        elif split_name == 'inference':
            line_starts = [ int(fs*(1-self.splits['test']) ) for fs in files_sizes  ]
            line_ends =  files_sizes
            #line_ends = [ ls+40 for ls,fs in zip(line_starts, files_sizes)  ]
            sampler = None
            ifc = self.inference_context_utt
            bs = 1
            shuffle = False

        li_dsets = [ SingleDataset(_f, self.tokenizer, line_start, line_end, ifc) 
                        for _f, line_start, line_end in zip(fns, line_starts, line_ends) ]
            
        concat_dset = torch.utils.data.ConcatDataset(li_dsets)
                                
        # if self.gpus <= 1 or split_name not in ['inference','test'] :
        #     sampler = SizedOrdered_Sampler(concat_dset, bs, shuffle=shuffle )
        # else:
        #     sampler = SizedOrdered_DistributedSampler( concat_dset, bs, shuffle=shuffle, gpus=self.gpus )
        sampler = None
 
        
        dataloader = torch.utils.data.DataLoader(concat_dset, batch_size=bs,
                num_workers=self.workers,
                sampler = sampler,
                collate_fn =  self.tokenizer.default_collate_pad,
                shuffle = shuffle
                )
        return dataloader

    def __call__(self):
        dict_dl = self.prepare_dataloaders()
        return dict_dl
    
class SingleDataset(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

        create a custom index which sorts the entries by their length
    """
    def __init__(self, file_path, tokenizer, line_start, line_end, inference_context_utt=0 ):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.line_start = line_start
        self.line_end = line_end

        self.inference_context_utt = inference_context_utt
                
        skiprows = self.line_start if self.line_start!=0 else None
        
        
        with open(self.fp, 'r') as f:
            if self.line_start == 0:
            
                self.data = pd.read_csv(file_path, sep=',', header=0, 
                    skiprows=skiprows, nrows=(self.line_end-self.line_start) )

            else: 
                names = open(file_path,"r").readline().strip().split(',')
                            
                self.data = pd.read_csv(file_path, sep=',', 
                    names=names, skiprows=skiprows,
                    nrows=(self.line_end-self.line_start) )
        

    def __len__(self):
        return (self.line_end - self.line_start)
    
    def __getitem__(self, index, pad_utterance=True):
        
        rst_rels, rst_ns, rst_pos, topics, topics_pos, utterance = self.getitem_extract_datum(index)
        
        if self.inference_context_utt != 0:
            
            utterance_context = ' '.join( utterance.split(' ')[:self.inference_context_utt] )
            
            encoded = self.getitem_tokenize(rst_rels,rst_ns, rst_pos,
                                         topics, 
                                        topics_pos, utterance_context,
                                        pad_utterance=False,
                                        generate_mode=True )

            encoded['orig_rst_rels'] = rst_rels
            encoded['orig_utt'] = utterance
            encoded['orig_topics'] = topics
        
        else:
        
            encoded = self.getitem_tokenize( rst_rels,  rst_ns, rst_pos,
                                         topics, 
                                        topics_pos, utterance,
                                        pad_utterance=pad_utterance )

        return encoded

    def getitem_extract_datum(self, index):
        
        datum = self.data[index:index+1]

        #region RST
        li_rst = json.loads(datum['rst'].values[0])  #list of dictionaries 
        rst_rels = [ _dict['rel'] for _dict in li_rst ]
        rst_ns = [ _dict['ns'] for _dict in li_rst ]
        rst_pos = [ _dict['pos'] for _dict in li_rst ]
        
            #sorting the order to be left to right in binary tree
        sorted_order = [i[0] for i in sorted(enumerate(rst_pos), key=lambda x: ( NLG_tokenizer.edukp_pos_sort_function(x[1]), x[1] ) )]
        rst_rels = [ rst_rels[idx] for idx in sorted_order ]
        rst_ns = [ rst_ns[idx] for idx in sorted_order ]
        rst_pos = [ rst_pos[idx] for idx in sorted_order ]
        #endregion

        
        #Topic scores
        topics_textrank = json.loads( datum['li_pos_kp'].values[0])
        topics_pos, topics = zip( *topics_textrank ) #top 3 important prhases from utterance
        topics_pos = tuple( int(pos) for pos in topics_pos )

        #Utterance
        utterance = ujson.loads( datum['txt_preproc'].values[0] )
        
        
        return rst_rels, rst_ns, rst_pos, topics, topics_pos, utterance

    def getitem_tokenize(self,  rst_rels, rst_ns, rst_pos ,topics, topic_pos,
        utterance,pad_utterance=True, generate_mode=False):
        
        encoded = self.tokenizer.encode(rst_rels, rst_ns, rst_pos ,
                        topics, topic_pos, utterance,
                        pad_utterance=pad_utterance, generate_mode=generate_mode)

        return encoded

class SizedOrdered_Sampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.
    #TODO; add this to pytorch. Sampler to sort nlp datasets by size
    Args:
        data_source (Dataset): dataset to sample from
    """
    
    def __init__(self, data_source, batch_size, shuffle) -> None:
        self.data_source = data_source
                        
        np_txt_lens = np.concatenate( [ds.np_textlens for ds in self.data_source.datasets] ).flatten()

        #Indices are sorted in order of the text lens of records in the datasets
        np_ordered_lens = np_txt_lens.argsort()
        
        # We Randomly re-arrange them in batches of batch size
        #li_chunked_lens = np.array_split( np_ordered_lens, self.__len__() //batch_size  )
        li_chunked_lens = [ np_ordered_lens[idx:idx+batch_size] for idx in range(0, np_ordered_lens.size - batch_size, batch_size) ]
        
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

    def __init__(self, dataset: Dataset, batch_size: int,
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
        
        #self.num_samples 
        #self.total_size = self.num_samples * self.num_replicas
        self.seed = seed
    
        # new code
        #self.dataset = dataset
        self.data_source = dataset
        np_txt_lens = np.concatenate( [ds.np_textlens for ds in self.data_source.datasets] ).flatten()

        #Indices are sorted in order of the text lens of records in the datasets
        np_ordered_lens = np_txt_lens.argsort()
        
            # We Randomly re-arrange them in batches of batch size
        li_chunked_lens = [ np_ordered_lens[idx:idx+batch_size] for idx in range(0, np_ordered_lens.size-batch_size, batch_size) ]

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
        #assert all( len(self.li_li_sizeorderedidx[idx])==self.num_samples for idx in range(len(self.num_replicas)) )
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
    trainer, training_module = TrainingModule.instatiate_trainer( tparams,  tb_logger, training_module)
    TrainingModule.start(trainer, tparams, training_module, mparams)
                
if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False) 
    
    # add model specific args
    mparams = NLG.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = TrainingModule.parse_train_specific_args(parent_parser)

    if tparams.mode == "test":
        assert tparams.gpus in [0,1]

    if tparams.gpus not in [0,1]:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '65302'

    try:
        main(vars(tparams), vars(mparams))
    except Exception:
        print(traceback.format_exc())


# dullduks server version 1 - No Freezing, Full RST

#   CUDA_VISIBLE_DEVICES=0 python3 train_nlg_v3.py --batch_size 3 --gpus 1 --accumulate_grad_batches 8 --freeze_pretrained 0 --scale_grad_by_freq 1 --max_input_len 220 --context_len '{ "rst":18, "topics":35 }' --workers 12 --version 2 --precision 16 --mode train_new --tag "gpt2 large . utterance 170 tokes long" --base_model_name "gpt2-large" --learning_rate 5e-4 --max_epochs 5

#   CUDA_VISIBLE_DEVICES=3,4 python3 train_nlg_v3.py --batch_size 8 --gpus 1 --accumulate_grad_batches 2 --freeze_pretrained 0 --scale_grad_by_freq 0 --max_input_len 240 --context_len '{ "rst":18, "topics":35 }' --workers 20 --version 5 --precision 16 --mode train_new --tag "gpt2 large . utterance 187 tokes long" --base_model_name "gpt2-large" --learning_rate 5e-4 --max_epochs 6
