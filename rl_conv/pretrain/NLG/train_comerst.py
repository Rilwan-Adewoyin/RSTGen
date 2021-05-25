#Script for training models related to predicting Key phrases from knowledge graph
# Example of how to use batch generate -> https://github.com/allenai/comet-atomic-2020/blob/master/models/comet_atomic2020_bart/generation_example.py

# Use this for finishing off code for fine-tuning https://github.com/allenai/comet-atomic-2020/blob/master/models/comet_atomic2020_bart/finetune.py


import os
os.environ['NCCL_SOCKET_IFNAME'] =  'lo' 
#os.environ['NCCL_SOCKET_IFNAME'] =  'enp3s0'
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple
import transformers
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pytorch_lightning.trainer.supporters import CombinedLoader

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    size = mask.size()
    if len( size ) == 2:
        bsz, src_len = size
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

    elif len(size) == 3:
        bsz, tgt_len , src_len = size
        
        expanded_mask = mask[:, None, :, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

transformers.models.bart.modeling_bart._expand_mask = _expand_mask

import utils_comerst as utils

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from torch.utils.data import DataLoader

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning import loggers as pl_loggers
import yaml
import glob
import ast
import regex as re
import random
import types
from sklearn import preprocessing as sklp

import ujson

from torch.nn import CrossEntropyLoss
from torch.utils.data._utils.collate import default_convert, default_collate
from torch.nn.utils.rnn import pad_sequence


import names
import pandas as pd

from pytorch_lightning.core.decorators import auto_move_data
from typing import Optional, Callable, Union, Optional, List, Iterable

from functools import lru_cache
import spacy

nlp = spacy.load("en_core_web_sm") 
components = ["parser","ner"]
# removing non needed componenets
for name in nlp.pipe_names:
    if name not in components:
        nlp.remove_pipe(name)


#region COMERST model and tokenizer
class COMERST(nn.Module):

    def __init__(self, 
                    base_model_name='bart_base', model_name="COMERST",
                    scale_grad_by_freq=False,
                    max_len_encoder=80,
                    max_len_decoder=20
                    ):
        super(COMERST, self).__init__()

        self.base_model_name = base_model_name   
        self.model_name = model_name
        self.scale_grad_by_freq = scale_grad_by_freq
        
        self.transformer = utils.load_pretrained_transformer(self.base_model_name, transformer=True)['transformer']
        
        self.tokenizer = COMERST_tokenizer(self.base_model_name,
                                            os.path.join( ("./models"), f"{model_name}_tokenizer"),
                                            max_len_encoder, max_len_decoder )
        
        self.transformer.resize_token_embeddings( len(self.tokenizer.base_tokenizer) )

        # region Extra embedding layers
        self.embedding_rst_rels = torch.nn.Embedding( len(self.tokenizer.rst_rel_li )+1, self.transformer.config.d_model, 
                                    padding_idx=len(self.tokenizer.rst_rel_li ), scale_grad_by_freq=self.scale_grad_by_freq )
        self.embedding_rst_rels.weight.data.normal_(mean=0.0, std=0.005)

            #+2 here because are node data is 0 indexed with node 30 as maximum node
        self.embedding_rst_pos = torch.nn.Embedding( self.tokenizer.rst_pos_maxidx + 1 +1 , self.transformer.config.d_model, 
                                    padding_idx=self.tokenizer.rst_pos_maxidx + 1 , scale_grad_by_freq=self.scale_grad_by_freq )
        self.embedding_rst_pos.weight.data.normal_(mean=0.0, std=0.005)

        self.embedding_rst_ns = torch.nn.Embedding( len(self.tokenizer.rst_ns_li )+1, self.transformer.config.d_model,
                                    padding_idx=len(self.tokenizer.rst_ns_li ), scale_grad_by_freq=self.scale_grad_by_freq )
        self.embedding_rst_ns.weight.data.normal_(mean=0.0, std=0.005)

        self.embedding_kp_score = torch.nn.Conv1d( 1, self.transformer.config.d_model , kernel_size=1, bias=False)
        self.embedding_kp_score.weight.data.normal_(mean=0.0, std=0.005)
        
        self.embedding_comet_rel = torch.nn.Embedding( len(self.tokenizer.atomic_rel_li ) + 1 , self.transformer.config.d_model,
                                    padding_idx=len(self.tokenizer.atomic_rel_li ) , scale_grad_by_freq=self.scale_grad_by_freq )
        self.embedding_comet_rel.weight.data.normal_(mean=0.0, std=0.005)

        self.embedding_tokentype = torch.nn.Embedding( 2, self.transformer.config.d_model, padding_idx=0 ) #0 is padding, 1 is rst relation
        self.embedding_comet_rel.weight.data.normal_(mean=0.0, std=0.005)

        # setting the pad token to 0 in posiiton_ids embedding and then freezing
        self.transformer.model.encoder.embed_positions.padding_idx = 0
        self.transformer.model.decoder.embed_positions.padding_idx = 0
        
        with torch.no_grad():
            self.transformer.model.encoder.embed_positions.weight[self.transformer.model.encoder.embed_positions.padding_idx].fill_(0)
            self.transformer.model.decoder.embed_positions.weight[self.transformer.model.decoder.embed_positions.padding_idx].fill_(0)

        #TODO: later initialize starting weight for special tokens to weight of pre-existing token like eos
        
        #endregion

        self.loss_fct = CrossEntropyLoss()

        self.transformer.model.encoder.forward = types.MethodType(utils.BART_encoder_forward, 
                                                    self.transformer.model.encoder )

        self.transformer.model.forward = types.MethodType(utils.BART_forward, 
                                                    self.transformer.model )
        
    def forward(self, input_, return_dict=None):

        return_dict = return_dict if return_dict is not None else self.transformer.config.use_return_dict

        
        #region forward pass
        
        input_rst = self.forward_embed_rst(**input_['rst'])
        input_comet = self.forward_embed_comet(**input_['comet'])
    
    
        # extracting labels
        labels_rst = input_rst.pop('labels')
        labels_comet = input_comet.pop('labels')

        output_rst = self.transformer.model.forward(
            **input_rst
        )
        lm_logits_rst = self.transformer.lm_head(output_rst[0]) + self.transformer.final_logits_bias

        output_comet = self.transformer.model.forward(
            **input_comet,
        )
        lm_logits_comet = self.transformer.lm_head(output_comet[0]) + self.transformer.final_logits_bias
        #endregion

        lm_loss = None
        lm_loss_rst= None
        lm_loss_comet =None

        if labels_rst is not None and  labels_comet is not None:
            #the labels are automatically aligned as per the GPT2 code
            #TODO: reevaluate whether bos or eos is the best method to use as start of output
                # right now we use the edu token to start sentences. The EDU token is just the bos token
            
            shift_logits_comet = lm_logits_comet[..., :-1, :].contiguous()
            shift_labels_comet = labels_comet[..., 1:].contiguous() 
            lm_loss_comet = self.loss_fct(shift_logits_comet.view(-1, self.transformer.config.vocab_size), shift_labels_comet.view(-1))

            shift_logits_rst = lm_logits_rst[..., :-1, :].contiguous()
            shift_labels_rst = labels_rst[..., 1:].contiguous() 
            lm_loss_rst = self.loss_fct(shift_logits_rst.view(-1, self.transformer.config.vocab_size), shift_labels_rst.view(-1))
                       
        if not return_dict:
            lm_loss = [lm_loss_rst, lm_loss_comet]
            _rst = (lm_logits_rst,) + output_rst[1:]
            _comet = (lm_logits_comet,) + output_comet[1:]
            return ((lm_loss,) + _rst + _comet ) if lm_loss is not None else (_rst, _comet)
        
        else:
            
            # return Seq2SeqLMOutput(
            #     loss=masked_lm_loss,
            #     logits=lm_logits,
            #     past_key_values=outputs.past_key_values,
            #     decoder_hidden_states=outputs.decoder_hidden_states,
            #     decoder_attentions=outputs.decoder_attentions,
            #     cross_attentions=outputs.cross_attentions,
            #     encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            #     encoder_hidden_states=outputs.encoder_hidden_states,
            #     encoder_attentions=outputs.encoder_attentions,
            #     )
            output = { 
                'lm_loss_rst':lm_loss_rst,
                'lm_loss_comet':lm_loss_comet,
                'lm_logits_rst':lm_logits_rst,
                        'lm_logits_comet':lm_logits_comet,
                        'rst_output':output_rst,
                        'comet_otuput':output_comet}

            return output

    def forward_embed_rst(self,  
                            head_ids,
                            head_treepos_ids,
                            #head_kpscore,
                            rst_rel_ids,
                            rst_treepos_ids,
                            rst_ns_ids,
                            tail_ids,
                            tail_treepos_ids,
                            tail_kp_score,
                            attention_mask_head,
                            attention_mask_rel, 
                            token_type_ids_head,
                            token_type_ids_rel,
                            position_ids_head,
                            labels):

        # region creating inputs_embeds
            # embed head ids
            # embed head treepos ids
            # embed kpscore
            # add positional encoding
        
        inputs_embed_head = self.transformer.model.shared( head_ids )
        inputs_embed_head += self.embedding_rst_pos( head_treepos_ids )
        inputs_embed_head += nn.Embedding.forward(self.transformer.model.encoder.embed_positions, position_ids_head )
        #inputs_embed_head += self.embedding_kp_score( head_kpscore )

        inputs_embed_rel = self.embedding_rst_rels( rst_rel_ids )
        inputs_embed_rel += self.embedding_rst_pos( rst_treepos_ids )
        inputs_embed_rel += self.embedding_rst_ns( rst_ns_ids )

        inputs_embeds = torch.cat( [ inputs_embed_head, inputs_embed_rel ], axis=-2)
        
        token_type_ids_enc = torch.cat( [token_type_ids_head, token_type_ids_rel], axis=1 )        
        inputs_embeds += self.embedding_tokentype( token_type_ids_enc )
        
        inputs_embeds *= self.transformer.model.encoder.embed_scale
        #endregion

        # region reforming attention mask
        _shape = attention_mask_head.shape[1] + attention_mask_rel.shape[1]
        attention_mask = torch.ones( [ attention_mask_head.shape[0], _shape , _shape ], dtype=torch.float, device=attention_mask_head.device )
        attention_mask[:, :attention_mask_head.shape[1], :attention_mask_head.shape[1]] = attention_mask_head
        #end region

            # Here we trim data of all columns which just have padding value 
                # This basically ensures that we only have padding at the ends of each sequence as opposed to in the middle 
            # Since heads_ids and rst_rel_ids are from a different space, we replace rst_rel_ids
            #   with a tensor of same shape that is filled with the value: 1+heads_ids.padding_value
        pdgidx = self.transformer.model.shared.padding_idx
        
        input_ids = torch.cat( [head_ids, torch.full_like( rst_rel_ids, pdgidx+1) ], axis=-1 )

        _, inputs_embeds, attention_mask = self.compress_padding( input_ids ,
                                                    pdgidx,inputs_embeds=inputs_embeds,
                                                     attention_mask=attention_mask  )

        # region creating decoder input embeds
            # embedding of tail_ids
            # adding a conditioning token at start of sequence to indicate 
            # note we don't add token type ids to output since, the tti for head and tail entities is the padding vector of 0s
        decoder_inputs_embeds = self.transformer.model.shared( tail_ids ) + \
                                self.embedding_rst_pos( tail_treepos_ids ) + \
                                    self.embedding_tokentype( torch.full_like( tail_ids  , 
                                        fill_value=self.embedding_tokentype.padding_idx ) )
                            
            #TODO: think of way to incorporate a keyphrase score prediction
        decoder_inputs_embeds = decoder_inputs_embeds * self.transformer.model.decoder.embed_scale
        #endregion

        
        return {
            'attention_mask': attention_mask,
            'inputs_embeds': inputs_embeds,
            'decoder_inputs_embeds': decoder_inputs_embeds,
            'labels': labels 
 
        }
    
    def forward_embed_comet(self, head_ids, head_treepos_ids, position_ids_head,
                            tail_ids,tail_treepos_ids,
                                rels_ids, rels_treepos_ids,
                                attention_mask,
                                token_type_ids_head,
                                token_type_ids_rel,
                                labels ):
        #region inputs_embeds
            # embed head_ids and 
            # emed rels_ids and tail_treepos_ids
            # embed token_type_ids
        inputs_embed_head = self.transformer.model.shared( head_ids )
        inputs_embed_head += self.embedding_rst_pos( head_treepos_ids )
        inputs_embed_head += nn.Embedding.forward(self.transformer.model.encoder.embed_positions, position_ids_head )

        inputs_embed_rel = self.embedding_comet_rel( rels_ids )
        inputs_embed_rel += self.embedding_rst_pos( rels_treepos_ids )
        inputs_embed_rel += self.embedding_rst_ns( torch.full_like( rels_treepos_ids , 
                                fill_value=self.embedding_rst_ns.padding_idx  )  )

        inputs_embeds = torch.cat( [ inputs_embed_head, inputs_embed_rel], axis=-2 )
        token_type_ids_enc = torch.cat( [token_type_ids_head, token_type_ids_rel], axis=-1 )        
        inputs_embeds += self.embedding_tokentype( token_type_ids_enc )

            # Here we trim data of all columns which just have padding value
            # Since heads_ids and rst_rel_ids are from a different space, we replace rst_rel_ids
            #   with a tensor of same shape that is filled with the value: 1+heads_ids.padding_value
        pdgidx = self.transformer.model.shared.padding_idx
        input_ids = torch.cat( [head_ids, torch.full_like( rels_ids, pdgidx+1) ], axis=-1 )
        _, inputs_embeds, attention_mask = self.compress_padding( input_ids ,
                                                    pdgidx,inputs_embeds=inputs_embeds,
                                                     attention_mask=attention_mask  )
        #endregion
        
        #region decoderinputs_embeds
            #token type ids does not have to be added to decoder input embeds
        decoder_inputs_embeds = self.transformer.model.shared( tail_ids ) + \
                                self.embedding_rst_pos( tail_treepos_ids ) + \
                                    self.embedding_tokentype( torch.full_like( tail_ids  , fill_value=self.embedding_tokentype.padding_idx ) )
                            
        decoder_inputs_embeds = decoder_inputs_embeds * self.transformer.model.decoder.embed_scale
        #endregion

        return {
            'attention_mask': attention_mask,
            
            'inputs_embeds': inputs_embeds,
            
            'decoder_inputs_embeds': decoder_inputs_embeds,

            'labels': labels 
        }

    def return_params(self):
        keys = ['base_model_name','max_len_encoder', 'max_len_decoder'
                        'scale_grad_by_freq','model_name']

        json_keys = []
        
        params = {
            k:self.__dict__[k] for k in keys if k in self.__dict__.keys()
        }

        json_params = {
            k:json.dumps(self.__dict__[k]) for k in json_keys if k in self.__dict__.keys()
        }

        json_params = {
            k:json.dumps(self.tokenizer.__dict__[k]) for k in json_keys if k in self.tokenizer.__dict__.keys()
        }

        params_ = {**params, **json_params}
        

        return params_

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(self.chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs
    
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def trim_batch( self,
        input_ids, pad_token_id, other_tensors=None):

        """Remove columns that are populated exclusively by pad_token_id"""
        keep_column_mask = input_ids.ne(pad_token_id).any(dim=1)
        if other_tensors is None:
            return input_ids[:, keep_column_mask]
        else:
            return (input_ids[:, keep_column_mask], *[ tens[:, keep_column_mask] for tens in other_tensors ]  )

    def compress_padding( self,
        input_ids, pad_token_id, inputs_embeds, attention_mask):
        """ First for each datum remove all padding due to the head parts
            Then use pad sequence to ensure they are all the same elnght"""
        
        """Remove columns that are populated exclusively by pad_token_id"""
        keep_column_mask = input_ids.ne(pad_token_id)
        

        inputs_embeds = self.compress_padding_inner(inputs_embeds, 1, keep_column_mask)
        attention_mask = self.compress_padding_inner(attention_mask, 2, keep_column_mask)
        
        return (input_ids, inputs_embeds, attention_mask  )  

    def compress_padding_inner( self, tensor_, compress_dims, keep_column_mask ):
        li_subtens = tensor_.unbind(dim=0)
        
        if compress_dims == 1:
            li_subtens = [ subtens[keep_column_mask[idx2]] for idx2, subtens in enumerate(li_subtens) ]
            batched_padded_subtens = pad_sequence(li_subtens, batch_first=True,padding_value=0.0) #this tensor only hass padingg at the end
        
        elif compress_dims == 2:
            max_len = keep_column_mask.sum(axis=1).max()
            li_subtens = [ subtens[ keep_column_mask[idx2], :][: , keep_column_mask[idx2] ] 
                for idx2, subtens in enumerate(li_subtens) ]
            li_padded_subtens = [ torch.nn.functional.pad( tens, (0, max_len-tens.shape[0] , 0, max_len-tens.shape[1]), value=0.0 ) 
                                for tens in li_subtens]
            batched_padded_subtens = torch.stack(li_padded_subtens)

        return batched_padded_subtens

    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
                
        parser.add_argument('--base_model_name', default='bart_base', required=False)

        parser.add_argument('--model_name', default='COMERST', required=False)
        
        #TODO: this is not implemented yet - the maximum lengths
        parser.add_argument('-el','--max_len_encoder', type= int, default=300 )
        parser.add_argument('-dl','--max_len_decoder', type= int, default=200 )
                      
        parser.add_argument('-sgbf','--scale_grad_by_freq', type=lambda x: bool(int(x)) , default=True, 
                help="Inverse the gradients to the emebdding layers based on the occurence of each index in the minibatch ")
        mparams = parser.parse_known_args( )[0]
       
        return mparams

class COMERST_tokenizer():
    """Rough Implmentation of the tokenizer for the NLG model

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """

    def __init__(self,
                 base_tokenizer_name='bart_base',
                 dir_tokenizer='./models/bart_tokenizer',
                 max_len_encoder=60,
                 max_len_decoder=60,
                 **kwargs ):
        
        #TOdo: ensure max_lens are being used

        self.base_tokenizer_name = base_tokenizer_name
        
        # region Setting up RST relation encoding

        self.rst_rel_li = ['Attribution',
            'Background','Cause','Comparison','Condition',
            'Contrast','Elaboration','Enablement','Evaluation',
            'Explanation','Joint','Manner-Means','Topic-Comment',
            'Summary','Temporal','Topic-Change','n','same-unit','textual-organization'] #Add this to savable config

        self.rst_rel_labeler = sklp.LabelEncoder()
        self.rst_rel_labeler.fit(  self.rst_rel_li )
        #endregion
        
        # region Setting up NS encoding 
        self.rst_ns_li = ['NN','NS','SN','a'] 
        self.rst_ns_labeler = sklp.LabelEncoder()
        self.rst_ns_labeler.fit( self.rst_ns_li  )

        # rst_pos
        #self.rst_pos_maxidx = 2*30 + 2  #original
        self.rst_pos_maxidx =2*( 2*30 + 2 )  #original
        
        #endregion

        # region Setting up CSKG relation encoding
                
        self.atomic_rel_li = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent',
                                'xNeed', 'xReact', 'xWant', 'AtLocation', 'ObjectUse', 'Desires',
                                'HasProperty', 'NotDesires', 'Causes', 'HasSubEvent', 'xReason',
                                'CapableOf', 'MadeUpOf', 'isAfter', 'isBefore', 'isFilledBy',
                                'HinderedBy']

        self.atomic_rel_labeler = sklp.LabelEncoder()
        self.atomic_rel_labeler.fit(  self.atomic_rel_li )
    
        #endregion

        # region Initalising base tokenzier
        self.base_tokenizer = utils.load_base_tokenizer(
            self.base_tokenizer_name, 
            dir_tokenizer,
            base_tokenizer_name,
        )

        self.token_bos = "<s>"
        self.token_eos = "</s>"
        self.token_edu = self.token_bos
        # endregion 

        #region properties and methods to be used when randomly swapping PersonX/PersonY for name/pronoun
        self.pattern_personx = re.compile("person[ ]*x", re.IGNORECASE)
        self.pattern_persony = re.compile("person[ ]*y", re.IGNORECASE)
        
        self.pattern_personx_possv = re.compile("person[ ]*x[]*'[]*s", re.IGNORECASE)
        self.pattern_persony_possv = re.compile("person[ ]*y[]*'[]*s", re.IGNORECASE)

        #TODO: expand to use of all adverbs by detecting what form verb takes and suggesting corret group of pronoun e.g. he/she/it runs. I/you/we/they run
        #self.personal_pronouns = [ "I", "you", "he", "she", "they"]
        
        self.personal_pronouns = [ "he", "she", "they"]

        self.pronmap_persnl_obj = { "I":"me", "we":"us", "you":"you", "she":"her", 
                                    "he":"him", "they":"them" }

        self.pronmap_persnl_possv = {"I":"my", "we":"our", "you":"your",
                                         "she":"her", "he":"his", "they":"their" }
        
        # for pronoun gender correction
        self.pattern_persnl_pronouns = re.compile(f"\b({'|'.join(self.personal_pronouns)})\b" ) 
        self.pattern_obj_pronouns = re.compile(f"\b({'|'.join(list(self.pronmap_persnl_obj.values()))})\b" )
        self.pattern_possv_pronouns = re.compile(f"\b({'|'.join(list(self.pronmap_persnl_possv.values()))})\b" )

        self.pronmap_gender_personal = {'male':"he", "female":"she" }
        self.pronmap_gender_obj = {'male':"him", "female":"her" }
        self.pronmap_gender_possv = {'male':"his", "female":"her" }

        self.func_dict = {  "name": self.name_sampler , "pronoun":self.pronoun_sampler}
        #endregion
        
    def tokenize_rst( self, li_edu_kp, li_rst, li_kp_score, target_edu_kp=None, target_edu_pos=None ):
        
        # region prepping rst tree info
        li_rst_rel = [ rst_node['rel'] for rst_node in li_rst ] 
        li_rst_pos = [ rst_node['pos'] for rst_node in li_rst ]
        li_rst_ns =  [ rst_node['ns'] for rst_node in li_rst ]

            #removing rst tree information for nodes that are too deep
        bool_filter =[pos <= self.rst_pos_maxidx for pos in li_rst_pos]
        li_rst_rel = [ rel for rel, bool_ in zip( li_rst_rel, bool_filter) if bool_==True]
        li_rst_pos = [ pos for pos, bool_ in zip( li_rst_pos, bool_filter) if bool_==True]
        li_rst_ns =  [ ns for ns, bool_ in zip( li_rst_ns, bool_filter) if bool_==True]
        

        # getting a list of all graph posiitons of edus in li_edus
        li_edukp_pos =  [ self.find_child_edus(pos, li_rst_pos ) for pos in li_rst_pos ]
        li_edukp_pos = sum( li_edukp_pos, [] )
        #endregion

        # Need to correct for mismatch between edu key phrase positions and rst tree 
            # edu keyphrases have unlimited node position length, whereas the rst tree was only defined up to node 32
            # so any  key phrase sequesnce with a node position > 64 can not be directly mapped to our rst trees
            # then after aligning key phrased and edu_node positions, we remove all over 64
        if len(li_edukp_pos) < len(li_edu_kp):
            # li_edukp is missing tree positions of size 64 and over. 
                # Based on this we can use a method to ensure the size 
                    # of li_edukp_pos and li_edu_kp is the same
            li_edukp_pos = self.correct_child_edus( li_edukp_pos, len(li_edu_kp) )
            li_edukp_pos.sort()

            np_edukp_pos = np.array(li_edukp_pos)

            bool_filter1 = np.where( np_edukp_pos<self.rst_pos_maxidx )
            li_edukp_pos = np_edukp_pos[bool_filter1].tolist()
            li_edu_kp = np.array(li_edu_kp)[bool_filter1].tolist()
   
        elif len(li_edukp_pos) > len(li_edu_kp):
            raise ValueError

        # region select target edu
            # optionally: randomly selecting a edu keyphrase node to be the target node for prediction
            # the other edu keyphrases form the relation information
        r_int = random.randint( 0, len(li_edukp_pos)-1 )

        if target_edu_kp == None:
                #extracting target data
            target_edu_kp = li_edu_kp.pop(r_int)
            target_edu_pos = li_edukp_pos.pop(r_int)
            target_edu_kpscore = li_kp_score.pop(r_int)

        #endregion

        #region tail
        # Encoded tail information. Adding special tokens
        # line beow list indicesq
        tail_ids = self.base_tokenizer.encode( target_edu_kp , add_prefix_space=True, return_tensors='pt' ).squeeze()
        #tail_treepos_id = torch.tensor( [ target_edu_pos ]*tail_ids.shape[-1] , type=torch.long )
        tail_treepos_ids = tail_ids.new_full( tail_ids.shape , target_edu_pos )
        tail_kp_score = torch.full( tail_ids.shape, target_edu_kpscore , dtype=torch.float32)
        #endregion
        
        # region head
            #    encode list of keyphrases and scores that are input to encoder
            # append edu token to start of each head
        li_head = li_edu_kp #adding prefix space since we use phrases not start of sequences
        li_head_ids = [ self.base_tokenizer.encode( head, add_prefix_space=True ) for head in li_head ]
        li_head_treepos_ids = [ [pos]*len(ids) for pos, ids in zip( li_edukp_pos, li_head_ids ) ] #creating a li of li of graph pos indexes for each keyphrase
        #li_head_kpscore =  [ [kpscore]*len(ids) for kpscore, ids in zip( li_kp_score, li_head_ids ) ]

            #flattening and converting to tensor
        head_ids = torch.tensor( sum(li_head_ids,[]), dtype=torch.long)
        head_treepos_ids = torch.tensor( sum(li_head_treepos_ids, []), dtype=torch.long)
        #head_kpscores = torch.tensor( sum( li_headkpscores, []), dtype=torch.long)

        #endregion 
       
        # region relation : encoded list of rst parent nodes information
        rst_rel_ids = torch.tensor(  [self.rst_rel_labeler.transform([rel]) for rel in li_rst_rel ], dtype=torch.long).squeeze()
        rst_treepos_ids = torch.tensor( li_rst_pos, dtype=torch.long )
        rst_ns_ids =  torch.tensor( [ self.rst_ns_labeler.transform([ns]) for ns in li_rst_ns ], dtype=torch.long).squeeze()
        #endregion

        #region attention mask
            #For rst we require 1) bidirectional attention over each keyphrase chunk
                                    # each keyphrase chunk can not attend directly to other keyphrase chunks
            #                   2) all encoder inputs can attend to rst relation tree info

        enc_input_dim = head_ids.shape[-1] + rst_rel_ids.shape[-1]

        #attention_mask = torch.zeros( [enc_input_dim, enc_input_dim], dtype=torch.long  )
        attention_mask_head = torch.zeros( [head_ids.shape[-1], head_ids.shape[-1]], dtype=torch.long  )

            # here we implement the diagonal attention for each sub keyphrase
            # NOTE: here each edu keyphrase ony refers to itself in a bidirectional manner, It does not refer to other keyphrases
            # it should also refer to the relations
        curr_bos_token_pos=0
        for hids in li_head_ids:

            len_ids = len(hids)
            _ =  attention_mask_head.new_ones( [len_ids, len_ids] )

            attention_mask_head[ curr_bos_token_pos:curr_bos_token_pos+len_ids, 
                curr_bos_token_pos : curr_bos_token_pos+len_ids   ] = _

            curr_bos_token_pos += len_ids
        
        # # here we ensure that all tokens can attend to the rel section
        # attention_mask[ :, head_ids.shape[-1]: ] = 1

        # # Ensuring rel token attends to all subphrase edus
        # attention_mask[ head_ids.shape[-1]: , : ] = 1

        # Create the 1dimensional mask representing tail attn. We appnd this later
        attention_mask_rel = torch.ones( [rst_rel_ids.shape[-1]] , dtype=torch.long )
        #endregion

        #region position_ids
            #RoBERTa never uses 0 and 1 positional ids, in ROBERTa, all pad tokens have position id of 1
                # , and the rest of the tokens have position ids in 
                # the range (2, seq_length - num_pad_tokens). It's implemented like this to
                #  match the original implementation in fairseq.
        
            #for position ids, it restarts from 1 for every new key phrase edu
            # the 2 offset is explained here https://github.com/huggingface/transformers/issues/10736
            # No positional ids for relation section

        position_ids_head = head_ids.new_full( (0,), 0.0, dtype=torch.long )

        for head in li_head_ids:
            new_ids = torch.arange( len( head ), dtype=torch.long ) + 2
            position_ids_head = torch.cat( [ position_ids_head, new_ids ]  )

        #endregion
        
        # region token type ids
        token_type_ids_head = torch.zeros( [head_ids.shape[-1]], dtype=torch.long  )
        token_type_ids_rel = torch.ones( [rst_rel_ids.shape[-1]], dtype=torch.long )
                
        #endregion 

        # region labels
        labels = tail_ids
        # endregion

        return {
            #head
            'head_ids': head_ids ,
            'head_treepos_ids': head_treepos_ids,
            #'head_kpscore': head_kpscore,

            #relation: tree information
            'rst_rel_ids': rst_rel_ids ,
            'rst_treepos_ids': rst_treepos_ids,
            'rst_ns_ids': rst_ns_ids,

            #tail
            'tail_ids': tail_ids.squeeze(),
            'tail_treepos_ids': tail_treepos_ids ,
            'tail_kp_score':tail_kp_score,

            'attention_mask_head': attention_mask_head,
            'attention_mask_rel': attention_mask_rel,
            'token_type_ids_head': token_type_ids_head,
            'token_type_ids_rel': token_type_ids_rel,

            'position_ids_head': position_ids_head,

            'labels':labels.squeeze()
        }

    def tokenize_comet( self, head, rel, tail ):
        
        # region randomising pronouns (randomly replacing occurences of PersonX or PersonY with names)
        head, rel, tail = self.tokenize_comet_person_randomizer(head, rel, tail)
        # endregion

        # region head tail rel
        head_ids = self.base_tokenizer.encode( head, add_prefix_space=True, return_tensors='pt')

        rels_ids = torch.tensor( self.atomic_rel_labeler.transform( [rel] ), dtype=torch.long )

        #tail_ids = self.base_tokenizer.encode( self.token_edu + " " + tail + " " + self.token_eos , add_prefix_space=False, return_tensor='pt')
        tail_ids = self.base_tokenizer.encode( tail, add_prefix_space=True, return_tensors='pt' )
        #endregion 

        #region treepos_ids
            # we imagine the cskg to have the same structure as rst tree
            # so relation is at parent node. the head and tail entity are at sibling nodes
        rels_treepos_ids = torch.randint(int( ( self.rst_pos_maxidx-2) /2 ), (1,), dtype=torch.long ) 
        head_treepos_ids = (rels_treepos_ids*2 + 1).expand( [head_ids.shape[-1]] )
        tail_treepos_ids = (rels_treepos_ids*2 + 2).expand( [tail_ids.shape[-1]] )
        #endregion

        #region attention mask
            # similar to the rst attention masking
            # the head attends to itself in bidirectional manner
            # the head attends to the relation
            
            #TODO: evaluate version where the relation attends to everything or where the relation only attends to itself
                #NOTE: currently using version where reltion attends to everything

        enc_input_dim = head_ids.shape[-1] + rels_ids.shape[-1]

        attention_mask = torch.ones( [enc_input_dim, enc_input_dim], dtype=torch.long  )

            # relation attention
        #_ =  attention_mask.new_zeros( [ rels_ids.shape[-1], head_ids.shape[-1] ] )
        #attention_mask[ -rels_ids.shape[-1]:,  : head_ids.shape[-1]]  = _
        #endregion

        #region token type ids
        token_type_ids_head = torch.zeros( [head_ids.shape[-1]], dtype=torch.long  )
        token_type_ids_rel = torch.ones( [rels_ids.shape[-1]], dtype=torch.long )
        #endregion

        #region position_ids
        position_ids_head = new_ids = torch.arange( head_ids.shape[-1] , dtype=torch.long ) + 2

        #region labels
        labels = tail_ids
        #endregion

        return {
            'head_ids':head_ids.squeeze(),
            'head_treepos_ids':head_treepos_ids,
            'position_ids_head': position_ids_head,

            'tail_ids':tail_ids.squeeze(),
            'tail_treepos_ids':tail_treepos_ids,
            
            'rels_ids':rels_ids,
            'rels_treepos_ids': rels_treepos_ids,

            'attention_mask': attention_mask,
            'token_type_ids_head': token_type_ids_head,
            'token_type_ids_rel': token_type_ids_rel,
            'labels':labels.squeeze()
        }

    def tokenize_comet_person_randomizer(self, head, rel, tail):
        
        #TODO: add in the condition that if her or him exist anywhere in the head or tail, we either leave as is or make that the pronoun depending on whether o or x is in the relation

        #region retreiving counts of each person
            # check any regex occurence of personX in either head or tail
            # "" personY
        personx_match_count_head = len( re.findall( self.pattern_personx , head ) )
        personx_match_count_tail = len( re.findall( self.pattern_personx , tail) )
        personx_match_count = personx_match_count_head + personx_match_count_tail 

        persony_match_count_head = len( re.findall( self.pattern_persony , head ) )
        persony_match_count_tail = len( re.findall( self.pattern_persony , tail) )
        persony_match_count = persony_match_count_head + persony_match_count_tail 

        if personx_match_count + persony_match_count == 0:
            return head, rel, tail
        #endregion
        
        # region choosing whether to use pronoun or name replacement
            #  sampling the replacement names/pronouns
                # ensure your model does not use the same pronoun for PersonX and Person Y
        
            # randomly sample a different name or pronoun for each personX/personY that occurs
                # we sample a personal pronoun. Each personal pronoun has a mapping to a different objective and possessive pronoun
                #                 

        # Check if person x/y occurs in 
        if personx_match_count > 0:
            # we randomly choose whether to sample using pronouns or an actual name
            x_sampling_method = random.choice(  list( self.func_dict ) )
            x_sampling_func = self.func_dict[x_sampling_method]
            x_sub = x_sampling_func()
        else:
            x_sub = ""
            
        if persony_match_count >0:
            y_sampling_method = random.choice(  list( self.func_dict ) )
            y_sampling_func = self.func_dict[y_sampling_method]
            y_sub = y_sampling_func(exclude_vals=[x_sub])

        #endregion

        # replace all occurences of personX and personY in head and tail
            # if pronoun used rmbr to handle for possessive 's. Essentially will need replace whole word including 's not just personX/Y

        # PersonX
        if personx_match_count > 0:

            # handling case where name was sampled in 
            if x_sampling_method == "name":
                
                x_sub, gender = x_sub
                if personx_match_count_head > 0:
                    
                    head = re.sub(self.pattern_personx, x_sub, head)

                    # fixing any gendered pronouns that exist in dataset
                    # get gender of x_sub
                    # use that to decide pronoun switch gender switch in tail
                    # we do switch if
                    #   the relation is an action personX is doing and there is no personY 
                    #   the relation is an action personY is doing and there is a personY
                if (persony_match_count==0) and (rel[0] == "o" and persony_match_count>0):
                    tail = re.sub( self.pattern_persnl_pronouns , self.pronmap_gender_personal[gender] , tail)
                    tail = re.sub( self.pattern_obj_pronouns, self.pronmap_gender_obj[gender] , tail)
                    tail = re.sub( self.pattern_possv_pronouns, self.pronmap_gender_possv[gender], tail)
                # elif (rel[0] == "o" and persony_match_count>0):
                #     tail = re.sub( self.pattern_obj_pronouns, self.pronmap_gender_obj[gender] , tail)

                if personx_match_count_tail > 0:
                    try:
                        tail =  re.sub(self.pattern_personx, x_sub, tail)
                    except Exception as e:
                        raise e

            # handling case where pronoun was sampled in 
            elif x_sampling_method == "pronoun":
                
                # substitutes for head
                if personx_match_count_head > 0:
                    # Do possessive substitute first
                    head = re.sub(self.pattern_personx_possv, self.pronmap_persnl_possv[x_sub], head )
                    
                    if  len( re.findall( self.pattern_personx , head) ) >0 :
                        
                        # first correct spelling for the person word
                        head = re.sub(self.pattern_personx, "PersonX", head )

                        # ascertain whether it is subject or object or possessive
                        
                        try:
                            dependency = next( token.dep_ for token in nlp(head) if token.text=="PersonX")
                        except StopIteration:
                            dependency = None
                            pass
                        # then do sub
                        if  dependency in ["nsubj","nsubjpass","csubj","agent","expl"]:
                            head = re.sub(self.pattern_personx, x_sub, head )
                        
                        elif dependency in ["dative","dobj","attr","oprd"]:
                            head = re.sub(self.pattern_personx, self.pronmap_persnl_obj[x_sub], head)
                        
                        # tail pronoun correction
                        # we do switch if
                        #   there is no person Y involved.
                        #   the relation is an action personY is doing and there is a personY
                        
                # substitutes for tail
                if personx_match_count_tail > 0:
                    # Do possessive substitute first
                    tail = re.sub(self.pattern_personx_possv, self.pronmap_persnl_possv[x_sub], tail )

                    #First check if the word person exists anymore
                    if  len( re.findall( self.pattern_personx , tail) ) >0 :

                        # # Person X is the subject in tail, if relation belong to xWant, xNeed etc
                        # if rel[0] == "x":
                        #     tail = re.sub(self.pattern_personx, x_sub, tail )
                        
                        # # Person X is the object in tail, if relation belong to oWant, oNeed etc
                        # elif rel[0]== "o":
                        #     tail = re.sub(self.pattern_personx, self.pronmap_persnl_obj[x_sub], tail )
                        
                        # # if relation not in o or x, use nltk to ascertain whether it is subject or object
                        # else:
                            # first correct spelling for the person word
                        tail = re.sub(self.pattern_personx, "PersonX", tail )

                        # ascertain whether it is subject or object or possessive
                        try:
                            dependency = next( token.dep_ for token in nlp(tail) if token.text=="PersonX")
                        except StopIteration:
                            dependency = None
                            pass

                        # then do sub
                        if  dependency in ["nsubj","nsubjpass","csubj","agent","expl"]:
                            tail = re.sub(self.pattern_personx, x_sub, tail )
                        
                        elif dependency in ["dative","dobj","attr","oprd"]:
                            tail = re.sub(self.pattern_personx, self.pronmap_persnl_obj[x_sub], tail)

                # Handling pronounds in tail when relation starts with o
                if (rel[0] == "o" and persony_match_count>0):
                    tail = re.sub( self.pattern_obj_pronouns, self.pronmap_persnl_obj[x_sub] , tail)

                # Handling pronouns in tail when no person Y involved
                if (persony_match_count==0 and persony_match_count==0):
                    tail = re.sub( self.pattern_persnl_pronouns , x_sub , tail)
                    tail = re.sub( self.pattern_obj_pronouns, self.pronmap_persnl_obj[x_sub] , tail)
                    tail = re.sub( self.pattern_possv_pronouns, self.pronmap_persnl_possv[x_sub], tail)

        # PersonY
        if persony_match_count >0:
            
            # handling case where name was sampled in 
            if y_sampling_method == "name":
                y_sub, gender = y_sub
                
                if persony_match_count > 0:
                    head = re.sub(self.pattern_persony, y_sub, head)
                if persony_match_count_tail > 0:
                    tail =  re.sub(self.pattern_persony, y_sub, tail)
                        #person y PRONOUNS IN TAIL
            
                #1)if personX is the actor and person Y is present then change pronouns in tail
                if (rel[0]=="x"):
                    tail = re.sub( self.pattern_obj_pronouns, self.pronmap_gender_obj[gender] , tail)

            # handling case where pronoun was sampled in 
            elif y_sampling_method == "pronoun":
                
                # substitutes for head occurences of person Y
                if persony_match_count_head > 0:
                    
                    if  len( re.findall( self.pattern_persony , head) ) >0 :
                    # Do possessive substitute first
                        head = re.sub(self.pattern_persony_possv, self.pronmap_persnl_possv[y_sub], head )
                    
                        #Do normal replace
                        #head = re.sub(self.pattern_persony, y_sub, head )

                    # Replace using part of speec detection
                        # first correct spelling for the person word
                        head = re.sub(self.pattern_persony, "PersonY", head )

                        # ascertain whether it is subject or object or possessive
                        try:
                            dependency = next( token.dep_ for token in nlp(head) if token.text=="PersonY")
                        except StopIteration:
                            dependency = None

                        # then do sub
                        if  dependency in ["nsubj","nsubjpass","csubj","agent","expl"]:
                            head = re.sub(self.pattern_persony, y_sub, head )
                        
                        elif dependency in ["dative","dobj","attr","oprd"]:
                            head = re.sub(self.pattern_persony, self.pronmap_persnl_obj[y_sub], head)

                        
                # substitutes for tail occurences of personY
                if persony_match_count_tail > 0 or persony_match_count>0:

                    # Do possessive peronY's substitute first and then normal person Y
                    tail = re.sub(self.pattern_persony_possv, self.pronmap_persnl_possv[y_sub], tail )
                    
                    #tail = re.sub(self.pattern_persony, y_sub, tail )

                    # Replace using part of speec detection
                    # first correct spelling for the person word
                    tail = re.sub(self.pattern_persony, "PersonY", tail )

                    # ascertain whether it is subject or object or possessive
                    try:
                        dependency = next( token.dep_ for token in nlp(tail) if token.text=="PersonY")
                    except StopIteration:
                        dependency = None

                    # then do sub
                    if  dependency in ["nsubj","nsubjpass","csubj","agent","expl"]:
                        tail = re.sub(self.pattern_persony, y_sub, tail )
                    
                    elif dependency in ["dative","dobj","attr","oprd"]:
                        tail = re.sub(self.pattern_persony, self.pronmap_persnl_obj[y_sub], tail)

           
            
                #person y PRONOUNS IN TAIL
                #1)if personX is the actor and person Y is present then change pronouns in tail
                if (rel[0]=="x"):
                    tail = re.sub( self.pattern_obj_pronouns, self.pronmap_persnl_obj[y_sub] , tail)
                    # tail = re.sub( self.pattern_persnl_pronouns, self.pronmap_gender_personal[gender] , tail)
                    # tail = re.sub( self.pattern_possv_pronouns, self.pronmap_gender_possv[gender], tail)


        return head, rel, tail

    def batch_tokenize_rst( self, li_li_head, li_li_rels, li_tail ):
        raise NotImplementedError
        #  should be able to perform batch encoding

        # take in a list of lists of head text / key phrase
        # take in a list of lists relations
        # take in a list of lists of nuclearity

    def batch_tokenize_comet( self, li_li_head, li_li_rels, li_tail_ent ):
        raise NotImplementedError
        #  should be able to perform batch encoding

        # take in a list of lists of head text / key phrase
        # take in a list of lists relations
        # take in a list of lists of nuclearity

    def find_child_edus(self, pos_parentnode, li_rst_pos):
        #returns the pos of any child elements of a parent node(rst) that are edus
               
        li_child_pos = [2*pos_parentnode+1, 2*pos_parentnode+2 ]

        li_child_edu_pos = [ pos for pos in li_child_pos if pos not in li_rst_pos]

        return li_child_edu_pos 
    
    def correct_child_edus(self, li_edukp_pos, goal_length ):
        #TODO: improve this methodology

        # here we essentially expand li_edukp_pos in a tree like manner until we reach a 
        # li_edukp with goal_length parent nodes

        
        # We chek if the list is of keyphrase pos is as long as goal length
            # if not we pop the final element from li_edukp_pos and then bring in the two child nodes
            # then we iteratively loop through 
        if len(li_edukp_pos) < goal_length:

            # position of indexes could have been incorrectly labelled as leaves
            _bool_idxs = np.where( np.array(li_edukp_pos) < int( (self.rst_pos_maxidx-2)/2 ),  )[0]    
            
            # if no idxs choosen then simply use all positions
            if len(_bool_idxs) != 0:
                idxs = np.arange( len(li_edukp_pos) )[ _bool_idxs ]
            else:
                idxs = np.arange( len(li_edukp_pos) )
            
            pos = li_edukp_pos.pop(idxs[-1])

            # Here we sample from positions, with more weight placed on larger positions
            # TODO; improve to  more weight placed on the righthand most positions
            #probs = list( np.arange(len(idxs), step=3 ) )
        
            new_child_nodes = self.find_child_edus( pos, li_edukp_pos )
            li_edukp_pos = new_child_nodes + li_edukp_pos
            
            #li_edukp_pos = 
            li_edukp_pos = self.correct_child_edus(li_edukp_pos, goal_length )
        
        return li_edukp_pos

    def name_sampler(self, exclude_vals=[] ):
        gender = random.choice(['male','female'])
        while True:
            name = names.get_first_name(gender=gender)
            if name not in exclude_vals:
                break

        return name, gender

    def pronoun_sampler(self, exclude_vals=[] ):
        # Sample from the list of personal pronouns which are also subject pronouns
        
        pronoun = random.choice( [ pron for pron in self.personal_pronouns if pron not in exclude_vals ] )
        return pronoun

# endregion

class TrainingModule(pl.LightningModule):

    def __init__(self, model_params, batch_size=20, 
                    dir_data_rst=None, 
                    dir_data_atomic2020=None,
                    accumulate_grad_batches=1,
                    max_epochs=25,
                    gpus=1, 
                    learning_rate=1e-3,
                    warmup_proportion=0.1,
                    workers=0,
                    lr_schedule='hard_restarts',
                    mode = 'train_new',
                    data_splits = {'train':0.6,'val':0.2,'test':0.2},
                    #dset_blend={'ATOMIC2020':0.5 , 'RST':0.5 },
                    tag='',
                    *args,
                    **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.gpus =  gpus
        self.model = COMERST( **model_params )
        self.mode = mode
        self.workers = workers
        self.data_splits = data_splits
        #self.dset_blend = dset_blend
        
        if self.mode in ['train_new','train_cont','test']:
            self.dir_data_rst = utils.get_path(dir_data_rst)
            self.dir_data_atomic2020 = dir_data_atomic2020
            self.create_data_loaders( )
            self.accumulate_grad_batches = accumulate_grad_batches
            self.tag = tag

        if self.mode in ['train_new','train_cont']:
            self.max_epochs = int( max_epochs )
            self.warmup_proportion = warmup_proportion
            self.lr_schedule = lr_schedule
            self.learning_rate = float( learning_rate )
        
            train_params_to_save = self.return_params()
            model_params_to_save = self.model.return_params()

            self.hparams.update({ **train_params_to_save, **model_params_to_save})

            #self.inference_samples = list( islice( self.inference_dl, 10 ) )
            #del self.inference_dl

        if self.mode in ['inference']:
            self.eval() 
            self.freeze() 

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data_rst', default="./dataset_keyphrase", help="Relative directory path of rst data")
        parser.add_argument('--dir_data_atomic2020', default="./dataset_atomic2020", help="Relative directory path for atomic2020data")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('-me','--max_epochs', default=28, type=int)
        parser.add_argument('-agb','--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('-bs','--batch_size', default=100, type=int)
        parser.add_argument('-lr','--learning_rate', default=5e-4, type=float)
        parser.add_argument('--warmup_proportion', default=0.15)
        parser.add_argument('--workers', default=12, type=int) 
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','train_cont','test','inference'])
        parser.add_argument('--splits', default={'train':0.6,'val':0.2,'test':0.2}, required=False, type=str )
        #parser.add_argument('--dset_blend', default={'ATOMIC2020':0.5, 'RST':0.5}, type=lambda _str: ast.literal_eval(_str), required=False)
        parser.add_argument('--version', default=0,required=False, type=int, help="The Experimental Versioning for this run" )
        parser.add_argument('--precision', default=16,required=False, type=int, help="Precision to use", choices=[16,32] )
        parser.add_argument('--tag', default='default model', required=False, type=str)
        parser.add_argument('--override',default=False, type = lambda x: bool(int(x)), choices=["0","1"] )
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
                    'learning_rate','precision','splits','tag']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'base_tokenizer_name','model_name','max_len_encoder','max_len_decoder'
                    ,'scale_grad_by_freq' ]} )
                
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
                    'learning_rate','precision','splits']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'base_tokenizer_name','loss_type','model_name','max_len_encoder','max_len_decoder']} )
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
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=2, 
            mode='min', dirpath=dir_checkpoints, 
            filename='{epoch:03d}_{val_loss:.5f}')
        
        checkpoint_callback._save_model  = types.MethodType(utils.monkey_save_model, checkpoint_callback) #monkey patch
        #checkpoint_callback._monitor_candidates = types.MethodType(utils._monitor_candidates, checkpoint_callback) # monkey patch

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            #monitor='val_loss_comet',
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode='min'
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(early_stop_callback)

        if tparams['gpus'] in [0,1]:
            accelerator=None
        else:
            accelerator = 'ddp'
        
        if tparams['mode'] in ["train_new"]:
            
            trainer = pl.Trainer.from_argparse_args(argparse.Namespace( **tparams),
                        progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                        default_root_dir=tparams['dir_checkpoints'],
                        #check_val_every_n_epoch=1,
                        logger=tb_logger,
                        #log_every_n_steps=20,
                        precision=tparams['precision'], callbacks=callbacks,
                        #accelerator='ddp2', amp_level='O2',# use_amp=True,
                        accelerator=accelerator,
                        #limit_train_batches =10,
                        #limit_val_batches = 10,
                        #val_check_interval=0.25,
                        val_check_interval=0.3,
                        num_sanity_val_steps=0, 
                        #track_grad_norm = True,
                        #overfit_batches=25,
                        #fast_dev_run=2, 
                        #log_gpu_memory=True,
                        reload_dataloaders_every_epoch=False,
                        multiple_trainloader_mode='max_size_cycle'
                        )

        elif tparams['mode'] in ["train_cont","inference"]:
            #restoring checkpoint             
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace( **tparams),
                    progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                    #check_val_every_n_epoch=1,
                    logger=tb_logger,
                    #log_every_n_steps=20,   
                    precision=tparams['precision'],
                    callbacks=callbacks,
                    #accelerator='ddp2',  amp_level='O2', # use_amp=True,
                    accelerator=accelerator,
                    #limit_train_batches = 0.4,
                    #val_check_interval=0.5,
                    #limit_val_batches = ,
                    val_check_interval=0.3,
                    num_sanity_val_steps=0,
                    #track_grad_norm = True,
                    #overfit_batches=5
                    #,fast_dev_run=2, 
                    #log_gpu_memory=True,
                    reload_dataloaders_every_epoch=True,
                    multiple_trainloader_mode='max_size_cycle'
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
           
        return trainer,training_module
    
    @staticmethod
    def get_ckpt_file(_dir_checkpoint,mode='best'):
        if mode=='best':
            checkpoint_yaml_file = os.path.join( _dir_checkpoint,"best_k_models.yaml" )
            scores_dict = yaml.load( open(checkpoint_yaml_file,"r") ) #key= ckptpath, value = val_loss
            best_ckpt_path = min(scores_dict, key=scores_dict.get)

            if torch.cuda.is_available():
                #checkpoint = torch.load(best_ckpt_path, map_location=lambda storage, loc: storage) )
                #checkpoint = torch.load(best_ckpt_path, map_location=lambda storage, loc: storage.cuda())  
                checkpoint = torch.load(best_ckpt_path, map_location='cpu' )  

            else:
                checkpoint = torch.load(best_ckpt_path, map_location='cpu')            
        else:
            raise NotImplementedError
        
        return checkpoint
        
    @staticmethod
    def start(trainer, tparams, training_module, mparams ):
        
        if tparams['mode'] in ['train_new','train_cont']:    
            trainer.fit( training_module )
        
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
    def load_comerst(model_name="COMERST", model_version=0, max_input_len=None):
        # Loading in NLG model
        checkpoint = TrainingModule.get_ckpt_file(f'./models/{model_name}/version_{model_version}/checkpoints')

        # Getting tparams
        tparams = {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
            'batch_size', 'learning_rate','precision','splits',
            'tag']}

        tparams['mode'] = 'inference'

        mparams =  {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
            'base_tokenizer_name','loss_type','model_name','max_len_encoder','max_len_decoder',
            'frst_version','scale_grad_by_freq']}
        
        mparams_json = {k:json.loads(v) for k,v in checkpoint['hyper_parameters'].items() if k in [] }

        mparams =  {**mparams, **mparams_json}
        
        if max_input_len != None:
            mparams['max_input_len'] = max_input_len
            
        # Loading Training Module
        training_module = TrainingModule(**tparams, model_params=mparams )
        training_module.load_state_dict(checkpoint['state_dict'])
        model = training_module.model

        # Deleting checkpoints to free up GPU space
        del checkpoint
        torch.cuda.empty_cache()
          
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model

    @staticmethod
    def model_name(mparams):
        return mparams['model_name']

    @auto_move_data
    def forward(self, input_):
        return self.model(input_)

    def step(self, batch, step_name):
        
        input_ = batch

        model_output = self.forward(input_) #(lm_logits and loss)
        lm_loss_rst = model_output['lm_loss_rst']
        lm_loss_comet = model_output['lm_loss_comet']
        
        #TODO: add scheme to change the weight of each loss as the model predicts more
        #v1
        #loss = lm_loss_rst/2 + lm_loss_comet/2
        #v2
        loss = lm_loss_rst*3/4 + lm_loss_comet/4
        loss_key = f"{step_name}_loss"
        
        output = {}
        if step_name == 'train':
            output["loss"] = loss

        else:       
            self.log( loss_key, loss)
            output[loss_key]=loss
        
        self.log( loss_key+"_rst", lm_loss_rst )
        self.log( loss_key+"_comet", lm_loss_comet )
            
        #here output dictionary instead
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
            if len(outputs) > 0:
                loss = torch.stack([x[f"{step_name}_loss"] for x in outputs]).mean()
                self.log(f"{step_name}_loss", loss, logger=True, prog_bar=True)
                            
    def create_data_loaders(self, shuffle=False, **kwargs ):
        
        pad_values = {'head_ids':self.model.transformer.model.shared.padding_idx , 
                        'head_treepos_ids':self.model.embedding_rst_pos.padding_idx, 
                        
                        'rst_rel_ids': self.model.embedding_rst_rels.padding_idx, 
                        'rst_treepos_ids': self.model.embedding_rst_pos.padding_idx,
                        'rst_ns_ids': self.model.embedding_rst_ns.padding_idx, 

                        'tail_ids': self.model.transformer.model.shared.padding_idx , 
                        'tail_treepos_ids':self.model.embedding_rst_pos.padding_idx ,
                        'tail_kp_score': 0,

                        'attention_mask': 0, 
                        'attention_mask_head': 0, 
                        'attention_mask_rel': 0, 
                        #'token_type_ids':self.model.embedding_tokentype.padding_idx , 
                        'token_type_ids_head':self.model.embedding_tokentype.padding_idx ,
                        'token_type_ids_rel':self.model.embedding_tokentype.padding_idx ,
                        'labels': self.model.loss_fct.ignore_index,

                        'position_ids_head':self.model.transformer.model.encoder.embed_positions.padding_idx if 
                                        self.model.transformer.model.encoder.embed_positions.padding_idx else 0  
                        }

        dg = DataLoaderGenerator(self.dir_data_rst, self.dir_data_atomic2020,
                self.batch_size, pad_values,
                self.model.tokenizer, 
                workers=self.workers, mode=self.mode, split=self.data_splits
                #dset_blend=self.dset_blend
                )

        
        if "train" in self.mode:
            self.train_dl = dg.prepare_dataloader_combined(shuffle=True, split_name='train')
            self.val_dl = dg.prepare_dataloader_combined(shuffle=True, split_name='val')
            self.test_dl = dg.prepare_dataloader_combined(shuffle=True, split_name='test')
            #self.inference_dl = dg.prepare_dataloader_combined(shuffle=True, split_name='inference')
        
        elif self.mode == "test":
            self.test_dl = dg.prepare_dataloader_combined(shuffle=True, split_name='test')

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    @lru_cache()
    def total_steps(self):

        ds_size = max( [ len(dl) for dl in self.train_dl.values()] ) // self.gpus
        steps = (ds_size * self.max_epochs) // (self.accumulate_grad_batches)
        return steps

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        warmup_steps = int( self.warmup_proportion*self.total_steps() )

        lr_schedule = get_cosine_schedule_with_warmup(optimizer, 
                        warmup_steps, self.total_steps(), 0.5 )

        return [optimizer], [{ "scheduler":lr_schedule ,"interval": "step", "monitor":"val_loss"}]
    
    def return_params(self):
        params = {}
        keys = ['batch_size','accumulate_grad_batches','learning_rate','max_epochs',
            'dir_data_rst','dir_data_atomic2020',
            'warmup_proportion','tag','version']
        
        params = {
            k:self.__dict__[k] for k in keys if k in self.__dict__.keys()
        }

        #json_keys = ['data_splits','dset_blend']

        json_keys = ['data_splits']
        
        jparams = {
            k:ujson.dumps(self.__dict__[k]) for k in keys if k in self.__dict__.keys()
        }

        params = {**params, **jparams}

        return params

class DataLoaderGenerator():
    """Handles the creation of dataloaders for a train, val and test set
    """
    def __init__(self, dir_data_rst, dir_data_atomic2020 ,batch_size,
                    pad_values ,
                    tokenizer, workers=0, mode='train_new',
                    splits={'train':0.6,'val':0.2,'test':0.2},
                    #dset_blend={'ATOMIC2020':0.5 , 'RST':0.5 },
                    **kwargs):
        
        self.dir_data_rst = dir_data_rst
        self.dir_data_atomic2020 = dir_data_atomic2020
        self.tokenizer = tokenizer
        self.splits = splits
        #self.dset_blend = dset_blend

        self.bs = batch_size
        self.workers_rst = int( workers/2) #if workers==0 else max( int( round( workers * (3/4), 0 ) ), 1 )
        self.workers_atomic = int( workers/2) #if workers==0 else max( workers - self.workers_rst, 1 )
        self.mode = mode
        self.pad_values = pad_values
        
    def prepare_dataloader_combined(self, shuffle=False, 
        split_name='train'):

        dataloder_rst = self.prepare_dloader_rst(shuffle, split_name)
        dataloader_atomic2020 = self.prepare_dloader_atomic2020(shuffle, split_name)

        output = {'comet':dataloader_atomic2020, 'rst':dataloder_rst }

        if split_name in ["val","test"] :
            output = CombinedLoader( output, "max_size_cycle")
        
        return output

    def prepare_dloader_rst(self, shuffle=False, 
        split_name='train'):

        """Prepares a dataloader given a directory of data for NLG language module
            # The current method takes a percentage of data from each subdirectory
            Args:
                dir_dset ([type]): [description]
        """
        #getting all files from all different subreddits/types of conversation
        fns = glob.glob(  os.path.join( utils.get_path(self.dir_data_rst),"*","*") )
        fns = [fn for fn in fns if os.path.split(fn)[-1]!="lock"]
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

        elif split_name == 'inference':
            line_starts = [ random.randrange( int(fs*(1-self.splits['test'])), fs) for fs in files_sizes  ]
            line_ends =  files_sizes
            shuffle = False

        li_dsets = [ SingleDataset_rst(_f, self.tokenizer, line_start, line_end) 
                        for _f, line_start, line_end in zip(fns, line_starts, line_ends) ]
        li_dsets = [dset for dset in li_dsets if dset.valid==True]

        if split_name == 'inference':
            random.sample(li_dsets,10)
            bs = 1
        else:
            bs = self.bs #* self.dset_blend['rst']

        concat_dset = torch.utils.data.ConcatDataset(li_dsets)
        
        dataloader = torch.utils.data.DataLoader(concat_dset, batch_size=bs,
            shuffle=shuffle, num_workers=self.workers_rst, 
            collate_fn=lambda batch: utils.default_collate_pad(batch, self.pad_values) )

        #TODO: change defualt collate to allow padding of elements for batching
        return dataloader

    def prepare_dloader_atomic2020(self, shuffle=False, 
        split_name='train'):

        """Prepares a dataloader given a directory of data for NLG language module
            # The current method takes a percentage of data from each subdirectory
            Args:
                dir_dset ([type]): [description]
        """
        #TODO: Clean the dataset of bad entries

        #getting all files from all different subreddits/types of conversation
       
        #defining starting line and total lines to use for dataset
        if split_name == 'train':
            shuffle = True
            fn = os.path.join( self.dir_data_atomic2020,"train_v2.csv" )
        
        elif split_name == 'val':
            fn = os.path.join( self.dir_data_atomic2020,"dev_v2.csv" )
            shuffle = False
       
        elif split_name == 'test':
            fn = os.path.join( self.dir_data_atomic2020,"test_v2.csv" )
            shuffle = False

        elif split_name == 'inference':
            fn = os.path.join( self.dir_data_atomic2020,"test_v2.csv" )
            shuffle = False

        dset = SingleDataset_atomic2020(fn, self.tokenizer )
                
        if split_name == 'inference':
            bs = 1
        else:
            bs = self.bs #*self.dset_blend['ATOMIC2020']

        dataloader = torch.utils.data.DataLoader(dset, batch_size=bs,
            shuffle=shuffle, num_workers= self.workers_atomic,
            collate_fn=lambda batch: utils.default_collate_pad( batch, self.pad_values) )
        
        return dataloader

class SingleDataset_rst(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

    """
    def __init__(self, file_path, tokenizer, line_start, line_end ):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.line_start = line_start
        self.line_end = line_end
                
        skiprows = self.line_start if self.line_start!=0 else None
        with open(self.fp, 'r') as f:
            if self.line_start == 0:
                
                #TODO: remove nrows
                self.data = pd.read_csv(file_path, sep=',', header=0, 
                    skiprows=skiprows,
                    nrows=(self.line_end-self.line_start),
                    #nrows=100
                     )

            else: 
                names = open(file_path,"r").readline().strip().split(',')
                            
                self.data = pd.read_csv(file_path, sep=',', 
                    names=names, skiprows=skiprows,
                    nrows=(self.line_end-self.line_start)
                    #nrows=100
                    ) 
                
        # TODO: add a check for lines which have keyphrases that are empty and remove them
        
        # filtering out lines which relate to long texts        
        if 'li_edus' in self.data.columns:
            #cls.data = cls.data[0:0]
            self.valid = True
            self.data  = self.data.loc[ (self.data['txt_preproc'].str.len() <= 250 ) & (~ self.data['li_edus'].isnull()) ]
        else:
            self.valid = False

    def __len__(self):
        return len(self.data)

 

    def __getitem__(self, index, pad_utterance=True):
        
        li_rst, li_edu_kp, li_kp_score = self.getitem_extract_datum(index)
        
        #encoding
        encoded = self.tokenizer.tokenize_rst( li_edu_kp = li_edu_kp ,
                                                     li_rst = li_rst,
                                                     li_kp_score = li_kp_score)

            # encoded May include some of the following
            #    return {
            #         #'input_ids': , 

            #         'li_head_ids': li_head_ids ,
            #         'li_head_pos_ids': li_head_pos_ids,
            #         'li_head_kpscore': li_head_kpscore,

            #         'tail_ids': tail_ids,
            #         'tail_pos_id': tail_pos_id ,
            #         'tail_kp_score': target_edu_kpscore

            #         'li_rst_rel_ids': li_rst_rel_ids ,
            #         'li_rst_pos_ids': li_rst_pos_ids,
            #         'li_rst_pos_ids': li_rst_ns_ids

            #     }   

        return encoded

    def getitem_extract_datum(self, index):

        datum = self.data[index:index+1]
        
        li_rst = ujson.loads(datum['rst'].values[0] )
        
        #list of dicts. Each dict contains keys:value => part of speech labelling system name
        try:
            li_dict_posname_likpscore = ujson.loads( datum['li_dict_posname_likpscore'].values[0] )
        except TypeError as e:
            print( datum['li_dict_posname_likpscore'] )
            print("len: ", len(self.data))
            raise Exception
        
        
        li_li_kp_score = [ self.pos_type_chooser( _dict ) for _dict in li_dict_posname_likpscore ]

        li_edu_kp, li_kp_score = list( zip(*li_li_kp_score) )

        li_edu_kp = list(li_edu_kp)
        #TODO: fixing gaps such as was nt and ca nt.

        li_kp_score = list(li_kp_score)

        return li_rst, li_edu_kp, li_kp_score
    
    def pos_type_chooser(self, _dict ):
        # This method chooses which part of speech annotation to choose for the keyphrase

        #_dict = {""pos1"": [[""first thing"", 0.4650969919261783], [""'s"", 0.06980608614764326]], ""pos0"": [[""That's the first thing"", 0.19999999999999993]] } 

        # Criteria for choosing: 
        #   For both pos options we select the first key_phrase, score couple
        #   Base on overlap
        
        # For Now just take the keyphrase given by pos1
        # TODO: implement choosing version
                
        kp = _dict['pos0'][0][0]
        kp_score = _dict['pos0'][0][1]

        return [kp, kp_score]

class SingleDataset_atomic2020(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

    """
    #TODO: think of way to balance ATOMIC and RST contribution
    #TODO: find research on multi task learning well
    def __init__(self, file_path, tokenizer ):
        self.fp = file_path
        self.tokenizer = tokenizer

        #with open(self.fp, 'r') as f:
        #TODO: remove nrows
        self.data = pd.read_csv(self.fp 
            #,nrows=100
            )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index,pad_utterance=True):
        
        head, rel, tail = self.getitem_extract_datum(index)
        
        #encoding
        encoded = self.tokenizer.tokenize_comet( head=head,
                                                        rel=rel,
                                                        tail=tail )

            # encoded May include some of the following
            # return {
            #     'li_head_ids':li_head_ids,
            #     'li_head_pos_ids':None,

            #     'tail_ids':tail_ids,
            #     'tail_pos_id':None,

            #     'li_rst_rel_ids': None,
            #     'li_rst_pos_ids': None,
            #     'li_rst_pos_ids':None,

            # }    

        return encoded

    def getitem_extract_datum(self, index):

        datum = self.data[index:index+1]

        #lists of length 1
        head_entity =   ujson.loads(datum['head'].values[0])
        relationship =  ujson.loads(datum['relation'].values[0]) 
        tail_entity =  ujson.loads(datum['tail'].values[0]) 
        
        return head_entity, relationship, tail_entity


def main(tparams={}, mparams={}):

    mparams['model_name'] = TrainingModule.model_name(mparams)
    
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
    mparams = COMERST.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = TrainingModule.parse_train_specific_args(parent_parser)

    if tparams.mode == "test":
        assert tparams.gpus in [0,1]

    # adjust to allow ddp to work on computer
    if tparams.gpus not in [0,1]:
        os.environ['MASTER_ADDR'] = 'localhost' #'127.0.0.1'
        os.environ['MASTER_PORT'] = '65302'

    main(vars(tparams), vars(mparams))

    # CUDA_VISIBLE_DEVICES=0,1 python3 train_comerst.py -bs 216 -agb 1 --gpus 2 --workers 12 --version 1 --precision 16 --mode train_new -lr 3e-4 -me 60 --tag "baseline"
    # CUDA_VISIBLE_DEVICES=0,1 python3 train_comerst.py -bs 130 -agb 3 --gpus 2 --workers 12 --version 2 --precision 16 --mode train_new -lr 3e-4 -me 15 --tag "weighting the rst loss at 3/4 and the comet at 1/4"
