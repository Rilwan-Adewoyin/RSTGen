#Script for training models related to predicting Key phrases from knowledge graph
# Example of how to use batch generate -> https://github.com/allenai/comet-atomic-2020/blob/master/models/comet_atomic2020_bart/generation_example.py

# Use this for finishing off code for fine-tuning https://github.com/allenai/comet-atomic-2020/blob/master/models/comet_atomic2020_bart/finetune.py

#region imports 
import os
#os.environ['NCCL_SOCKET_IFNAME'] =  'lo' 
#os.environ['NCCL_SOCKET_IFNAME'] =  'eth'
#os.environ['NCCL_SOCKET_IFNAME'] =  "enp226s0f0"
#os.environ['NCCL_IB_DISABLE'] ="1"#
#os.environ['CUDA_LAUNCH_BLOCKING']='1' 
import json
import torch
import argparse
from torch._C import Value
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple
import transformers
from transformers import get_cosine_schedule_with_warmup
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.distributed import _get_rank
from transformers.modeling_outputs import Seq2SeqLMOutput
from sklearn.utils import shuffle as skl_shuffle
from itertools import islice
import math
from itertools import groupby
import copy
from types import MethodType
from collections import defaultdict
import pke
from pke.data_structures import Candidate, Document
from timeout_timer import timeout, TimeoutInterrupt


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

#patching the pke reader 
def read(self, text, **kwargs):
    """Read the input file and use spacy to pre-process.
    Spacy model selection: By default this function will load the spacy
    model that is closest to the `language` parameter ('fr' language will
    load the spacy model linked to 'fr' or any 'fr_core_web_*' available
    model). In order to select the model that will be used please provide a
    preloaded model via the `spacy_model` parameter, or link the model you
    wish to use to the corresponding language code
    `python3 -m spacy link spacy_model lang_code`.
    Args:
        text (str): raw text to pre-process.
        max_length (int): maximum number of characters in a single text for
            spacy (for spacy<3 compatibility, as of spacy v3 long texts
            should be splitted in smaller portions), default to
            1,000,000 characters (1mb).
        spacy_model (model): an already loaded spacy model.
    """

    spacy_model = kwargs.get('spacy_model', None)

    if spacy_model is None:
        try:
            spacy_model = spacy.load(str2spacy(self.language),
                                     disable=['ner', 'textcat', 'parser'])
        except OSError:
            logging.warning('No spacy model for \'{}\' language.'.format(self.language))
            logging.warning('Falling back to using english model. There might '
                'be tokenization and postagging errors. A list of available '
                'spacy model is available at https://spacy.io/models.'.format(
                    self.language))
            spacy_model = spacy.load(str2spacy('en'),
                                     disable=['ner', 'textcat', 'parser'])
        if int(spacy.__version__.split('.')[0]) < 3:
            sentencizer = spacy_model.create_pipe('sentencizer')
        else:
            sentencizer = 'sentencizer'
        spacy_model.add_pipe(sentencizer)
        if 'max_length' in kwargs and kwargs['max_length']:
            spacy_model.max_length = kwargs['max_length']

    #spacy_model = fix_spacy_for_french(spacy_model)
    spacy_doc = spacy_model(text)
    
    retokenize(spacy_doc)
    
    sentences = []
    for sentence_id, sentence in enumerate(spacy_doc.sents):
        sentences.append({
            "words": [token.text for token in sentence],
            "lemmas": [token.lemma_ for token in sentence],
            # FIX : This is a fallback if `fix_spacy_for_french` does not work
            "POS": [token.pos_ or token.tag_ for token in sentence],
            "char_offsets": [(token.idx, token.idx + len(token.text))
                             for token in sentence]
        })

    doc = Document.from_sentences(
        sentences, input_file=kwargs.get('input_file', None), **kwargs)

    return doc

def retokenize(doc):
    position = [token.i for token in doc if token.i!=0 and "'" in token.text]
    with doc.retokenize() as retokenizer:
        for pos in position:
            try:
                retokenizer.merge(doc[pos-1:pos+1])
            except ValueError:
                pass

def get_n_best(self, n=10, redundancy_removal=False, stemming=False):
    """Returns the n-best candidates given the weights.
    Args:
        n (int): the number of candidates, defaults to 10.
        redundancy_removal (bool): whether redundant keyphrases are
            filtered out from the n-best list, defaults to False.
        stemming (bool): whether to extract stems or surface forms
            (lowercased, first occurring form of candidate), default to
            False.
    """

    # sort candidates by descending weight
    best = sorted(self.weights, key=self.weights.get, reverse=True)

    # remove redundant candidates
    if redundancy_removal:

        # initialize a new container for non redundant candidates
        non_redundant_best = []

        # loop through the best candidates
        for candidate in best:

            # test wether candidate is redundant
            if self.is_redundant(candidate, non_redundant_best):
                continue

            # add the candidate otherwise
            non_redundant_best.append(candidate)

            # break computation if the n-best are found
            if len(non_redundant_best) >= n:
                break

        # copy non redundant candidates in best container
        best = non_redundant_best

    # get the list of best candidates as (lexical form, weight) tuples
    n_best = [(u, self.weights[u]) for u in best[:min(n, len(best))]]

    # replace with surface forms if no stemming
    if not stemming:
        n_best = [  (''.join( [ " "+sf if (sf not in [".","(",")","[","]","?","{","}","!"]) else sf for sf in self.candidates[u].surface_forms[0] ]).strip(' ') ,
                   self.weights[u]) for u in best[:min(n, len(best))]]

    # return the list of best candidates
    return n_best

pke.readers.RawTextReader.read = read
pke.base.LoadFile.get_n_best = get_n_best

def reset(self):
    self.sentences = []
    self.candidates = defaultdict(Candidate)
    self.weights = {}
    #self.graph = nx.Graph()
    self.graph.clear()

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
from pathlib import Path


import names
import pandas as pd

from pytorch_lightning.core.decorators import auto_move_data
from typing import Optional, Callable, Union, Optional, List, Iterable

from functools import lru_cache, reduce
import spacy
from pytorch_lightning.core.saving import save_hparams_to_yaml

nlp = spacy.load("en_core_web_sm") 
components = ["parser","ner"]

#endregion 

# removing non needed componenets
for name in nlp.pipe_names:
    if name not in components:
        nlp.remove_pipe(name)

class EmbeddingRstPos(nn.Module):
    def __init__(self, lr_func , max_rst_index=62, max_rst_level=8, rst_encoding_ndim=768,
                    ):
        super(EmbeddingRstPos, self).__init__()

        self.max_rst_index = max_rst_index
        self.max_rst_level = max_rst_level
        self.left_right_seq_from_root_to_edu_pos = lr_func

        self.fixed_rst_encoding = self.make_rst_encoding( )
        self.ffd = torch.nn.Linear(self.max_rst_level, rst_encoding_ndim, bias=False )
        
        self.padding_idx = self.fixed_rst_encoding.padding_idx

    def forward(self, x ):
        while x.max() >= self.max_rst_index:
            x = torch.where( x<self.max_rst_index, x, torch.ceil( (x-2)/2 ).long() )
   

        x = self.fixed_rst_encoding(x)
        x = self.ffd( x )
        return x
    
    def make_rst_encoding(self):
        
        embedding_weight = torch.zeros( 
                                (self.max_rst_index, self.max_rst_level ),
                                dtype = torch.float )
        
        # zero index embedding vector
        zero_embedding = np.zeros( [self.max_rst_level] )

        split_dir_numb = {'L':-1, 'R':1}
        
        # for each embedding
        for idx in range(self.max_rst_index):
            
            idx_embedding = copy.deepcopy( zero_embedding )
            
            # Determine the sequence of lefts and rights to reach node    
            left_rights_from_root_to_pos = self.left_right_seq_from_root_to_edu_pos( idx )
            
            # Convert sequence of LRs to a sequence of -1 and 1s and 0s
            for idx1, val in enumerate(left_rights_from_root_to_pos):
                idx_embedding[idx1] = split_dir_numb[val]

            # set this as the new embedding
            embedding_weight[idx] = torch.FloatTensor( idx_embedding )

        fixed_rst_encoding = torch.nn.Embedding.from_pretrained( embedding_weight ,
                                    freeze=True, padding_idx=0)

        return fixed_rst_encoding
    
def clamp_values(x, max):

    #clamps values in a tree method where the parent tree nodes is the evel
        # to reduce to
    # we use this since the rst positions in our tree are often too large 
    # for torch.long to handle
    while x.max() >= max:
        x = np.where( x<=max, x, np.floor_divide(x-1,2) )
    
    return x.astype( int )

MAX_LONG_VALUE = torch.iinfo(torch.long).max


#region COMERST model and tokenizer
class COMERST(nn.Module):

    def __init__(self, 
                    base_model_name='bart_base', model_name="COMERST",
                    scale_grad_by_freq=False,
                    max_len_head=20,
                    max_len_tail=20,
                    max_edu_nodes_to_select=-1,
                    filter_atomic_rels=False,
                    dict_embed_mnorms = {},
                    relation_embedding = "flattened",
                    attention_type = 1,
                    freeze_embeds=False,
                    rst_pos_embed_type = 1
                    ):
        super(COMERST, self).__init__()

        self.base_model_name = base_model_name   
        self.model_name = model_name
        self.scale_grad_by_freq = scale_grad_by_freq
        self.relation_embedding = relation_embedding
        self.attention_type = attention_type
        self.freeze_embeds = freeze_embeds
        
        self.transformer = utils.load_pretrained_transformer(self.base_model_name, transformer=True)['transformer']
        self.transformer.comerst = lambda : self
        self.rst_pos_embed_type = rst_pos_embed_type

        self.tokenizer = COMERST_tokenizer(self.base_model_name,
                                            os.path.join( ("./models"), f"{model_name}_tokenizer"),
                                            max_len_head, max_len_tail,
                                            max_edu_nodes_to_select,
                                            filter_atomic_rels,
                                            self.relation_embedding,
                                            attention_type=self.attention_type,
                                            rst_pos_embed_type=rst_pos_embed_type )
        
        self.transformer.resize_token_embeddings( len(self.tokenizer.base_tokenizer) )

        # region Extra embedding layers
        
            # The rst_parent_nodes can go up to N, but the edu_positions can then go up to N*2 +2
                #+1+1 here because max_node =N, but our node data is 0 indexed  padding index is final node
        if self.rst_pos_embed_type == 1:
            self.embedding_rst_pos = torch.nn.Embedding( (2*self.tokenizer.rst_pos_maxidx +2 ) + 1 +1 , self.transformer.config.d_model, 
                                    padding_idx=(2*self.tokenizer.rst_pos_maxidx+2)+1 , scale_grad_by_freq=self.scale_grad_by_freq )

        elif self.rst_pos_embed_type == 2:
            
            self.embedding_rst_pos = EmbeddingRstPos(  self.tokenizer.left_right_seq_from_root_to_edu_pos,
                                                    max_rst_index=self.tokenizer.rst_pos_maxidx,
                                                        max_rst_level = self.tokenizer.node_level(self.tokenizer.rst_pos_maxidx),
                                                rst_encoding_ndim=self.transformer.config.d_model)
            

        self.embedding_rst_ns = torch.nn.Embedding( len(self.tokenizer.rst_ns_li )+1, self.transformer.config.d_model,
                                    padding_idx=len(self.tokenizer.rst_ns_li ), scale_grad_by_freq=self.scale_grad_by_freq )

        # self.embedding_kp_score = torch.nn.Conv1d( 1, self.transformer.config.d_model , kernel_size=1, bias=False)
        # self.embedding_kp_score.weight.data.normal_(mean=0.0, std=0.005)

        # relations embedding
        if self.relation_embedding == "flattened":
            rel_embed_len = len(self.tokenizer.rst_rel_li ) + len(self.tokenizer.atomic_rel_li ) + 1
            rel_pad_idx = len(self.tokenizer.rst_rel_li ) + len(self.tokenizer.atomic_rel_li ) 
            self.embedding_rels = torch.nn.Embedding( rel_embed_len, self.transformer.config.d_model, padding_idx=rel_pad_idx, scale_grad_by_freq=self.scale_grad_by_freq   )
            with torch.no_grad():
                self.embedding_rels.weight[ self.embedding_rels.padding_idx ].fill_(0)
                
            # In this embedding comet takes the first set of embedding indices. RST takes the second set of embedding indices
        
        elif self.relation_embedding == "hierarchical1":
            # This is the normal embedding layer for COMET model
                # We allow one extra embedding position to act as a domain shift embedding position from COMET to RST - essentially a bias term
            rel_embed_len_atomic = len(self.tokenizer.atomic_rel_li ) + 1 + 1
            rel_pad_idx_atomic = len(self.tokenizer.atomic_rel_li ) + 1
            self.embedding_rels_atomic = torch.nn.Embedding( rel_embed_len_atomic, self.transformer.config.d_model, padding_idx=rel_pad_idx_atomic, scale_grad_by_freq=self.scale_grad_by_freq   )

            #  This is an embedding layer that maps each RST relationship to a set of weights over the atomic relationships
            rel_embed_len_rst = len(self.tokenizer.atomic_rel_li ) + 1
            rel_pad_idx_rst = len(self.tokenizer.atomic_rel_li )
            self.embedding_rels_rst = torch.nn.Embedding( rel_embed_len_rst, rel_embed_len_atomic , padding_idx=rel_pad_idx_rst, scale_grad_by_freq=self.scale_grad_by_freq   )

            with torch.no_grad():
                self.embedding_rels_atomic.weight[ self.embedding_rels_atomic.padding_idx ].fill_(0)
                self.embedding_rels_rst.weight[ self.embedding_rels_rst.padding_idx ].fill_(0)
        
        elif self.relation_embedding == "hierarchical2":
            # This is the normal embedding layer for COMET model
                # We allow one extra embedding position to act as a domain shift embedding position from COMET to RST
            rel_embed_len_atomic = len(self.tokenizer.atomic_rel_li ) + 1 + 1
            rel_pad_idx_atomic = len(self.tokenizer.atomic_rel_li ) + 1
            self.embedding_rels_atomic = torch.nn.Embedding( rel_embed_len_atomic, self.transformer.config.d_model, padding_idx=rel_pad_idx_atomic, scale_grad_by_freq=self.scale_grad_by_freq   )

            #  This is an embedding layer that maps each RST relationship to a set of weights over the atomic relationships
            rel_embed_len_rst = len(self.tokenizer.atomic_rel_li ) + 1
            rel_pad_idx_rst = len(self.tokenizer.atomic_rel_li )
            self.embedding_rels_rst_l1 = torch.nn.Embedding( rel_embed_len_rst, 80 , padding_idx=rel_pad_idx_rst, scale_grad_by_freq=self.scale_grad_by_freq   )            

            self.embedding_rels_rst = torch.nn.Sequential(
                self.embedding_rels_rst_l1,
                torch.nn.Linear( 80, 80 , bias=True ) ,
                torch.nn.Tanh(),
                torch.nn.Linear( 80, rel_embed_len_atomic , bias=True ) ,
                torch.nn.Tanh(),
            )
            self.embedding_rels_rst.padding_idx = self.embedding_rels_rst[0].padding_idx

            with torch.no_grad():
                self.embedding_rels_atomic.weight[ self.embedding_rels_atomic.padding_idx ].fill_(0)
                self.embedding_rels_rst[0].weight[ self.embedding_rels_rst.padding_idx ].fill_(0) 
          
        # setting the pad token to 0 in posiiton_ids embedding and then freezing
        self.transformer.model.encoder.embed_positions.padding_idx = 0
        self.transformer.model.decoder.embed_positions.padding_idx = 0
        
        with torch.no_grad():
            self.transformer.model.encoder.embed_positions.weight[self.transformer.model.encoder.embed_positions.padding_idx].fill_(0)
            self.transformer.model.decoder.embed_positions.weight[self.transformer.model.decoder.embed_positions.padding_idx].fill_(0)

        #TODO: freezing weights
        if self.freeze_embeds:
            utils.freeze_params(self.transformer.model.shared)
            utils.freeze_params( self.transformer.lm_head )

            for d in [self.transformer.model.encoder, self.transformer.model.decoder]:
                utils.freeze_params(d.embed_positions)
                
                utils.freeze_params(d.embed_tokens)
        
        #endregion

        # region embedding layer normalization
        mapping = {"word":self.transformer.model.shared,
        
            "pe":self.transformer.model.encoder.embed_positions ,
            "pd":self.transformer.model.decoder.embed_positions ,

            "r_pos": self.embedding_rst_pos ,
            "r_ns": self.embedding_rst_ns,
            
            }
        if self.relation_embedding == 'flattened':
            mapping['rels'] = self.embedding_rels

        elif self.relation_embedding == 'hierarchical1':
            mapping["rels_h1"] = self.embedding_rels_rst

        elif self.relation_embedding == 'hierarchical2':
            mapping["rels_h2"] = self.embedding_rels_rst[0]

        for key, val in dict_embed_mnorms.items():
            mapping[key].max_norm = val

        del mapping
            
        #end region

        self.loss_fct = CrossEntropyLoss()

        # region adapting existing methods
        self.transformer.model.encoder.forward = types.MethodType(utils.BART_encoder_forward, 
                                                    self.transformer.model.encoder )

        self.transformer.model.forward = types.MethodType(utils.BART_forward, 
                                                    self.transformer.model )
        
        self.transformer.prepare_inputs_for_generation = types.MethodType(
            utils.prepare_inputs_for_generation, self.transformer
        )

        self.transformer.greedy_search = types.MethodType(utils.greedy_search, 
                                                    self.transformer)
        
        #endregion

        self.generate = self.transformer.generate
        self.config = self.transformer.config
        self.get_encoder = self.transformer.get_encoder
        self.get_decoder = self.transformer.get_decoder

        self.pad_values = {'head_ids':self.transformer.model.shared.padding_idx , 
                        'head_treepos_ids':self.embedding_rst_pos.padding_idx, 
                        
                        'rst_treepos_ids': self.embedding_rst_pos.padding_idx,
                        'rst_ns_ids': self.embedding_rst_ns.padding_idx, 

                        'tail_ids': self.transformer.model.shared.padding_idx , 
                        'tail_treepos_ids':self.embedding_rst_pos.padding_idx ,
                        #'tail_kp_score': 0,

                        'attention_mask': 0, 
                        'attention_mask_head': 0, 
                        'attention_mask_rel': 0, 
                        
                        'labels': self.loss_fct.ignore_index,

                        'position_ids_head':self.transformer.model.encoder.embed_positions.padding_idx if 
                                        self.transformer.model.encoder.embed_positions.padding_idx else 0  
                        }
        
        if self.relation_embedding == 'flattened':
            self.pad_values['rst_rel_ids'] = self.embedding_rels.padding_idx
        
        elif self.relation_embedding == "hierarchical1":
            self.pad_values['rst_rel_ids'] = self.embedding_rels_rst.padding_idx
        
        elif self.relation_embedding == "hierarchical2":
            self.pad_values['rst_rel_ids'] = self.embedding_rels_rst[0].padding_idx


    def forward(self, input_, return_dict=None):

        return_dict = return_dict if return_dict is not None else self.transformer.config.use_return_dict

        labels_rst = None
        labels_comet = None
        output_rst = []
        output_comet = []

        #region forward pass
        if 'rst' in input_:
            input_rst = self.forward_embed_rst(**input_['rst'])
            labels_rst = input_rst.pop('labels',None)
            output_rst = self.transformer.model.forward(
                            **input_rst)
            lm_logits_rst = self.transformer.lm_head(output_rst[0]) + self.transformer.final_logits_bias

        if 'comet' in input_:
            input_comet = self.forward_embed_comet(**input_['comet'])
            labels_comet = input_comet.pop('labels',None)
            output_comet = self.transformer.model.forward(
                            **input_comet)
            lm_logits_comet = self.transformer.lm_head(output_comet[0]) + self.transformer.final_logits_bias
        
        # endregion
        lm_loss_rst = None
        lm_loss_comet = None

        #region Calculation Losses
        if labels_comet is not None:
            #the labels are automatically aligned as per the GPT2 code
            #TODO: reevaluate whether bos or eos is the best method to use as start of output
                # right now we use the edu token to start sentences. The EDU token is just the bos token
            shift_logits_comet = lm_logits_comet[..., :-1, :].contiguous()
            shift_labels_comet = labels_comet[..., 1:].contiguous() 
            lm_loss_comet = self.loss_fct(shift_logits_comet.view(-1, self.transformer.config.vocab_size), shift_labels_comet.view(-1))

        if  labels_rst is not None:
            shift_logits_rst = lm_logits_rst[..., :-1, :].contiguous()
            shift_labels_rst = labels_rst[..., 1:].contiguous() 
            lm_loss_rst = self.loss_fct(shift_logits_rst.view(-1, self.transformer.config.vocab_size), shift_labels_rst.view(-1))
        #endregion

        if not return_dict:
            lm_loss = {}
            if lm_loss_comet is not None:
                lm_loss['comet'] = lm_loss_comet 
            if lm_loss_rst is not None:
                lm_loss['rst'] =  lm_loss_rst

            _rst = (lm_logits_rst,) + output_rst[1:]
            _comet = (lm_logits_comet,) + output_comet[1:]
            return ((lm_loss,) + _rst + _comet ) if len(lm_loss) != 0 else (_rst, _comet)
            
        
        else:
            s1s_output = {}

            if lm_loss_rst is not None:
                s1s_output_rst =  Seq2SeqLMOutput(
                    loss=lm_loss_rst,
                    logits=lm_logits_rst,
                    past_key_values=output_rst.past_key_values,
                    decoder_hidden_states=output_rst.decoder_hidden_states,
                    decoder_attentions=output_rst.decoder_attentions,
                    cross_attentions=output_rst.cross_attentions,
                    encoder_last_hidden_state=output_rst.encoder_last_hidden_state,
                    encoder_hidden_states=output_rst.encoder_hidden_states,
                    encoder_attentions=output_rst.encoder_attentions,
                    )
                s1s_output['rst'] = s1s_output_rst

            if lm_loss_comet is not None:
                s1s_output_comet =  Seq2SeqLMOutput(
                    loss=lm_loss_comet,
                    logits=lm_logits_comet,
                    past_key_values=output_comet.past_key_values,
                    decoder_hidden_states=output_comet.decoder_hidden_states,
                    decoder_attentions=output_comet.decoder_attentions,
                    cross_attentions=output_comet.cross_attentions,
                    encoder_last_hidden_state=output_comet.encoder_last_hidden_state,
                    encoder_hidden_states=output_comet.encoder_hidden_states,
                    encoder_attentions=output_comet.encoder_attentions,
                    )
                s1s_output['comet'] = s1s_output_comet
            
            return s1s_output 

    def forward_embed_rst(self,  
                            head_ids,
                            head_treepos_ids,
                            #head_kpscore,
                            rst_rel_ids,
                            rst_treepos_ids,
                            rst_ns_ids,
                            tail_ids,
                            tail_treepos_ids,
                            #tail_kp_score,
                            attention_mask_head,
                            attention_mask_rel, 

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

        if self.relation_embedding == "flattened":  
            #TODO: make sure padding vectors dont get gi      
            inputs_embed_rel = self.embedding_rels( rst_rel_ids )
        elif self.relation_embedding == "hierarchical1":
            _ = self.embedding_rels_rst( rst_rel_ids )
            inputs_embed_rel = torch.matmul( _, self.embedding_rels_atomic.weight.detach() ) # embedding vector times all the embedding vectors from the comet things
        elif self.relation_embedding == "hierarchical2":
            _ = self.embedding_rels_rst( rst_rel_ids )
            inputs_embed_rel = torch.matmul( _, self.embedding_rels_atomic.weight.detach() ) 

        inputs_embed_rel += self.embedding_rst_pos( rst_treepos_ids )
        inputs_embed_rel += self.embedding_rst_ns( rst_ns_ids )

        inputs_embeds = torch.cat( [ inputs_embed_head, inputs_embed_rel ], axis=-2)
        
        inputs_embeds *= self.transformer.model.encoder.embed_scale
        #endregion

        # region reforming attention mask
        _shape = attention_mask_head.shape[1] + attention_mask_rel.shape[1]
        attention_mask = torch.zeros( [ attention_mask_head.shape[0], _shape , _shape ], dtype=torch.float, device=attention_mask_head.device )
        attention_mask[:, :attention_mask_head.shape[1], :attention_mask_head.shape[1]] = attention_mask_head
        
        if self.attention_type == 1:
            attention_mask[:, -attention_mask_rel.shape[1]:, :] = 1 

        elif self.attention_type==2:
            attention_mask[  : , -attention_mask_rel.shape[1]: , -attention_mask_rel.shape[1]: ] = 1
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
                                self.embedding_rst_pos( tail_treepos_ids ) 

                            
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
                                labels ):
        #region inputs_embeds
            # embed head_ids and 
            # emed rels_ids and tail_treepos_ids
            
        inputs_embed_head = self.transformer.model.shared( head_ids )
        inputs_embed_head += self.embedding_rst_pos( head_treepos_ids )
        inputs_embed_head += nn.Embedding.forward(self.transformer.model.encoder.embed_positions, position_ids_head )

        if self.relation_embedding == "flattened":
            inputs_embed_rel = self.embedding_rels( rels_ids )
        else:
            inputs_embed_rel = self.embedding_rels_atomic(rels_ids)

        inputs_embed_rel += self.embedding_rst_pos( rels_treepos_ids )
        inputs_embed_rel += self.embedding_rst_ns( torch.full_like( rels_treepos_ids , 
                                fill_value=self.embedding_rst_ns.padding_idx  )  )

        inputs_embeds = torch.cat( [ inputs_embed_head, inputs_embed_rel], axis=-2 )
        
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
                                self.embedding_rst_pos( tail_treepos_ids ) 
                            
        decoder_inputs_embeds = decoder_inputs_embeds * self.transformer.model.decoder.embed_scale
        #endregion

        return {
            'attention_mask': attention_mask,
            
            'inputs_embeds': inputs_embeds,
            
            'decoder_inputs_embeds': decoder_inputs_embeds,

            'labels': labels 
        }

    def return_params(self):
        keys = ['base_model_name','max_len_head', 'max_len_tail',
                        'scale_grad_by_freq','model_name',
                        'filter_atomic_rels','max_edu_nodes_to_select',
                        'relation_embedding','attention_type',
                        'freeze_embeds','rst_pos_embed_type']

        json_keys = ['dict_embed_mnorms']
        
        params = {
            k:self.__dict__[k] for k in keys if 
                k in self.__dict__.keys() }

        tokenizer_params = {
            k:self.tokenizer.__dict__[k] for k in keys if 
                k in self.tokenizer.__dict__.keys() }

        params = {**params, **tokenizer_params}

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
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(self.chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = utils.trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

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

    def generate_from_dloaderbatch(
            self, 
            batch,
            comet_or_rst="comet",
            device='cuda:0',
            generation_kwargs = {}
            ):

        """[summary]

        
            [generation_kwargs]: {
                     do_sample: Optional[bool] = None,
                    diversity_penalty: Optional[float] = None
                    early_stopping: Optional[bool] = None,
                    num_beams: Optional[int] = None,
                    temperature: Optional[float] = None,
                    top_k: Optional[int] = None,
                    top_p: Optional[float] = None,
                    repetition_penalty: Optional[float] = None,
                    bad_words_ids: Optional[Iterable[int]] = None,
                    bos_token_id: Optional[int] = None,
                    pad_token_id: Optional[int] = None,
                    eos_token_id: Optional[int] = None,
                    length_penalty: Optional[float] = None,
                    no_repeat_ngram_size: Optional[int] = None,
                    encoder_no_repeat_ngram_size: Optional[int] = None,
                    num_return_sequences: Optional[int] = None,
            }
            Returns:
        """
        for key in batch:
            batch[key] = batch[key].to(device)

        with torch.no_grad():
            #examples = queries
            tail_treepos_ids = batch['tail_treepos_ids'][:,:1]

            if comet_or_rst == "comet":
                input_ = self.forward_embed_comet( **batch )
            
            elif comet_or_rst == "rst":
                input_ = self.forward_embed_rst( **batch )

            # removing labels and decoder_inputs_embeds
            labels = input_.pop('labels')
            __ = input_.pop('decoder_inputs_embeds')

            #inserting decoder_input_ids
            decoder_start_token_id = self.config.eos_token_id
            bos_token_id = self.config.bos_token_id
            decoder_start_token_id = self.transformer._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
            decoder_input_ids = (
                torch.ones(( input_['inputs_embeds'].shape[0], 1), dtype=torch.long, device=input_['inputs_embeds'].device) * decoder_start_token_id
                )
            input_['decoder_input_ids'] = decoder_input_ids


            summaries = self.transformer.generate(
                decoder_start_token_id=decoder_start_token_id,
                comet_or_rst = comet_or_rst,
                tail_treepos_ids= tail_treepos_ids,
                **generation_kwargs,
                **input_,
                )

            decs = self.tokenizer.base_tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        
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
        parser.add_argument('-mlh','--max_len_head', type= int, default=20 )
        parser.add_argument('-mlt','--max_len_tail', type= int, default=20 )
        parser.add_argument('-mnts','--max_edu_nodes_to_select', type=int, default=4, )
        parser.add_argument('-far','--filter_atomic_rels', type=lambda x: bool(int(x)), default=False, )
        parser.add_argument('-dem','--dict_embed_mnorms', type=lambda x: ujson.decode(x), default={})
        parser.add_argument('-re', '--relation_embedding', type=str, choices=['flattened','hierarchical1','hierarchical2'], default='flattened' )
        parser.add_argument('-at','--attention_type',type=int, choices=[1,2,3], default=1)
        parser.add_argument('-fe','--freeze_embeds',type=lambda x: bool(int(x)), default=False)
        parser.add_argument('-rpet','--rst_pos_embed_type', type=int, default=1)
        
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
                 max_len_head=20,
                 max_len_tail=20,
                 max_edu_nodes_to_select=-1,
                 filter_atomic_rels=False,
                 relation_embedding = "flattened",
                 randomize_comet_pronouns=False,
                 remove_to=False,
                 attention_type=1, 
                 rst_pos_embed_type = 1,
                 **kwargs ):
        
        #TOdo: ensure max_lens are being used

        self.base_tokenizer_name = base_tokenizer_name
        self.max_len_head = max_len_head
        self.max_len_tail = max_len_tail
        self.max_edu_nodes_to_select = max_edu_nodes_to_select
        self.filter_atomic_rels = filter_atomic_rels
        self.randomize_comet_pronouns = randomize_comet_pronouns
        self.remove_to = remove_to
        self.attention_type = attention_type


        # region Setting up CSKG relation encoding
        self.filter_atomic_rels = filter_atomic_rels
        if self.filter_atomic_rels == False:      
            self.atomic_rel_li = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent',
                                'xNeed', 'xReact', 'xWant', 'AtLocation', 'ObjectUse', 'Desires',
                                'HasProperty', 'NotDesires', 'Causes', 'HasSubEvent', 'xReason',
                                'CapableOf', 'MadeUpOf', 'isAfter', 'isBefore', 'isFilledBy',
                                'HinderedBy']
        
        elif self.filter_atomic_rels == True :
            self.atomic_rel_li = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent',
                                        'xNeed', 'xReact', 'xWant', 'Desires', 'HasProperty',
                                        'NotDesires','Causes','HasSubEvent','xReason',
                                        'CapableOf','MadeupOf','isAfter','isBefore', 'HinderedBy' ]

        self.atomic_rel_labeler = sklp.LabelEncoder()
        self.atomic_rel_labeler.fit(  self.atomic_rel_li )
    
        #endregion

        # region Setting up RST relation encoding

        self.rst_rel_li = ['Attribution',
            'Background','Cause','Comparison','Condition',
            'Contrast','Elaboration','Enablement','Evaluation',
            'Explanation','Joint','Manner-Means','Topic-Comment',
            'Summary','Temporal','Topic-Change','n','same-unit','textual-organization'] #Add this to savable config

        self.rst_rel_labeler = sklp.LabelEncoder()
        self.rst_rel_labeler.fit(  self.rst_rel_li )
        if relation_embedding == "flattened":
            self.rst_rel_labeler.starting_idx = len(self.atomic_rel_li)
        else:
            self.rst_rel_labeler.starting_idx = 0

        self.rst_rel_labeler.transform_patch  = types.MethodType(utils.transform_patch, self.rst_rel_labeler) #monkey patch
        self.rst_rel_labeler.inverse_transform_patch = types.MethodType(utils.inverse_transform_patch, self.rst_rel_labeler)

        #endregion

        # region Setting up RST NS encoding 
        self.rst_ns_li = ['NN','NS','SN','a'] 
        self.rst_ns_labeler = sklp.LabelEncoder()
        self.rst_ns_labeler.fit( self.rst_ns_li  )

        # rst_pos
        if rst_pos_embed_type == 1:
            self.rst_pos_maxidx = 30  #same as in train_nlg. Our model can only, model sentences with 5 rst tree depth
        
        elif rst_pos_embed_type == 2:
            self.rst_pos_maxidx = 4094
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

        # region Initialise base keyphrase extractor
        self.setup_keyphrase_extractor()
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
        

        # getting a list of all graph posiitons of edus in li_edus
        li_edukp_pos =  [ self.find_child_edus(pos, li_rst_pos ) for pos in li_rst_pos ]
        li_edukp_pos = sum( li_edukp_pos, [] )
        #TODO: need to reorder the edukp pos from left to right on a flattened binary tree
 
        li_edukp_pos.sort(key=self.edukp_pos_sort_function)
        #endregion

        # region correct for mismatch between edu key phrase positions and rst tree 
            # edu keyphrases have unlimited node position length, whereas the rst tree was only defined up to node 32
            # so any  key phrase sequesnce with a node position > 64 can not be directly mapped to our rst trees
            # then after aligning key phrased and edu_node positions, we remove all over 64
        if len(li_edukp_pos) < len(li_edu_kp):
            # li_edukp is missing tree positions of size 64 and over. 
                # Based on this we can use a method to ensure the size 
                    # of li_edukp_pos and li_edu_kp is the same
                
            li_edukp_pos = self.correct_child_edus( li_edukp_pos, len(li_edu_kp) )
            li_edukp_pos.sort(key=self.edukp_pos_sort_function)

            np_edukp_pos = np.array(li_edukp_pos)

            bool_filter1 = np.where( np_edukp_pos<= 2*self.rst_pos_maxidx + 2 )
            
            li_edukp_pos = np_edukp_pos[bool_filter1].tolist()
            li_edu_kp = np.array(li_edu_kp)[bool_filter1].tolist()
   
        elif len(li_edukp_pos) > len(li_edu_kp):
            raise ValueError
        
        #region - Selecting a subtree from the  RST tree to focus on
            
            # randomly select a span of edu nodes at random
                # from between covering 2 edus' to covering maxspan edus
                    #max span is a function of maximum tree node. i.e. max_idx=30 -> tree of 5 depth -> max edu count = 32
                    # however a max_span of 32 is rather large, so instead we restrict it to a number between 4 and 10
            # Then select smallest sub-tree that connects the edu nodes
                # The edu node with the lowest number position = anchor
                # Find this achor's parent and check if each other edu in
                #  span is a child of achor's parent node... If not then do the same with the grandparent.
                #  Once a n-level parent node is found that connects, this is the new ROOT node....
                #  Now retrieve all intermediate parent nodes between ROOT node and edus in span ------
            
        if self.max_edu_nodes_to_select==-1:
            max_edu_nodes_to_select = len( li_edukp_pos ) 
        else:
            max_edu_nodes_to_select = min( len( li_edukp_pos ), self.max_edu_nodes_to_select )

        edu_count_to_select = random.randint(2, max_edu_nodes_to_select)
        start_edu_node_idx = random.randint(0, (max_edu_nodes_to_select-1)-(edu_count_to_select-1) )
        
        li_edu_kp = li_edu_kp[ start_edu_node_idx: start_edu_node_idx+edu_count_to_select]
        li_edukp_pos = li_edukp_pos[ start_edu_node_idx: start_edu_node_idx+edu_count_to_select]
        
            # reduce the rst tree information
                # to only include the smallest subtree of parent nodes that connect the edu nodes
        li_rst_pos, li_rst_rel, li_rst_ns = self.smallest_spanning_subtree(li_edukp_pos, li_rst_pos, li_rst_rel, li_rst_ns  ) 

        #endregion


        # region select target edu
            # optionally: randomly selecting a edu keyphrase node to be the target node for prediction
            # the other edu keyphrases form the relation information

            # we aim to select an edu node that is at least two words long if one exists otherwise select any

        if target_edu_kp == None:
            
            # indexes for phrases in li_edukp_pos that contain at least two words
            idxs_of_valid_phrases = [ idx for idx, phrase in enumerate(li_edu_kp) if len(phrase.split(' '))>1 ]

            if len(idxs_of_valid_phrases)>0:
                r_int = random.choice(idxs_of_valid_phrases)
            else:
                r_int = random.randint(0, len(li_edu_kp)-1 )

            target_edu_kp = li_edu_kp.pop(r_int)
            target_edu_pos = li_edukp_pos.pop(r_int)
            target_edu_kpscore = li_kp_score.pop(r_int)
        #endregion

        #region tail
        # Encoded tail information. Adding special tokens
        # line beow list indicesq
        tail_ids = self.base_tokenizer.encode( target_edu_kp , add_prefix_space=True, return_tensors='pt', truncation=True, max_length=self.max_len_tail ).squeeze()
        tail_treepos_ids = tail_ids.new_full( tail_ids.shape , target_edu_pos )
        #tail_kp_score = torch.full( tail_ids.shape, target_edu_kpscore , dtype=torch.float32)
        #endregion


        
        # region head
            #    encode list of keyphrases and scores that are input to encoder
            # append edu token to start of each head
        li_head = li_edu_kp #adding prefix space since we use phrases not start of sequences
        li_head_ids = [ self.base_tokenizer.encode( head, add_prefix_space=True,truncation=True, max_length=self.max_len_head  ) for head in li_head ]
        li_head_treepos_ids = [ [pos]*len(ids) for pos, ids in zip( li_edukp_pos, li_head_ids ) ] #creating a li of li of graph pos indexes for each keyphrase
        #li_head_kpscore =  [ [kpscore]*len(ids) for kpscore, ids in zip( li_kp_score, li_head_ids ) ]

            #flattening and converting to tensor
        head_ids = torch.tensor( sum(li_head_ids,[]), dtype=torch.long)
        head_treepos_ids = torch.tensor( sum(li_head_treepos_ids, []), dtype=torch.long)
        #head_kpscores = torch.tensor( sum( li_headkpscores, []), dtype=torch.long)

        #endregion 
       
        # region relation : encoded list of rst parent nodes information
        rst_rel_ids = torch.tensor(  [self.rst_rel_labeler.transform_patch([rel]) for rel in li_rst_rel ], dtype=torch.long).squeeze(dim=-1)
        rst_treepos_ids = torch.tensor( li_rst_pos, dtype=torch.long )
        rst_ns_ids =  torch.tensor( [ self.rst_ns_labeler.transform([ns]) for ns in li_rst_ns ], dtype=torch.long).squeeze(dim=-1)
        #endregion

        #region attention mask
            #For rst we require 1) bidirectional attention over each keyphrase chunk
                                    # each keyphrase chunk can not attend directly to other keyphrase chunks
            #                   2) all encoder inputs can attend to rst relation tree info

        
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
            #'tail_kp_score':tail_kp_score,

            'attention_mask_head': attention_mask_head,
            'attention_mask_rel': attention_mask_rel,
            'position_ids_head': position_ids_head,

            'labels':labels.squeeze()
        }
        
    def tokenize_rst_v2( self, li_rst, dict_pos_edu, target_edu_kp=None, target_edu_pos=None ):
        
        # In this one we will predict the key-phrase for a 3/4edu chunk using information from another 3/4edu chunk
            # - at first we will only train with one other chunk. so one to one.
            # Need to select information to make head
            # Then need to select information that will make tail
            # If node position numbers are too deep, then need to maybe move them up into higher positions
                # Allow tree rst position embedding to extend to a larger number. Then put a method in place to clamp the max value
        try:
            with timeout(seconds=5):
                dict_pos_edu = {int(k):v for k,v in dict_pos_edu.items()}

                # region prepping rst tree info
                li_rst_rel = [ rst_node['rel'] for rst_node in li_rst ] 
                li_rst_pos = [ rst_node['pos'] for rst_node in li_rst ]
                li_rst_ns =  [ rst_node['ns'] for rst_node in li_rst ]

                    # getting a list of all graph posiitons of edus in li_edus
                li_edu_pos =  [ pos for pos, edu in dict_pos_edu.items() ]
                li_edu_pos.sort(key=lambda edu_pos: ( self.edukp_pos_sort_function(edu_pos), edu_pos ) )
                #endregion

                # region - Selecting a subtree from the  RST tree to focus on
                    
                    # head -> randomly select a span of 3or4 edu nodes at random
                    # tail -> select the following 3 or 4 edu span
                    # rel -> select smallest sub-tree of li_rst that coverse both edus    
                            # anchor1 = Find parent node for the nodes in the head
                            # anchor2 = Find parent node for the nodes in the tail                            
                            #  Find  achor1 parent and check if this is parent for anchor2
                                # If not then do the same with the grandparent. and repeat
                            #  Once a n-level parent node is found that connects, this is the ROOT node r 
                            #  Now retrieve all intermediate parent nodes between ROOT node and each anchor node
                    
                    # remember the input key phrases are shortened text, but the output keyphrase is normal english
                min_edus_in_chunk = 3
                edus_in_tail = min_edus_in_chunk
                count_edus_in_record = len( li_edu_pos )


                if self.max_edu_nodes_to_select==-1:
                    max_edu_nodes_to_select_for_head =  count_edus_in_record - edus_in_tail
                else:
                    max_edu_nodes_to_select_for_head = min(  self.max_edu_nodes_to_select , count_edus_in_record - edus_in_tail )

                li_tailkp_score = []
                li_headkp_score = []
                #attempts = 1
                
                # Extracting list of tailkp and headkp. repeat until adequate keyphrases are selected
                #while len(li_tailkp_score)==0 or len(li_headkp_score)==0 and attempts<=5:
                
                #for attempt in range(3):
                count_edus_for_head = random.randint(min_edus_in_chunk, max_edu_nodes_to_select_for_head)
                start_edu_node_idx_for_head = random.randint(0, count_edus_in_record-edus_in_tail-count_edus_for_head )
                
                final_edu_node_idx_for_head = start_edu_node_idx_for_head+count_edus_for_head
                tail_node_end_idx = final_edu_node_idx_for_head + edus_in_tail
                
                #TODO: make sure nodes selected arent too far apart
                li_edu_pos_for_head = li_edu_pos[ start_edu_node_idx_for_head: final_edu_node_idx_for_head ]
                li_edu_pos_for_tail = li_edu_pos[  final_edu_node_idx_for_head: tail_node_end_idx ]
                
                    # reduce the rst tree information
                        # to only include the smallest subtree of parent nodes that connect the edu nodes
                all_nodes_included = li_edu_pos[ start_edu_node_idx_for_head: tail_node_end_idx ]
                li_rst_pos_, li_rst_rel_, li_rst_ns_ = self.smallest_spanning_subtree( all_nodes_included, li_rst_pos, li_rst_rel, li_rst_ns  ) 
                
                # Extracting keyphrases for  tails and ehads
                    # We sample the top two, then select one at random
                    # NOTE: IDEA During inference we can use beam search to sample multiple key phrases
                tail_text = ' '.join( [ dict_pos_edu[pos] for pos in li_edu_pos_for_tail] )
                    
                    # We sample the top three, then select  two at random
                head_text = ' '.join( [ dict_pos_edu[pos] for pos in li_edu_pos_for_head] )
                
                li_tailkp_score = self.key_phrase_extract( tail_text, n_best= 2, shorten=False )
                li_headkp_score = self.key_phrase_extract( head_text, n_best= 3 ) 

                # Checking non empty kps are returned

                if len(li_tailkp_score)==0 or len(li_headkp_score)==0:
                    return {
                        #head
                        'head_ids': None ,
                        'head_treepos_ids': None,
                        #'head_kpscore': head_kpscore,

                        #relation: tree information
                        'rst_rel_ids': None ,
                        'rst_treepos_ids': None,
                        'rst_ns_ids': None,

                        #tail
                        'tail_ids': None,
                        'tail_treepos_ids': None ,
                        #'tail_kp_score':tail_kpscore,

                        'attention_mask_head': None,
                        'attention_mask_rel': None,
                        'position_ids_head': None,

                        'labels':None
                    }

                #region encoding tail

                tail_kp, tail_kpscore = random.choice( li_tailkp_score )

                    # Encoded tail information. Adding special tokens
                tail_ids = self.base_tokenizer.encode( tail_kp, add_prefix_space=True, 
                    return_tensors='pt', truncation=True, max_length=self.max_len_tail ).squeeze()
                
                

                tail_edus_parent_pos = self.lowest_shared_parent_node(li_edu_pos_for_tail)
                tail_edus_parent_pos = clamp_values( np.asarray(tail_edus_parent_pos),  MAX_LONG_VALUE ).item()
                tail_treepos_ids = tail_ids.new_full( tail_ids.shape , tail_edus_parent_pos )

                #tail_kp_score = torch.full( tail_ids.shape, tail_kpscore , dtype=torch.float32)
                #endregion

                # region encoding head
                # We sample two keyphrases from the 3 best ones

                li_headkp_score = random.sample( li_headkp_score, k=min(2,len(li_headkp_score)) )

                    # encode list of keyphrases and scores that are input to encoder
                    # append edu token to start of each head
                li_head_ids = [ self.base_tokenizer.encode( head, add_prefix_space=True, truncation=True, 
                                max_length=self.max_len_head  ) for head, score in li_headkp_score ]
                head_edus_parent_pos = self.lowest_shared_parent_node(li_edu_pos_for_head)
                li_li_head_treepos_ids = [ [head_edus_parent_pos]*len(ids) for  ids in li_head_ids  ] #creating a li of li of graph pos indexes for each keyphrase
                #li_head_kpscore =  [ [kpscore]*len(ids) for kpscore, ids in zip( li_kp_score, li_head_ids ) ]

                    #flattening and converting to tensor
                head_ids = torch.tensor( sum(li_head_ids,[]), dtype=torch.long)
                li_head_treepos_ids = sum(li_li_head_treepos_ids, [])
                li_head_treepos_ids = clamp_values( np.asarray(li_head_treepos_ids), MAX_LONG_VALUE)
                head_treepos_ids = torch.tensor( li_head_treepos_ids, dtype=torch.long)
                #head_kpscores = torch.tensor( sum( li_headkpscores, []), dtype=torch.long)

                #endregion 
            
                # region relation : encoded list of rst parent nodes information
                rst_rel_ids = torch.tensor(  [self.rst_rel_labeler.transform_patch([rel]) for rel in li_rst_rel_ ], dtype=torch.long).squeeze(dim=-1)[:10]
                rst_treepos_ids = torch.tensor( clamp_values(np.asarray(li_rst_pos_),MAX_LONG_VALUE), dtype=torch.long )[:10]
                rst_ns_ids =  torch.tensor( [ self.rst_ns_labeler.transform([ns]) for ns in li_rst_ns_], dtype=torch.long).squeeze(dim=-1)[:10]
                #endregion

                #region attention mask
                    #For rst we require 
                        # 1) bidirectional attention over each keyphrase chunk
                        # each keyphrase chunk can not attend directly to other keyphrase chunks
                        # 2) all encoder inputs can attend to rst relation tree info
                
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
                            
                # region labels
                labels = tail_ids
                # endregion
        except TimeoutInterrupt as e:
            return {
                #head
                'head_ids': None ,
                'head_treepos_ids': None,
                #'head_kpscore': head_kpscore,

                #relation: tree information
                'rst_rel_ids': None ,
                'rst_treepos_ids': None,
                'rst_ns_ids': None,

                #tail
                'tail_ids': None,
                'tail_treepos_ids': None ,
                #'tail_kp_score':tail_kpscore,

                'attention_mask_head': None,
                'attention_mask_rel': None,
                'position_ids_head': None,

                'labels':None
            }

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
            #'tail_kp_score':tail_kpscore,

            'attention_mask_head': attention_mask_head,
            'attention_mask_rel': attention_mask_rel,
            'position_ids_head': position_ids_head,

            'labels':labels.squeeze()
        }

    def tokenize_comet( self, head, rel, tail  ):
        
        # region possibly removing the to at the start of the tail
        if self.remove_to:
            if tail[:3] == "to ":
                tail = tail[3:]

        # endregion

        # region randomising pronouns (randomly replacing occurences of PersonX or PersonY with names)
            #Do this at random, so model can still predict for PersonX PersonY in dset
        if random.random() > 0.33 and self.randomize_comet_pronouns:
            head, rel, tail = self.tokenize_comet_person_randomizer(head, rel, tail)
        # endregion
        
        # region head tail rel
        head_ids = self.base_tokenizer.encode( head, add_prefix_space=True, return_tensors='pt', truncation=True, max_length=self.max_len_head)

        rels_ids = torch.tensor( self.atomic_rel_labeler.transform( [rel] ), dtype=torch.long )
        rels_ids = rels_ids

        #tail_ids = self.base_tokenizer.encode( self.token_edu + " " + tail + " " + self.token_eos , add_prefix_space=False, return_tensor='pt')
        tail_ids = self.base_tokenizer.encode( tail, add_prefix_space=True, return_tensors='pt', truncation=True, max_length=self.max_len_tail )
        #endregion 

        #region treepos_ids
            # we imagine the cskg to have the same structure as rst tree
            # so relation is at parent node. the head and tail entity are at sibling nodes
        rels_treepos_ids = torch.randint( self.rst_pos_maxidx, (1,), dtype=torch.long ) 
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

        if self.attention_type == 1:
            attention_mask = torch.ones( [enc_input_dim, enc_input_dim], dtype=torch.long  )
        if self.attention_type == 2:
            attention_mask = torch.ones( [enc_input_dim, enc_input_dim], dtype=torch.long  )
            attention_mask[ -1:, :-1 ] = 0

            # relation attention
        #_ =  attention_mask.new_zeros( [ rels_ids.shape[-1], head_ids.shape[-1] ] )
        #attention_mask[ -rels_ids.shape[-1]:,  : head_ids.shape[-1]]  = _
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

            # position of indexes which could have been incorrectly labelled as leaves
            _bool_idxs = np.where( np.array(li_edukp_pos) > self.rst_pos_maxidx ,  )[0]    
            
            # if no idxs choosen then simply use all positions
            if len(_bool_idxs) != 0:
                idxs = np.arange( len(li_edukp_pos) )[ _bool_idxs ]
            else:
                idxs = np.arange( len(li_edukp_pos) )
            
            # here we sample the edukp_pos (from the ones that could have been incorrectly labelled)
                # , with more weight placed on the deeper edukp_pos
            #pos = li_edukp_pos.pop(idxs[-1])
            li_edukp_pos_filt = [ li_edukp_pos[i] for i in idxs]
            level_of_each_edukp_pos = [ math.floor( math.log( pos+1 , 2 ) ) for pos in li_edukp_pos_filt]
            maxl = max(level_of_each_edukp_pos)
                #twice as likely to pick node from a level l+1 compared to node from level l
            weights_for_each_edukp = [ 0.5**(maxl-level) for level in level_of_each_edukp_pos  ]

            idx = random.choices(idxs, weights=weights_for_each_edukp, k=1)[0]
            pos = li_edukp_pos.pop(idx)

            # extend tree by replacing selected node with child nodes
            new_child_nodes = self.find_child_edus( pos, li_edukp_pos )
            li_edukp_pos = new_child_nodes + li_edukp_pos
            
            
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

    def setup_keyphrase_extractor(self):
        spacy_kwargs = {'disable': ['ner', 'textcat', 'parser']}
        self.spacy_model = spacy.load('en_core_web_sm', **spacy_kwargs)
        exceptions = self.spacy_model.Defaults.tokenizer_exceptions
        filtered_exceptions = {k:v for k,v in exceptions.items() if "'" not in k}
        self.spacy_model.tokenizer = spacy.tokenizer.Tokenizer(self.spacy_model.vocab, rules = filtered_exceptions)
        self.spacy_model.add_pipe('sentencizer')
        self.kp_extractor = pke.unsupervised.TextRank()
        
        self.kp_extractor.reset = MethodType(reset, self.kp_extractor)
        self.kp_extractor.pos_set = {'NOUN',  'PROPN' ,'ADJ','VERB','PRON','SYM','ADV','DET', 'PUNCT', 'PART'}
        
        self.kp_extractor.pos_to_remove_from_kp = ['ADV','DET','X']
        self.kp_extractor.tags_to_remove = ['VBZ']

    def key_phrase_extract(self, str_utterance, n_best=3, window=3, shorten=True):
        self.kp_extractor.reset()
        self.kp_extractor.load_document(input=str_utterance, language='en',normalization=None, spacy_model=self.spacy_model)
        self.kp_extractor.candidate_selection(pos=self.kp_extractor.pos_set)
        self.kp_extractor.candidate_weighting(window=window, normalized=True, pos=self.kp_extractor.pos_set)

        li_kp_score = self.kp_extractor.get_n_best(n=n_best, redundancy_removal=True)

        if shorten==True:
            li_kp_score = [ [ self.key_phrase_shortener(kp) ,score] for kp, score in li_kp_score ]
        else:
            li_kp_score = [ [ kp ,score] for kp, score in li_kp_score ]
        return li_kp_score
    
    def key_phrase_shortener(self, key_phrase):
        #This is akin to extracting the AMR of a sentence
    
        doc = self.spacy_model(key_phrase)
        
        shortened_kp = ' '.join( [ tkn.text for tkn in doc if ( tkn.pos_ not in self.kp_extractor.pos_to_remove_from_kp) and (tkn.tag_ not in self.kp_extractor.tags_to_remove)  ] )
        if len(shortened_kp) == 0:
            return key_phrase
            
        return shortened_kp

    def rst_pos_bounding(self, rst_pos):

        rst_pos = torch.where( rst_pos>self.tokenizer.rst_pos_maxidx, torch.ceil( (rst_pos-2)/2 ).long()  ,rst_pos )

        return rst_pos

    @lru_cache()
    def left_right_seq_from_root_to_edu_pos(self, edukp_pos: int):
            # from root_pos find the sequence of left/rights down the tree to each edukp_pos

        parent_pos = edukp_pos
        li_leftright_seq = [] #sequence of left-rights to get from the root to the edukp_pos

        while abs(parent_pos)!=0:
            parent_pos = (parent_pos-1 )/2
            # child node is left child node if (child_node_pos-1 /2)==int
            # child node is right child node if (child_node_pos-1 /2)=/int
            if parent_pos.is_integer():
                child_position_rel_to_parent = 'L'
            else:
                child_position_rel_to_parent = 'R'
            
            li_leftright_seq = [child_position_rel_to_parent] + li_leftright_seq

            parent_pos = math.floor(parent_pos)
        
        return li_leftright_seq

    @lru_cache()
    def edukp_pos_sort_function(self, edukp_pos: int):
        # We use a sorting function to know tree leftright order of edukp_pos
            # sort_function
            # from root_pos find the sequence of left/rights down the tree to each edukp_pos
            # Then use the 1/2, 1/4 method to calculate edukpos float representtion on flat line
            # Then retun this float
            # NOTE: intuition -> imageine binary tree is collapsed to a flatline. root=0 , left/right from parent= +/- 0.5^n

        li_leftright_seq = self.left_right_seq_from_root_to_edu_pos(edukp_pos) 
        
        # Now calculate the flattened position using the sequence of left and rights
        _ = {'L':-1, 'R':+1}
        li_flattened_pos_contributions = [  _[direction]*(0.5**(idx+1)) for idx,direction in enumerate(li_leftright_seq)  ]
        flattened_pos = sum(li_flattened_pos_contributions)

        return flattened_pos

    def smallest_spanning_subtree(self, li_edu_pos, li_rst_pos=None, li_rst_rel=None, li_rst_ns=None  ):
        """Given a list of edukp_pos, find the smallest spanning subtree that connects these edukp
            Then slice li_rst_pos, li_rst_rel, li_rst_ns to reflect this reduced tree
        """

        smallest_edu_pos = min(li_edu_pos)

        # finding root node
        # check if each other edukkp_pos can be reached from the root node
        root_found = False
        candidiate_root_node = self.parent_node_pos(smallest_edu_pos)

        while root_found==False:
            root_found = all( self.node_x_reachable_from_node_y( candidiate_root_node, pos )[0] for pos in li_edu_pos )
            
            if root_found == False:
                candidiate_root_node = self.parent_node_pos(candidiate_root_node)
            
            elif root_found == True:
                #find the nodes that
                nodes_in_minimum_spanning_tree = [ self.node_x_reachable_from_node_y( candidiate_root_node, pos )[1] 
                                                        for pos in li_edu_pos   ]
                nodes_in_minimum_spanning_tree = sum(nodes_in_minimum_spanning_tree, [])
                nodes_in_minimum_spanning_tree = list(set(nodes_in_minimum_spanning_tree))
                nodes_in_minimum_spanning_tree.sort(key=lambda edu_pos: ( self.edukp_pos_sort_function(edu_pos), edu_pos ) )

                if not nodes_in_minimum_spanning_tree == li_rst_pos:
                    bool_filter = [ pos in nodes_in_minimum_spanning_tree for pos in li_rst_pos ]
                    li_rst_rel = [ rel for rel, bool_ in zip( li_rst_rel, bool_filter) if bool_ ] 
                    li_rst_pos = [ pos for pos, bool_ in zip( li_rst_pos, bool_filter) if bool_]
                    li_rst_ns =  [ ns for ns, bool_ in zip( li_rst_ns, bool_filter) if bool_]
        
        # clipping the length of these
            # Removing all nodes that are below a certain depth
            # This is to match the rst pos embedding  embedding layer behaviour which only 
            # models rst_pos up to a certain level
        if max(li_rst_pos) > self.rst_pos_maxidx:
            l = [ [rel, pos, ns] for rel, pos, ns in zip(li_rst_rel, li_rst_pos, li_rst_ns) if self.node_level(pos)<=self.rst_pos_maxidx ] 
            li_rst_rel, li_rst_pos, li_rst_ns = [ list(t) for t in zip( *l ) ]

        return li_rst_pos, li_rst_rel, li_rst_ns

    def lowest_shared_parent_node(self, li_edupos ):
        """Given a list of edukp_pos, find the smallest spanning subtree that connects these edukp
            Then slice li_rst_pos, li_rst_rel, li_rst_ns to reflect this reduced tree
        """
        smallest_edu_pos = min( li_edupos )

        # finding root node
        # check if each other edukkp_pos can be reached from the root node
        root_found = False
        candidiate_root_node = self.parent_node_pos(smallest_edu_pos)

        while root_found==False:
            root_found = all( self.node_x_reachable_from_node_y( candidiate_root_node, pos )[0] for pos in li_edupos )
            
            if root_found == False:
                candidiate_root_node = self.parent_node_pos(candidiate_root_node)
        
        return candidiate_root_node
        
    @lru_cache(maxsize=4000)
    def parent_node_pos(self, node_pos):
        """Gets the parent node position for any node_pos

        Args:
            node_pos ([type]): [description]

        Returns:
            [type]: [description]
        """
        if node_pos == 0:
            return node_pos

        #parent_node = int( math.ceil( ( node_pos -2 ) /2 ) )    
        
        parent_node =  (node_pos -1) // 2 
        #Integer division to avoid erroneos approximations
        #   # it automatically rounds down
        #   therefore we use node_pos -1 
        #   $ therefore we do'nt math.ceil anymore  

        return parent_node
    
    @lru_cache()
    def node_level(self, node_pos):
        val = math.floor( math.log( node_pos+1 , 2 ) )
        
        return val

    @lru_cache(maxsize=4000)
    def node_x_reachable_from_node_y(self, nodex, nodey):
        """returns (bool, [sequence showing path from nodex down tre to nodey])

            Args:
                nodex ([type]): [description]
                nodey ([type]): [description]

            Raises:
                ValueError: [description]
                NotImplementedError: [description]
                NotImplementedError: [description]
                Exception: [description]

            Returns:
                [type]: [description]
        """

        #Find path by calculating the sequence of Left and Rights between a nodey and node x
        #Convert this path to the nodes that it represents
        #We assume node x is reachable until we know it is not.

        parent_path = []
        curr_node = nodey
        reachable=True

        if nodex == nodey:
            return ( True, [] )

        while reachable==True and nodex not in parent_path:
        
            curr_node = self.parent_node_pos(curr_node)
            
            parent_path = [curr_node] + parent_path

            # Check if new curr node is on same level as nodex.
                # if yes At which point we know nodex is not directly reachable if curr_node is not nodex
            if self.node_level(curr_node) == self.node_level(nodex):
                reachable = (curr_node == nodex)

        
        return reachable, parent_path
        
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
                    loss_weight_rst = 0.5,
                    loss_weight_comet = 0.5,
                    randomize_comet_pronouns = True,
                    remove_to = False,
                    tag='',
                    *args,
                    **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.gpus =  gpus
        self.model = COMERST( **model_params )
        self.randomize_comet_pronouns = randomize_comet_pronouns
        self.model.tokenizer.randomize_comet_pronouns = self.randomize_comet_pronouns
        self.remove_to = remove_to
        self.model.tokenizer.remove_to = self.remove_to
        
        self.mode = mode
        self.workers = workers
        self.data_splits = data_splits
        self.loss_weight_rst = loss_weight_rst
        self.loss_weight_comet = loss_weight_comet
        
        
        if self.mode in ['train_new','train_cont','test']:
            self.dir_data_rst = dir_data_rst
            self.dir_data_atomic2020 = dir_data_atomic2020
            self.create_data_loaders(  )
            
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
            self.save_hyperparameters({ **train_params_to_save, **model_params_to_save})

            #self.save_hyperparameters()
            # save_hparams_to_yaml(f"{self.logger.log_dir}/hparams.yaml",
            #                   self.hparams)

            self.inference_samples = list( islice( self.inference_dl, 10 ) )
            del self.inference_dl

        if self.mode in ['inference']:
            self.eval() 
            self.freeze() 

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data_rst', default="./dataset_keyphrase_v2", help="Relative directory path of rst data")
        parser.add_argument('--dir_data_atomic2020', default="./dataset_atomic2020", help="Relative directory path for atomic2020data")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('-me','--max_epochs', default=28, type=int)
        parser.add_argument('-agb','--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('-bs','--batch_size', default=100, type=int)
        parser.add_argument('-lr','--learning_rate', default=5e-4, type=float)
        parser.add_argument('--warmup_proportion', default=0.25)
        parser.add_argument('--workers', default=12, type=int) 
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','train_cont','test','inference'])
        parser.add_argument('--splits', default={'train':0.6,'val':0.2,'test':0.2}, required=False, type=str )
        parser.add_argument('--version', default=0,required=False, type=int, help="The Experimental Versioning for this run" )
        parser.add_argument('--precision', default=16,required=False, type=int, help="Precision to use", choices=[16,32] )
        parser.add_argument( '-lwr','--loss_weight_rst',default=0.5, required=False, type=float)
        parser.add_argument( '-lwc','--loss_weight_comet',default=0.5, required=False, type=float)
        parser.add_argument( '--rcp', '--randomize_comet_pronouns', default=True, type= lambda x : bool(int(x)), help="remove all 'to' from the start of the comet phrases" )
        parser.add_argument( '-rmto', '--remove_to', default=False, type= lambda x : bool(int(x)) )
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
                    'learning_rate','precision','splits','tag','loss_weight_rst','loss_weight_comet']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'base_tokenizer_name','model_name','max_len_head','max_len_tail'
                    ,'scale_grad_by_freq','filter_atomic_rels','max_edu_nodes_to_select', 'relation_embedding']} )
                
                mparams_json = {k:json.loads(v) for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'dict_embed_mnorms'] }
        
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
                    'learning_rate','precision','splits','loss_weight_rst','loss_weight_comet']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'base_tokenizer_name','loss_type','model_name','max_len_head','max_len_tail']} )
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
            patience=5,
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
                        #accelerator='ddp2', amp_level='O2',
                        accelerator=accelerator,
                        #limit_train_batches =5,
                        #limit_val_batches = 5,
                        val_check_interval=0.3,
                        num_sanity_val_steps=0, 
                        #overfit_batches=25,
                        reload_dataloaders_every_epoch=True,
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
            scores_dict = yaml.load( open(checkpoint_yaml_file,"r"), Loader = yaml.FullLoader ) #key= ckptpath, value = val_loss
            best_ckpt_path = min(scores_dict, key=scores_dict.get)

            if os.path.exists(best_ckpt_path) == False:
                root_dir = Path(__file__).resolve().parents[4]
                best_ckpt_path = os.path.join( str(root_dir), best_ckpt_path[ best_ckpt_path.index('mastering-conversation'): ] )

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
    def load_comerst(model_name="COMERST", model_version=0, device="cuda:0" ):
        # Loading in NLG model
        checkpoint = TrainingModule.get_ckpt_file(f'./models/{model_name}/version_{model_version}/checkpoints')

        # Getting tparams
        tparams = {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
            'batch_size', 'learning_rate','precision','splits',
            'tag','loss_weight_rst','loss_weight_comet','randomize_comet_pronouns','remove_to']}

        tparams['mode'] = 'inference'

        mparams =  {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
            'base_tokenizer_name','loss_type','model_name','max_len_head','max_len_tail',
            'frst_version','scale_grad_by_freq','max_edu_nodes_to_select','filter_atomic_rels',
            'relation_embedding','attention_type','freeze_embeds','rst_pos_embed_type']}
        
        mparams_json = {k:json.loads(v) for k,v in checkpoint['hyper_parameters'].items() if k in [] }

        mparams =  {**mparams, **mparams_json}
                    
        # Loading Training Module
        training_module = TrainingModule(**tparams, model_params=mparams )
        training_module.load_state_dict(checkpoint['state_dict'])
        model = training_module.model

        # Deleting checkpoints to free up GPU space
        del checkpoint
        torch.cuda.empty_cache()
          
        #if torch.cuda.is_available():
        if device != 'cpu' and torch.cuda.is_available():
            model = model.to(device)
        
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
        loss = None
        
        if 'comet' in model_output:
            lm_loss_comet = model_output['comet'].loss * self.loss_weight_comet
            loss = lm_loss_comet
        
        if 'rst' in model_output:
            lm_loss_rst = model_output['rst'].loss *self.loss_weight_rst
            if loss == None:
                loss = lm_loss_rst
            else:
                loss = loss + lm_loss_rst
        
        loss_key = f"{step_name}_loss"
        
        output = {}
        if step_name == 'train':
            output["loss"] = loss

        else:       
            self.log( loss_key, loss)
            output[ loss_key ]=loss


        if 'rst' in model_output:
            self.log( loss_key+"_rst", lm_loss_rst )
        elif 'comet' in model_output:
            self.log( loss_key+"_comet", lm_loss_comet )
            # here output dictionary instead
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

        if step_name == "val" and _get_rank() == 0 :
            bad_words = ['"']
            bad_words_ids = [ self.model.tokenizer.base_tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in bad_words ]
            
            generation_kwargs = {'num_beams':1, 'temperature':1.2, 'repitition_penalty':1.0, 
                                'early_stopping':False, 'do_sample':False, 'no_repeat_ngram_size':3, 
                                'num_return_sequences':1, 'bad_words_ids':bad_words_ids,
                                'min_length':3, 'max_length':20 } #'max_length':30,'min_length':4

            # At end of validation loop produce some quantitative examples of model's performance

            # Making directory for inference testing results
            dir_infer = os.path.join(self.trainer.log_dir,"inference")
            if not os.path.exists(dir_infer):
                os.makedirs(dir_infer,exist_ok=True)
        
            # data holders
            li_comet_heads = []
            li_comet_rels = []
            li_comet_preds = []
            li_comet_tails = []

            li_rst_heads = []
            li_rst_rels = []
            li_rst_preds = []
            li_rst_tails = []

            for idx, batch in enumerate( self.inference_samples ):
                
                # COMET testing
                batch_comet = batch['comet']
                heads_comet = self.model.tokenizer.base_tokenizer.decode( batch_comet['head_ids'][0], skip_special_tokens=True  ).strip() 
                rels_comet = self.model.tokenizer.atomic_rel_labeler.inverse_transform( batch_comet['rels_ids'].cpu().numpy() )[0].tolist()
                tails_comet =  self.model.tokenizer.base_tokenizer.decode( batch_comet['tail_ids'][0] , skip_special_tokens=True  ).strip() 

                preds = self.model.generate_from_dloaderbatch( batch_comet, comet_or_rst="comet", generation_kwargs=generation_kwargs )[0]
                
                li_comet_heads.append(heads_comet.strip())
                li_comet_rels.append(rels_comet)
                li_comet_tails.append(tails_comet.strip())
                li_comet_preds.append(preds)

                # RST Testing
                    #TODO: update this 
                # batch_rst = batch['rst']

                # preds = self.model.generate_from_dloaderbatch( batch_rst, comet_or_rst="rst",
                #     generation_kwargs=generation_kwargs )[0]
                # heads_rst = self.model.tokenizer.base_tokenizer.decode( batch_rst['head_ids'][0],  skip_special_tokens=False ).split('</s><s>') 
                # heads_rst = [ _.strip("<s>").strip("</").strip() for _ in heads_rst ]
                # heads_treepos_rst = batch_rst['head_treepos_ids'].cpu().tolist()[0]
                # heads_treepos_rst = [ key for key, group in groupby(heads_treepos_rst) ]
                

                # rels_ids_rst = self.model.tokenizer.rst_rel_labeler.inverse_transform_patch( batch_rst['rst_rel_ids'].cpu().squeeze(dim=0)  ).tolist()


                # rels_treepos_rst = batch_rst['rst_treepos_ids'][0].tolist()
                # rels_ns_rst = self.model.tokenizer.rst_ns_labeler.inverse_transform(batch_rst['rst_ns_ids'].cpu().squeeze(dim=0)).tolist()

                # tails_rst = self.model.tokenizer.base_tokenizer.decode( batch_rst['tail_ids'][0], skip_special_tokens=True ).strip()
                # tail_treepos_ids_rst = batch_rst['tail_treepos_ids'].cpu().numpy()[0][0].tolist()

                # li_rst_heads.append( [ {pos:head } for pos,head in zip(heads_treepos_rst, heads_rst)  ] )
                # li_rst_rels.append( [ {pos:rel} for pos, rel,ns in zip(rels_treepos_rst, rels_ids_rst, rels_ns_rst) ] )
                # li_rst_tails.append( {tail_treepos_ids_rst:tails_rst.strip()} )
                # li_rst_preds.append( {tail_treepos_ids_rst:preds.strip() }  )


            # Adding records from this epoch to files
            for idx in range(len(self.inference_samples)):
                if 'comet' in batch:
                    fp_comet = os.path.join(dir_infer, f"example_comet_{idx:03d}.csv")
                    
                    # comet- If file for example idx does not exists we add the true observed records
                    if not os.path.exists(fp_comet):
                        
                        df_comet = pd.DataFrame(columns=[ 'epoch', 'head','rels','tail', 'preds'])                    
                        
                        head = li_comet_heads[idx]
                        rels = li_comet_rels[idx]
                        tail = li_comet_tails[idx]
                        preds = li_comet_preds[idx]
                                            
                        datum = { 'epoch': 0,
                                    'head': head,
                                    "rels": rels,
                                    "tail":tail,
                                    "preds":preds }
                    
                        df_comet = df_comet.append(datum, ignore_index=True)
                        df_comet.to_csv( fp_comet, index=False)

                    # comet - adding to preds
                    df_comet = pd.read_csv(fp_comet)    
                    datum_comet = {
                        'epoch':df_comet['epoch'].max()+1,
                        'head': '',
                        'rels':'',
                        'tail':'',
                        'preds':li_comet_preds[idx] }

                    df_comet = df_comet.append(datum_comet, ignore_index=True)
                    df_comet.to_csv( fp_comet, index=False)

                # if 'rst' in batch:
                #     fp_rst = os.path.join(dir_infer, f"example_rst_{idx:03d}.csv")
                #     # rst - If file for example idx does not exists we add the true observed records
                #     if not os.path.exists(fp_rst):
                        
                #         df_rst = pd.DataFrame(columns=[ 'epoch', 'head','rels','tail', 'preds'])                    
                        
                #         head = li_rst_heads[idx]
                #         rels = li_rst_rels[idx]
                #         tail = li_rst_tails[idx]
                #         preds = li_rst_preds[idx]
                                            
                #         datum = { 'epoch': 0,
                #                     'head': head,
                #                     "rels": rels,
                #                     "tail":tail,
                #                     "preds":preds }
                    
                #         df_rst = df_rst.append(datum, ignore_index=True)
                #         df_rst.to_csv( fp_rst, index=False)

                #     # rst - adding to preds
                #     df_rst = pd.read_csv(fp_rst)    
                #     datum_rst = {
                #         'epoch':df_rst['epoch'].max()+1,
                #         'head': '',
                #         'rels':'',
                #         'tail':'',
                #         'preds':li_rst_preds[idx] }

                #     df_rst = df_rst.append(datum_rst, ignore_index=True)
                #     df_rst.to_csv( fp_rst, index=False)

    def create_data_loaders(self, shuffle=False, **kwargs ):
        
        dg = DataLoaderGenerator(self.dir_data_rst, self.dir_data_atomic2020,
                self.batch_size, self.model.pad_values,
                self.model.tokenizer, 
                workers=self.workers, mode=self.mode, split=self.data_splits,
                #dset_blend=self.dset_blend
                )

        
        if "train" in self.mode:
            self.train_dl = dg.prepare_dataloader_combined(True, 'train', loss_weight_rst=self.loss_weight_rst, loss_weight_comet=self.loss_weight_comet)
            self.val_dl = dg.prepare_dataloader_combined(True, 'val', loss_weight_rst=self.loss_weight_rst, loss_weight_comet=self.loss_weight_comet)
            self.test_dl = dg.prepare_dataloader_combined(False, 'test', loss_weight_rst=self.loss_weight_rst, loss_weight_comet=self.loss_weight_comet)
            self.inference_dl = dg.prepare_dataloader_combined(True, 'inference', loss_weight_rst=self.loss_weight_rst, loss_weight_comet=self.loss_weight_comet)        
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

        ds_size = max( [ len(dl) for dl in self.train_dl.values()] ) // max(1,self.gpus)
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
            'warmup_proportion','tag','version','loss_weight_rst','loss_weight_comet',
            'randomize_comet_pronouns','remove_to']
        
        params = {
            k:self.__dict__[k] for k in keys if k in self.__dict__.keys()
        }

        #json_keys = ['data_splits','dset_blend']

        json_keys = ['data_splits']
        
        jparams = {
            k:ujson.dumps(self.__dict__[k]) for k in json_keys if k in self.__dict__.keys()
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
                    **kwargs):
        
        self.dir_data_rst = dir_data_rst
        self.dir_data_atomic2020 = dir_data_atomic2020
        self.tokenizer = tokenizer
        self.splits = splits
        self.randomize_comet_pronouns = self.tokenizer.randomize_comet_pronouns
        self.remove_to = self.tokenizer.remove_to
        

        self.bs = batch_size
        self.workers_rst = int(  workers/2 ) #if workers==0 else max( int( round( workers * (3/4), 0 ) ), 1 )
        self.workers_atomic = int( workers/2 ) #if workers==0 else max( workers - self.workers_rst, 1 )
        self.mode = mode
        self.pad_values = pad_values
        
    def prepare_dataloader_combined(self, shuffle=False, 
        split_name='train', **kwargs):

        output = {}
        if kwargs.get('loss_weight_rst',0.5) != 0.0:
            dataloder_rst = self.prepare_dloader_rst(shuffle, split_name)
            output['rst'] = dataloder_rst
        if kwargs.get('loss_weight_comet',0.5) != 0.0:
            dataloader_atomic2020 = self.prepare_dloader_atomic2020(shuffle, split_name)
            output['comet'] = dataloader_atomic2020
        

        if split_name in ["val","test","inference"] :
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
            pin_memory=True
        
        elif split_name == 'val':
            line_starts = [ int(fs*self.splits['train']) for fs in files_sizes  ]
            line_ends = [ ls+int(fs*self.splits['val']) for ls,fs in zip(line_starts, files_sizes)  ]
            shuffle = False
            pin_memory=False

        elif split_name == 'test':
            line_starts = [ int(fs*(1-self.splits['test']) ) for fs in files_sizes  ]
            line_ends = files_sizes
            shuffle = False
            pin_memory=False

        elif split_name == 'inference':
            line_starts = [ random.randrange( int(fs*(1-self.splits['test'])), fs) for fs in files_sizes  ]
            line_ends =  files_sizes
            shuffle = False
            pin_memory=False

        li_dsets = [ SingleDataset_rst_v2(_f, self.tokenizer, line_start, line_end) 
                        for _f, line_start, line_end in zip(fns, line_starts, line_ends) ]
        
        # remove invalid dataset
        li_dsets = [dset for dset in li_dsets if dset.valid_dset==True]

        if split_name == 'inference':
            li_dsets = random.sample(li_dsets,min(20,len(li_dsets)) )
            bs = 1
        else:
            bs = self.bs

        concat_dset = torch.utils.data.ConcatDataset(li_dsets)
        
        dataloader = torch.utils.data.DataLoader(concat_dset, batch_size=bs,
            shuffle=shuffle, num_workers=self.workers_rst, 
            collate_fn=lambda batch: utils.default_collate_pad(batch, self.pad_values),
            pin_memory=pin_memory, prefetch_factor=1, timeout=30 )

        #TODO: change defualt collate to allow padding of elements for batching
        return dataloader

    def prepare_dloader_atomic2020(self, shuffle,
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
            shuffle = shuffle
            fn = os.path.join( self.dir_data_atomic2020,"train_v2.csv" )
            drop_duplicates = False
            randomize_comet_pronouns = self.randomize_comet_pronouns

        elif split_name == 'val':
            fn = os.path.join( self.dir_data_atomic2020,"dev_v2.csv" )
            shuffle = shuffle
            drop_duplicates = False
            randomize_comet_pronouns = self.randomize_comet_pronouns

        elif split_name == 'test':
            fn = os.path.join( self.dir_data_atomic2020,"test_v2.csv" )
            shuffle = shuffle
            drop_duplicates = False
            randomize_comet_pronouns = self.randomize_comet_pronouns

        elif split_name == 'inference':
            fn = os.path.join( self.dir_data_atomic2020,"test_v2.csv" )
            shuffle = shuffle
            drop_duplicates = True
            randomize_comet_pronouns = self.randomize_comet_pronouns

        dset = SingleDataset_atomic2020(fn, self.tokenizer,  drop_duplicates=drop_duplicates )
                
        if split_name == 'inference':
            bs = 1
        else:
            bs = self.bs #*self.dset_blend['ATOMIC2020']

        dataloader = torch.utils.data.DataLoader(dset, batch_size=bs,
            shuffle=shuffle, num_workers= self.workers_atomic,
            collate_fn=lambda batch: utils.default_collate_pad( batch, self.pad_values),
            pin_memory=True, prefetch_factor=1, timeout=15 )
        
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
            self.valid_dset = True
            # filtering out long lines
            self.data  = self.data.loc[ (self.data['txt_preproc'].str.len() <= 225 ) & (~ self.data['li_edus'].isnull()) ]

            # only select columns that have at least two non null phrases in li_edus

            
        else:
            self.valid_dset = False

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

        li_dict_posname_likpscore = ujson.loads( datum['li_dict_posname_likpscore'].values[0] )
        
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

class SingleDataset_rst_v2(torch.utils.data.Dataset):
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
                        
        # filtering out lines which relate to long texts        
        if 'dict_pos_edu' in self.data.columns:
            self.valid_dset = True
            
            self.data  = self.data[ ~self.data['dict_pos_edu'].isnull() ]
            
        else:
            self.valid_dset = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index, pad_utterance=True):

        li_rst, dict_pos_edu = self.getitem_extract_datum(index)
        
        encoded = self.tokenizer.tokenize_rst_v2( li_rst = li_rst ,
                                                     dict_pos_edu = dict_pos_edu )

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
        
        li_rst = json.loads( datum['rst'].values[0] )
        dict_pos_edu = json.loads( datum['dict_pos_edu'].values[0] )

        # li_rst = json.loads( datum['rst'].values[0] )
        # dict_pos_edu = json.loads( datum['dict_pos_edu'].values[0] )
        

        return li_rst, dict_pos_edu
    
class SingleDataset_atomic2020(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

    """
    #TODO: think of way to balance ATOMIC and RST contribution
    #TODO: find research on multi task learning well
    def __init__(self, file_path, tokenizer, drop_duplicates=False, sample_size=None ):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.drop_duplicates = drop_duplicates #used for test set. In the case our model uses greedy decoding. Filters on duplicate head and tails
        self.randomize_comet_pronouns = self.tokenizer.randomize_comet_pronouns
        

        #TODO: remove nrows
        self.data = pd.read_csv(self.fp 
            #,nrows=100
            )
        
        if self.drop_duplicates:
            self.data = skl_shuffle(self.data)
            self.data = self.data.drop_duplicates(subset=['head','relation'],
                keep='first',ignore_index=True)
        
        if self.tokenizer.filter_atomic_rels == True:
            self.data = self.data.loc[ self.data.relation.str.strip('"').str.contains('|'.join(self.tokenizer.atomic_rel_li),regex=True) ]

        if sample_size != None:
            assert type(sample_size) == int
            self.data = skl_shuffle(self.data)
            self.data = self.data[:sample_size]


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
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

# 1 - Baseline
# CUDA_VISIBLE_DEVICES=0,1 python3 train_comerst.py -lwr 0.5 -lwc 0.5 -mlh 20 -mlt 20 -bs 260 -ments 5 -far 0 -agb 1 --gpus 2 --workers 12 --version 1 --precision 16 --mode train_new -lr 3e-4 -me 20 --tag "baseline"

# 2 - Baseline w\ reduced feature set size
# CUDA_VISIBLE_DEVICES=1 python3 train_comerst.py -lwr 0.5 -lwc 0.5 -mlh 20 -mlt 20 -bs 260 -ments 5 -far 1 -agb 1 --gpus 1 --workers 12 --version 2 --precision 16 --mode train_new -lr 1e-4 -me 20 --tag "reduced feature set size"

# 3 - Baseline w\ reduced feature set size and starting variance of 1.5 and max_norm set to r_pos, r_ns, rels 2 2 5
# CUDA_VISIBLE_DEVICES=1 python3 train_comerst.py -isv 1.5 -dem {\"r_pos\":2, \"r_ns\":2, \"rels\":5} -lwr 0.5 -lwc 0.5 -mlh 20 -mlt 20 -bs 260 -ments 5 -far 1 -agb 1 --gpus 1 --workers 12 --version 3 --precision 16 --mode train_new -lr 3e-4 -me 30 --tag "Baseline w\ reduced feature set size and starting variance of 1.5 and max_norm set to r_pos, r_ns, rels 2 2 5"

# 4 - Baselines w\ reduced feature set size and starting variance of 1.5 an max_norm to 3 3 4 to r_pos, r_ns, rels 3 3 3
# CUDA_VISIBLE_DEVICES=3 python3 train_comerst.py -isv 1.5 -dem {\"r_pos\":3, \"r_ns\":3, \"rels\":3} -1wr 0.5 -lwc 0.5 -mlh 20 -mlt 20 -bs 260 -ments 5 -far 1 -agb 1 --gpus 1 --workers 12 --version 4 --precision 16 --mode train_new -lr 3e-4 -me 30 --tag "Baseline w\ reduced feature set size and starting variance of 1.5 and max_norm set to r_pos, r_ns, rels 3 3 3"

# 5 - Map the relations from RST onto the COMET embedding space - using embedding matrix to map from rst rels to comet rels
# CUDA_VISIBLE_DEVICES=3 python3 train_comerst.py -isv 1.0 -rmto 1 -at 2 -re hierarchical1 -1wr 0.5 -lwc 0.5 -mlh 20 -mlt 20 -bs 260 -ments 5 -far 1 -agb 1 --gpus 1 --workers 6 --version 5 --precision 16 --mode train_new -lr 5e-5 -me 40 --tag "Map the relations from RST onto the COMET embedding space - using embedding matrix to map from rst rels to comet rels, with fixed attention and 'to' removed from comet"

# 6 - Map the relations from RST onto the COMET embedding space - using embedding matrix, slp and tanh layer to map
# CUDA_VISIBLE_DEVICES=5 python3 train_comerst.py -fe 1 -sgbf 1 -isv 0.005 -rmto 1 -at 2 -re hierarchical2 -1wr 0.4 -lwc 0.6 -mlh 20 -mlt 20 -bs 260 -ments 5 -far 1 -agb 1 --gpus 1 --workers 6 --version 6 --precision 16 --mode train_new -lr 5e-5 -me 40 --tag "Map the relations from RST onto the COMET embedding space - using embedding matrix, slp and tanh layer to map, with fixed attention and 'to' removed from comet"

# 7 - Same as #3, but only trained on RST data
# CUDA_VISIBLE_DEVICES=1 python3 train_comerst.py -isv 1.5 -dem {\"r_pos\":2, \"r_ns\":2, \"rels\":5} -lwr 1.0 -lwc 0.0 -mlh 20 -mlt 20 -bs 360 -ments 5 -far 1 -agb 1 --gpus 1 --workers 12 --version 7 --precision 16 --mode train_new -lr 5e-4 -me 30 --tag "Baseline w\ reduced feature set size and starting variance of 1.5 and max_norm set to r_pos, r_ns, rels 2 2 5. Only trained on RST data"

# 8 - Same as #3, but only trained on COMET data
# CUDA_VISIBLE_DEVICES=4 python3 train_comerst.py -isv 1.5 -dem {\"r_pos\":2, \"r_ns\":2, \"rels\":5} -lwr 0.0 -lwc 1.0 -mlh 20 -mlt 20 -bs 540 -ments 5 -far 1 -agb 1 --gpus 1 --workers 12 --version 8 --precision 16 --mode train_new -lr 5e-4 -me 30 --tag "Baseline w\ reduced feature set size and starting variance of 1.5 and max_norm set to r_pos, r_ns, rels 2 2 5. Only trained on COMET data"

# 9 - Same as #3, but masking fixed and attention method used where relation embeddings don't attend to text embeddings 
# CUDA_VISIBLE_DEVICES=4 python3 train_comerst.py -isv 1.3 -rmto 1 -at 2 -dem "{\"r_pos\":2, \"r_ns\":2, \"rels\":5}" -lwr 0.5 -lwc 0.5 -mlh 20 -mlt 20 -bs 240 -ments 5 -far 1 -agb 1 --gpus 1 --workers 6 --version 9 --precision 16 --mode train_new -lr 1e-4 -me 40 --tag "Baseline w\ reduced feature set size and starting variance of 1.5 and max_norm set to r_pos, r_ns, rels 2 2 5. masking fixed and attention method used where relation embeddings don't attend to text embeddings, with fixed attention and 'to' removed from comet"

# 10 - Same as #9, but using full comet relations set 
# CUDA_VISIBLE_DEVICES=1 python3 train_comerst.py -isv 1.3 -rmto 1 -at 2 -dem "{\"r_pos\":2, \"r_ns\":2, \"rels\":5}" -lwr 0.5 -lwc 0.5 -mlh 20 -mlt 20 -bs 240 -ments 5 -far 0 -agb 1 --gpus 1 --workers 6 --version 10 --precision 16 --mode train_new -lr 1e-4 -me 40 --tag "Baseline w\ reduced feature set size and starting variance of 1.5 and max_norm set to r_pos, r_ns, rels 2 2 5. masking fixed and attention method used where relation embeddings don't attend to text embeddings and full comet relations, with fixed attention and 'to' removed from comet"

# 11 - Same as #10, but no maxnorms, lower initial starting variance,  no scaling embeddings by frequency
# CUDA_VISIBLE_DEVICES=1 python3 train_comerst.py -sgbf 0 -isv 0.005 -rmto 1 -at 2 -lwr 0.5 -lwc 0.5 -mlh 20 -mlt 20 -bs 240 -ments 5 -far 0 -agb 1 --gpus 1 --workers 6 --version 11 --precision 16 --mode train_new -lr 5e-4 -me 40 --tag "Baseline w\ reduced feature set size and starting variance of 1.5 and max_norm set to r_pos, r_ns, rels 2 2 5. masking fixed and attention method used where relation embeddings don't attend to text embeddings and full comet relations, with fixed attention and 'to' removed from comet"

# 13 - Same as #10, but medium and embeddings frozen
# CUDA_VISIBLE_DEVICES=3 python3 train_comerst.py  --gpus 1 --version 13 -fe 1 -sgbf 1 -isv 0.005 -rmto 1 -at 2 -lwr 0.5 -lwc 0.5 -mlh 20 -mlt 20 -bs 260 -ments 5 -far 0 -agb 1 --workers 6  --precision 16 --mode train_new -lr 5e-4 -me 40 --tag "Baseline w\ reduced feature set size . masking fixed and attention method used where relation embeddings don't attend to text embeddings and full comet relations, with fixed attention and 'to' removed from comet"

# 14 - Same as #10, but medium and embeddings frozen weighted more towards comet
# CUDA_VISIBLE_DEVICES=3 python3 train_comerst.py  --gpus 1 --version 13 -fe 1 -sgbf 1 -isv 0.005 -rmto 1 -at 2 -lwr 0.35 -lwc 0.65 -mlh 20 -mlt 20 -bs 260 -ments 5 -far 0 -agb 1 --workers 6  --precision 16 --mode train_new -lr 5e-4 -me 40 --tag "Baseline w\ reduced feature set size . masking fixed and attention method used where relation embeddings don't attend to text embeddings and full comet relations, with fixed attention and 'to' removed from comet"

# 6 - Same as 6 but fixed embedding 
# CUDA_VISIBLE_DEVICES=5 python3 train_comerst.py -isv 0.005 -fe 1 -sgbf 1 -isv 0.005 -rmto 1 -at 2 -re hierarchical2 -1wr 0.4 -lwc 0.6 -mlh 20 -mlt 20 -bs 260 -ments 5 -far 1 -agb 1 --gpus 1 --workers 6 --version 6 --precision 16 --mode train_new -lr 5e-5 -me 40 --tag "Map the relations from RST onto the COMET embedding space - using embedding matrix, slp and tanh layer to map, with fixed attention and 'to' removed from comet"

# 101 - New model. New Keyphrase v2 dataset and With novel position embedding to allow all positions to be embeded
# CUDA_VISIBLE_DEVICES=1 python3 train_comerst.py -rpet 2 -fe 1 -sgbf 1 -rmto 1 -at 2 -re hierarchical2 -1wr 0.5 -lwc 0.5 -mlh 20 -mlt 20 -bs 280 -far 0 -agb 1 --gpus 1 --workers 12 --version 101 --precision 16 --mode train_new -lr 1e-4 -me 20 --tag "New model. New Keyphrase v2 dataset and With novel position embedding to allow all positions to be embeded"
