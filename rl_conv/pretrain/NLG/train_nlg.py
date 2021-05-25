import os
from typing import Optional, Tuple

#os.environ["NCCL_DEBUG"]="INFO"
#os.environ["NCCL_DEBUG_SUBSYS"]="ALL"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["NCCL_P2P_LEVEL"] = "3"
#os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_SOCKET_IFNAME'] =  'lo' 
#os.environ['NCCL_SOCKET_IFNAME'] =  'enp3s0'
#os.environ['CUDA_LAUNCH_BLOCKING']="1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing as mp

import numpy as np
import warnings
import sklearn
import gc

import glob
import pandas as pd
import json
from functools import lru_cache
from typing import List
import pickle

from itertools import cycle, islice
from torch.utils.data._utils.collate import default_convert, default_collate
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import CrossEntropyLoss

from sklearn import preprocessing as sklp

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import utils_nlg as utils
import random 

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from transformers import Adafactor
from transformers.generation_beam_search import BeamHypotheses

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.core.decorators import auto_move_data
from collections import OrderedDict
import yaml
import ast
import types

from copy import deepcopy

from itertools import permutations, combinations, combinations_with_replacement
from typing import Optional, Callable, Union, Optional, List, Iterable

from transformers.generation_logits_process import (
    TopKLogitsWarper,
    TopPLogitsWarper,
)

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

def _monitor_candidates(self, trainer):
    ckpt_name_metrics = deepcopy(trainer.logger_connector.logged_metrics)
    ckpt_name_metrics.update(trainer.logger_connector.progress_bar_metrics)
    ckpt_name_metrics.update(trainer.logger_connector.callback_metrics)

    return ckpt_name_metrics


#Monkey patching the forward on gpt
def forward(
    self,
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    input_embeds=None,
    position_embeds=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **kwargs):
    
    input_ids = None if input_ids is not None else input_ids # Our model should ignore any input_ids entered into the model

    #self.register_buffer("position_ids",position_ids)

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and input_embeds is not None:
        raise ValueError("You cannot specify both input_ids and input_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif input_embeds is not None:
        input_shape = input_embeds.size()[:-1]
        batch_size = input_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or input_embeds")

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1]) #.to(input_embeds.device)

    if position_ids is not None and input_embeds is not None:
        raise ValueError("You cannot specify both input_ids and input_embeds at the same time")
    elif position_ids is not None:
        input_shape = position_ids.size()
        position_ids = position_ids.view(-1, input_shape[-1])
    elif position_embeds is not None:
        pass
    else:
        raise ValueError("You have to specify either input_ids or input_embeds")

    if past_key_values is None:
        past_length = 0
        past_key_values = [None] * len(self.h)
    else:
        past_length = past_key_values[0][0].size(-2)
    
    if position_ids is None and position_embeds is None:

        device = input_ids.device if input_ids is not None else input_embeds.device
        position_ids = torch.arange(past_length, input_shape[-1] + past_length,
            dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # Attention mask.
    if attention_mask is not None:
        attention_mask = attention_mask[:, None, :, :] # adding head dimension
        attention_mask = attention_mask.type_as(input_embeds) # fp16 compatibility
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

    if input_embeds is None:
        input_embeds = self.wte(input_ids)

    if position_embeds is None:
        position_embeds = self.wpe(position_ids)    

    hidden_states = input_embeds + position_embeds

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
        return {
             k:v for k,v in zip(
                 ['hidden_states', 'presents', 'all_hidden_states', 'all_self_attentions'],
                 [hidden_states, presents, all_hidden_states, all_self_attentions]
             ) if v is not None
        }

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    ) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    return logits

def calc_banned_ngram_tokens(prev_input_ids, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]

    generated_ngrams = [{} for _ in range(num_hypos)]

    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    
    return banned_tokens

def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_tokens):
            # if bad word tokens are longer than prev tokens they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice, banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens

def set_scores_to_inf_for_banned_tokens(scores: torch.Tensor, banned_tokens: List[List[int]]) -> None:
    """Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be
    a list of list of banned tokens to ban in the format [[batch index, vocabulary position],...]
        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return
    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))
    # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
    # [ 0  1  1 ]
    # [ 0  0  0 ]
    # [ 1  0  0 ]

    banned_mask = torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    scores.masked_fill_(banned_mask, -float("inf"))

class NLG(nn.Module):
    """NLG unit
    """

    def __init__(self, 
        base_model_name= 'distilgpt2', model_name="NLG",
        reset_base_transformer=True,
        fda=False, frst=True, ftopic=True, frst_version=0,
        max_input_len=264 , freeze_pretrained=0,
        scale_grad_by_freq=False, **kwargs ):
            #base model uses same code as 'microsoft/DialoGPT-small'
        super(NLG, self).__init__()
        
        self.base_model_name = base_model_name   
        self.model_name = model_name
        self.freeze_pretrained = freeze_pretrained
        
        self.fda = fda 
        self.frst = frst
        self.ftopic = ftopic
        self.frst_version = frst_version
        self.scale_grad_by_freq = scale_grad_by_freq

        # Retreive/Instantiate base transformer
        self.transformer = utils.load_pretrained_transformer(self.base_model_name, transformer=True)['transformer']    
        # self._use_cache = False


        self.nlg_tokenizer = NLG_tokenizer(base_model_name,
                                os.path.join( ("./models"), f"{model_name}_tokenizer"),
                                fda=fda, frst=frst, ftopic=ftopic,
                                frst_version=frst_version, max_input_len=max_input_len,
                                 **kwargs)
        

        self.transformer.resize_token_embeddings( len(self.nlg_tokenizer.e2m_tokenizer) )
        self.transformer.transformer.forward = types.MethodType(forward,self.transformer.transformer) #monkey patch
        
        self.config= self.transformer.config

        # Embedding Layers
        self.embd_outp_dim = self.transformer.config.n_embd
        special_token_count = sum( [self.fda, self.frst, self.ftopic] )

        if self.fda:
            self.embedding_das = torch.nn.Conv1d( 12, self.embd_outp_dim, kernel_size=1, bias=False )
            self.embedding_das.weight.data.normal_(mean=0.0, std=0.005) #use smaller start variance to minimize init impact of new info
        

        if self.frst:
            self.embedding_rst_rels = torch.nn.Embedding( len(self.nlg_tokenizer.rst_rel_li )+1, self.embd_outp_dim, padding_idx=len(self.nlg_tokenizer.rst_rel_li ), scale_grad_by_freq=self.scale_grad_by_freq )
            self.embedding_rst_rels.weight.data.normal_(mean=0.0, std=0.005)

        if self.frst_version == 1:
            self.embedding_rst_ns = torch.nn.Embedding( len(self.nlg_tokenizer.rst_ns_li )+1, self.embd_outp_dim, padding_idx=len(self.nlg_tokenizer.rst_ns_li ),scale_grad_by_freq=self.scale_grad_by_freq )
            self.embedding_rst_ns.weight.data.normal_(mean=0.0, std=0.005)

            self.embedding_rst_pos = torch.nn.Embedding( self.nlg_tokenizer.rst_pos_maxidx + 1 + 1 , self.embd_outp_dim, padding_idx=self.nlg_tokenizer.rst_pos_maxidx + 1 , scale_grad_by_freq=self.scale_grad_by_freq )
            self.embedding_rst_pos.weight.data.normal_(mean=0.0, std=0.005)

        if self.ftopic:
            self.embedding_topics_score = torch.nn.Conv1d( 1, self.embd_outp_dim, kernel_size=1, bias=False)
            self.embedding_topics_score.weight.data.normal_(mean=0.0, std=0.005)

    
        self.token_type_embeddings = torch.nn.Embedding( special_token_count + self.nlg_tokenizer.context_len['topics']//2, self.embd_outp_dim, scale_grad_by_freq=self.scale_grad_by_freq) #The maximum value this can take is based on the different types of input
        self.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.005)
        # 1 for each of da, rst and + 1 for each topic phrase (note that each topic phrase includes a <topic> token.
        #      therefore the largest number of different topics is topic_ctx//2 if every topic only has one word)

        self.loss_fct = CrossEntropyLoss()

        if self.freeze_pretrained == 0:
            # Freeze no weights
            pass
        
        if self.freeze_pretrained == 1:
        #Freeze all weights except for: New embedding layers, language model head, wte
            #freezing gpt layers weights       
            for name, param in self.transformer.transformer.named_parameters(): 
                if 'wte' not in name:
                    param.requires_grad = False

        #Initialising the embeddings for new special tokens [da, rst, ta] to the values used for endoftext
            #TODO: Experiment to what degree this initialisation affects performance
        with torch.no_grad():
            # initialising to eos token value
            self.transformer.transformer.wte.weight[-special_token_count:,:] = self.transformer.transformer.wte.weight[-special_token_count-1:-special_token_count,:] 

        self.transformer.tie_weights()

    def get_output_embeddings(self):
        return self.transformer.lm_head

    @torch.no_grad()
    def generate(self, 
        input_,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
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
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_specific_kwargs 
        ) -> torch.LongTensor:

        r""" Generates sequences for models with a LM head."""

        #Half Embedded the inputs to get input_embed
        input_ids = input_['tknzd_utt'] #need to make sure no padding is done here ??

        input_ = self.forward_embedding(input_) 
        
        input_embeds = input_['input_embeds']
        attention_mask = input_['attention_mask']
        position_embeds = input_['position_embeds']
        token_type_ids = None
        # Need to get the input ids (tknzd_utterance)
        
        #region - (original code) parameter init and checks
        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_embeds.shape[0]  #changed here : overriden by the input batch_size
        else:
            batch_size = 1
        
        assert batch_size == 1 #changed here

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"
        #endregion

        #region - (original code) sort pad_token_id and handle case of encoder-decoder

        # create attention mask if necessary
        # # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        # if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        #     attention_mask = input_ids.ne(pad_token_id).long()
        # elif attention_mask is None:
        #     attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            # logger.warning(
            #     "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            # )
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        # if self.config.is_encoder_decoder:
        #     if decoder_start_token_id is None:
        #         decoder_start_token_id = bos_token_id

        #     assert (
        #         decoder_start_token_id is not None
        #     ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
        #     assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
        #     assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

        #     # get encoder and store encoder outputs
        #     encoder = self.get_encoder()

        #     encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)
        #endregion
        
        #region  -(Reshaping tensors that need it and some more encoder-decoder logic)

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_embeds_len = input_embeds.shape[-2]
            input_embeds_dim = input_embeds.shape[-1]

            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            input_embeds = input_embeds.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_embeds_len, input_embeds_dim) #Change
            # attention_mask = attention_mask.unsqueeze(1).expand(
            #     batch_size, effective_batch_mult * num_beams, input_ids_len, input_ids_len
            # )
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_embeds_len, input_embeds_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            input_embeds = input_embeds.contiguous().view(
                effective_batch_size * num_beams, input_embeds_len, input_embeds_dim
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len, dim1)
            # attention_mask = attention_mask.contiguous().view(
            #     effective_batch_size * num_beams, input_ids_len, input_ids_len
            # )  # shape: (batch_size * num_return_sequences * num_beams, cur_len, dim1)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_embeds_len, input_embeds_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len, dim1)

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            assert (
                batch_size == encoder_outputs[0].shape[0]
            ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )
            # expand encoder_outputs
            encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

        else:
            encoder_outputs = None
            cur_len = input_ids.shape[-1]

        #endregion

        #TODO: batch size is calculated on line 280 uses first index of input_ids or set to 1
        # input_embeds should be two dimensionl
        if num_beams > 1:
            output = self._generate_beam_search(  
                input_ids = input_ids,
                input_embeds = input_embeds,
                position_embeds=position_embeds,
                attention_mask = attention_mask,
                token_type_ids = None,

                    cur_len=cur_len,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=do_sample,
                    early_stopping=early_stopping,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    bad_words_ids=bad_words_ids,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    batch_size=effective_batch_size,
                    num_return_sequences=num_return_sequences,
                    length_penalty=length_penalty,
                    num_beams=num_beams,
                    vocab_size=vocab_size,
                    encoder_outputs=encoder_outputs,
                    use_cache=use_cache,
                    model_specific_kwargs=model_specific_kwargs)
                 
                #may need to add other special tokens to the mix here

        else:
            output = self._generate_no_beam_search(
                input_ids = input_ids,
                input_embeds = input_embeds,
                position_embeds=position_embeds,
                attention_mask = attention_mask,
                token_type_ids = None,

                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,

                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )

        return output
        
    def _generate_no_beam_search(
        self,
        
        input_ids,
        input_embeds,
        position_embeds,
        token_type_ids,
        attention_mask,

        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        encoder_outputs,
        use_cache,

        **model_specific_kwargs,
     ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        #TODO: Add inteoperatability with past for quicker generation
        past = (encoder_outputs, None) if encoder_outputs is not None else None
        try:
            print(past.shape)
        except Exception as e:
            pass

        while cur_len < max_length:

            model_inputs = self.prepare_inputs_for_generation(
                input_ids, input_embeds, past=past, attention_mask=attention_mask,
                position_embeds=position_embeds , 
                token_type_ids = token_type_ids, use_cache=use_cache, **model_specific_kwargs
            )
           
            outputs = self(input_= model_inputs, skip_embed1=True )         
            lm_logits = outputs[0]

            next_token_logits = lm_logits[:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
            #if True:
            #if False: #Added by me: debugging
                past = outputs[1]
                #raise Exception(f'{past.shape}')

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            new_tokens = tokens_to_add[None, ...]

            # definining new inputs to append to old inputs
            input_ids = torch.cat( [input_ids, new_tokens ],axis=1 ).contiguous()
            input_embeds = torch.cat( [input_embeds, self.transformer.transformer.wte(new_tokens)], axis=1 ) # (batch, 1)

            # Under new token_type_id scheme, we do not add a token type to the utterance part
            # Since we do not change the input_embeds, we can use the input embeds from previous round
            
            # Position ids
                # creating position ids for utterance
            position_ids_utt =  torch.arange( 0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            position_ids_utt = torch.stack( input_embeds.shape[0]*[position_ids_utt]).contiguous() #repeating position_ids for each beam
            position_embeds_utt = self.transformer.transformer.wpe(position_ids_utt) 
            
                # Creating zero value position embeds for context
            position_embeds_context = position_embeds_utt.new_full( [position_embeds_utt.shape[0],
                                                                        self.nlg_tokenizer.context_len_pre_utterance,
                                                                          position_embeds_utt.shape[-1] ] , 0.0) 
            position_embeds = torch.cat([position_embeds_context, position_embeds_utt] , axis=1)
            
            # Making attention mask
                # new token should attend too all prev utterance & all context  except for the padded sections of topics
                # First copy the old attn_mask for all the old tokens #(0) 
                # best way to do this is to just copy the mask used for the previous utterance token (1)
                # Then ensure the new token attends to itself (2)
                # Then all previous tokens should not attend to the new utterance token (3)
                
            old_attn_shape = attention_mask.shape #bs, old_seq_len, old_seq_len
            _ = old_attn_shape
            new_attn_mask = attention_mask.new_empty( [_[0],_[1]+1,_[2]+1] )
            
            new_attn_mask[ :, :-1, :-1] = attention_mask                    #(0)
            new_attn_mask[:, -1:, :-1  ] = new_attn_mask[:, -2:-1, :-1 ]    #(1)
            new_attn_mask[:, -1:, -1: ] = 1.0                                 #(2)
            new_attn_mask[:, :-1, -1: ] = 0.0                               #(3)
            
            attention_mask = new_attn_mask.contiguous()
            
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            # if False: #self.config.is_encoder_decoder is False:
            #     attention_mask = torch.cat(
            #         [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            #     )

        return input_ids

    def _generate_beam_search(
        self,
        input_ids,
        input_embeds,
        attention_mask,
        position_embeds,

        token_type_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        use_cache,
        model_specific_kwargs,
         ):
        """ Generate sequences for each example with beam search.
        """

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = (encoder_outputs, None) if encoder_outputs is not None else None
        #past = None #Added by me: debugging
        #use_cache = False #Added by me: debugging
        # done sentences
        done = [False for _ in range(batch_size)]

        # print(cur_len)
        # print(max_length)

        while cur_len < max_length:
        
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, input_embeds, past=past, attention_mask=attention_mask,
                position_embeds=position_embeds , 
                token_type_ids = token_type_ids, use_cache=use_cache, **model_specific_kwargs
            )
            outputs = self(input_=model_inputs, skip_embed1 = True )  # (batch_size * num_beams, cur_len, vocab_size)
            lm_logits = outputs[0]
            next_token_logits = lm_logits[:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache): 
            #if False: #Added by me: debugging
                past = outputs[1]
            if self.config.is_encoder_decoder and do_sample is False:
                # TODO (PVP) still a bit hacky here - there might be a better solution
                next_token_logits = self.adjust_logits_during_generation(
                    next_token_logits, cur_len=cur_len, max_length=max_length   
                )

            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            scores = self.postprocess_next_token_scores(
                scores=scores,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=num_beams,
            )

            assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
                scores.shape, (batch_size * num_beams, vocab_size)
            )

            if do_sample:
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # Temperature
                if temperature != 1.0:
                    _scores = _scores / temperature
                # Top-p/top-k filtering
                _scores = top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence, add a pad token
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content, this will get added to next_batch_beam
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # once the beam for next step is full, don't add more tokens to it.
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if we are done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batchbeam_idx
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch and update current length
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)

            #region changed : creating / adding to model inputs
            new_tokens = beam_tokens.unsqueeze(1)
            input_embeds = torch.cat( [input_embeds, self.transformer.transformer.wte( new_tokens ) ], axis=1 ) # (batch, 1)

            # Under new token_type_id scheme, we do not add a token type to the utterance part
            # Since we do not change the input_embeds, we can use the input embeds from previous round
            
            # Position ids
                # creating position ids for utterance
            position_ids_utt =  torch.arange( 0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            position_ids_utt = torch.stack( input_embeds.shape[0]*[position_ids_utt]).contiguous() #repeating position_ids for each beam
            position_embeds_utt = self.transformer.transformer.wpe(position_ids_utt) 
            
                # Creating zero value position embeds for context
            position_embeds_context = position_embeds_utt.new_full( [ position_embeds_utt.shape[0],
                                                                        self.nlg_tokenizer.context_len_pre_utterance ,
                                                                          position_embeds_utt.shape[-1] ] , 0.0) 
            position_embeds = torch.cat([position_embeds_context, position_embeds_utt] , axis=1)

            # Making attention mask
                # new token should attend too all prev utterance & all context  except for the padded sections of topics
                # First copy the old attn_mask for all the old tokens #(0) 
                # best way to do this is to just copy the mask used for the previous utterance token (1)
                # Then ensure the new token attends to itself (2)
                # Then all previous tokens should not attend to the new utterance token (3)
                
            old_attn_shape = attention_mask.shape #bs, old_seq_len, old_seq_len
            _ = old_attn_shape
            new_attn_mask = attention_mask.new_empty( [_[0],_[1]+1,_[2]+1] )
            
            new_attn_mask[ :, :-1, :-1] = attention_mask                    #(0)
            new_attn_mask[:, -1:, :-1  ] = new_attn_mask[:, -2:-1, :-1 ]    #(1)
            new_attn_mask[:, -1:, -1: ] = 1                                 #(2)
            new_attn_mask[:, :-1, -1: ] = 0.0                               #(3)
            
            attention_mask = new_attn_mask.contiguous()

            cur_len = cur_len + 1
            #endregion
            
            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)

            # extend attention_mask for new generated input if only decoder
            if False: # self.config.is_encoder_decoder is False: 
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

        return decoded    

    def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        """
        Enforce the repetition penalty (from the `CTRL paper <https://arxiv.org/abs/1909.05858>`__).
        """
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty

    def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
            ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores,
                batch_size,
                num_beams,
                input_ids,
                repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = calc_banned_ngram_tokens(
                input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")

        if bad_words_ids is not None:
            # Exclude EOS token (already processed)
            bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
            # calculate a list of banned tokens according to bad words
            banned_tokens = calc_banned_bad_words_ids(input_ids.tolist(), bad_words_ids)
            # Modify the scores in place by setting the banned tokens logits to `-inf`
            set_scores_to_inf_for_banned_tokens(scores, banned_tokens)

        return scores

    def _use_cache(self, outputs, use_cache):
        """During generation, decide whether to pass the `past` variable to the next forward pass."""
        if len(outputs) <= 1 or use_cache is False:
            return False
        if hasattr(self.transformer.config, "mem_len") and self.transformer.config.mem_len == 0:
            return False
        return True

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PretrainedModel.beam_search` or :meth:`~transformers.PretrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
                
        parser.add_argument('--base_model_name', default='distilgpt2', required=False)
        parser.add_argument('--reset_base_transformer', default=False, required=False, type=bool)
        parser.add_argument('-fp','--freeze_pretrained', default=0, required=False, type=int )

        parser.add_argument('--model_name', default='NLG', required=False)
        parser.add_argument('--loss_type', default='CrossEntropy', required=False, 
            choices=['CrossEntropy','UtteranceSimilarity']) 
        
        parser.add_argument( '-fda' ,'--fda', type= lambda x: bool(int(x) )
             ,help="whether or not to include da in feature", default=True  )
        
        parser.add_argument( '-frst' ,'--frst', type= lambda x: bool(int(x) )
             ,help="whether or not to include rst info in feature", default=True  )

        parser.add_argument( '-ftopic' ,'--ftopic', type= lambda x: bool(int(x) )
             ,help="whether or not to include topic info in feature", default=True  )

        parser.add_argument('-cl','--context_len',type= lambda x: json.loads(x) )

        parser.add_argument( '-frstv' ,'--frst_version', type= int, choices=[0,1] 
             ,help="which version of frst to use", default=0  ) 
             #version 0, only uses the relations
             #version 1, uses relations and ns and position information
        
        
        parser.add_argument('-mil','--max_input_len', type=int, default=264)
        
        parser.add_argument('-sgbf','--scale_grad_by_freq', type=lambda x: bool(int(x)) , default=False, 
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

        #print(input_.keys())
        #outputs = self.transformer( input_embeds=input_['input_embeds'],
        transformer_outputs = self.transformer.transformer.forward( 
                                    # input_embeds=input_['input_embeds'],
                                    # attention_mask = input_['attention_mask'],
                                    # position_embeds = input_['position_embeds'],
                                    # token_type_ids = None, #token type embedding new (This gpt implementation incorrectly uses same embedding layer as for input)
                                    #                         # Further we handle token_type embedding in forward_embedding layer
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
                        
            return (lm_logits, loss) 

        else:
            output = (lm_logits,) + transformer_outputs[1:]

            return output

    def forward_embedding(self, input_):
        #Partially does the emebddign for our new inputs to the transformer

        # Creating embedded inputs and attention mask
        if self.fda and self.frst:
            input_ = self.forward_embedding_drt( input_ )
        elif self.fda==False and self.frst:
            input_ = self.forward_embedding_rt( input_ )
        return input_ 

    def forward_embedding_drt(self, input_):
        """Performs the input embedding and token type embedding calcs

            Args:
                input_ ([type]): [description]

            Returns:
                input_embedded [type]: [description]
                
                Note: logic for token_type_ids embedding is usually performed in transformer, but has been moved here
                    since gpt-2 code indicates that 
        """
        #TODO: need to add the rst_rels and ns options
        
        # da token embedding
        da_start_embed = self.transformer.transformer.wte( input_['da_start_token'] ).unsqueeze(1)
        das_embed = self.embedding_das(input_['tnsr_das']).permute(0,2,1).contiguous()

        # rst token emebedding
        rst_start_embed = self.transformer.transformer.wte( input_['rst_start_token'] ) 
        rst_rel_embed = self.embedding_rst_rels( input_['tnsr_rst_rels'] ) #.permute(0,2,1) # (bs, channels, seq_len)
        rst_embed = rst_rel_embed
        
        if self.frst_version == 1:
            rst_ns_embed = self.embedding_rst_ns( input_['tnsr_rst_ns'] ) #.permute(0,2,1)
            rst_pos_embed = self.embedding_rst_pos( input_['tnsr_rst_pos'] ) #.permute(0,2,1)
            rst_embed += rst_ns_embed + rst_pos_embed

        #topic  embedding
        topics_phrase_embed = self.transformer.transformer.wte(input_['tnsr_topics_phrase']  )  
        topics_score_embed = self.embedding_topics_score( input_['tnsr_topics_score']).permute(0,2,1)

        topics_embed = topics_phrase_embed + topics_score_embed

        utt_embed = self.transformer.transformer.wte(input_['tknzd_utt'] ) 

        input_embeds = torch.cat(
            [da_start_embed, das_embed,
             rst_start_embed, rst_embed,
             topics_embed,
             utt_embed], axis = 1
            ).contiguous() #dim [bs, seq_len, dim1]
        
            # Token type embedding is only added to the context section
        token_type_embedding = self.token_type_embeddings( input_['token_type_ids'] )
        input_embeds[ :self.context_len_pre_utterance, :] += token_type_embedding
        
        input_['input_embeds'] = input_embeds

        # Positional Embedding
            # Creating zero'd positional_embedding for context section,
            #  while utterance section uses values from embedding matrix
        utt_position_embeds = self.transformer.transformer.wpe(input_['position_ids']) 
        shape = utt_position_embeds.shape
        shape[1] = input_embeds.shape[1] - shape[1] # this dimensions of context
        ctx_position_embeds = utt_position_embeds.new_zeros(  shape  ) #ctx has no position embedding 
        input_['position_embeds'] = torch.cat( [ctx_position_embeds, utt_position_embeds], axis=1 ) 

    
        
        return input_

    def forward_embedding_rt(self, input_):
        """Performs the input embedding and token type embedding calcs

            Args:
                input_ ([type]): [description]

            Returns:
                input_embedded [type]: [description]
                
                Note: logic for token_type_ids embedding is usually performed in transformer, but has been moved here
                    since gpt-2 code indicates that 
        """
        new_input = {}
        new_input['attention_mask'] = input_['attention_mask']
        new_input['labels'] = input_['labels']

        # rst token emebedding
        rst_start_embed = self.transformer.transformer.wte( input_['rst_start_token'] ) 
        rst_rel_embed = self.embedding_rst_rels( input_['tnsr_rst_rels'] ) #.permute(0,2,1) # (bs, channels, seq_len)
        rst_embed = rst_rel_embed
        
        if self.frst_version == 1:
            rst_ns_embed = self.embedding_rst_ns( input_['tnsr_rst_ns'] ) #.permute(0,2,1)
            rst_pos_embed = self.embedding_rst_pos( input_['tnsr_rst_pos'] ) #.permute(0,2,1)
            rst_embed = rst_embed + rst_ns_embed + rst_pos_embed

        
        # topic embedding
        topics_score_embed =  self.embedding_topics_score( input_['tnsr_topics_score']).permute(0,2,1) 
        topics_phrase_embed = self.transformer.transformer.wte(input_['tnsr_topics_phrase']  )  #this is contiguous from the tokenizer
        
        topics_embed = topics_phrase_embed + topics_score_embed
        
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
        input_embeds[ :, :self.nlg_tokenizer.context_len_pre_utterance, :] += token_type_embedding

        #input_['input_embeds'] = input_embeds
        new_input['input_embeds'] = input_embeds

        # position embedding
        utt_position_embeds = self.transformer.transformer.wpe(input_['position_ids']) 

        _ = utt_position_embeds.shape
        shape = [_[0], self.nlg_tokenizer.context_len_pre_utterance ,_[2]]
        ctx_position_embeds = utt_position_embeds.new_zeros(  shape  ) #ctx has no position embedding 
        #input_['position_embeds'] = torch.cat( [ctx_position_embeds, utt_position_embeds], axis=1 ) 
        new_input['position_embeds'] = torch.cat( [ctx_position_embeds, utt_position_embeds], axis=1 ) 
        

        return new_input

    def return_params(self):
        keys = ['base_model_name','freeze_pretrained','max_input_len',
                        'fda','frst','ftopic','frst_version','scale_grad_by_freq']

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
            rst_pos, topics, topics_score, 
            prompt, generation_params={}):
        """Given an encoded input, outputs predictions up until an <EOS> is emmitted
                The input will have been encoded using nlg_tokenizer
                
            Raises:
                Exception: [description]

            Returns:
                [type]: [description]
        """


        # output = self.generate(encoded_input, **generation_params)
        # decoded_text = self.nlg_tokenizer.e2m_tokenizer.decode(output[0],skip_special_tokens=True)
        
        # Example 
        # rst_rels = ['Contrast','Temporal']
        # rst_ns = ['NS','NS']
        # rst_pos = [0,1]
        # topics = ('Bitcoin','a new watch')
        # topics_score = (1.20, 1.501)
        # prompt = ""

        #type checks
        if type(topics) == list:
            topics = tuple(topics)
        
        if type(topics_score) == list:
            topics_score = tuple(topics_score)       

        # default generation params
            #TODO: add these to config so they are automatically done
        if 'bad_words_ids' not in generation_params:
            bad_words = ["<|rst|>","<|ta|>",r"\n", "\s"," \s", ". \s", "|", '\\n', "\\", "\\t", "#|"]
            bad_words_ids = [model.nlg_tokenizer.e2m_tokenizer.encode(bad_word, add_prefix_space=False) for bad_word in bad_words]
            bad_words_ids = [model.nlg_tokenizer.e2m_tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in bad_words]
            bad_words_ids = bad_words_ids + [[526], [55],[8172], [3467], [59], [6852], [7479],[7879],[13426],[17405],[91],[8614],[930],[10],[9],[12],[1303],[2],[4242], [2235],[46424]]
            generation_params['bad_words_ids'] = bad_words_ids
        else:
            bad_words_ids = None

        default_generation_params = {'num_beams':1, 'temperature':1.2, 'repitition_penalty':2.0, 
                            'early_stopping':True, 'do_sample':False, 'no_repeat_ngram_size':3, 'bad_words_ids':bad_words_ids, 'max_length':300 } #,'min_length':4

        for key,value in default_generation_params.items():
            if key not in generation_params:
                generation_params[key] = value


        encoded_input = self.nlg_tokenizer.encode_v2_exda(rst_rels, rst_ns, rst_pos ,
                                                    topics, topics_score, prompt,
                                                    pad_utterance=False, generate_mode=True)

            # Add batch dimension to data and moving to GPU
        device = next(self.parameters()).device
        for key in ['tnsr_rst_rels', 'tnsr_rst_ns', 'tnsr_rst_pos',
                    'tnsr_topics_phrase','tnsr_topics_score','tknzd_utt',
                    'position_ids','token_type_ids',
                        'attention_mask','rst_start_token']:
            encoded_input[key] = torch.unsqueeze( encoded_input[key],axis=0).to(device)

            # Generating Text
        output = self.generate(encoded_input, **generation_params)
        gen_text = self.nlg_tokenizer.e2m_tokenizer.decode(output[0],skip_special_tokens=True)

        return gen_text

    def prepare_inputs_for_generation(self, input_ids, input_embeds, position_embeds,
            attention_mask,token_type_ids ,past=None, **kwargs):
        
        # only last token for input_ids if past is defined in kwargs
        if past != None:
            #input_ids = input_ids[:, -1].unsqueeze(-1)
            input_embeds = input_embeds[:, -1, :].unsqueeze(-2)
            position_embeds = position_embeds[:, -1, :].unsqueeze(-2)
            attention_mask = attention_mask[ :, -1, :].unsqueeze(-2)

            #TODO: may also have to crop the input_embeds and attneiton_mask
        
        return {
            #"input_ids": input_ids,
            'input_embeds':input_embeds,
            "attention_mask": attention_mask,
            'position_embeds': position_embeds,
            "past_key_values": past,
            "token_type_ids": None,

        }

class NLG_tokenizer():
    """Rough Implmentation of the tokenizer for the NLG model

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """

    def __init__(self,
                 e2m_base_model_name='distilgpt2',
                 dir_tokenizer='./models/NLG_tknzr',
                 fda = True,
                 frst = True,
                 frst_version = 0,
                 ftopic = True,
                 max_input_len = 216,                 
                 **kwargs
                 ):

        self.e2m_base_model_name = e2m_base_model_name
        self.fda = fda
        self.frst = frst
        self.frst_version = frst_version
        self.ftopic = ftopic
        self.special_token_count = sum( [self.fda, self.frst, self.ftopic] )

        assert max_input_len < 1025
        self.max_input_len = max_input_len  #self.e2m_tokenizer.max_len
        
        # Setting up RST utilities
        if self.frst:
            self.rst_rel_li = ['Attribution',
                'Background','Cause','Comparison','Condition',
                'Contrast','Elaboration','Enablement','Evaluation',
                'Explanation','Joint','Manner-Means','Topic-Comment',
                'Summary','Temporal','Topic-Change','n','same-unit','textual-organization'] #Add this to savable config

            self.rst_rel_labeler = sklp.LabelEncoder()
            self.rst_rel_labeler.fit(  self.rst_rel_li )

            #TODO: here
            self.rst_ns_li = ['NN','NS','SN','a'] 
            self.rst_ns_labeler = sklp.LabelEncoder()
            self.rst_ns_labeler.fit( self.rst_ns_li  )

            self.rst_pos_maxidx = 30


        # Setting up context lengths
        if self.fda and self.frst and self.ftopic:
            self.context_len = kwargs.get( 'context_len' , { 'da':2, 'rst':6, 'topics':20 } ) #add this to config
        
        elif self.fda==False and self.frst and self.ftopic:
            self.context_len = kwargs.get( 'context_len', { 'rst':6, 'topics':16 } ) #add this to config )
        
        self.context_len_pre_utterance =  sum(self.context_len.values())

        self.context_len['utt'] = self.max_input_len - self.context_len_pre_utterance

        # Initalising tokenzier
        if os.path.isdir(dir_tokenizer):
            self.e2m_tokenizer = AutoTokenizer.from_pretrained(dir_tokenizer,use_fast=False)

        # retreiving base tokenizer from online or from local distillgpt2
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
                    []}
            if self.fda:
                special_tokens_dict['additional_special_tokens'].append('<|da|>')
            if self.frst:
                special_tokens_dict['additional_special_tokens'].append('<|rst|>')
            if self.ftopic:
                special_tokens_dict['additional_special_tokens'].append('<|ta|>')
            
            if str(special_tokens_dict['additional_special_tokens']) != \
                    self.e2m_tokenizer.special_tokens_map.get('additional_special_tokens',''):
                
                num_added_toks = self.e2m_tokenizer.add_special_tokens(special_tokens_dict)
                os.makedirs(dir_tokenizer)
                
                self.e2m_tokenizer.init_kwargs['name_or_path'] = dir_tokenizer
                self.e2m_tokenizer.init_kwargs['special_tokens_map_file'] = os.path.join(dir_tokenizer,"special_tokens_map.json")
                
                self.e2m_tokenizer.save_pretrained(dir_tokenizer)
                config.save_pretrained(dir_tokenizer)

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

    def encode_v2( self, das ,rst_rels, rst_ns, rst_pos, topics, topics_score,
             utterance, pad_utterance,generate_mode):

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


        #Getting Vectors
        tnsr_das = self.encode_da( das ) #dims (1, 20), 

        #Getting Vectors
        if self.frst_version == 0:
            tnsr_rst_rels, rst_pad_count = self.encode_rst_v2(rst_rels, max_len=self.context_len['rst'] - 1)   # dims (max_padding, n) #not we do rt_len-1 since we add the rst start token outside this method
            # dummy tensors
            tnsr_rst_ns = torch.zeros([1])
            tnsr_rst_pos = torch.zeros([1])

        elif self.frst_version == 1:
            tnsr_rst_rels, rst_pad_count, tnsr_rst_ns, tnsr_rst_pos = self.encode_rst_v2_full(rst_rels, rst_ns, rst_pos ,max_len=self.context_len['rst'] - 1)   # dims (max_padding, n) #not we do rt_len-1 since we add the rst start token outside this method


        tnsr_topics_phrase, tnsr_topics_score, topics_pad_count, ta_tokens_pos, ta_phrase_lens  = self.encode_topic_v2( topics, topics_score, max_len=self.context_len['topics'], padding_token=padding_token) # dims (max_padding, 13) 
        tknzd_utt, utt_pad_count = self.encode_utterance_v2(utterance, pad_utterance)
                            
        # Building Attention Mask

            # calc the ending cumulative dim for da, rst, topics, utt, segments,
        d_dim = self.context_len['da']
        dr_dim = da_len + self.context_len['rst']       # d_dim + tnsr_rst_rels.shape[0]+ 1
        drt_dim = dr_dim + self.context_len['topics']    # dr_dim + tnsr_topics_phrase.shape[1]

        if pad_utterance == True:
            utt_dim = drt_dim + self.context_len['utt']  
        else:
            utt_dim = drt_dim + tknzd_utt.shape[-1]

        # New Version
        attn_mask = torch.tril( torch.ones([utt_dim, utt_dim]), diagonal=drt_dim )
                   #correcting for padding in rst section
        attn_mask[ dr_dim-rst_pad_count:dr_dim ,:] = 0
        attn_mask[ :, dr_dim-rst_pad_count:dr_dim] = 0
                     #correcting for padding in topic section
        attn_mask[ drt_dim-topics_pad_count: drt_dim, :  ] = 0
        attn_mask[ :, drt_dim-topics_pad_count: drt_dim ] = 0

            # Implementing causal masking for each topic subphrase 
                # First, each topic subphrase only attends to other words within that topic subphrase (including ta token) and not the other topcics
                # So set the template attn to 0 
        attn_mask[ dr_dim:drt_dim, dr_dim:drt_dim ] = 0
                #Second each topic phrase has causal masking on the tokens within the topic phrase
                # use ta_tokens_pos
                # essentially for idx i, i+1
                # add a tril att attn_mask[ i:i+1 , i:i+1 ] so that in each topic phrase each word only attmeds to the previous words
        for ta_idx, phrase_len in zip( ta_tokens_pos, ta_phrase_lens):
            attn_mask[ r_dim+ta_idx:r_dim+ta_idx+phrase_len, r_dim+ta_idx:r_dim+ta_idx+phrase_len ] = torch.tril( attn_mask.new_ones( [phrase_len,phrase_len]  )  )

                #correcting for padding in and after utterance section
                    #when the utterance padding starts then we mask
        attn_mask[ utt_dim-utt_pad_count: , : ] = 0

        
        #Creating labels/targets for GPT Language Model Head
        labels = -100* torch.ones( size=[1, self.max_input_len], dtype = torch.long  ) 
        labels[0][drt_dim:utt_dim-utt_pad_count] = tknzd_utt[: utt_dim-utt_pad_count-drt_dim]

        # Creating Positional Emebeddings
            # ALL words in drt get a positional encoding of 0 -> No positional meaning
            # utterance has normal positional encoding        
        
        position_ids_utt =  torch.arange( 0, utt_dim-drt_dim   , dtype=torch.long)
        position_ids = position_ids_utt

        # Creating Token Type Ids
            # 0:da, 
            # 1:rst, 
            # n<m for each word in each topic phrase i of length m_i including leading <ta> where 2>n>=2+topics_len//2
        
        token_type_ids_d = torch.full( [self.context_len['da'] ], 0 ,dtype=torch.long)
        token_type_ids_r = torch.full( [self.context_len['rst']], 1 ,dtype=torch.long)
                
        _ = torch.zeros( [self.context_len['topics'] ], dtype=torch.long)
        _[ ta_tokens_pos ] = 1
        token_type_ids_t =  _.cumsum(axis=0) + 1
        token_type_ids = torch.cat( [token_type_ids_d, token_type_ids_r, token_type_ids_t ] )

        return { 'da_start_token':da_start_token, 'tnsr_das':tnsr_das,
                'tnsr_rst_ns':tnsr_rst_ns, 'tnsr_rst_pos':tnsr_rst_pos,
                 'rst_start_token':rst_start_token, 'tnsr_rst_rels':tnsr_rst_rels,
                 'tnsr_topics_phrase':tnsr_topics_phrase.contiguous(),
                  'tnsr_topics_score': tnsr_topics_score,
                 'tknzd_utt':tknzd_utt.contiguous(),
                 'attention_mask':attn_mask.contiguous(),
                 'labels':labels,
                 'position_ids':position_ids.contiguous(),
                 'token_type_ids':token_type_ids.contiguous()
                 }

    def encode_v2_exda( self, rst_rels, rst_ns , rst_pos, topics, topics_score, 
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
        padding_token =  self.e2m_tokenizer.encode("<|endoftext|>",return_tensors="pt") 


        #Getting Vectors
        if self.frst_version == 0:
            tnsr_rst_rels, rst_pad_count = self.encode_rst_v2(rst_rels, max_len=self.context_len['rst'] - 1)   # dims (max_padding, n) #not we do rt_len-1 since we add the rst start token outside this method
            # dummy tensors
            tnsr_rst_ns = torch.zeros([1])
            tnsr_rst_pos = torch.zeros([1])

        elif self.frst_version == 1:
            tnsr_rst_rels, rst_pad_count, tnsr_rst_ns, tnsr_rst_pos = self.encode_rst_v2_full(rst_rels,rst_ns, rst_pos, max_len=self.context_len['rst'] - 1)   # dims (max_padding, n) #not we do rt_len-1 since we add the rst start token outside this method
        
        tnsr_topics_phrase, tnsr_topics_score, topics_pad_count, ta_tokens_pos, ta_phrase_lens  = self.encode_topic_v2( topics, topics_score, max_len=self.context_len['topics'], padding_token=padding_token) # dims (max_padding, 13) 
        tknzd_utt, utt_pad_count = self.encode_utterance_v2(utterance, pad_utterance, generate_mode)

            # calc the ending cumulative dim for, rst, topics, utt, segments,
        r_dim = self.context_len['rst']       # tnsr_rst_rels.shape[0]
        rt_dim = r_dim +self.context_len['topics']    # dr_dim + tnsr_topics_phrase.shape[1]
        
        if pad_utterance == True:
            utt_dim = rt_dim + self.context_len['utt']  
        else:
            utt_dim = rt_dim + tknzd_utt.shape[-1]

        # Building Attention Mask
        attn_mask = torch.tril( torch.ones([utt_dim, utt_dim]), diagonal=rt_dim )
            #correcting for padding in rst section
        attn_mask[ r_dim-rst_pad_count:r_dim ,:] = 0
        attn_mask[ :, r_dim-rst_pad_count:r_dim] = 0

            #correcting for padding in topic section
        attn_mask[ rt_dim-topics_pad_count: rt_dim, :  ] = 0
        attn_mask[ :, rt_dim-topics_pad_count: rt_dim ] = 0

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
        attn_mask[ utt_dim-utt_pad_count: , : ] = 0


        #Creating labels/targets for GPT Language Model Head
        try:
            
            labels = -100* torch.ones( size=[1, utt_dim], dtype = torch.long  ) 

            labels[0][rt_dim:utt_dim-utt_pad_count] =  tknzd_utt[ : utt_dim-utt_pad_count-rt_dim ]
        
        except Exception:
            labels = None
        
        # Creating Positional Emebeddings
            # ALL words in drt get a positional encoding of 0 -> No positional meaning
            # utterance has normal positional encoding        
        position_ids_utt =  torch.arange( 0, utt_dim-rt_dim, dtype=torch.long)
        position_ids = position_ids_utt

        # Creating Token Type Ids
        #     1:rst, 
        #     n<m for each word in each topic phrase i of length m_i including leading <ta> where 4>=n>=4+topics_len//2
        #     3:utterance part

        token_type_ids_r = torch.full( [self.context_len['rst']], 0 ,dtype=torch.long)
        
        _ = torch.zeros( [self.context_len['topics'] ], dtype=torch.long)
        _[ ta_tokens_pos ] = 1
        token_type_ids_t =  _.cumsum(axis=0)

        token_type_ids = torch.cat( [token_type_ids_r, token_type_ids_t] ) 


        return { 'rst_start_token':rst_start_token, 'tnsr_rst_rels':tnsr_rst_rels,
                'tnsr_rst_ns':tnsr_rst_ns, 'tnsr_rst_pos':tnsr_rst_pos,
                 'tnsr_topics_phrase':tnsr_topics_phrase.contiguous(),
                  'tnsr_topics_score': tnsr_topics_score,
                 'tknzd_utt':tknzd_utt.contiguous(),
                 'attention_mask':attn_mask.contiguous(),
                 'labels':labels,
                 'position_ids':position_ids.contiguous(),
                 'token_type_ids':token_type_ids.contiguous()
                 }

    def encode_rst_v2(self,rst_rels, max_len=8):
        """Converts rst_rels in a series of vectors

            Args:
                rst_rels ([type]): [description]
                max_padding ([type]): padding amount
                rst_pos ([type]): [description]
        """
        rst_rel_encoded = self.rst_rel_labeler.transform(rst_rels) #.reshape( [1 , -1] )
        tnsr_rels = torch.LongTensor( rst_rel_encoded )
        
        #Padding out to max_len length
        _len = tnsr_rels.shape[0]
        diff = (max_len - _len)
        if diff > 0:
            tnsr_rels = torch.cat([tnsr_rels, torch.full( [diff], len(self.rst_rel_li ), dtype=torch.long)] , axis=-1 ) 
        elif diff == 0:
            pass
        else:
            tnsr_rels = tnsr_rels[ :max_len]
            diff = 0

        return tnsr_rels, diff

    def encode_rst_v2_full(self,rst_rels, rst_ns, rst_pos, max_len=8):
        """Converts rst_rels in a series of vectors

            Args:
                rst_rels ([type]): [description]
                max_padding ([type]): padding amount
                rst_pos ([type]): [description]
                rst_ns
            Also includes an encoding for rst_ns and rst_pos
        """

        tnsr_rels, diff = self.encode_rst_v2(rst_rels, max_len=max_len)

        # Encoding the rst ns 
            #Encoded to the sequence of integers representing the ns values
        rst_ns_encoded = self.rst_ns_labeler.transform( rst_ns ) #.reshape( [1,-1] )  
        tnsr_ns = torch.LongTensor(rst_ns_encoded)

        # Encoding the rst position
        tnsr_pos = torch.LongTensor( rst_pos ) #.reshape([1,-1])

        # padding ns and pos
            # The ns and pos embedding layer uses the index value 0 as a padding index
            # For this index the vector is initialized to zer0 and as such never updates
        
        len_ =  tnsr_ns.shape[0]
        if len_ > max_len:
            tnsr_ns = tnsr_ns[:max_len]
            tnsr_pos = tnsr_pos[:max_len]
        
        elif len_ < max_len:
            _ = max_len-len_
            tnsr_ns = torch.cat( [tnsr_ns, torch.full([_],len(self.rst_ns_li ))])
            tnsr_pos = torch.cat( [tnsr_pos, torch.full([_],self.rst_pos_maxidx+1)])


        return tnsr_rels, diff, tnsr_ns, tnsr_pos

    def encode_da(self, das):
        """[summary]

        Args:
            das ([type]): [list of da probabilites]
        """
        #TODO: add some normalization of da probabilities here
        tnsr_das = torch.unsqueeze( torch.FloatTensor( das), axis=-1 ) #dim [encode_dim1, 1]
        
        return tnsr_das

    def encode_topic_v2(self, topics, topics_score, padding_token, max_len=16):
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
                                            max_length = self.context_len['topics'],
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
        #TODO: here check the ta_phrase_lens above is not a problem
        topics_score = [ [score]*phrase_len for score, phrase_len in zip(topics_score, ta_phrase_lens) ]
        topics_score = sum(topics_score,[]) #flattening list
        tnsr_score = torch.unsqueeze( torch.FloatTensor( topics_score ) , dim=0 ) # shape (1, topic_count) #pytorch has convolution dims opposite to tf
        
        
        #Padding out to max_len
        _len = dict_encoding['length']
        diff = (max_len - _len)[0]
        if diff>0:
            topic_phrases = torch.cat( [ topic_phrases, torch.ones([diff], dtype=torch.int64 )*padding_token[0]] , axis=-1 )
            tnsr_score = torch.cat( [tnsr_score, torch.zeros( [1, diff] ) ], axis=-1 )
        else:
            topic_phrases = topic_phrases[:max_len]
            tnsr_score = tnsr_score[:, :max_len ]
            diff = 0

        return topic_phrases , tnsr_score, diff, ta_idxs, ta_phrase_lens

    def encode_utterance_v2(self, utterance, pad=True, generate_mode=False):
        #pad: 
        #   set to True during training to ensure all batches have the same length
        #   set to False in the case of Generation in order to work with huggingface .generate()
        #TODO: change to be able to handle batches of data
        #TODO: When you move to training on large seequences performing variable batch sizes to reduce time
        if generate_mode == False:
            utterance ='<|endoftext|>' + utterance + '<|endoftext|>'
            add_prefix_space=False
        else:
            utterance ='<|endoftext|>' + utterance
            add_prefix_space = False
            

        if pad == True:
            encoded = self.e2m_tokenizer( utterance, add_special_tokens=False,
                                        return_attention_mask = False, 
                                        
                                        padding='do_not_pad',
                                        truncation=True, 
                                        max_length= self.context_len['utt'],
                                                                                
                                        return_tensors='pt',
                                        return_length=True,
                                        return_token_type_ids=None,
                                        add_prefix_space = add_prefix_space
                                        )
            
            #tokenizer length usually returns the padded length as opposed the original length.
            #So using 'do_not_pad' and then padding manually
            tknzd_utt_no_pad_len = encoded['length'][0]
            pad_count = self.context_len['utt'] - tknzd_utt_no_pad_len            
            tknzd_utt = torch.cat( [ encoded['input_ids'], torch.LongTensor(1,pad_count).fill_(self.e2m_tokenizer.eos_token_id) ],axis=-1 )[0]
                                           
        
        elif pad == False:
                        
            encoded = self.e2m_tokenizer( utterance, add_special_tokens=False,
                                        return_attention_mask = False, 
                                        padding='do_not_pad',
                                        truncation=True, 
                                        max_length= self.context_len['utt'],
                                        return_tensors='pt',
                                        return_length=True,
                                        return_token_type_ids=None,
                                        add_prefix_space=add_prefix_space)
            tknzd_utt_no_pad_len = encoded['length'][0]
            pad_count = 0

        
            tknzd_utt = encoded['input_ids'][0]
        
        return tknzd_utt, pad_count

class TrainingModule(pl.LightningModule):

    def __init__(self, model_params, batch_size=20, 
                    dir_data=None, 
                    accumulate_grad_batches=1,
                    max_epochs=25,
                    gpus=1, 
                    learning_rate=1e-3,
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
            self.create_data_loaders( )
            self.accumulate_grad_batches = accumulate_grad_batches
            self.tag = tag

        if self.mode in ['train_new','train_cont']:
            self.max_epochs = max_epochs
            self.warmup_proportion = warmup_proportion
            self.lr_schedule = lr_schedule
            self.learning_rate = learning_rate
        
            train_params_to_save = self.return_params()
            model_params_to_save = self.model.return_params()

            try:
                self.hparams = { **train_params_to_save, **model_params_to_save}
            except Exception as e:
                self.hparams.update({ **train_params_to_save, **model_params_to_save})

            self.inference_samples = list( islice( self.inference_dl, 10 ) )
            bad_words = ["<|rst|>","<|ta|>",r"\n" ] 
            bad_words_ids = [self.model.nlg_tokenizer.e2m_tokenizer.encode(bad_word, add_prefix_space=False) for bad_word in bad_words]
            bad_words_ids = [self.model.nlg_tokenizer.e2m_tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in bad_words]
            bad_words_ids = bad_words_ids + [[526], [55],[8172]]
            

            generation_params = {'num_beams':1, 'temperature':1.1, 'repitition_penalty':1.2, 
                                'top_k': 50, 'top_p':0.85,
                                'length_penalty':1.5, 'early_stopping':True,
                                'do_sample':True, 'bad_words_ids':bad_words_ids, 'no_repeat_ngram_size':3,
                                'min_length':5, 'max_length':80  } 
                                
                                # 'max_length':self.model.nlg_tokenizer.max_input_len  } 
            
            self.inference_generation_params = generation_params

            del self.inference_dl

        if self.mode in ['inference']:
            self.eval() 
            self.freeze() 

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data', default="./dataset/reddit_large_annotated_long3", help="Relative directory path for datafiles")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('-me','--max_epochs', default=32, type=int)
        parser.add_argument('-agb','--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('-bs','--batch_size', default=5, type=int)
        parser.add_argument('-lr','--learning_rate', default=5e-4, type=float)
        parser.add_argument('--warmup_proportion', default=0.15)
        parser.add_argument('--workers', default=16, type=int) #TODO: change to 6
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','train_cont','test','inference'])
        parser.add_argument('--lr_schedule', default='cosine_warmup', required=False, choices =['cosine_warmup','LROnPlateau','hard_restarts','constant'])
        parser.add_argument('--splits', default={'train':0.6,'val':0.2,'test':0.2}, required=False, type=str )
        parser.add_argument('--version', default=None,required=False, type=int, help="The Experimental Versioning for this run" )
        parser.add_argument('--precision', default=16,required=False, type=int, help="Precision to use", choices=[16,32] )
        parser.add_argument('-opt','--optimizer_type', default="AdamW",required=False, type=str, help="Optimizer to use", choices=["AdamW","Adafactor"] )
        parser.add_argument('--tag',default='',required=True, type=str)
        parser.add_argument('--override',default=False, type = lambda x: bool(int(x)), choices=["0","1"] )
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
                    'base_model_name','loss_type','model_name','fda','frst','ftopic','max_input_len',
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
                    'base_model_name','loss_type','model_name','fda','frst','ftopic','max_input_len']} )
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
        
        checkpoint_callback._save_model  = types.MethodType(monkey_save_model,checkpoint_callback) #monkey patch
        checkpoint_callback._monitor_candidates = types.MethodType(_monitor_candidates, checkpoint_callback) # monkey patch

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
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
                        check_val_every_n_epoch=1, logger=tb_logger,
                        #log_every_n_steps=20,
                        precision=tparams['precision'], callbacks=callbacks,
                        accelerator=accelerator,
                        #limit_train_batches =10,
                        #limit_val_batches = 10,
                        val_check_interval=0.2,
                        num_sanity_val_steps=0, 
                        #track_grad_norm = True,
                        #overfit_batches=25,
                        #fast_dev_run=2, 
                        #log_gpu_memory=True
                        )

        elif tparams['mode'] in ["train_cont","inference"]:
            #restoring checkpoint             
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            #training_module.load_state_dict(checkpoint['state_dict'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace( **tparams),
                    progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                    check_val_every_n_epoch=1, logger=tb_logger,
                    log_every_n_steps=20,   
                    precision=tparams['precision'],
                    callbacks=callbacks,
                    accelerator=accelerator,
                    #limit_train_batches = 0.4,
                    #val_check_interval=0.5,
                    #limit_val_batches = ,
                    val_check_interval=0.2,
                    num_sanity_val_steps=0,
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
    def load_nlgmodel(model_name="NLG_rt", model_version=11,max_input_len=None):
        # Loading in NLG model
        checkpoint = TrainingModule.get_ckpt_file(f'./models/{model_name}/version_{model_version}/checkpoints')

        # Getting tparams
        tparams = {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
            'batch_size', 'lr_schedule', 'learning_rate','precision','splits','optimizer_type',
            'tag']}

        tparams['mode'] = 'inference'

        mparams =  {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
            'base_model_name','loss_type','model_name','fda','frst','ftopic','max_input_len',
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
          
        if torch.cuda.is_available():
            nlg_model =nlg_model.cuda()
        
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
        
        # if step_name == "val" and rank_zero_only.rank == 0 :
             
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
        #             #utterance = self.nlg_tokenizer.e2m_tokenizer.decode( 
                    
        #             datum = { 'val_round': -1,
        #                         'rst_rels': sum( rst_rels, ()),
        #                         "topics": sum( topics, () ),
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
        #                             skip_special_tokens=False)
        #         datum = {
        #             'val_round':df['val_round'].max()+1,
        #             'rst_rels': '',
        #             'topics':'',
        #             'utterance':json.dumps(decoded_text) }
        #         df = df.append(datum, ignore_index=True)
        #         df.to_csv( fp, index=False)
        #         # Saving to file
                   
    def create_data_loaders(self, shuffle=False, **kwargs):
       
        dg = DataLoaderGenerator(self.dir_data,  self.batch_size, self.model.nlg_tokenizer, 
                workers=self.workers, mode=self.mode, split=self.data_splits,
                fda=self.model.fda, frst=self.model.frst,
                ftopic=self.model.ftopic,
                inference_context_utt=self.inference_context_utt)

        _dict_dl = dg()
        self.train_dl = _dict_dl['train_dl']
        self.val_dl = _dict_dl['val_dl']
        self.test_dl = _dict_dl['test_dl']
        self.inference_dl = _dict_dl['inference_dl']

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
            optimizer = torch.optim.Adafactor(
                self.model.parameters(), lr=self.learning_rate,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=True,
                warmup_init=False
                )

            return [optimizer]
            raise NotImplementedError

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
                    fda=True, frst=True, ftopic=True,
                    inference_context_utt=0,
                    **kwargs):
        
        self.dir_data = dir_data
        self.tokenizer = tokenizer
        self.splits = splits

        self.bs = batch_size
        self.workers  = workers
        self.mode = mode

        self.fda = fda
        self.frst = frst 
        self.ftopic = ftopic
        
        self.inference_context_utt = inference_context_utt

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

    def prepare_dataloader(self, dir_data, shuffle=False, 
        split_name='train'):

        """Prepares a dataloader given a directory of data for NLG language module
            # The current method takes a percentage of data from each subdirectory
            Args:
                dir_dset ([type]): [description]
        """
        #getting all files from all different subreddits/types of conversation
        fns = glob.glob(  os.path.join( utils.get_path(dir_data),"*","*") )
        fns = [fn for fn in fns if os.path.split(fn)[-1]!="lock"]
        #getting number of utterances records in each file
        files_sizes = [ int(fn[-10:]) for fn in fns]

        #defining starting line and total lines to use for dataset
        if split_name == 'train':
            line_starts = [0]*len(files_sizes)
            line_ends = [ ls+int(fs*self.splits['train']) for ls,fs in zip(line_starts, files_sizes)  ]
            shuffle = True
            ifc = 0
        
        elif split_name == 'val':
            line_starts = [ int(fs*self.splits['train']) for fs in files_sizes  ]
            line_ends = [ ls+int(fs*self.splits['val']) for ls,fs in zip(line_starts, files_sizes)  ]
            shuffle = False
            ifc = 0

        elif split_name == 'test':
            line_starts = [ int(fs*(1-self.splits['test']) ) for fs in files_sizes  ]
            line_ends = files_sizes
            shuffle = False
            ifc = 0

        elif split_name == 'inference':
            line_starts = [ random.randrange( int(fs*(1-self.splits['test'])), fs) for fs in files_sizes  ]
            line_ends =  files_sizes
            shuffle = False
            ifc = self.inference_context_utt

        li_dsets = [ SingleDataset(_f, self.tokenizer, line_start, line_end, self.fda, self.frst, self.ftopic, ifc) 
                        for _f, line_start, line_end in zip(fns, line_starts, line_ends) ]

        if split_name == 'inference':
            random.sample(li_dsets,10)
            bs = 1
        else:
            bs = self.bs

        concat_dset = torch.utils.data.ConcatDataset(li_dsets)

        dataloader = torch.utils.data.DataLoader(concat_dset, batch_size=bs,
            shuffle=shuffle, num_workers=self.workers, collate_fn=default_collate)
        
        return dataloader

    def __call__(self):
        dict_dl = self.prepare_dataloaders()
        return dict_dl
    
class SingleDataset(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

    """
    def __init__(self, file_path, tokenizer, line_start, line_end, fda, 
                    frst, ftopic, inference_context_utt=0 ):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.line_start = line_start
        self.line_end = line_end

        self.fda = fda
        self.frst = frst 
        self.ftopic = ftopic

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
    
    def __getitem__(self, index,pad_utterance=True):
        
        das, rst_rels, rst_ns, rst_pos, topics, topics_score, utterance = self.getitem_extract_datum(index)
        
        if self.inference_context_utt != 0:
            utterance_context = ' '.join( utterance.split(' ')[:self.inference_context_utt] )
            
            encoded = self.getitem_tokenize(das, rst_rels,rst_ns, rst_pos,
                                         topics, 
                                        topics_score, utterance_context,
                                        pad_utterance=False,
                                        generate_mode=True )

            encoded['orig_rst_rels'] = rst_rels
            encoded['orig_utt'] = utterance
            encoded['orig_topics'] = topics
        
        else:
        
            encoded = self.getitem_tokenize(das, rst_rels,  rst_ns, rst_pos,
                                         topics, 
                                        topics_score, utterance,
                                        pad_utterance=pad_utterance )

            # encoded May include some of the following
            #( da_start_token, tnsr_das,    
             #rst_start_token, tnsr_rst_rels,
             #tnsr_topics_phrase, tnsr_topics_score, 
             # tknzd_utt,
             # attn_mask
             # labels)      

        return encoded

    def getitem_extract_datum(self, index):

        datum = self.data[index:index+1]

        #Dialogue Act
        if self.fda:
            das = json.loads(datum['li_da'].values[0])
        else:
            das = None
        
        #RST
        if self.frst:
            li_rst = json.loads(datum['rst'].values[0])  #list of dictionaries 
        
            # relation r is the relation between Discourse Unit A and Discourse Unit B 
            # ns is the nuclearity value for a relation r from unit A to  unit B. (e.g. sn means that A is the sattellite and B is the nucleiod)
            # pos is the position in a binary tree of Discourse Unit A

            rst_rels = [ _dict['rel'] for _dict in li_rst ]
            rst_ns = [ _dict['ns'] for _dict in li_rst ]
            rst_pos = [ _dict['pos'] for _dict in li_rst ]
        
        else:
            rst_rels = None
            rst_ns = None
            rst_pos = None
        
        #Topic scores
        if self.ftopic:
            topics_textrank = json.loads(datum['topic_textrank'].values[0])

            topics, topics_score = zip( *topics_textrank ) #top 3 important words from utterance

        else:
            topics = None
            topics_score = None
        
        #Utterance
        utterance = json.loads( datum['txt_preproc'].values[0] )
        
        
        return das, rst_rels, rst_ns, rst_pos, topics, topics_score, utterance

    def getitem_tokenize(self, das, rst_rels, rst_ns, rst_pos ,topics, topics_score,
        utterance,pad_utterance=True, generate_mode=False):
        
        if self.fda and self.frst:
            encoded = self.tokenizer.encode_v2(das, rst_rels,  rst_ns, rst_pos,
                        topics, topics_score, utterance,
                        pad_utterance=pad_utterance,
                        generate_mode=generate_mode)

        elif self.fda == False and self.frst:
            encoded = self.tokenizer.encode_v2_exda(rst_rels, rst_ns, rst_pos ,
                        topics, topics_score, utterance,
                        pad_utterance=pad_utterance, generate_mode=generate_mode)

        return encoded

def main(tparams={}, mparams={}):
    #gc.collect()
    #torch.cuda.empty_cache()

    #Adapting Model Name to handle testing of different scenarios
    if mparams['fda'] and mparams['frst'] and mparams['ftopic']:
        mparams['model_name'] = f"{mparams['model_name']}_drt"
    
    elif not mparams['fda'] and mparams['frst'] and mparams['ftopic']:
        mparams['model_name'] = f"{mparams['model_name']}_rt"
     
    else:
        NotImplementedError
    
    
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
        os.environ['MASTER_ADDR'] = 'localhost' #'127.0.0.1'
        os.environ['MASTER_PORT'] = '65302'

    main(vars(tparams), vars(mparams))


# dullduks server version 1 - No Freezing, Full RST

# CUDA_VISIBLE_DEVICES=1 python3 train_nlg.py -bs 100 -agb 1 --gpus 1 -fda 0 -fp 0 -frstv 1 --workers 8 --version 1 --precision 16 --mode train_new -lr 4e-4 -me 60 -mil 160 --tag "no freezing full rst" --base_model_name "distilgpt2"
# CUDA_VISIBLE_DEVICES=1 python3 train_nlg.py -bs 100 -agb 1 --gpus 1 -fda 0 -fp 0 -frstv 1 --workers 8 --version 11 --precision 16 --mode train_new -lr 1e-5 -me 60 -mil 160 --tag "no freezing full rst, lower learning rate" --base_model_name "distilgpt2"
# CUDA_VISIBLE_DEVICES=1 python3 train_nlg.py -bs 60 -agb 2 --gpus 1 -fda 0 -fp 0 -frstv 1 --workers 8 --version 12 --precision 16 --mode train_new -lr 1e-4 -me 90 -mil 160 --tag "no freezing full rst, lower learning rate but using normal sized gpt and full sized dset" --base_model_name "gpt2" --dir_data "./dataset/reddit_large_annotated_long2"
# CUDA_VISIBLE_DEVICES=1 python3 train_nlg.py -bs 64 -agb 5 --gpus 1 -fda 0 -fp 0 -frstv 1 -sgbf 1 --workers 4 --version 13 --precision 16 --mode train_new -lr 5e-4 -me 50 -mil 160 --tag "no freezing full rst, normal sized gpt and proper full sized dset, inverse_grad_freq used in embedding layer " --base_model_name "gpt2" --dir_data "./dataset/reddit_large_annotated_fixed"

# CUDA_VISIBLE_DEVICES=1 python3 train_nlg.py -bs 86 -agb 6 --gpus 1 -fda 0 -fp 0 -frstv 1 -sgbf 1 -cl '{ "rst":16, "topics":30 }'  --workers 4 --version 14 --precision 16 --mode train_new -lr 6e-4 -me 50 -mil 200 --tag "no freezing full rst, distilgpt and proper full sized dset (with additions from new data_v2), inverse_grad_freq used in embedding layer, larger context_len for rst and topics " --base_model_name "distilgpt2" --dir_data "./dataset/reddit_large_annotated_fixed"
# CUDA_VISIBLE_DEVICES=1 python3 train_nlg.py -bs 44 -agb 6 --gpus 1 -fda 0 -fp 0 -frstv 1 -sgbf 1 -cl '{ "rst":16, "topics":30 }'  --workers 4 --version 15 --precision 16 --mode train_new -lr 6e-4 -me 50 -mil 200 --tag "no freezing full rst, gpt and proper full sized dset (with additions from new data_v2), inverse_grad_freq used in embedding layer, larger context_len for rst and topics " --base_model_name "gpt2" --dir_data "./dataset/reddit_large_annotated_fixed"

# CUDA_VISIBLE_DEVICES=0,1 python3 train_nlg.py -bs 86 -agb 3 --gpus 1 -fda 0 -fp 0 -frstv 1 -sgbf 1 -cl '{ "rst":16, "topics":30 }' --workers 6 --version 16 --precision 16 --mode train_new -lr 6e-4 -me 50 -mil 200 --tag "no freezing full rst, gpt2 and proper full sized dset (with iteration 2 of new data_v2), inverse_grad_freq used in embedding layer, larger context_len for rst and topics " --base_model_name "gpt2" --dir_data "./dataset_v2/reddit_large_annotated"
# CUDA_VISIBLE_DEVICES=0,1 python3 train_nlg.py -bs 48 -agb 4 --gpus 2 -fda 0 -fp 0 -frstv 1 -sgbf 1 -cl '{ "rst":16, "topics":30 }' --workers 6 --version 161 --precision 16 --mode train_new -lr 6e-4 -me 50 -mil 200 --tag "no freezing full rst, gpt2 and proper full sized dset (with iteration 2 of new data_v2), inverse_grad_freq used in embedding layer, larger context_len for rst and topics, amended issue where txt_preproc was not being josn parsed so qoutation marks existed around text " --base_model_name "gpt2" --dir_data "./dataset_v2/reddit_large_annotated"
# CUDA_VISIBLE_DEVICES=0 python3 train_nlg.py -bs 24 -agb 10 --gpus 1 -fda 0 -fp 0 -frstv 1 -sgbf 1 -cl '{ "rst":16, "topics":30 }' --workers 6 --version 17 --precision 16 --mode train_new -lr 18e-4 -me 70 -mil 200 --tag "no freezing full rst, gpt2 and proper full sized dset (with iteration 2 of new data_v2), inverse_grad_freq used in embedding layer, larger context_len for rst and topics " --base_model_name "gpt2-medium" --dir_data "./dataset_v2/reddit_large_annotated"

# python3 train_nlg.py -bs 112 -agb 1 --gpus 2 -fda 0 --workers 16 --version 41 -opt AdamW --precision 16 --mode test


# dullduks server version 2 - No Freezing, partial RST
# CUDA_VISIBLE_DEVICES=0 python3 train_nlg.py -bs 300 -agb 1 --gpus 1 -fda 0 -fp 0 -frstv 0 --workers 8 --version 2 --precision 16 --mode train_new -lr 4e-4 -me 60 -mil 1660 --tag "no freezing partial rst" --base_model_name "distilgpt2"
# python3 train_nlg.py -bs 40 -agb 1 --gpus 2 -fda 0 --workers 16 --version 42 -opt AdamW --precision 16 --mode test

# version 3 - Freezing, Full RST
# CUDA_VISIBLE_DEVICES=1,2 python3 train_nlg.py -bs 40 -agb 2 --gpus 2 -fda 0 -fp 1 -frstv 1 --workers 8 --version 3 --precision 16 --mode train_new -lr 1e-5 -me 80 -mil 160 --tag "freezing, full rst" --base_model_name "distilgpt2"
# CUDA_VISIBLE_DEVICES=1 python3 train_nlg.py -bs 60 -agb 2 --gpus 1 -fda 0 -fp 1 -frstv 1 --workers 8 --version 31 --precision 16 --mode train_new -lr 1e-4 -me 80 -mil 160 --tag "freezing, full rst, gpt2, dataset with sentences with two sections" --base_model_name "gpt2" --dir_data "./dataset/reddit_large_annotated_long2"
# CUDA_VISIBLE_DEVICES=1 python3 train_nlg.py -bs 64 -agb 5 --gpus 1 -fda 0 -fp 1 -frstv 1 -sgbf 0 --workers 4 --version 32 --precision 16 --mode train_new -lr 5e-4 -me 80 -mil 160 --tag "freezing, full rst, gpt2, full dataset" --base_model_name "gpt2" --dir_data "./dataset/reddit_large_annotated_fixed"
# CUDA_VISIBLE_DEVICES=1 python3 train_nlg.py -bs 64 -agb 5 --gpus 1 -fda 0 -fp 1 -frstv 1 -sgbf 1 --workers 4 --version 33 --precision 16 --mode train_new -lr 5e-4 -me 80 -mil 160 --tag "freezing, full rst, gpt2, full dataset, inverse_freq_grad to embedding" --base_model_name "gpt2" --dir_data "./dataset/reddit_large_annotated_fixed"
# CUDA_VISIBLE_DEVICES=1 python3 train_nlg.py -bs 50 -agb 6 --gpus 1 -fda 0 -fp 1 -frstv 1 -sgbf 1 -cl '{ "rst":16, "topics":30 }'  --workers 4 --version 35 --precision 16 --mode train_new -lr 8e-4 -me 20 -mil 200 --tag "partial freezing full rst, distilgpt and proper full sized dset (with additions from new data_v2), inverse_grad_freq used in embedding layer, larger context_len for rst and topics " --base_model_name "distilgpt2" --dir_data "./dataset/reddit_large_annotated_fixed"