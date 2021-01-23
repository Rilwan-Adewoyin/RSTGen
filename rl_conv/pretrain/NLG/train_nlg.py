import os
#os.environ['export NCCL_DEBUG']='INFO'
#os.environ['NCCL_IB_DISABLE'] = "1"
#export NCCL_DEBUG=INFO
#os.environ['NCCL_P2P_DISABLE']='1'

os.environ['CUDA_LAUNCH_BLOCKING']="1"

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
from torch.nn import CrossEntropyLoss

from sklearn import preprocessing as sklp

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
import utils_nlg as utils
import random 

import transformers
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from transformers import Adafactor, AdamW

from pytorch_lightning import loggers as pl_loggers
from collections import OrderedDict
import yaml
import ast
import types
from functools import wraps
from copy import deepcopy

from itertools import permutations, combinations, combinations_with_replacement
from typing import Optional, Callable, Union, Optional, List, Iterable
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
    
    input_ids = None # Our model should ignore any input_ids entered into the model

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
        attention_mask = attention_mask.type_as(inputs_embeds) # fp16 compatibility
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

def get_output_embeddings(self):
    return self.lm_head


class NLG(nn.Module):
    """NLG unit
    """

    def __init__(self, 
        base_model_name= 'distilgpt2', model_name="NLG",
        reset_base_transformer=True, loss_type="CrossEntropy",
        fda=True, frst=True, ftopic=True, max_input_len=264 ,**kwargs ):
            #base model uses same code as 'microsoft/DialoGPT-small'
        super(NLG, self).__init__()
        
        self.base_model_name = base_model_name   
        self.model_name = model_name
        
        self.fda = fda 
        self.frst = frst
        self.ftopic = ftopic

    
        # Retreive/Instantiate base transformer
        self.transformer = utils.load_pretrained_transformer(self.base_model_name, transformer=True)['transformer']    
        

        self.nlg_tokenizer = NLG_tokenizer(base_model_name,
                                os.path.join( ("./models"), f"{model_name}_tokenizer"),
                                fda=fda, frst=frst, ftopic=ftopic,
                                 **kwargs)
        
        self.transformer.resize_token_embeddings( len(self.nlg_tokenizer.e2m_tokenizer) )
        self.transformer.forward = types.MethodType(forward,self.transformer) #monkey patch
        self.transformer.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        # For compatibility with hugging face generate
        self.transformer.get_output_embeddings = lambda : self.lm_head
        # Embedding Layers
        
        self.embd_outp_dim = self.transformer.config.n_embd
        if self.fda and self.frst:
            self.embedding_das = torch.nn.Conv1d( 12, self.embd_outp_dim, kernel_size=1 )
            self.embedding_rst_rels = torch.nn.Conv1d( 19, self.embd_outp_dim, kernel_size=1 )
            self.token_type_embeddings = torch.nn.Embedding( 4 + self.nlg_tokenizer.context_len['topics']//2, self.embd_outp_dim) #The maximum value this can take is based on the different types of input
                                            #1 for each of da, rst, utterance and + 1 for each topic phrase (note that each topic phrase includes a <topic> token.
                                            #      therefore the largest number of different topics is topic_ctx//2 if every topic only has one word)
        
        if self.fda == False and self.frst:
            self.embedding_rst_rels = torch.nn.Conv1d( 19, self.embd_outp_dim, kernel_size=1 )
            self.token_type_embeddings = torch.nn.Embedding( 3 + self.nlg_tokenizer.context_len['topics']//2, self.embd_outp_dim) #The maximum value this can take is based on the different types of input
                                            #1 for each of da, rst, utterance and + 1 for each topic phrase (note that each topic phrase includes a <topic> token.
                                            #      therefore the largest number of different topics is topic_ctx//2 if every topic only has one word)
        
        self.embedding_topics_score = torch.nn.Conv1d( 1, self.embd_outp_dim, kernel_size=1)
        
        self.lm_head = nn.Linear( self.embd_outp_dim, self.transformer.config.vocab_size, bias=False  )
        self.lm_head.weight.data.normal_(mean=0.0, std=0.02)
        

        
        self.loss_type = loss_type 
        self.loss_fct = CrossEntropyLoss()

    @torch.no_grad()
    def generate(self, 
        _input,
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
        input_ = self.forward_embedding(encoded_input) 
        input_embeds = input_['input_embeds']
        attention_mask = input_['attn_mask']
        position_ids = input_['position_ids']
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
            batch_size = inputs_embeds.shape[0]  #changed here : overriden by the input batch_size
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
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
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
            input_embeds_dim = input_embeds.shape[-1]

            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            input_embeds = input_embeds.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len, input_embeds_dim) #Change
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            input_embeds = input_embeds.contiguous().view(
                effective_batch_size * num_beams, input_ids_len, input_embeds_dim
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len, dim1)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len, input_ids_len
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
                inputs_ids = input_ids,
                inputs_embeds = input_embeds,
                position_ids=position_ids,
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
                    model_specific_kwargs=model_specific_kwargs,
                **generate_params )
                #may need to add other special tokens to the mix here

        else:
            output = self._generate_no_beam_search(
                inputs_ids = input_ids,
                inputs_embeds = input_embeds,
                position_ids=position_ids,
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
        position_ids,
        token_type_ids,

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
        attention_mask,
        use_cache,

        **model_specific_kwargs,
     ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = (encoder_outputs, None) if encoder_outputs is not None else None

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask,
                position_ids=position_ids ,use_cache=use_cache, 
                token_type_ids = token_type_ids,
                **model_specific_kwargs
            )

            lm_logits = self(model_inputs, True )
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
                past = outputs[1]

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

            # definining new inputs to append to old inputs
            next_input_embed =  self.trasnformer.wte(tokens_to_add) # (batch, 1)

            if self.fda and self.frst:
                next_token_type_ids = tokens_to_add.new(next_input_embed.shape[1])._fill(4) #Token type id of context utterance
            elif self.fda==False and self.frst:
                next_token_type_ids = tokens_to_add.new(next_input_embed.shape[1])._fill(3) #Token type id of context utterance

            #Joining new outputs to existing or redifining old inputs
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1) 
            input_embeds = torch.cat( [input_embeds, next_input_embed ], axis=-2 ).contiguous()
            
            position_ids = torch.arange(1, input_embeds.shape[-2], dtype=torch.long, device=input_embeds.device )
            position_ids = position_ids.unsqueeze(0).view(-1, input_embeds.shape[-2])

            token_type_ids = torch.cat( [token_type_ids, next_token_type_ids[None, ...] ] )
                
                #The new utterance token will be able to attend to all prev values
            old_attnm_shape = attention_mask.shape() #bs, old_seq_len, old_seq_len
            _ = old_attnm_shape.shape
            new_attn_mask = torch.emmpty( [_[0],_[1]+1,_[2]+2], device=old_attnm_shape.device )
            
            new_mask_col = attention_mask.new( attention_mask.shape[-2] )._fill(0)
            new_mask_row = attention_mask.new( attention_mask.shape[-1] )._fill(1)
            new_attn_mask[:, -1:, : ] = new_mask_col.repeat([_[0],1,1])
            new_attn_mask[:, :, -1: ] = new_mask_row.repeat([_[0],1,1])
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
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids

    def _generate_beam_search(
        self,
        input_ids,
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
        attention_mask,
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

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask,
                position_ids=position_ids , 
                token_type_ids = token_type_ids, use_cache=use_cache, **model_specific_kwargs
            )
            lm_logits = self(model_inputs, True)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = lm_logits[:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
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
            cur_len = cur_len + 1

            #region changed : creating / adding to model inputs
            input_embeds = input_embeds[beam_idx, :, :]
            next_input_embed =  self.trasnformer.wte(beam_tokens.unsqueeze(1)) # (batch, 1)
            input_embeds = torch.cat([input_embeds, next_input_embed ] , dim=-2 )

            if self.fda and self.frst:
                next_token_type_ids = tokens_to_add.new(next_input_embed.shape[1])._fill(4) #Token type id of context utterance
            elif self.fda==False and self.frst:
                next_token_type_ids = tokens_to_add.new(next_input_embed.shape[1])._fill(3) #Token type id of context utterance
            token_type_ids = torch.cat( [token_type_ids, next_token_type_ids[None, ...] ] )

            #Joining new outputs to existing or redifining old inputs                        
            position_ids = torch.arange(1, input_embeds.shape[-2], dtype=torch.long, device=input_embeds.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_embeds.shape[-2])
                       
                #The new utterance token will be able to attend to all prev values
            old_attnm_shape = attention_mask.shape() #bs, old_seq_len, old_seq_len
            _ = old_attnm_shape.shape
            new_attn_mask = torch.emmpty( [_[0],_[1]+1,_[2]+2], device=old_attnm_shape.device)
            
            new_mask_col = attention_mask.new( attention_mask.shape[-2] )._fill(0)
            new_mask_row = attention_mask.new( attention_mask.shape[-1] )._fill(1)
            new_attn_mask[:, -1:, : ] = new_mask_col.repeat([_[0],1,1])
            new_attn_mask[:, :, -1: ] = new_mask_row.repeat([_[0],1,1])
            attention_mask = new_attn_mask.contiguous()

            #endregion
            

            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
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

    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
                
        parser.add_argument('--base_model_name', default='distilgpt2', required=False)
        parser.add_argument('--reset_base_transformer', default=False, required=False, type=bool)
        parser.add_argument('--model_name', default='NLG', required=False)
        parser.add_argument('--loss_type', default='CrossEntropy', required=False, 
            choices=['CrossEntropy','UtteranceSimilarity']) 
        
        parser.add_argument( '-fda' ,'--fda', type= lambda x: bool(int(x) )
             ,help="whether or not to include da in feature", default=True  )
        
        parser.add_argument( '-frst' ,'--frst', type= lambda x: bool(int(x) )
             ,help="whether or not to include rst info in feature", default=True  )

        parser.add_argument( '-ftopic' ,'--ftopic', type= lambda x: bool(int(x) )
             ,help="whether or not to include topic info in feature", default=True  )
        
        parser.add_argument('-mil','--max_input_len', type=int, default=264)
        
        mparams = parser.parse_known_args( )[0]
       
        return mparams

    # def __call__(self, input_, skip_embed1=False):

    #     if skip_embed1 == False:
    #         input_ = self.forward_embedding(input_)
        
    #     outputs = self.transformer(input_)
    #     lm_logits = self.lm_head( outputs[0] )

    #     return lm_logits # Only returns the logits from the output layer


    def forward_embedding(self, input_):
        #Partially does the emebddign for our new inputs to the transformer

        # Creating embedded inputs and attention mask
        if self.fda and self.frst:
            input_ = self.layer_embedding( input_ )
        elif self.fda==False and self.frst:
            input_ = self.layer_embedding_exda( input_ )
        
        return input_

    def forward(self, input_, skip_embed1=False):
        """[summary]

        Args:
            input_ (torch.tensor): dict of inputs

        Returns:
            [type]: [description]
        """
        # Handles embedding of our new non word features
        if skip_embed1 == False:
            input_ = self.forward_embedding(input_)
                
        # Feed input to distilgpt2
        
        outputs = self.transformer( inputs_embeds=input_['input_embeds'],
                                    attention_mask = input_['attn_mask'],
                                    position_ids=input_['position_ids'], #check pos_ids are being removed
                                    token_type_ids = None, #token type embedding new (This gpt implementation incorrectly uses same embedding layer as for input)
                                                            # Further we handle token_type embedding in forward_embedding layer
                                    return_dict=False)
        hidden_states = outputs[0]
        lm_logits = self.lm_head( hidden_states )

        if 'labels' in input_:
            
            if self.loss_type == "CrossEntropy":      
        
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = input_['labels'][..., 1:].contiguous() 
                
                loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            elif self.loss_type == "UtteranceSimilarity":
                raise NotImplementedError
    
            return (lm_logits, loss) 

        else:
            return lm_logits
    
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
        token_type_embedding = self.token_type_embeddings( input_['token_type_ids'] )

        input_embeds += token_type_embedding
        input_['input_embeds'] = input_embeds
        
        return input_

    def layer_embedding_exda(self, input_):
        """Performs the input embedding and token type embedding calcs

            Args:
                input_ ([type]): [description]

            Returns:
                input_embedded [type]: [description]
                
                Note: logic for token_type_ids embedding is usually performed in transformer, but has been moved here
                    since gpt-2 code indicates that 
        """
        # token embedding

        rst_start_embed = self.transformer.wte( input_['rst_start_token'] ).unsqueeze(1)
        rst_embed = self.embedding_rst_rels( input_['tnsr_rst_rels'] ).permute(0,2,1) # (bs, channels, seq_len)

        topics_phrase_embed = self.transformer.wte(input_['tnsr_topics_phrase']  )  #TODO: Add positional encoding to each sub-phrase
        topics_score_embed = self.embedding_topics_score( input_['tnsr_topics_score']).permute(0,2,1)

        topics_embed = topics_phrase_embed + topics_score_embed

        
        utt_embed = self.transformer.wte(input_['tknzd_utt'] )

        input_embeds = torch.cat(
            [rst_start_embed, rst_embed,
             topics_embed,
             utt_embed], axis = 1
            ) #dim [bs, 1024, dim1]
        
        # token type embeddings
        token_type_embedding = self.token_type_embeddings( input_['token_type_ids'] )

        input_embeds += token_type_embedding
        input_['input_embeds'] = input_embeds
        
        
        return input_

    def return_params(self):
        params = {}

        params['base_model_name'] = self.base_model_name
        params['loss_type'] = self.loss_type 
        params['max_input_len'] = self.nlg_tokenizer.max_input_len
        params['fda'] = self.fda
        params['frst'] = self.frst
        params['ftopic'] = self.ftopic

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

    def prepare_inputs_for_generation(self, input_ids, input_embeds, attention_mask, position_ids ,past=None, **kwargs):
        
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            input_embeds = input_embeds[:, -1, :].unsqueeze(-2)
            #TODO: may also have to crop the input_embeds and attneiton_mask
        
        return {
            "input_ids": input_ids,
            'input_embeds':input_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,

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
                 ftopic = True,
                 max_input_len = 216,                 
                 **kwargs
                 ):


        self.e2m_base_model_name = e2m_base_model_name

        self.fda = fda
        self.frst = frst
        self.ftopic = ftopic

        self.max_input_len = max_input_len #512 #self.e2m_tokenizer.max_len
    
        # Setting up RST utilities
            #TODO: add case when rel == 'n' when it could not be classified
        if self.frst:
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

        # Setting up context lengths
        if self.fda and self.frst and self.ftopic:
            self.context_len = { 'da':2, 'rst':8, 'topics':16 } #add this to config
        
        elif self.fda==False and self.frst and self.ftopic:
                self.context_len = { 'rst':6, 'topics':14 } #add this to config
        
        self.context_len['utt'] = self.max_input_len - sum(self.context_len.values())

        
        
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
                self.e2m_tokenizer.save_pretrained(dir_tokenizer)
                config.save_pretrained(dir_tokenizer)

    def rst_vectors(version="combinations", relations="all", **kwargs):
        """
            Allows the user to select partiuclar rst_vectors in order to control their output

            version: rule to decide how to compose relations
            relations: A list of the relations to utilise
         """
        assert ( count>0 and count<7 )

        if relations == "all":
            relations = [[rel] for rel in self.rst_rel_li]
        
        assert type(relations)== list

        if version=="combinations":
            
            combination_count = kwargs.get('combinatoric_count',3)
            iter_rst_comb = combinations( rst_rel_li, combination_count )
            li_rst_comb =  list(iter_rst_comb)
            li_rst_comb = random(li_rst_comb)

            li_rst_comb = li_rst_comb[: kwargs.get("return_count",10) ]           


            rst_rel_encoded = self.rst_rel_binarizer.transform( li_rst_comb ) #list of list of each relation
        
        elif version=="permutations":
            
            combination_count = kwargs.get('combinatoric_count',3)
            iter_rst_perm = permutations( rst_rel_li, combination_count )
            li_rst_perm =  list(iter_rst_perm)
            li_rst_perm = random(li_rst_perm)

            li_rst_perm = li_rst_perm [: kwargs.get("return_count",10) ]           

            rst_rel_encoded = self.rst_rel_binarizer.transform( li_rst_perm ) #list of list of each relation

        elif version=="combinations_with_replacement":
            
            combination_count = kwargs.get('combinatoric_count',3)
            iter_rst_combwr = combinations_with_replacement( rst_rel_li, combination_count )
            li_rst_combwr =  list(iter_rst_combwr)
            li_rst_combwr = random(li_rst_combwr)

            li_rst_combwr = li_rst_combwr[: kwargs.get("return_count",10) ]           


            rst_rel_encoded = self.rst_rel_binarizer.transform( li_rst_perm ) #list of list of each relation

        return rst_rel_encoded


    def __call__(self, das=None, rst_rels=None, topics=None, topics_score=None, 
                    utterance=None, prev_das=None, prev_rst=None,
                    stem_context_utterance = -1, pad_utterance=True):
        
        #Stem context utterance decides whether or not to reduce the size of the context.
            #This is only helpful when evaluating how model produces output given a fixed small starting to the sentence

        if stem_context_utterance != -1:
            utterance = ' '.join( utterance.split(' ')[:stem_context_utterance] )

        if self.fda and self.frst and self.ftopic:

            outp = self.encode_v2(das ,rst_rels,topics, topics_score, 
                    utterance,pad_utterance)
        
        elif self.fda==False and self.frst and self.ftopic:
            
            outp = self.encode_v2_exda(rst_rels, topics, topics_score, 
                    utterance,pad_utterance)
        else:
            NotImplementedError
            
        return outp

    def encode_v2( self, das ,rst_rels,topics, topics_score, 
                    utterance, pad_utterance):

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
        tnsr_rst_rels, rst_pad_count = self.encode_rst_v2(rst_rels, max_padding=self.context_len['rst'] - 1)   # dims (max_padding, n) #not we do rt_len-1 since we add the rst start token outside this method
        tnsr_topics_phrase, tnsr_topics_score, topics_pad_count, ta_tokens_pos  = self.encode_topic_v2( topics, topics_score, max_padding=self.context_len['topics'], padding_token=padding_token) # dims (max_padding, 13) 
        tknzd_utt, utt_pad_count = self.encode_utterance_v2(utterance, pad_utterance)
                            
        # Building Attention Mask

            # calc the ending cumulative dim for da, rst, topics, utt, segments,
        d_dim = self.context_len['da']
        dr_dim = da_len + self.context_len['rst']       # d_dim + tnsr_rst_rels.shape[0]+ 1
        drt_dim = dr_dim + self.context_len['topics']    # dr_dim + tnsr_topics_phrase.shape[1]
        utt_dim = drt_dim + self.context_len['utt']  

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

        #Creating labels/targets for GPT Language Model Head
        labels = -100* torch.ones( size=[1, self.max_input_len], dtype = torch.long  ) 
        
        labels[0][drt_dim:utt_dim] = tknzd_utt[:utt_dim- drt_dim]


        # Creating Positional Emebeddings
            # ALL words in drt get a positional encoding of 0 -> No positional meaning
            # utterance has normal positional encoding        
        position_ids_drt = torch.zeros([drt_dim], dtype=torch.long) 
        position_ids_utt =  torch.arange( 1, utt_dim-drt_dim + 1  , dtype=torch.long)
    
        position_ids = torch.cat([position_ids_drt,position_ids_utt], axis=-1)


        # Creating Token Type Ids
            # 1:da, 
            # 2:rst, 
            # n<m for each word in each topic phrase i of length m_i including leading <ta> where 4>=n>=4+topics_len//2
            # 3:utterance part

        token_type_ids_d = torch.zeros( [self.context_len['da'] ] , dtype=torch.long) + 1
        token_type_ids_r = torch.zeros( [self.context_len['rst']], dtype=torch.long) + 2
        token_type_ids_utt = torch.zeros( [self.context_len['utt']], dtype=torch.long ) + 3
        
        _ = torch.zeros( [self.context_len['topics'] ], dtype=torch.long)
        _[ ta_tokens_pos ] = 1
        token_type_ids_t =  _.cumsum(axis=0) + 3
        token_type_ids = torch.cat( [token_type_ids_d, token_type_ids_r,\
                    token_type_ids_t, token_type_ids_utt] ) 

        return { 'da_start_token':da_start_token, 'tnsr_das':tnsr_das,
                 'rst_start_token':rst_start_token, 'tnsr_rst_rels':tnsr_rst_rels,
                 'tnsr_topics_phrase':tnsr_topics_phrase, 'tnsr_topics_score': tnsr_topics_score,
                 'tknzd_utt':tknzd_utt,
                 'attn_mask':attn_mask,
                 'labels':labels,
                 'position_ids':position_ids,
                 'token_type_ids':token_type_ids
                 }

    def encode_v2_exda( self,rst_rels,topics, topics_score, 
                    utterance, pad_utterance):

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
        rst_start_token = self.e2m_tokenizer.encode("<|rst|>")[0] 
        padding_token =  self.e2m_tokenizer.encode("<|endoftext|>") 


        #Getting Vectors
        tnsr_rst_rels, rst_pad_count = self.encode_rst_v2(rst_rels, max_padding=self.context_len['rst'] - 1)   # dims (max_padding, n) #not we do rt_len-1 since we add the rst start token outside this method
        tnsr_topics_phrase, tnsr_topics_score, topics_pad_count, ta_tokens_pos  = self.encode_topic_v2( topics, topics_score, max_padding=self.context_len['topics'], padding_token=padding_token) # dims (max_padding, 13) 
        tknzd_utt, utt_pad_count = self.encode_utterance_v2(utterance, pad_utterance)
                            
        # Building Attention Mask

            # calc the ending cumulative dim for, rst, topics, utt, segments,
        r_dim = self.context_len['rst']       # tnsr_rst_rels.shape[0]
        rt_dim = r_dim +self.context_len['topics']    # dr_dim + tnsr_topics_phrase.shape[1]
        utt_dim = rt_dim + self.context_len['utt']  

                
            # creating mask
        attn_mask = torch.tril( torch.ones([self.max_input_len,self.max_input_len]))
                    #pre_utterance general masking
        attn_mask[ :rt_dim , :rt_dim ] = 1 

                    #correcting for padding in rst section
        attn_mask[ r_dim-rst_pad_count:r_dim ,:] = 0
        attn_mask[ :, r_dim-rst_pad_count:r_dim] = 0

                    #correcting for padding in topic section
        attn_mask[ rt_dim-topics_pad_count: rt_dim, :  ] = 0
        attn_mask[ :, rt_dim-topics_pad_count: rt_dim ] = 0
                
                #correcting for padding in and after utterance section
        attn_mask[ utt_dim-utt_pad_count: , : ] = 0

        #Creating labels/targets for GPT Language Model Head
        labels = -100* torch.ones( size=[1, self.max_input_len], dtype = torch.long  ) 

        labels[0][rt_dim:utt_dim] =  tknzd_utt[ : utt_dim- rt_dim ]
        
        # Creating Positional Emebeddings
            # ALL words in drt get a positional encoding of 0 -> No positional meaning
            # utterance has normal positional encoding        
        position_ids_rt = torch.zeros([rt_dim], dtype=torch.long) 
        position_ids_utt =  torch.arange( 1, utt_dim-rt_dim + 1  , dtype=torch.long)
        position_ids = torch.cat([position_ids_rt,position_ids_utt], axis=-1)

        # Creating Token Type Ids
            # 1:rst, 
            # n<m for each word in each topic phrase i of length m_i including leading <ta> where 4>=n>=4+topics_len//2
            # 3:utterance part

        token_type_ids_r = torch.zeros( [self.context_len['rst']], dtype=torch.long) + 1
        token_type_ids_utt = torch.zeros( [self.context_len['utt']  ], dtype=torch.long ) + 2
        
        _ = torch.zeros( [self.context_len['topics'] ], dtype=torch.long)
        _[ ta_tokens_pos ] = 1
        token_type_ids_t =  _.cumsum(axis=0) + 2

        token_type_ids = torch.cat( [token_type_ids_r,\
                            token_type_ids_t, token_type_ids_utt] ) 


        return { 'rst_start_token':rst_start_token, 'tnsr_rst_rels':tnsr_rst_rels,
                 'tnsr_topics_phrase':tnsr_topics_phrase, 'tnsr_topics_score': tnsr_topics_score,
                 'tknzd_utt':tknzd_utt,
                 'attn_mask':attn_mask,
                 'labels':labels,
                 'position_ids':position_ids,
                 'token_type_ids':token_type_ids
                 }

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

    def encode_utterance_v2(self, utterance, pad=True):
        #pad: 
        #   set to True during training to ensure all batches have the same length
        #   set to False in the case of Generation in order to work with huggingface .generate()
        #TODO: change to be able to handle batches of data
        #TODO: When you move to training on large seequences performing variable batch sizes to reduce time

        utterance ='<|endoftext|>' + utterance + '<|endoftext|>'
        
        if pad == True:
            encoded = self.e2m_tokenizer( utterance, add_special_tokens=False,
                                        return_attention_mask = False, 
                                        
                                        padding='do_not_pad',
                                        truncation=True, 
                                        max_length= self.context_len['utt'],
                                                                                
                                        return_tensors='pt',
                                        return_length=True,
                                        return_token_type_ids=None)
            
            # Find a way to get utterance length without tokeinzer
            tknzd_utt_no_pad_len = encoded['length'][0]
            pad_count = (self.context_len['utt']) - tknzd_utt_no_pad_len            
            encoded['input_ids'] = torch.cat( [ encoded['input_ids'], torch.LongTensor(1,pad_count).fill_(self.e2m_tokenizer.eos_token_id) ],axis=-1 )
                                           
        
        elif pad == False:
            encoded = self.e2m_tokenizer( utterance, add_special_tokens=False,
                                        return_attention_mask = False, 
                                        padding='do_not_pad',
                                        truncation=True, 
                                        max_length= self.context_len['utt'],
                                        return_tensors='pt',
                                        return_length=True,
                                        return_token_type_ids=None)
            tknzd_utt_no_pad_len = encoded['length'][0]
            pad_count = 0

        
        tknzd_utt = encoded['input_ids'][0]
        
        return tknzd_utt, pad_count

class TrainingModule(pl.LightningModule):

    def __init__(self, model_params, batch_size=20, 
                    dir_data=None, 
                    accumulate_grad_batches=1,
                    max_epochs=100,
                    gpus=1, 
                    learning_rate=1e-3,
                    warmup_proportion=0.1,
                    workers=0,
                    lr_schedule='hard_restarts',
                    mode = 'train_new',
                    data_splits = {'train':0.6,'val':0.2,'test':0.2},
                    optimizer_type="AdamW",
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
            self.create_data_loaders(self.workers)
            self.accumulate_grad_batches = accumulate_grad_batches

        if self.mode in ['train_new','train_cont']:
            self.max_epochs = max_epochs
            self.warmup_proportion = warmup_proportion
            self.lr_schedule = lr_schedule
            self.learning_rate = learning_rate
        
            train_params_to_save = self.return_params()
            model_params_to_save = self.model.return_params()
            self.save_hyperparameters( train_params_to_save )
            self.save_hyperparameters( model_params_to_save )
        
        if self.mode in ['inference']:
            self.eval() 
            self.freeze() 

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data', default="./dataset/reddit_large_annotated", help="Relative directory path for datafiles")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('--max_epochs', default=80, type=int)
        parser.add_argument('-agb','--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('-bs','--batch_size', default=5, type=int)
        parser.add_argument('--learning_rate', default=2e-4, type=float)
        parser.add_argument('--warmup_proportion', default=0.15)
        parser.add_argument('--workers', default=16, type=int) #TODO: change to 6
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--mode',default='train_new', type=str, choices=['train_new','train_cont','test','inference'])
        parser.add_argument('--lr_schedule', default='constant', required=False, choices =['LROnPlateau','hard_restarts','constant'])
        parser.add_argument('--splits', default={'train':0.6,'val':0.2,'test':0.2}, required=False, type=str )
        parser.add_argument('--version', default=None,required=False, type=int, help="The Experimental Versioning for this run" )
        parser.add_argument('--precision', default=16,required=False, type=int, help="Precision to use", choices=[16,32] )
        parser.add_argument('-opt','--optimizer_type', default="AdamW",required=True, type=str, help="Optimizer to use", choices=["AdamW","Adafactor"] )
        
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
        pl.seed_everything(10)
        if tparams['mode'] in ["train_new"]:
            training_module = TrainingModule(**tparams, model_params=mparams  )
            
        elif tparams['mode'] in ["test", "train_cont", "inference"]:
            
            #Retreiving best checkpoint from specific model-version
            # checkpoint_yaml_file = os.path.join( tparams['dir_checkpoints'],"best_k_models.yaml" )
            # scores_dict = yaml.load( open(checkpoint_yaml_file,"r") ) #key= ckptpath, value = val_loss
            # best_ckpt_path = min(scores_dict, key=scores_dict.get)


            # if torch.cuda.is_available():
            #     checkpoint = torch.load(best_ckpt_path)
            # else:
            #     checkpoint = torch.load(best_ckpt_path, map_location=torch.device('cpu'))


            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            #restore/update param files from the checkpoint
            try:
                tparams.update ( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'batch_size', 'lr_schedule', 'learning_rate','precision','splits','optimizer_type']} )

                mparams.update( {k:v for k,v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name','loss_type','model_name','fda','frst','ftopic','max_input_len']} )
            except KeyError:
                pass
            del checkpoint
            #torch.cuda.empty_cache()
            
            #Restore/update Training Module
            training_module = TrainingModule(**tparams, model_params=mparams)
            

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
            patience=30,
            verbose=False,
            mode='min'
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
                        #accelerator='ddp2', amp_level='O2',# use_amp=True,
                        accelerator='dp',
                        #limit_train_batches = 0.8,
                        #track_grad_norm = True,
                        #overfit_batches=5,
                        #fast_dev_run=2, 
                        #log_gpu_memory=True
                        )

        if tparams['mode'] in ["train_cont","test","inference"]:
            #restoring checkpoint             
            checkpoint = TrainingModule.get_ckpt_file( tparams['dir_checkpoints'])

            training_module.load_state_dict(checkpoint['state_dict'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace( **tparams),
                    progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                    check_val_every_n_epoch=1, logger=tb_logger,
                    log_every_n_steps=1,   
                    precision=tparams['precision'],callbacks=callbacks,
                    #accelerator='ddp2',  amp_level='O2', # use_amp=True,
                    accelerator='dp',
                    #limit_train_batches = 0.8,
                    #limit_val_batches = ,
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

            #del checkpoint
            #torch.cuda.empty_cache()

        return trainer
    
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
            training_module.eval() 
            training_module.freeze() 
            raise NotImplementedError   


    def step(self, batch, step_name):
        
        input_= batch
        _, loss = self.forward(input_) #(lm_logits and loss)
        loss_key = f"{step_name}_loss"
        
        if step_name == 'train':
            str_loss_key = "loss"
        else:
            str_loss_key = loss_key       
            #self.log( str_loss_key, loss, sync_dist=True)
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
            #self.log(f"{step_name}_loss", loss, logger=True, prog_bar=True, sync_dist=True)
            self.log(f"{step_name}_loss", loss, logger=True, prog_bar=True)

    def create_data_loaders(self, shuffle=False, **kwargs):
       
        dg = DataLoaderGenerator(self.dir_data,  self.batch_size, self.model.nlg_tokenizer, 
                workers=self.workers, mode=self.mode, split=self.data_splits,
                fda=self.model.fda, frst=self.model.frst,
                ftopic=self.model.ftopic)
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
        
        if self.optimizer_type == "AdamW":
            
            optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
            
            warmup_steps = self.warmup_proportion*self.total_steps()

            #lr_schedule = get_constant_schedule_with_warmup( optimizer, warmup_steps )

            last_epoch = self.current_epoch if self.current_epoch != 0 else -1

            lr_schedule = get_cosine_schedule_with_warmup(optimizer, 
                            warmup_steps, self.total_steps(), 3, last_epoch=last_epoch )

            return [optimizer], [{ "scheduler":lr_schedule ,"interval": "step"}]
        
        elif self.optimizer_type == "Adafactor":
            optimizer = Adafactor(
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
        params['batch_size'] = self.batch_size
        params['accumulate_grad_batches'] = self.accumulate_grad_batches
        params['lr_schedule'] = self.lr_schedule 
        params['learning_rate'] = self.learning_rate
        params['max_epochs'] = self.max_epochs
        params['warmup_proportion'] = self.warmup_proportion 
        params['optimizer_type'] = self.optimizer_type

        return params

class DataLoaderGenerator():
    """Handles the creation of dataloaders for a train, val and test set
    """
    def __init__(self, dir_data, batch_size,
                    tokenizer, 
                    workers=0, mode='train_new',
                    splits={'train':0.6,'val':0.2,'test':0.2},
                    fda=True, frst=True, ftopic=True,
                    **kwargs):
        
        self.dir_data = dir_data
        self.tokenizer = tokenizer
        self.splits = splits
        #label_mapping = json.load(open(utils.get_path("../DialogueAct/label_mapping.json"),"r"))     

        self.bs = batch_size
        self.workers  = workers
        self.mode = mode

        self.fda = fda
        self.frst = frst 
        self.ftopic = ftopic

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

        li_dsets = [ SingleDataset(_f, self.tokenizer, line_start, line_end, self.fda, self.frst, self.ftopic) 
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
    def __init__(self, file_path, tokenizer, line_start, line_end, fda, frst, ftopic, stem_context_utterance = -1 ):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.line_start = line_start
        self.line_end = line_end

        self.fda = fda
        self.frst = frst 
        self.ftopic = ftopic

        # Used to stem the context utterance for observing model's performance with varying levels of context input
        self.stem_context_utterance = stem_context_utterance
        
        skiprows = self.line_start if self.line_start!=0 else None
        with open(self.fp, 'r') as f:
            if self.line_start == 0:
            
                self.data = pd.read_csv(file_path, sep=',', header=0, 
                    skiprows =skiprows, nrows=(self.line_end-self.line_start) )
            
            else: 
                #names = ['text','subreddit','txt_preproc','rst','li_da','topic_textrank']
                names = open(file_path,"r").readline().strip().split(',')
                            
                self.data = pd.read_csv(file_path, sep=',', 
                    names = names,
                    skiprows =skiprows, nrows=(self.line_end-self.line_start) )
                    
    def __len__(self):
        return (self.line_end - self.line_start)
    
    def __getitem__(self, index):
        
        das, rst_rels, topics, topics_score,utterance = self.getitem_extract_datum(index)
                    
        encoded = self.getitem_tokenize(das, rst_rels, topics, 
                                    topics_score, utterance,
                                    pad_utterance=True )

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
            try:
                li_rst = json.loads(datum['rst'].values[0])  #list of dictionaries 
            
            except json.decoder.JSONDecodeError as e:
                li_rst = ast.literal_eval(datum['rst'].values[0])
        
            rst_rels = [ [ _dict['rel'] for _dict in li_rst ] ]
            rst_ns = [ [_dict['ns']] for _dict in li_rst ]
            rst_pos = [ _dict['pos'] for _dict in li_rst ]
        
        else:
            rst_rels = None
            rst_ns = None
            rst_pos = None
        
        if self.ftopic:
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
        else:
            topics = None
            topics_score = None
        
        #Utterance
        utterance = datum['txt_preproc'].values[0].strip('\"')
        
        return das, rst_rels, topics, topics_score,utterance

    def getitem_tokenize(self, das, rst_rels, topics, topics_score,utterance,pad_utterance=True):
        encoded = self.tokenizer(das, rst_rels, topics, 
                                    topics_score, utterance,
                                    prev_das=None, prev_rst=None,
                                    stem_context_utterance=self.stem_context_utterance, pad_utterance=pad_utterance)
        return encoded

def main(tparams={}, mparams={}):
    gc.collect()
    torch.cuda.empty_cache()

    #Adapting Model Name to handle testing of different scenarios
    if not mparams['fda']:
        mparams['model_name'] = f"{mparams['model_name']}_exda"
    if not mparams['frst']:
        mparams['model_name'] = f"{mparams['model_name']}_exrst"
    if not mparams['ftopic']:
        mparams['model_name'] = f"{mparams['model_name']}_extopic"
                
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
    trainer = TrainingModule.instatiate_trainer( tparams,  tb_logger, training_module)
    TrainingModule.start(trainer, tparams, training_module)
                
if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False) 
    
    # add model specific args
    mparams = NLG.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = TrainingModule.parse_train_specific_args(parent_parser)

    if tparams.gpus not in [0,1]:
        #mp.set_start_method('spawn')
        #os.environ['NCCL_P2P_DISABLE']='1'
        os.environ['MASTER_ADDR'] = 'localhost' #'127.0.0.1'
        os.environ['MASTER_PORT'] = '65302'
        #os.environ['LOCAL_RANK'] = '1'
        #os.environ['NCCL_SOCKET_IFNAME'] = "eth0"
        pass


    main(vars(tparams), vars(mparams))


# CUDA_VISIBLE_DEVICES=0,1,2 python3 train_nlg.py -bs 24 -agb 3 --gpus 3 

# Training with no DA
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_nlg.py -bs 28 -agb 2 --gpus 2 -fda 0 --workers 16 --version 40