from ctypes import ArgumentError
import os
# os.environ['NCCL_SOCKET_IFNAME'] = 'ib0' #'lo' eno1,eno2, eno3,eno4,ens2,ens2d1
# os.environ['NCCL_SOCKET_IFNAME'] = 'lo' #'lo' eno1,eno2, eno3,eno4,ens2,ens2d1

# os.environ['NCCL_DEBUG'] = "INFO"
# os.environ['NCCL_SOCKET_IFNAME'] = 'enp226s0f0'
# os.environ['NCCL_SOCKET_IFNAME'] = 'enp226s0f1'
# os.environ['NCCL_SOCKET_IFNAME'] = 'docker0'
os.environ['TOKENIZERS_PARALLELISM'] = "true"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


from transformers.utils import logging
from seg_bot_segmenter import Segmenter, Lang, PointerNetworks
from utils_nlg_v3 import EmbeddingRstPos, RstModelMixin, DegenerateLossMixin
import utils_nlg_v3 as utils
import operator
import functools
import string
import argparse
import copy
import glob
import inspect
import json
import pickle
import random
import traceback
import types
import contextlib
import time
from itertools import islice
from pathlib import Path
from typing import (Any, Dict, Iterator, List, Optional, OrderedDict, TypeVar)
import einops
import numpy as np
import pandas as pd
import ujson
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss

from transformers import GPT2Config, GPT2TokenizerFast
from transformers.modeling_outputs import (ModelOutput,
                                           BaseModelOutputWithPastAndCrossAttentions,
                                           CausalLMOutputWithCrossAttentions
                                           )
from transformers import GPT2LMHeadModel
from transformers.optimization import Adafactor, AdafactorSchedule, AdamW, get_cosine_schedule_with_warmup
from transformers.tokenization_utils_base import AddedToken
import sys
from torch.utils.data import Sampler
from pytorch_lightning.utilities.distributed import _get_rank

import torch
import pytorch_lightning as pl
import torch_optimizer as toptim
T_co = TypeVar('T_co', covariant=True)

mp1 = os.path.abspath(os.path.join('..'))
mp2 = "../DockerImages/feng_hirst_rst_parser"
mp3 = "../DockerImages/feng_hirst_rst_parser/src"
mp4 = "../DockerImages/feng_hirst_rst_parser/model"
modules_paths = [mp1, mp2, mp3, mp4]
for path_ in modules_paths:
    if path_ not in sys.path:
        sys.path.append(path_)
        
from DockerImages.feng_hirst_rst_parser.src.parse2 import DiscourseParser

logger = logging.get_logger(__name__)
from nltk.tokenize.treebank import TreebankWordDetokenizer

from torch import nn
from transformers.modeling_utils import Conv1D

from utils_nlg_v3 import SizedOrderedBatchSampler,SizedOrderedDistributedBatchSampler

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype):
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
        raise ValueError(
            "Encoder Attention mask should have three dimensions Decoder Attention mask should have two dimensions")

    attention_mask = (1.0 - attention_mask) * -10000.0

    return attention_mask

# Monkey patched forward method for GPT2
def GPT2_forward(
    self,
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    position_embeds=None,
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
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError(
            "You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_embeds is None and position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].size(-2)

    if position_embeds is None and position_ids is None:
        position_ids = torch.arange(
            past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # GPT2Attention mask.
    if attention_mask is not None:
        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")

        attention_mask = _expand_mask(attention_mask, self.dtype)

    # If a 2D ou 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.add_cross_attention and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (
            encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_shape, device=device)
        encoder_attention_mask = self.invert_attention_mask(
            encoder_attention_mask)
    else:
        encoder_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    if position_embeds is None:
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

        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure layer_past is on same device as hidden_states (might not be correct)
            if layer_past is not None:
                layer_past = tuple(past_state.to(hidden_states.device)
                                   for past_state in layer_past)
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if getattr(self.config, "gradient_checkpointing", False) and self.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                None,
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

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + \
                (outputs[2 if use_cache else 1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + \
                    (outputs[3 if use_cache else 2],)

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = self.ln_f(hidden_states)

    if hidden_states.numel() != torch.prod(torch.tensor(output_shape)):
        new_batch_size = hidden_states.numel()//torch.prod(torch.tensor(output_shape))
        output_shape = torch.Size((new_batch_size, ) +  output_shape[1:])
    hidden_states = hidden_states.view(*output_shape)
    
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )

class RSTGPT2_Config(GPT2Config):

    def __init__(self,
                 base_model_name='gpt2',
                 model_name="RSTGPT2",
                 scale_grad_by_freq=True,
                 max_len_rst=36,
                 max_len_kp=64,
                 max_len_utt=270,
                 rst_tree_aligned_attention=False,
                 rst_segment_method='None',
                 max_rst_pos=4094,
                 
                 embed_tkn_pdrop=0.15,
                 embed_kp_pdrop=0.15,
                 embed_rst_pdrop=0.15,

                 attn_pdrop=0.15,
                 resid_pdrop=0.15,
                 
                 ull_loss_tkn=False,
                 prev_context_len=None,

                 **kwargs):

        super().__init__( attn_pdrop=attn_pdrop,
                            resid_pdrop=resid_pdrop,
                            **kwargs)

        self.base_model_name = base_model_name
        self.model_name = model_name
        self.scale_grad_by_freq = scale_grad_by_freq
        self.max_len_utt = max_len_utt
        self.max_len_rst = max_len_rst
        self.max_len_kp = max_len_kp
        self.rst_tree_aligned_attention = rst_tree_aligned_attention
        self.rst_segment_method = rst_segment_method
        self.rst_rel_li = ['Attribution',
                           'Background', 'Cause', 'Comparison', 'Condition',
                           'Contrast', 'Elaboration', 'Enablement', 'Evaluation',
                           'Explanation', 'Joint', 'Manner-Means', 'Topic-Comment',
                           'Summary', 'Temporal', 'Topic-Change', 'same-unit', 'textual-organization']
        self.rst_ns_li = ['NN', 'NS', 'SN'] 
        self.max_rst_pos = max_rst_pos
        self.rst_added_tokens = 2
        self.vocab_size = self.vocab_size + self.rst_added_tokens

        self.embed_tkn_pdrop=embed_tkn_pdrop
        self.embed_kp_pdrop=embed_kp_pdrop
        self.embed_rst_pdrop=embed_rst_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

        self.ull_loss_tkn = ull_loss_tkn
        self.prev_context_len = prev_context_len

        self.pad_token_id= self.eos_token_id

        self.pad_values = {'rst_start_token': self.eos_token_id,
                           'rst_rel': len(self.rst_rel_li),
                           'rst_ns': len(self.rst_ns_li),
                           'rst_pos': self.max_rst_pos-1,

                           'key_phrase_ids': self.eos_token_id,
                           'li_kprstpos': self.max_rst_pos-1,

                            'position_ids_keyphrase':self.n_ctx-1,
                            'position_ids_utt':self.n_ctx-1,

                           'input_ids_utt': self.eos_token_id,
                           'attention_mask': 0.0,

                           'labels': -100, #self.model.loss_fct.ignore_index,

                           'edu_rstpos': -1,

                           'context_rst_rstpos':-1,
                           'context_kp_rstpos':-1
                           }

        self.pad_maxlens = {
            'rst_start_token': 1,
            
            'rst_rel': self.max_len_rst-1,
            'rst_ns': self.max_len_rst-1,
            'rst_pos': self.max_len_rst-1,

            'key_phrase_ids': self.max_len_kp,
            'li_kprstpos': self.max_len_kp,
            'input_ids_utt': self.max_len_utt,
            'labels': self.max_len_rst + self.max_len_kp + self.max_len_utt,

            'attention_mask': self.max_len_rst + self.max_len_kp + self.max_len_utt,


            'position_ids_keyphrase':self.max_len_kp,
            'position_ids_utt':self.max_len_utt ,

            'edu_rstpos': self.max_rst_pos // 2, #40,
            'context_rst_rstpos':self.max_len_rst,
            'context_kp_rstpos':self.max_len_kp
            }




class RSTGPT2(DegenerateLossMixin, RstModelMixin, GPT2LMHeadModel):

    def __init__(self,
                 config: RSTGPT2_Config):

        super().__init__(config)

        self.base_model_name = config.base_model_name
        self.model_name = config.model_name
        self.scale_grad_by_freq = config.scale_grad_by_freq
        self.max_len_rst = config.max_len_rst
        self.max_len_kp = config.max_len_kp
        self.max_len_utt = config.max_len_utt
        self.rst_tree_aligned_attention = config.rst_tree_aligned_attention
        self.rst_segment_method = config.rst_segment_method

        self.transformer.forward = types.MethodType(
            GPT2_forward, self.transformer)

        self.embed_rst_rels = torch.nn.Embedding(len(self.config.rst_rel_li)+1,
                                                 self.config.n_embd, padding_idx=len(
                                                 self.config.rst_rel_li),
                                                 scale_grad_by_freq=self.scale_grad_by_freq)
        self.embed_rst_rels.weight.data.normal_(
            mean=0.0, std=self.config.initializer_range/3)

        self.embed_rst_ns = torch.nn.Embedding(len(self.config.rst_ns_li)+1,
                                                    self.config.n_embd, padding_idx=len(
                                                    self.config.rst_ns_li),
                                               scale_grad_by_freq=self.scale_grad_by_freq)
        self.embed_rst_ns.weight.data.normal_(
            mean=0.0, std=self.config.initializer_range/3)

        self.embed_rst_pos = EmbeddingRstPos(max_rst_index=self.config.max_rst_pos,
                                             max_rst_level=RSTTokenizer.node_level(
                                                 self.config.max_rst_pos),
                                             rst_encoding_ndim=self.config.n_embd,
                                             init_val=0.05,
                                             std=self.config.initializer_range/3)

        self.loss_fct = CrossEntropyLoss()

        # generation params
        self.generation_params = {  # 'num_beams': 1,
            'bad_words_ids': [[50257], [50258]],
            'early_stopping': True,
            'no_repeat_ngram_size': 3,
            'min_length': 5, 'max_length': 260, 'pad_token_id':self.config.eos_token_id }
        
        self.correct_attn_bias()

    def correct_attn_bias( self  ):
         
        for idx in range(len(self.transformer.h) ):
            
            _ = self.transformer.h[idx].attn.bias.shape

            self.transformer.h[idx].attn.register_buffer(
                        "bias",
                        torch.ones_like(self.transformer.h[idx].attn.bias, dtype=torch.uint8).view(
                            1, 1, _[-1], _[-1]
                        ),
                    )

    def forward(self,
                rst_start_token_id=None,
                rst_rel=None,
                rst_ns=None,
                rst_pos=None,
                key_phrase_ids=None,
                li_kprstpos=None,
                position_ids_keyphrase=None,
                position_ids_utt=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,

                input_ids_utt=None,
                # context_rstpos=None,
                context_rst_rstpos = None,
                context_kp_rstpos = None,
                edu_rstpos=None,
                curr_edu_pos=None,

                head_mask=None,
                past_key_values=None,
                inputs_embeds=None,
                position_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds == None and position_embeds == None:
            inputs_embeds, position_embeds = self.embed(
                rst_start_token_id,
                rst_rel,
                rst_ns,
                rst_pos,
                key_phrase_ids,
                li_kprstpos,
                input_ids_utt,
                position_ids_keyphrase,
                position_ids_utt,
                **kwargs)

        transformer_outputs = self.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,

            inputs_embeds=inputs_embeds,
            position_embeds=position_embeds,

            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states) 

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        loss = None
        lm_loss = None
        ul_loss_token = None
        if labels is not None:       

            # Shift so that tokens < n predict n
            shift_lm_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            lm_loss = self.loss_fct(
                shift_logits.view(-1, shift_lm_logits.size(-1)), shift_labels.view(-1))

            if self.config.ull_loss_tkn:
                
                #Replacing the kp start tokens with pad tokens
                kp_cands = key_phrase_ids.masked_fill( key_phrase_ids==self.tokenizer.keyphrase_start_token_id.to(key_phrase_ids.device),
                                                         self.config.pad_values['key_phrase_ids'] )
                ul_loss_token = self._compute_token_level_unlikelihood_loss(    
                                    shift_lm_logits, shift_labels, kp_cands, tkn_pad_idx=self.config.pad_values['input_ids_utt'] )
                loss = lm_loss + ul_loss_token
            else:
                loss = lm_loss

        if not return_dict:
            raise NotImplementedError("return dict")
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        output = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions)
        
        output['ll_loss']=lm_loss
        output['ul_loss_token']=ul_loss_token

        output['curr_edu_pos'] = curr_edu_pos
        # output['context_rstpos'] = 
        output['context_rst_rstpos'] =context_rst_rstpos
        output['context_kp_rstpos'] =context_kp_rstpos
        output['edu_rstpos'] = edu_rstpos
        output['li_gen_text'] = kwargs.get('li_gen_text',None)

        return output

    def embed(
        self,
        rst_start_token_id,
        rst_rel,
        rst_ns,
        rst_pos,
        key_phrase_ids,
        li_kprstpos,
        input_ids_utt,
        position_ids_keyphrase,
        position_ids_utt,
        **kwargs
     ):
        # RST context embedding
        rst_start_token_embed = self.transformer.wte(rst_start_token_id)
        # rst_rel_embed = self.embed_rst_rels(rst_rel)
        # rst_ns_embed = self.embed_rst_ns(rst_ns)
        rst_rel_embed   = self.discrete_embedding_dropout( rst_rel, self.embed_rst_rels, self.config.embed_rst_pdrop)
        rst_ns_embed    = self.discrete_embedding_dropout( rst_ns, self.embed_rst_ns, self.config.embed_rst_pdrop )

        rst_pos_embed = self.embed_rst_pos(rst_pos)

        rst_embed = (rst_rel_embed + rst_ns_embed + rst_pos_embed)

        # Key Phrase context embedding
        # keyphrase_phrase_embed = self.transformer.wte( key_phrase_ids)
        keyphrase_phrase_embed = self.discrete_embedding_dropout(key_phrase_ids, self.transformer.wte, self.config.embed_kp_pdrop)
        keyphrase_rst_pos_embed = self.embed_rst_pos(li_kprstpos)
        keyphrase_embed = keyphrase_rst_pos_embed + keyphrase_phrase_embed

        # input_id embedding
        # utt_inputs_embeds = self.transformer.wte(input_ids_utt)
        utt_inputs_embeds = self.discrete_embedding_dropout(input_ids_utt, self.transformer.wte, self.config.embed_tkn_pdrop)

        inputs_embeds = torch.cat([
            rst_start_token_embed,
            rst_embed,
            keyphrase_embed,
            utt_inputs_embeds,
        ], axis=-2)

        # Position Embedding
        # position_embed_kp_utt = self.transformer.wpe(position_ids_kp_utt)
        # _ = position_embed_kp_utt.shape
        position_embed_kp = self.transformer.wpe(position_ids_keyphrase)
        position_embed_utt = self.transformer.wpe(position_ids_utt)
        _ = position_embed_kp.shape
        position_embed_rst = position_embed_kp.new_zeros(
            [_[0], 1+rst_rel_embed.shape[1], _[2]])
        position_embed = torch.cat(
            [position_embed_rst, position_embed_kp, position_embed_utt ], axis=1)

        return inputs_embeds, position_embed

    def generate_plus(self, encoded_input, generation_params=None, tokenizer=None, segmenter=None):
        
        assert tokenizer!=None
        self.tokenizer = tokenizer
        
        if self.rst_tree_aligned_attention:
            
            #TODO: change punctuation system to use the indices and not rely upon the text decoding
            # Setting up cue_phrase, discourse_marker helpers for segmentation calculators
            if not hasattr(self, 'dict_len_phrasetok'):
                tknzed_cue_phrases = [ tokenizer.encode(' '+phrase) for phrase in self.cue_phrases ] + [ tokenizer.encode(phrase) for phrase in self.cue_phrases ]
                li_li_tknzed_discourse_markers = [ [tokenizer.encode(' '+dm) for dm in dms ] + [tokenizer.encode(dm) for dm in dms ]  for dn_type,dms in self.discourse_markers.items() ]

                tknzed_discourse_markers = sum(li_li_tknzed_discourse_markers,[])

                combined_phrases = tknzed_cue_phrases + tknzed_discourse_markers
            
                self.dict_len_phrasetok = { length:[ torch.tensor(tknzed_phrase,dtype=torch.long,device=next(self.transformer.parameters()).device)
                                                        for tknzed_phrase in combined_phrases if len(tknzed_phrase)==length ]
                                                    for length in range(1,max( [ len(tknzed_phrase) for tknzed_phrase in combined_phrases]) )}
            if segmenter != None:
                self.segmenter = segmenter
            
            else:
                if self.rst_segment_method == "fenghirst":
                    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                        self.segmenter = DiscourseParser(
                            verbose=False, skip_parsing=True, global_features=False)

                elif self.rst_segment_method == "segbot":
                    self.segmenter = Segmenter()
                    self.segmenter.to(self.device)

                else:
                    raise NotImplementedError(
                        f"Invalid segmentation method specified - curr val={self.rst_segment_method}")

        generation_params = generation_params if generation_params else self.generation_params

        # moving to gpu
        for k in list(encoded_input.keys()):
            encoded_input[k] = encoded_input[k].to(self.device)

        # generating
        with torch.no_grad():
            input_ids = encoded_input.get('input_ids_utt')

            inputs_embeds, position_embeds = self.embed(**encoded_input)

            encoded_input['inputs_embeds'], encoded_input['position_embeds'] = inputs_embeds, position_embeds
            
            self.gen_key_phrase_ids  =  encoded_input['key_phrase_ids']
            
            output = self.generate(
                input_ids, use_cache=True, **encoded_input, **generation_params)
            output = output[0]

        decoded_text = tokenizer.decode(output,
                                                skip_special_tokens=True)

        if hasattr(self, 'prev_edu_pos'):
            if not hasattr(self.tokenizer, 'li_prev_edu_pos'):
                self.tokenizer.li_prev_edu_pos = []
            self.tokenizer.li_prev_edu_pos.append(self.prev_edu_pos)
            del self.prev_edu_pos
            # pass
        
        if hasattr(self, 'consecutive_steps_prev_edu_used' ):
            del self.consecutive_steps_prev_edu_used

        if hasattr(self, 'gen_key_phrase_ids'):
            self.gen_key_phrase_ids = self.gen_key_phrase_ids.to('cpu')
            del self.gen_key_phrase_ids
            
        
        #unloading segmenter  
        if self.rst_segment_method == "fenghirst" and segmenter==None:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                self.segmenter.unload()
            del self.segmenter

        elif self.rst_segment_method == "segbot" and segmenter==None:
            self.segmenter.to('cpu')
            del self.segmenter
        
        return decoded_text
       
    
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
                
        # input_ids is essentially input_ids_utt. It does not include the context information
        if self.rst_tree_aligned_attention:
                            
            # update in prepare_inputs_for_generation
            # use curr_edu_pos in _update model kwargs in order to create new attn matrix
            curr_edu_pos, li_gen_text = self.get_curr_edu_pos(
                input_ids, 
                kwargs.get('edu_rstpos'),
                li_gen_text=kwargs.get('li_gen_text'))

            if past is not None:
                _ = input_ids.shape
                position_embeds = self.transformer.wpe(
                    input_ids.new_full((_[0], 1), _[1]))

                # only last token for inputs_ids if past is defined in kwargs
                input_ids = input_ids[:, -1].unsqueeze(-1)
                inputs_embeds = self.transformer.wte(input_ids)
                attention_mask = kwargs.get('attention_mask')[..., -1:, :]

            elif past is None:
                # getting length of generated utterance ids that were not provided as context
                # this is the length of the context
                utt_ctx_len = kwargs.get('input_ids_utt').shape[1]
                utt_len = input_ids.shape[1]
                new_utt_len = utt_len - utt_ctx_len
                _ = new_utt_len

                if _ != 0:
                    inputs_embeds_utt = self.transformer.wte(
                        input_ids[..., -_:])
                    inputs_embeds = torch.cat(
                        [kwargs.get('inputs_embeds'), inputs_embeds_utt], axis=1)

                    position_ids_utt = torch.arange(
                        utt_ctx_len, utt_len, device=input_ids.device)
                    position_embeds_utt = self.transformer.wpe(
                        position_ids_utt)
                    position_embeds = torch.cat(
                        [kwargs.get('position_embeds'), position_embeds_utt])
                else:
                    inputs_embeds = kwargs.get('inputs_embeds')
                    position_embeds = kwargs.get('position_embeds')

                attention_mask = kwargs.get('attention_mask')
        else:
            curr_edu_pos = None
            li_gen_text = None
            if past is not None:
                _ = input_ids.shape
                position_embeds = self.transformer.wpe(
                    input_ids.new_full((_[0], 1), _[1]))

                input_ids = input_ids[:, -1].unsqueeze(-1)
                inputs_embeds = self.transformer.wte(input_ids)
                attention_mask = kwargs.get('attention_mask')[..., -1:, :]

            elif past is None:
                # this is the length of the context
                utt_ctx_len = kwargs.get('input_ids_utt').shape[1]
                utt_len = input_ids.shape[1]
                new_utt_len = utt_len - utt_ctx_len
                _ = new_utt_len

                if _ != 0:
                    inputs_embeds_utt = self.transformer.wte(
                        input_ids[..., -_:])
                    inputs_embeds = torch.cat(
                        [kwargs.get('inputs_embeds'), inputs_embeds_utt], axis=1)

                    position_ids_utt = torch.arange(
                        utt_ctx_len, utt_len, device=input_ids.device)
                    position_embeds_utt = self.transformer.wpe(
                        position_ids_utt)
                    position_embeds = torch.cat(
                        [kwargs.get('position_embeds'), position_embeds_utt])
                else:
                    inputs_embeds = kwargs.get('inputs_embeds')
                    position_embeds = kwargs.get('position_embeds')

                attention_mask = kwargs.get('attention_mask')

        return {
            # "input_ids": None,
            'inputs_embeds': inputs_embeds,
            'position_embeds': position_embeds,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            'curr_edu_pos': curr_edu_pos,
            'li_gen_text':li_gen_text
        }
    
    def _update_model_kwargs_for_generation(self,
                                            outputs: ModelOutput, 
                                            model_kwargs: Dict[str, Any], 
                                            is_encoder_decoder: bool = False
                                            ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder and not self.rst_tree_aligned_attention:

            # attention mask
            attention_mask = model_kwargs["attention_mask"]

            # TODO: convert this to a 3d expansion
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, 1, 0, 0), value=0)  #  context and utterance attention to new token

            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, 0, 0, 1), value=1)  # new token to context and utterance attention

        elif not is_encoder_decoder and self.rst_tree_aligned_attention:

            # attention mask
            curr_utt_len = model_kwargs.get(
                # 'attention_mask').shape[1]+1 - model_kwargs['context_rstpos'].shape[1]
                'attention_mask').shape[1] + 1 - model_kwargs['context_rst_rstpos'].shape[1] - model_kwargs['context_kp_rstpos'].shape[1]

            
            attention_mask = self.tokenizer.prepare_attention_mask(
                curr_edu_pos=outputs['curr_edu_pos'],
                # context_rstpos=model_kwargs['context_rstpos'],
                context_rst_rstpos = model_kwargs['context_rst_rstpos'],
                context_kp_rstpos = model_kwargs['context_kp_rstpos'],
                prev_mask=model_kwargs.get('attention_mask'),
                curr_utt_len=curr_utt_len,
                training=False)
                        
        model_kwargs["attention_mask"] = attention_mask
        
        # update li_gen_text
        model_kwargs['li_gen_text'] = outputs.get('li_gen_text', None)
                       
        return model_kwargs

    @classmethod
    def load_model_tokenizer(cls, model_name="RSTGPT2", model_version=None, mparams_new={}, device="cuda:0"):

        if model_version != None:
            # load from a pretrained RSTGPT2
            checkpoint = RSTGPT2_TrainingModule.get_ckpt_file(
                f'./models/{model_name}/version_{model_version}/checkpoints')

            mparams = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                'base_model_name', 'model_name', 'max_len_kp',
                'max_len_rst', 'max_len_utt',
                'scale_grad_by_freq', 'rst_tree_aligned_attention']}

            for key, value in mparams_new.items():
                mparams[key] = value

            mconfig = RSTGPT2_Config.from_pretrained(
                mparams['base_model_name'], **mparams)

            model = RSTGPT2(mconfig)

            # Loading Training Module
            training_module = RSTGPT2_TrainingModule(
                mconfig, mode='inference', model=model)
            training_module.load_state_dict(checkpoint['state_dict'])

            model = training_module.model
            tok = training_module.tokenizer
            model.correct_attn_bias()

            # Deleting checkpoints to free up GPU space
            del checkpoint
            torch.cuda.empty_cache()

            # if torch.cuda.is_available():
            if device != 'cpu' and torch.cuda.is_available():
                model = model.to(device)

            return model, tok

        else:
            raise ValueError(
                "At least one of model_version or mconfig must not be None ")

    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--base_model_name',
                            default='gpt2', required=False)
        parser.add_argument('--model_name', default='RSTGPT2', required=False)
        parser.add_argument('--max_len_utt', type=int, default=270)
        parser.add_argument('--max_len_rst', type=int, default=36)
        parser.add_argument('--max_len_kp', type=int, default=64)
        parser.add_argument('--scale_grad_by_freq', action='store_true', default=True,
                            help="Inverse the gradients to the emebdding layers based on the occurence of each index in the minibatch ")
        parser.add_argument('--rst_tree_aligned_attention',default=False, action='store_true')
        parser.add_argument('--rst_segment_method', type=str,default='segbot', choices=['None', 'fenghirst', 'segbot'])
        #Regularization params
        parser.add_argument('--attn_pdrop', type=float, default=0.1)
        parser.add_argument('--resid_pdrop', type=float, default=0.1)

        parser.add_argument('--embed_tkn_pdrop',type=float, default=0.15, help="We drop specific indices from embedding")
        parser.add_argument('--embed_kp_pdrop',type=float, default=0.15, help="We drop specific indices from embedding")
        parser.add_argument('--embed_rst_pdrop',type=float, default=0.15, help="We drop specific indices from embedding")

        parser.add_argument('--ull_loss_tkn', default=False, action='store_true')
        parser.add_argument('--prev_context_len', type=int, default=0, help="lookback length for ull token loss")

        mparams = parser.parse_known_args()[0]
        return mparams
       

class RSTTokenizer(GPT2TokenizerFast, utils.EffeciencyMixin, utils.RstTokenizerMixin):
    rst_tree_aligned_attention = False
    sample_kps = False
    # Setting up RST2

    rst_rel_li = ['Attribution',
                  'Background', 'Cause', 'Comparison', 'Condition',
                  'Contrast', 'Elaboration', 'Enablement', 'Evaluation',
                  'Explanation', 'Joint', 'Manner-Means', 'Topic-Comment',
                  'Summary', 'Temporal', 'Topic-Change', 'same-unit', 'textual-organization']  # Add this to savable config

    rst_rel_labeler = LabelEncoder()
    rst_rel_labeler.fit(rst_rel_li)

    rst_ns_li = ['NN', 'NS', 'SN']
    rst_ns_labeler = LabelEncoder()
    rst_ns_labeler.fit(rst_ns_li)

    # Setting up context lengths
    max_len_rst = 30
    max_len_kp = 64
    max_len_utt = 240
    max_rst_pos = 4094

    special_token_count = 2

    rst_start_token = "<|rst|>"
    keyphrase_start_token = "<|kp|>"

    def __init__(self, *args, **kwargs):
        
        super().__init__( *args, **kwargs)
        
        self.max_len_rst = kwargs.get('max_len_rst', self.max_len_rst)
        self.max_len_kp = kwargs.get('max_len_kp',self.max_len_kp)
        self.max_len_utt = kwargs.get('max_len_utt', self.max_len_utt)
        self.max_rst_pos = kwargs.get('max_rst_pos', self.max_rst_pos )
        self.rst_tree_aligned_attention = kwargs.get('rst_tree_aligned_attention',self.rst_tree_aligned_attention)
        self.sample_kps = kwargs.get('sample_kps', self.sample_kps)

        self.detok = TreebankWordDetokenizer()


    def encode_input(self, rst_rel, rst_ns, rst_pos, li_kp, li_kprstpos,
                     utterance=None, utterance_prompt=None, dict_pos_edu=None,
                     max_len_rst=None, max_len_kp=None,
                     exclude_from_output=[], device=None ):
        """
            This version is a smaller output space than v1, by dropping rst_pos and rst_ns
            Return 
            w/o da
            dictionary
            attention_mask : Bidirectional up to bos token, Causal Up till EOS, 0s till end of padding

        Note this method returns integer encodings for tokens that will be processed by BERT embedding layer
            and possibly one-hot encoded vectors that will not be encoded by same pert embedding layer
        """
        # Preprocessing
        if utterance != None:
            utterance = utterance.lstrip(string.punctuation+" ")

        # Encoding rst, keyphrase and utterance info
        rst_rel, rst_ns, rst_pos, rst_pad_len = self.encode_rst(
            rst_rel, rst_ns, rst_pos, max_len_rst)

        key_phrase_ids, li_kprstpos, ta_tokens_pos, kp_phrase_lens, kp_pad_len = self.encode_keyphrase(
            li_kp, li_kprstpos, max_len_kp)

        input_ids_utt, labels, utt_len = self.encode_utterance(utterance, utterance_prompt,
                                                               context_len= 1 + rst_rel.shape[-1] + 
                                                               key_phrase_ids.shape[-1] )

        # Lengths of each input
        r_len = 1 + rst_rel.shape[-1]
        rt_len = r_len + key_phrase_ids.shape[-1]
        rtu_len = rt_len + utt_len

        # Building position ids
        if kp_phrase_lens.shape[0] != 0:

            position_ids_keyphrase = torch.cat(
                [torch.arange(tpl, dtype=torch.long) for tpl in kp_phrase_lens])

        else:
            position_ids_keyphrase = input_ids_utt.new_full([0], 0)

        position_ids_keyphrase = torch.cat([position_ids_keyphrase, torch.full(
                [kp_pad_len], self.pad_values['position_ids_keyphrase'])])

        position_ids_utt = torch.arange(0, utt_len, dtype=torch.long)
        # position_ids_kp_utt = torch.cat(
        #     (position_ids_keyphrase, position_ids_utt))

        # Building Attention Mask
        # prepending 0 to rst_pos in order to factor in
        # here
        attention_mask = self.prepare_attention_mask(
            r_len, rt_len, rtu_len,
            ta_tokens_pos, kp_phrase_lens,
            rst_pos=torch.cat([rst_pos[0:1], rst_pos]),
            li_kprstpos=li_kprstpos,
            dict_pos_edu=dict_pos_edu,
            training=(utterance != None and utterance_prompt == None),
            utterance_ids=input_ids_utt
        )

        attention_mask = self.prepare_attention_mask_handle_padding(
            attention_mask,
            r_len, rst_pad_len, max_len_rst,
            rt_len, kp_pad_len, max_len_kp)
        
        output = OrderedDict(
                { 
                    'rst_start_token_id': self.rst_start_token_id,

                    'rst_rel': rst_rel, 'rst_ns': rst_ns, 'rst_pos': rst_pos,

                    'key_phrase_ids': key_phrase_ids.contiguous(),
                    'li_kprstpos': li_kprstpos.contiguous(),

                    'position_ids_keyphrase':position_ids_keyphrase.contiguous(),
                    'position_ids_utt':position_ids_utt.contiguous(),

                    'input_ids_utt': input_ids_utt.contiguous(),

                    'labels': labels,

                    'attention_mask': attention_mask

                  }

        ) 

        # #ensuring interoparability with huggingface generate code
        # if utterance_prompt != None:
        #     output['input_ids'] = input_ids_utt.contiguous()

        if self.rst_tree_aligned_attention:
            
            output['context_rst_rstpos'] = torch.cat([rst_pos[0:1], rst_pos] )
            output['context_kp_rstpos']  = li_kprstpos
            dict_pos_edu =  self.detokenize_dict_pos_edu(dict_pos_edu)
            dec_rst_pos = [self.clamp_values(np.array(int(key)), utils.MAX_LONG_VALUE).item(
                0) for key in dict_pos_edu.keys()]

            output['edu_rstpos'] = torch.tensor(sorted(dec_rst_pos, key=RSTTokenizer.edukp_pos_sort_function,
                                                       ),
                                                dtype=torch.long)
        # moving to devie
        if device != None:
            for key in output:
                if output[key] != None:
                    output[key] = output[key].to(device).unsqueeze(0)

        # excluding items from output
        for key in exclude_from_output:
            output.pop(key, None)
            
        return output

    def encode_rst(self, rst_rels, rst_ns, rst_pos, variable_padding_size=None):
        """Converts rst_rels in a series of vectors

            Args:
                rst_rels ([type]): [description]
                max_padding ([type]): padding amount
                rst_pos ([type]): [description]
                rst_ns
            Also includes an encoding for rst_ns and rst_pos
        """

        rst_rel_encoded = self.rst_rel_labeler.transform(
            rst_rels)
        tnsr_rels = torch.LongTensor(rst_rel_encoded)

        # Encoding the rst ns
        rst_ns_encoded = self.rst_ns_labeler.transform(
            rst_ns)
        tnsr_ns = torch.LongTensor(rst_ns_encoded)

        tnsr_pos = torch.LongTensor(
            [RSTTokenizer.clamp_value(val) for val in rst_pos])

        # padding ns and pos
        # The ns and pos embedding layer uses the index value 0 as a padding index
        # For this index the vector is initialized to zer0 and as such never updates
        len_ = tnsr_rels.shape[0]

        if variable_padding_size != None:

            pad_len = variable_padding_size - 1

            if len_ > pad_len:
                tnsr_rels = tnsr_rels[:pad_len]
                tnsr_ns = tnsr_ns[:pad_len]
                tnsr_pos = tnsr_pos[:pad_len]
                diff = 0

            elif len_ < pad_len:
                diff = pad_len - len_
                tnsr_rels = torch.nn.functional.pad(
                    tnsr_rels, (0, diff), value=self.pad_values['rst_rel'])
                tnsr_ns = torch.nn.functional.pad(
                    tnsr_ns, (0, diff), value=self.pad_values['rst_ns'])
                tnsr_pos = torch.nn.functional.pad(
                        tnsr_pos, (0, diff), value=self.pad_values['rst_pos'])
            else:
                diff = 0

        else:
            if len_ > self.max_len_rst - 1:
                tnsr_rels = tnsr_rels[:self.max_len_rst-1]
                tnsr_ns = tnsr_ns[:self.max_len_rst-1]
                tnsr_pos = tnsr_pos[:self.max_len_rst-1]
                diff = 0

            elif len_ < self.max_len_rst - 1:
                diff = self.max_len_rst - 1 - len_
                tnsr_rels = torch.nn.functional.pad(
                    tnsr_rels, (0, diff), value=self.pad_values['rst_rel'])
                tnsr_ns = torch.nn.functional.pad(
                    tnsr_ns, (0, diff), value=self.pad_values['rst_ns'])
                tnsr_pos = torch.nn.functional.pad(
                    tnsr_pos, (0, diff), value=self.pad_values['rst_pos'])
            else:
                diff = 0

        return tnsr_rels, tnsr_ns, tnsr_pos, diff

    def encode_keyphrase(self, key_phrases, li_kprstpos, variable_padding_size=None):
        """[summary]

            Args:
                keyphrase ([type]): [list of keyphrase (phrases or words)]
                keyphrase_score ([type]): [list of float scores for each topic relevancy]

            Raises:
                Exception: [description]

            Returns:
                [type]: [description]
        """
        if len(key_phrases) != 0:

            # max_len = min(variable_padding_size,
            #               self.max_len_kp) if variable_padding_size else self.max_len_kp
            
            max_len = variable_padding_size
            # Shuffling the order of the keyphrases during training
            
            if self.sample_kps == True:
                
                # counts =  np.array( [ (1+idx)//7+1 for idx in range(len(key_phrases)) ] )
                counts = np.array( [ 2**(1+idx) for idx in range(len(key_phrases)+1) ] ) #Larger weight on choosing key_phrases that occur later in text
                p = counts/np.sum(counts)
                p = p.astype('float64')

                possible_amounts_to_keep = list(range(len(key_phrases)+1)) # indexes
                # kps_to_keep = random.randint( 1 ,len(key_phrases) ) 
                kps_to_keep = np.random.choice( np.array(possible_amounts_to_keep), size=1, replace=False, p=p )[0]# Larger weight on choosing to keep a larger amount

                p = counts[:-1]/np.sum(counts[:-1])
                p = p.astype('float64')
                indexes = list(range(len(key_phrases))) # indexes
                sampled_idxs = np.random.choice(indexes, size=kps_to_keep, replace=False, p=p ).tolist()
                
                if len(sampled_idxs) != 0: 
                    key_phrases, li_kprstpos = list(zip(*[(key_phrases[idx], li_kprstpos[idx]) for idx in sampled_idxs]))
                else:
                    key_phrases = []
                    li_kprstpos = []
           

            if len(key_phrases) > 0 :
                str_keyphrases = '<|kp|> ' + '<|kp|> '.join(key_phrases)

                key_phrase_ids = self.encode(str_keyphrases, add_special_tokens=False,
                                            truncation=True,
                                            padding='do_not_pad',
                                            return_tensors='np',
                                            max_length=max_len,
                                            return_special_tokens_mask=False)[0]
            
            else:
                key_phrase_ids = np.array([self.pad_values['li_kprstpos']]*max_len,dtype=int)

            # Repeating each score in the case where the score is allocated to a phrase topic which is broken down into constituent words
            # e.g. keyphrase - ["fast car", "motorbike", "long rail road"], scores = [0.9, 0.4, 0.2] -> scores = [0.9, 0.9, 0.9, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2]
            # have to do it after tokenization due to bytepair encoding
            # get index of where <|kp|> tokens occur
            kp_idxs = np.where(
                key_phrase_ids == self.keyphrase_start_token_id_np)[0]

            key_phrase_ids = torch.LongTensor(key_phrase_ids)

            # filtering out idxs if index is larger than padding marker
            kp_idxs = kp_idxs[kp_idxs < max_len]

            # get difference in index position between <|kp|> tag n and <|kp|> tag n+1 ( for final tag use difference between tag and end of list)
            kp_phrase_lens = np.diff(kp_idxs, append=key_phrase_ids.numel())

            # copies each score phrase_len times to cover that phrase and handles case where there is no phrase
            _ = [[pos]*phrase_len for pos,
                 phrase_len in zip(li_kprstpos, kp_phrase_lens)]
            li_kprstpos = functools.reduce(operator.iconcat, _, [])

            tnsr_rst_pos = torch.LongTensor(
                [RSTTokenizer.clamp_value(val) for val in li_kprstpos])

            # Adding padding to key_phrase
            diff = max_len - key_phrase_ids.shape[-1]

            key_phrase_ids = torch.nn.functional.pad(
                key_phrase_ids, (0, diff), value=self.pad_values['key_phrase_ids'])
            tnsr_rst_pos = torch.nn.functional.pad(
                tnsr_rst_pos, (0, diff), value=self.pad_values['li_kprstpos'])
        
        else:
            max_len = min(variable_padding_size,
                          self.max_len_kp) if variable_padding_size else self.max_len_kp

            key_phrase_ids = torch.full(
                size=(variable_padding_size,), fill_value=self.pad_values['key_phrase_ids'])
            tnsr_rst_pos = torch.full(
                size=(variable_padding_size,), fill_value=self.pad_values['li_kprstpos'])
            kp_idxs = np.array([])
            kp_phrase_lens = np.array([])
            diff = variable_padding_size

        return key_phrase_ids, tnsr_rst_pos, kp_idxs, kp_phrase_lens, diff

    def encode_utterance(self, utterance=None, utterance_prompt=None, context_len=None):

        # Creating labels/targets
        if utterance_prompt != None:
            utterance_prompt = self.eos_token + utterance_prompt

            utt_prompt_tok_ids = self.encode(
                utterance_prompt,
                add_special_tokens=False,
                return_attention_mask=False,
                padding='do_not_pad',
                truncation=False,
                max_length=self.max_len_utt,
                return_tensors='pt' )[0]

            utt_ids = utt_prompt_tok_ids.contiguous()
            labels = None
            utt_len = utt_ids.shape[-1]

        elif utterance != None:
            utterance = self.eos_token + utterance + self.eos_token
            utt_tok_ids = self.encode(
                utterance,
                add_special_tokens=False,
                padding='do_not_pad',
                truncation=True,
                max_length=self.max_len_utt,
                return_tensors='pt' )[0]

            utt_ids = utt_tok_ids.contiguous()
            labels = torch.cat(
                [torch.full(size=(context_len,), fill_value=-100, dtype=torch.long), utt_ids])
            utt_len = utt_ids.shape[-1]

        return utt_ids, labels, utt_len

    def prepare_attention_mask(self, r_len=None, rt_len=None, rtu_len=None,
                               ta_tokens_pos=None, kp_phrase_lens=None,
                               rst_pos=None, li_kprstpos=None,
                               dict_pos_edu=None,

                               # generation
                               curr_edu_pos=None,
                               curr_utt_len=None,
                            #    context_rstpos=None,
                               context_kp_rstpos = None,
                               context_rst_rstpos=None,
                               prev_mask=None,
                               training=True,

                               # correcting generation,
                               utterance_ids=None
                               ):

        if self.rst_tree_aligned_attention == False:

            if prev_mask == None:
                # template #[batch_size, num_heads, from_seq_length, to_seq_length]
                attention_mask = torch.zeros([rtu_len, rtu_len])

                # RST section fully attends to itself
                attention_mask[:r_len, :r_len] = 1

                # Implementing causal masking for each kp
                # First each kp only attends to other words within that kp (including ta token) and the RST info
                # So set the template attn to 0 in this range within kp range
                # attention_mask[r_len:rt_len, r_len:rt_len] = 0
                attention_mask[r_len:rt_len, :r_len] = 1

                # Second each kp has causal masking on the tokens within the topic phrase
                for ta_idx, phrase_len in zip(ta_tokens_pos, kp_phrase_lens):
                    s_idx = r_len+ta_idx
                    e_idx = s_idx+phrase_len
                    attention_mask[s_idx:e_idx, s_idx:e_idx] = \
                        torch.tril(attention_mask.new_ones(
                            [phrase_len, phrase_len]))

                # Implementing causal attention mask for text
                attention_mask[rt_len:, :] = torch.tril(
                    torch.ones([rtu_len-rt_len, rtu_len]), diagonal=rt_len)

            # TODO: make sure padding is handled in prev_mask
            else:
                dims = prev_mask.shape()
                
                prev_mask = torch.nn.functional.pad(prev_mask, (0, 1), value=0) # No old index attends to the new word
                attention_mask = torch.cat(
                    [prev_mask, prev_mask.new_ones([dims[0], 1, dims[-1]+1 ])], axis=-1)                

        else:
            if prev_mask == None and curr_edu_pos == None:  # training

                attention_mask = torch.zeros([rtu_len, rtu_len])  # template
                # Detecting which node each context should attend to in  O(n)
                # pos should be ordered based on left to right along an imaginary tree
                # so first re-order in terms of node depth (which is just re-ordering by value )
                # Then starting from the end of the list find the direct tree of nodes to the parent
                # Store each pos and parent in a dictionary of keys=child, value=parent
                dict_rstpos_parents = torch.nn.ParameterDict()

                # region Creating attention matrix for the rst/kp context over the rst/kp context
                all_pos_context = torch.cat((rst_pos, li_kprstpos))

                # creating dictionary indicating the parents of each rst node pos in the context
                for pos in all_pos_context:
                    if pos not in dict_rstpos_parents:
                        dict_rstpos_parents[str(pos)] = torch.nn.parameter.Parameter(torch.tensor(
                            RSTTokenizer.seq_from_root_to_edu_pos(pos) + [int(pos)], dtype=torch.long), requires_grad=False)

                # Creating vector indicating which rst node pos attend to which other positions
                li_tens_pos = []
                for pos in all_pos_context:

                    # parent tree for current position
                    li_parent_tree = dict_rstpos_parents[str(pos)]

                    pos_tree_aligned_attn = (
                        all_pos_context[..., None] == li_parent_tree).any(-1).squeeze()  # Creates a boolean vector indicating where model can attend

                    # concatenating an extra section to account for no attendance
                    li_tens_pos.append(pos_tree_aligned_attn)

                attention_mask_context = torch.stack(li_tens_pos, dim=0).to(
                    torch.float)  # shape( rt_len, rt_len )
                # endregion

                # region Creating the attention matrix for the utterance section over the rst/kp context
                li_attn_vectors = []

                # li_pos_edu_idslen_ids will be a list  containing the rst_pos, edu_ids and ids_len for each edu

                li_pos_edu_idslen_ids = sorted([[str(self.clamp_values(np.array(int(pos)), utils.MAX_LONG_VALUE).item(0)), edu, None, None] for pos, edu in dict_pos_edu.items()],
                                               key=lambda x: RSTTokenizer.edukp_pos_sort_function(
                    int(x[0])))

                # Adding special tokens to edu to mirror the ecnoded utterance with special tokens
                li_pos_edu_idslen_ids[0][1] = self.eos_token + \
                    li_pos_edu_idslen_ids[0][1].lstrip(string.punctuation+" ")

                if training:  # eos_token only at end for training
                    li_pos_edu_idslen_ids[-1][1] = li_pos_edu_idslen_ids[-1][1] + \
                        self.eos_token

                # dictionary containing the list of rst trees from nodes to root
                for idx in range(len(li_pos_edu_idslen_ids)):
                    # Find the tokenized length of each edu

                    if idx != 0:
                        li_pos_edu_idslen_ids[idx][1] = " " + \
                            li_pos_edu_idslen_ids[idx][1]

                    li_pos_edu_idslen_ids[idx][3] = self.encode(
                        li_pos_edu_idslen_ids[idx][1], add_special_tokens=False)

                    li_pos_edu_idslen_ids[idx][2] = len(
                        li_pos_edu_idslen_ids[idx][3])

                    pos = li_pos_edu_idslen_ids[idx][0]

                    if pos not in dict_rstpos_parents:
                        dict_rstpos_parents[pos] = torch.nn.parameter.Parameter(torch.tensor(
                            RSTTokenizer.seq_from_root_to_edu_pos(int(pos)) + [int(pos)], dtype=torch.long), requires_grad=False)
                # endregion

                # region EDU tokenization may be different from text tokenization due to the RST parser
                    # evening up the tokenization lengths
                _len = sum(item[2] for item in li_pos_edu_idslen_ids)
                if _len != rtu_len - rt_len and training:

                    # Find the Index of positions where the utterance indices match
                    edu_ids_ = [item[3] for item in li_pos_edu_idslen_ids]
                    edu_ids_flat = sum(edu_ids_, [])
                    edu_ids_len_cum = np.cumsum(
                        [item[2] for item in li_pos_edu_idslen_ids])

                    # list of tuples containing pairs where indices represent matching tokens
                    matching_indexes = []
                    max_non_matches_till_id1_skip = 4
                    # This is 0(n^2)
                    for idx1, id1 in enumerate(utterance_ids):

                        id1_matches_checked = 0
                        for idx2, id2 in enumerate(edu_ids_flat):

                            # skip until the most recently matched index for the edu phrases
                            if len(matching_indexes) > 0:
                                if idx2 <= matching_indexes[-1][1]:
                                    continue

                            # if index match then add to record
                            if id1 == id2:
                                matching_indexes.append([idx1, idx2])
                                max_non_matches_till_id1_skip = 4  # reset back to original vale
                                break

                            id1_matches_checked += 1
                            # if we are unable to find a match for id1 in 3-4 idx2  then we skip to the next id1
                            # we also increase the max non matches by one, to handle any increase variation between sentences
                            if id1_matches_checked == max_non_matches_till_id1_skip:
                                max_non_matches_till_id1_skip += 1
                                break

                    # Find the lengths of sequential unmatched indices
                    matching_indexes_utt, matching_indexes_edus = zip(
                        *matching_indexes)

                    # indexes required to arrive at next matching token for each sequence
                    matching_indices_utt_diff = np.diff(matching_indexes_utt)
                    matching_indices_edus_diff = np.diff(matching_indexes_edus)

                    # differences in index distance required for each sequence
                    li_diff_utt_edu = matching_indices_utt_diff - matching_indices_edus_diff

                    # Reduce spans of edu lengths
                    # cycle through the mismatched chunks
                    # For each mismatched chunk decide which edu this corresponds to
                    # Then change the length in li_pos_edu_idslen_ids
                    for matched_token_idx, diff_utt_edu in enumerate(li_diff_utt_edu):

                        if diff_utt_edu != 0:

                            # Find out which edu the matched_token is positioned in, within the edu form of utterance
                            token_idx_in_flattened_edu = matching_indexes_edus[matched_token_idx]

                            edu_idx = np.searchsorted(
                                edu_ids_len_cum, token_idx_in_flattened_edu, side='right', sorter=None)

                            # find the edu that the prev match word occurs in
                            li_pos_edu_idslen_ids[edu_idx][2] += diff_utt_edu
                # endregion

                # region creating attn vectors for each edu
                    # For each edu create attn vector to (rst and kp) with rst pos in the set of edu's parents pos
                for pos, edu_txt, edu_txt_len, edu_ids in li_pos_edu_idslen_ids:
                    if edu_txt_len <= 0:
                        continue
                    li_parent_tree = dict_rstpos_parents[pos]

                    pos_tree_aligned_attn = (
                        all_pos_context[..., None] == li_parent_tree).any(-1).squeeze()

                    # Repeating by tokenized length of EDU
                    pos_tree_aligned_attn = einops.repeat(pos_tree_aligned_attn,
                                                          'd -> l d', l=edu_txt_len)

                    li_attn_vectors.append(pos_tree_aligned_attn)

                # need to transpose since _expand_mask requires masks to be bsz, 1, tgt_len, src_len
                attention_mask_utt_context = torch.cat(li_attn_vectors, dim=0).to(
                    torch.float)  # shape( rtu_len-rt_len, rt_len )

                if attention_mask_utt_context.shape[0] > rtu_len - rt_len:
                    attention_mask_utt_context = attention_mask_utt_context[:rtu_len-rt_len, :]

                # endregion

                attention_mask_utt = torch.tril(
                    torch.ones((rtu_len-rt_len, rtu_len-rt_len)))

                # region Combining both sections of attention
                try:
                    attention_mask[..., :rt_len,
                                   :rt_len] = attention_mask_context
                    attention_mask[..., rt_len:, rt_len:] = attention_mask_utt
                    attention_mask[..., rt_len:,
                                   :rt_len] = attention_mask_utt_context
                except Exception as e:
                    a = 1
                    pass
                # correcting the kp phrases to have causal attention
                # Second each kp has causal masking on the tokens within the topic phrase
                for ta_idx, phrase_len in zip(ta_tokens_pos, kp_phrase_lens):
                    s_idx = r_len+ta_idx
                    e_idx = s_idx+phrase_len
                    attention_mask[s_idx:e_idx, s_idx:e_idx] = \
                        torch.tril(attention_mask.new_ones(
                            [phrase_len, phrase_len]))

            else:
                # all_pos = context_rstpos
                li_batch_new_attn = []

                # creating new attention mask for every element in curr_edu_pos
                for pos in curr_edu_pos:

                    li_parent_tree = torch.tensor(RSTTokenizer.seq_from_root_to_edu_pos(
                        pos.item()) + [pos.item()], device=prev_mask.device)

                    # pos_tree_aligned_attn = (
                    #     context_rstpos[..., None] == li_parent_tree).any(-1).squeeze()

                    pos_tree_aligned_attn_rst = (
                        context_rst_rstpos[..., None] == li_parent_tree).any(-1).squeeze()

                    pos_tree_aligned_attn_kp = (
                        context_kp_rstpos[..., None] == li_parent_tree).any(-1).squeeze()
                    
                    pos_tree_aligned_attn = torch.cat( [pos_tree_aligned_attn_rst, pos_tree_aligned_attn_kp] )

                    li_batch_new_attn.append(
                        pos_tree_aligned_attn.unsqueeze(0))

                # this context below does not include the utterance provided as context
                attention_mask_context = torch.stack(
                    li_batch_new_attn, dim=0).float()  # shape( bs, 1 , context )

                # next word should attend to all previous utteranec words under causal attention

                attention_mask = torch.cat((attention_mask_context, attention_mask_context.new_ones(
                    (attention_mask_context.shape[0], 1, curr_utt_len))), axis=-1)

                # appending to new attention_mask if it exists otherwise just return the attention
                prev_mask = torch.nn.functional.pad(prev_mask, (0, 1), value=0)

                attention_mask = torch.cat([prev_mask, attention_mask], axis=1)

        return attention_mask

    def prepare_attention_mask_handle_padding(self, attention_mask,
                                              r_len, rst_pad_len, max_len_rst,
                                              rt_len, kp_pad_len, max_kp_len):
        # Changing attention masks to compensate for the Variable RST batching
        if max_len_rst != None and rst_pad_len != 0:
            attention_mask[:, r_len-rst_pad_len:r_len] = 0
            attention_mask[r_len-rst_pad_len:r_len, :] = 0

        if max_kp_len != None and kp_pad_len != 0:
            attention_mask[:, rt_len-kp_pad_len:rt_len] = 0
            attention_mask[rt_len-kp_pad_len:rt_len, :] = 0

        return attention_mask

    def detokenize_dict_pos_edu( self, dict_pos_edu ):
        
        for pos in dict_pos_edu.keys():
            dict_pos_edu[pos] = self.detok.detokenize(  dict_pos_edu[pos].split() )
            
        return dict_pos_edu

    @classmethod
    def from_pretrained(cls,
                        dir_tokenizer="./tokenizers/RSTGPT2",
                        base_tokenizer_name="gpt2",
                        rst_params={},
                        **kwargs):  # max_len_rst, max_len_kp, max_rst_depth, max_len_utt, max_rst_pos

        if os.path.exists(dir_tokenizer):
            tokenizer = super(RSTTokenizer, cls).from_pretrained(
                dir_tokenizer, local_files_only=True, **kwargs, **rst_params)

        else:

            additional_special_tokens = kwargs.pop(
                'additional_special_tokens', [])
            at_rst_start = AddedToken(cls.rst_start_token, lstrip=False, rstrip=False) if isinstance(
                cls.rst_start_token, str) else cls.rst_start_token
            at_topic_start = AddedToken(cls.keyphrase_start_token, lstrip=False, rstrip=False) if isinstance(
                cls.keyphrase_start_token, str) else cls.keyphrase_start_token
            additional_special_tokens = [
                at_rst_start, at_topic_start] + additional_special_tokens

            cls = super(RSTTokenizer, cls).from_pretrained(base_tokenizer_name,
                                                           additional_special_tokens=additional_special_tokens,
                                                           )

            cls.save_pretrained(dir_tokenizer)
            tokenizer = cls

        tokenizer.rst_start_token_id = torch.full( (1,), 50257 , dtype=torch.long )
        tokenizer.keyphrase_start_token_id = torch.full( (1,), 50258 , dtype=torch.long )        
        tokenizer.keyphrase_start_token_id_np = tokenizer.keyphrase_start_token_id.numpy()

        for k, v in rst_params.items():
            setattr(tokenizer, k, v)

        return tokenizer

class RSTGPT2_TrainingModule(pl.LightningModule):

    def __init__(self,
                 mconfig,
                 batch_size=20,
                 dir_data=None,
                 accumulate_grad_batches=1,
                 max_epochs=25,
                 gpus=1,
                 gpu_nodes = 1,        
                 tpu_nodes=None,
                 tpu_cores=None,
                 learning_rate=1e-4,
                 warmup_proportion=0.1,
                 workers=0,
                 mode='train_new',
                 tag='',
                 optimizer='adafactor',
                 model = None,
                 tokenizer = None,
                 debugging=False,
                 **kwargs):

        super().__init__()

        self.batch_size = batch_size

        self.mode = mode
        self.workers = workers
        self.optimizer = optimizer
        self.dir_checkpoints = kwargs.get('dir_checkpoints')
        self.debugging = debugging

        # Setting up gpu or tpu params
        self.accelerator = "gpu"*(gpus!=None and gpu_nodes!=None) + "tpu"*(tpu_nodes!=None and tpu_cores!=None)
        if self.accelerator == "gpu":
            self.units = gpus #units per node
            self.nodes = gpu_nodes
        elif self.accelerator == "tpu":
            self.units = tpu_cores 
            self.nodes = tpu_nodes
        else:
            raise ArgumentError("User must either have tpu or gpu set up")


        if tokenizer == None:
            self.tokenizer = RSTTokenizer.from_pretrained(f"./tokenizers/{mconfig.model_name}",
                                                         base_tokenizer_name=mconfig.base_model_name,
                                                         rst_params={name: getattr(mconfig, name) for name in ['max_len_rst',
                                                                                                               'max_len_kp',
                                                                                                               'max_rst_pos',
                                                                                                               'max_len_utt',
                                                                                                               'rst_tree_aligned_attention'] if hasattr(mconfig, name)
                                                                     })
        else:
            self.tokenizer = tokenizer
        
        if model is not None:
            self.model = model
        else:
            mconfig.vocab_size = mconfig.vocab_size-2
            self.model = RSTGPT2.from_pretrained(
                mconfig.base_model_name, config=mconfig)
            mconfig.vocab_size = mconfig.vocab_size+2
            self.model.config.vocab_size = mconfig.vocab_size
            self.model.resize_token_embeddings(self.model.config.vocab_size)
        self.model.correct_attn_bias()

        # self.pad_values = {'rst_start_token': mconfig.eos_token_id,
        #                    'rst_rel': self.model.embed_rst_rels.padding_idx,
        #                    'rst_ns': self.model.embed_rst_ns.padding_idx,
        #                    'rst_pos': self.model.embed_rst_pos.padding_idx,

        #                    'key_phrase_ids': mconfig.eos_token_id,
        #                    'li_kprstpos': self.model.embed_rst_pos.padding_idx,

        #                     'position_ids_keyphrase':mconfig.n_ctx-1,
        #                     'position_ids_utt':mconfig.n_ctx-1,

        #                    'input_ids_utt': mconfig.eos_token_id,
        #                    'attention_mask': 0.0,

        #                    'labels': self.model.loss_fct.ignore_index,

        #                    'edu_rstpos': -1,

        #                    'context_rst_rstpos':-1,
        #                    'context_kp_rstpos':-1
        #                    }
        # self.tokenizer.pad_values = self.pad_values

        # self.pad_maxlens = {
        #     'rst_start_token': 1,
            
        #     'rst_rel': mconfig.max_len_rst-1,
        #     'rst_ns': mconfig.max_len_rst-1,
        #     'rst_pos': mconfig.max_len_rst-1,

        #     'key_phrase_ids': mconfig.max_len_kp,
        #     'li_kprstpos': mconfig.max_len_kp,
        #     'input_ids_utt': mconfig.max_len_utt,
        #     'labels': mconfig.max_len_rst + mconfig.max_len_kp + mconfig.max_len_utt,

        #     'attention_mask': mconfig.max_len_rst + mconfig.max_len_kp + mconfig.max_len_utt,


        #     'position_ids_keyphrase':mconfig.max_len_kp,
        #     'position_ids_utt':mconfig.max_len_utt ,

        #     'edu_rstpos': mconfig.max_rst_pos // 2, #40,
        #     'context_rst_rstpos':mconfig.max_len_rst,
        #     'context_kp_rstpos':mconfig.max_len_kp
        #     }

        # self.tokenizer.pad_maxlens = self.pad_maxlens
        
        self.tokenizer.pad_values = self.model.config.pad_values
        self.tokenizer.pad_maxlens = self.model.config.pad_maxlens
        self.model.tokenizer = self.tokenizer

        if self.mode in ['train_new', 'train_cont', 'test']:
            self.dir_data = utils.get_path(dir_data, _dir=True, relative=True)
            self.accumulate_grad_batches = accumulate_grad_batches
            self.tag = tag

            self.dg = DataLoaderGenerator(self.dir_data,  self.batch_size, self.tokenizer,
                                          workers=self.workers, mode=self.mode, 
                                          units=self.units,
                                          nodes=self.nodes,
                                          pad_maxlens=self.model.config.pad_maxlens, 
                                          pad_values=self.model.config.pad_values
                                          )

            if self.mode == "test":
                self.create_data_loaders(['test'])
            else:
                self.max_epochs = max_epochs
                self.warmup_proportion = warmup_proportion
                self.learning_rate = learning_rate

                self.create_data_loaders(['train', 'val', 'inference'])
                self.tokenizer.inference_mode = True
                self.inference_samples = list(islice(self.inference_dl, 2))
                self.tokenizer.inference_mode = False
                del self.inference_dl



                # train_params_to_save = self.return_params()
                # mparams_to_save = {param: getattr(mconfig, param) for param in list(filter(
                #     lambda p: p not in ['self', 'kwargs'], list(inspect.signature(RSTGPT2_Config.__init__).parameters.keys())))}

                # self.hparams.update({**train_params_to_save, **mparams_to_save})
                # pl.core.saving.save_hparams_to_yaml(os.path.join(os.path.dirname(
                #     kwargs['dir_checkpoints']), "hparams.yaml"), self.hparams)

                # mconfig_dict = self.model.config.to_dict()
                mconfig_dict = {param: getattr(mconfig, param) for param in list(filter(
                    lambda p: p not in ['self', 'kwargs'], list(inspect.signature(RSTGPT2_Config.__init__).parameters.keys())))}
                self.hparams.update( **mconfig_dict )
                ignore_list = ['mconfig','dir_data','gpus','gpu_nodes','tpu_nodes','tpu_cores','workers','debugging','model','model_dir','tokenizer','override']
                self.save_hyperparameters(ignore=ignore_list)

        elif self.mode in ['inference']:
            self.eval()
            self.freeze()

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data', default="./dataset_v3_2",
                            help="Relative directory path for datafiles")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('--max_epochs', default=50, type=int)
        parser.add_argument('--accumulate_grad_batches', default=3, type=int)
        parser.add_argument('--batch_size', default=20, type=int)

        parser.add_argument('--learning_rate', default=4e-3, type=float)
        parser.add_argument('--workers', default=16,
                            type=int)  # TODO: change to 6
        
        parser.add_argument('--gpu_nodes',default=None, type=int )
        parser.add_argument('--gpus', default=None, type=int)
        parser.add_argument('--tpu_nodes',default=None, type=int )
        parser.add_argument('--tpu_cores', default=None, type=int, )


        parser.add_argument('--mode', default='train_new', type=str,
                            choices=['train_new', 'train_cont', 'test', 'inference'])
        parser.add_argument('--version', default=None, required=False,
                            type=int, help="The Experimental Versioning for this run")
        parser.add_argument('--precision', default=16, required=False,
                            type=int, help="Precision to use", choices=[16, 32])
        parser.add_argument('--tag', default='', required=True, type=str)
        parser.add_argument('--override', default=False,
                            type=lambda x: bool(int(x)), choices=[0, 1])
        parser.add_argument('--optimizer', default='adafactor',
                            choices=['adafactor', 'adamw'])

        parser.add_argument('--val_check_interval', type=float, default=0.33) 
        parser.add_argument('--debugging', action='store_true', default=False) 
        

        tparams = parser.parse_known_args()[0]

        return tparams

    @staticmethod
    def instatiate_training_module(tparams=None, mparams=None):
        """Create training module

        Args:
            tparams ([type]): [description]
        """

        if tparams['mode'] in ["train_new"]:
            mconfig = RSTGPT2_Config.from_pretrained(
                mparams['base_model_name'], **mparams)

            training_module = RSTGPT2_TrainingModule(mconfig, **tparams)
            training_module.model.correct_attn_bias()


        elif tparams['mode'] in ["train_cont", "inference"]:

            checkpoint = RSTGPT2_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            # restore/update param files from the checkpoint
            if "hyper_parameters" in checkpoint.keys() and tparams['override'] == False:
                tparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                    'learning_rate', 'splits', 'tag', 'optimizer']})

                mparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name', 'model_name', 'max_len_utt', 'max_len_rst', 'max_len_kp',
                     'rst_tree_aligned_attention']})

                mparams = mparams

            else:
                print("param files not found utilsing default or user entered params\n")

            mconfig = RSTGPT2_Config(**mparams)

            # Restore/update Training Module
            training_module = RSTGPT2_TrainingModule(mconfig, **tparams)

            training_module.load_state_dict(checkpoint['state_dict'])
            training_module.model.correct_attn_bias()

        elif tparams['mode'] in ["test"]:

            checkpoint = RSTGPT2_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            # restore/update param files from the checkpoint
            try:
                tparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                    'learning_rate', 'precision']})

                mparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name', 'model_name', 'max_len_utt', 'max_len_rst', 'max_len_kp']})
            except KeyError:
                pass

            # Restore/update Training Module
            training_module = RSTGPT2_TrainingModule(
                **tparams, mparams=mparams)
            training_module.load_state_dict(checkpoint['state_dict'])

        else:
            raise ValueError(
                "tparams['mode'] must be in range [train_new, train_cont, test, inference]")

        return training_module

    @staticmethod
    def instatiate_trainer(tparams, tb_logger, training_module, debugging=False):
        """[summary]

            Creates The Trainer and callbacks
        """
        dir_checkpoints = tparams['dir_checkpoints']

        tparams.pop('tpu_nodes')
        tparams.pop('tpu_cores')
        tparams.pop('gpu_nodes')
        tparams.pop('gpus')
            

        trainer_vars = { 'accelerator': training_module.accelerator,
                        'strategy': 'ddp',
                        'plugins': DDPPlugin(find_unused_parameters=False),
                        'num_nodes':training_module.nodes}

        k = 'gpus' if training_module.accelerator=='gpu' else 'tpu_cores'
        trainer_vars[k]= training_module.units

        checkpoin_valloss = ModelCheckpoint(monitor='val_loss', 
                                            save_top_k=2,
                                            mode='min', dirpath=dir_checkpoints,
                                            filename='{epoch:03d}_{val_loss:.5f}')
        
        checkpoint_interval = ModelCheckpoint(
            filename="interval-{epoch}-{step}",
            monitor="step",
            mode="max",
            every_n_val_epochs=1,
            save_top_k=1/tparams['val_check_interval']
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience = 1/tparams['val_check_interval'],       
            verbose=False,
            mode='min'
        )
        callbacks = [checkpoin_valloss, early_stop_callback, checkpoint_interval]


        if tparams['mode'] in ["train_new"]:
            # Creating Callbacks

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    default_root_dir=tparams['dir_checkpoints'],
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks,
                                                    # Training
                                                    val_check_interval=tparams['val_check_interval'],
                                                    limit_train_batches=35 if debugging else 1.0,
                                                    limit_val_batches=20 if debugging else 1.0,

                                                    log_every_n_steps= 100,
                                                    
                                                    replace_sampler_ddp=False,
                                                    num_sanity_val_steps=0,
                                                    **trainer_vars
                                                    )
            trainer.logger.log_hyperparams(training_module.hparams)

        elif tparams['mode'] in ["train_cont", "inference"]:
            # restoring checkpoint
            checkpoint = RSTGPT2_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks,
                                                    val_check_interval=0.33,
                                                    num_sanity_val_steps=0,
                                                    replace_sampler_ddp=False,
                                                    **trainer_vars
                                                    )

            trainer.scaler.load_state_dict(checkpoint['native_amp_scaling_state'])
            trainer.logger.log_hyperparams(training_module.hparams)

            # load callback states
            trainer.on_load_checkpoint(checkpoint)

            try:
                trainer.current_epoch = checkpoint['epoch']
                trainer.global_step = checkpoint['global_step']

            except Exception:
                trainer.fit_loop.current_epoch = checkpoint['epoch']
                trainer.fit_loop.global_step = checkpoint['global_step']


            del checkpoint
            # torch.cuda.empty_cache()

        elif tparams['mode'] in ["test"]:

            # restoring checkpoint
            checkpoint = RSTGPT2_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            training_module.load_state_dict(checkpoint['state_dict'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    check_val_every_n_epoch=1,
                                                    checkpoint_callback=False,
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    )

            # load callback states
            trainer.on_load_checkpoint(checkpoint)

        return trainer, training_module

    @staticmethod
    def get_ckpt_file(_dir_checkpoint, mode='best'):
        if mode == 'best':
            checkpoint_yaml_file = os.path.join(_dir_checkpoint, "best_k_models.yaml")
            # key= ckptpath, value = val_loss
            scores_dict = yaml.load(open(checkpoint_yaml_file, "r"), Loader=yaml.FullLoader)
            best_ckpt_path = min(scores_dict, key=scores_dict.get)

            if os.path.exists(best_ckpt_path) == False:
                root_dir = Path(__file__).resolve().parents[4]
                best_ckpt_path = os.path.join(
                    str(root_dir), best_ckpt_path[best_ckpt_path.index('mastering-conversation'):])

            checkpoint = torch.load(best_ckpt_path, map_location='cpu')

        else:
            raise NotImplementedError

        return checkpoint

    @staticmethod
    def start(trainer, tparams, training_module, mparams):

        if tparams['mode'] in ['train_new', 'train_cont']:
            trainer.fit(training_module)

        if tparams['mode'] in ["test"]:

            checkpoint = RSTGPT2_TrainingModule.get_ckpt_file(
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

    def forward(self, input_):
        # with torch.cuda.amp.autocast(enabled=True):
        return self.model(**input_, return_dict=True)

    def step(self, batch, step_name):

        model_output = self.forward(batch)
        output = {}
        
        if step_name == 'train':
            output["loss"] = model_output.loss
            # self.log( "loss", model_output.loss, sync_dist=True)
            self.log( "loss", model_output.loss, sync_dist=False, rank_zero_only=True)


        else:
            loss_key = f"{step_name}_loss"
            # self.log( loss_key, model_output.loss, sync_dist=True)
            self.log( loss_key, model_output.loss, sync_dist=False, rank_zero_only=True)
            output[loss_key]= model_output.loss
        
        self.log('step', self.global_step, prog_bar=False, 
            logger=False, on_step=True, on_epoch=False, 
            sync_dist=False, rank_zero_only=True)

        return output

    def training_step(self, batch, batch_idx):
        output = self.step(batch, "train")
        return output

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
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

    def epoch_end_log(self, outputs, step_name):
        if not hasattr(self,'w_rank'):
            self.w_rank = _get_rank()

        if step_name == "train":
            pass
        else:
            loss = torch.stack([x[f"{step_name}_loss"]for x in outputs]).mean()

            self.log(f"{step_name}_loss", loss, prog_bar=True, sync_dist=True)

        if step_name == "val":# _get_rank() == 0:
        # if step_name == "val" and self.trainer.global_rank == 0:

            # Making directory if it doesnt exist
            dir_infer = os.path.join(self.trainer.log_dir, "inference")

            if not os.path.exists(dir_infer):
                os.makedirs(dir_infer, exist_ok=True)

            # Adding true values and making csv files if thy dont already exists
            for idx, encoded_input in enumerate(self.inference_samples):

                encoded_input = {k: v.detach().clone() if isinstance(
                    v, torch.Tensor) else copy.deepcopy(v) for k, v in encoded_input.items()}

                fp = os.path.join(dir_infer, f"example_{idx:03d}.csv")

                # If the file does not exists we add the true observed records
                if not os.path.exists(fp):

                    df = pd.DataFrame(columns=['epoch', 'rst_rels', 'keyphrase', 'utterance',
                                               'dict_pos_edu', 'li_kprstpos',
                                               'rst_ns','rst_pos',
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
                        "utterance": utterance,
                        'rst_rels': ', '.join(rst_rels),
                        'rst_ns': ', '.join(rst_ns),
                        'rst_pos': rst_pos,

                        "keyphrase": ', '.join(keyphrase),
                        
                        "dict_pos_edu": json.dumps(dict_pos_edu),

                        "li_kprstpos": json.dumps(orig_li_kprstpos),
                    }

                    df = df.append(datum, ignore_index=True)
                    if self.w_rank==0:
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
                generation_params['max_length'] = 160
                
                generation_params['max_time'] = 45
                try:
                    decoded_text = self.model.generate_plus(
                        encoded_input, generation_params, self.tokenizer)
                except Exception as e:
                    decoded_text = "ERROR"

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
                if self.w_rank==0:
                    pd.DataFrame.from_records([datum]).to_csv(
                        fp, index=False, mode='a', header=False)
                # Saving to file
                pass
        else:
            pass
        a = 1

    def create_data_loaders(self, modes):

        if 'train' in modes:
            self.train_dl = self.dg.prepare_dataloader(
                split_name='train', debugging=self.debugging)
            self.train_dl_gen_count = 0

        if 'val' in modes:
            self.val_dl = self.dg.prepare_dataloader(
                split_name='val', debugging=self.debugging)
        if 'test' in modes:
            self.test_dl = self.dg.prepare_dataloader(
                split_name='test', debugging=self.debugging)
        if 'inference' in modes:
            self.inference_dl = self.dg.prepare_dataloader(
                split_name='inference',
                debugging=self.debugging)

    def train_dataloader(self):
        if self.train_dl_gen_count == 0:
            self.train_dl_gen_count += 1
            return self.train_dl
        else:
            self.train_dl = self.dg.prepare_dataloader(
                split_name='train', debugging=self.debugging)
            return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

    def total_steps(self):
        ds_size = len(self.train_dl)
        steps = (ds_size * self.max_epochs) // (self.accumulate_grad_batches)
        return steps

    def configure_optimizers(self):
        #optimal params for adafactor https://github.com/huggingface/transformers/pull/10526#issuecomment-804652154
    
        optimizer = Adafactor(self.model.parameters(), scale_parameter=True,
                                relative_step=True, warmup_init=True, lr=None,
                                weight_decay=0.01) # Works better for small models

        # optimizer = Adafactor(self.model.parameters(), scale_parameter=False,
        #                         relative_step=False, warmup_init=False, lr=1e-3,
        #                         weight_decay=0.01                         ) # works better for bigger models

        # lr_scheduler = AdafactorSchedule(optimizer)
        
        if self.mode == "train_cont":
            # restore the optimizers
            checkpoint = self.get_ckpt_file(self.dir_checkpoints)
            optimizer_states = checkpoint['optimizer_states']
            optimizer.load_state_dict(optimizer_states[0])
   
            # # restore the lr schedulers
            # if 'lr_schedulers' in checkpoint:
            #     lr_scheduler_states = checkpoint['lr_schedulers']
            #     lr_scheduler.load_state_dict(lr_scheduler_states[0])
        
        return {'optimizer': optimizer, "interval": "step", "monitor": "val_loss"}
        
        # return {'optimizer': optimizer, "lr_scheduler": lr_scheduler, "interval": "step", "monitor": "val_loss"}

    def return_params(self):
        params = {}
        keys = ['batch_size', 'accumulate_grad_batches', 'learning_rate', 'max_epochs', 'dir_data','tag']

        params = {
            k: self.__dict__[k] for k in keys if k in self.__dict__.keys()
        }

        return params

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:

        res = super().on_save_checkpoint(checkpoint)

        next( filter( lambda cb: isinstance(cb, ModelCheckpoint) and cb.monitor=="val_loss" , self.trainer.callbacks) ).to_yaml()
        return res

class DataLoaderGenerator():
    """Handles the creation of dataloaders for a train, val and test set
    """

    def __init__(self, dir_data, batch_size,
                 tokenizer,
                 workers=0, mode='train_new',
                 units=1,
                 nodes=1,
                 pad_values={},
                 pad_maxlens={},
                 
                 **kwargs):

        self.dir_data = dir_data
        self.tokenizer = tokenizer
        self.splits = {'train': 0.6, 'val': 0.2, 'test': 0.2}

        self.batch_size = batch_size
        self.workers = workers
        self.mode = mode

        self.nodes =nodes
        self.units = units
        self.pad_values = pad_values
        self.pad_maxlens = pad_maxlens

    def prepare_dataloader(self,
                           split_name='train',
                           debugging=False):
        """Prepares a dataloader given a directory of data for NLG language module
            # The current method takes a percentage of data from each subdirectory
            Args:
                dir_dset ([type]): [description]
        """
        seed = random.randint(1,1000)
        dir_data = self.dir_data

        # getting all files from all different subreddits/types of conversation
        fns = glob.glob(os.path.join(utils.get_path(dir_data,_dir=True, relative=True), "*", "*"))
        
        
        fns = [fn for fn in fns if os.path.split(
            fn)[-1] != "lock" and "dict_len" not in fn] #[:5] #debugging

        if debugging:
            fns = fns[:5]

        # getting number of utterances records in each file
        files_sizes = [int(fn[-10:]) for fn in fns]
        

        # defining starting line and total lines to use for dataset
        if split_name == 'train':
            line_starts = [0]*len(files_sizes)
            line_ends = [ls+int(fs*self.splits['train'])
                         for ls, fs in zip(line_starts, files_sizes)]
            inference = False
            bs = self.batch_size
            shuffle = True
            sample_kps = True

            def collate_fn(
                batch): return self.tokenizer.default_collate_pad(batch)

        elif split_name == 'val':
            line_starts = [int(fs*self.splits['train']) for fs in files_sizes]
            line_ends = [ls+int(fs*self.splits['val'])
                         for ls, fs in zip(line_starts, files_sizes)]

            shuffle = False
            inference = False
            bs = self.batch_size
            sample_kps = False

            def collate_fn(
                batch): return self.tokenizer.default_collate_pad(batch)

        elif split_name == 'test':
            line_starts = [int(fs*(1-self.splits['test']))
                           for fs in files_sizes]
            line_ends = files_sizes
            inference = False
            bs = self.batch_size

            def collate_fn(batch): return self.tokenizer.default_collate_pad(
                batch)
            shuffle = False
            sample_kps=False
            

        elif split_name == 'inference':
            line_starts = [int(fs*(1-self.splits['test']))
                           for fs in files_sizes]
            line_ends = files_sizes
            sampler = None
            inference = True
            shuffle = False
            bs = 1
            sample_kps=False
            def collate_fn(
                batch): return self.tokenizer.default_collate_pad(batch)

        li_dsets = [ SingleDataset(_f, copy.deepcopy( self.tokenizer ), line_start, line_end, inference, sample_kps=sample_kps)
                    for _f, line_start, line_end in zip(fns, line_starts, line_ends)]

        concat_dset = torch.utils.data.ConcatDataset(li_dsets)

        if self.units*self.nodes <= 1 and split_name not in ['inference', 'test']:
            sampler = SizedOrderedBatchSampler(concat_dset, bs, drop_last=True, shuffle=shuffle)
            bs = 1

        elif self.units*self.nodes > 1 and split_name not in ['inference', 'test']:
            sampler = SizedOrderedDistributedBatchSampler(concat_dset, bs, drop_last=True, shuffle=shuffle, units=self.units, seed=seed, nodes=self.nodes)
            bs = 1
        else:
            sampler = None

        dataloader = torch.utils.data.DataLoader(concat_dset,
                                                batch_size = bs,
                                                 num_workers=self.workers,
                                                 batch_sampler=sampler,
                                                 pin_memory=True,
                                                 collate_fn=collate_fn,
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

    def __init__(self, file_path, tokenizer, line_start, line_end, inference, **kwargs):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.line_start = line_start
        self.line_end = line_end
        self.inference = inference
        self.tokenizer.sample_kps = kwargs.get('sample_kps',False)

        skiprows = self.line_start if self.line_start != 0 else None
        with open(self.fp, 'r') as f:
            if self.line_start == 0:

                self.data = pd.read_csv(file_path, sep=',', header=0,
                                        skiprows=skiprows, nrows=(self.line_end-self.line_start))

            else:
                names = open(file_path, "r").readline().strip().split(',')

                self.data = pd.read_csv(file_path, sep=',',
                                        names=names, skiprows=skiprows,
                                        nrows=(self.line_end-self.line_start))

        fp_cached_order = os.path.join(os.path.dirname(
            file_path), f"gpt2_dict_lens_{line_start}_to_{line_end}.pkl")

        # # resetting the cached order files
        # if os.path.exists( fp_cached_order):
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
            # np_rstlens and np_keyphrase_lens used by the sampler

            # len of rst
            self.np_rstlens = np.array(
                [1 + len(json.loads(rst)) for rst in self.data.rst.values.tolist()])

            # len of keyphrase
            li_li_pos_kp = [json.loads(
                li_pos_kp) for li_pos_kp in self.data.li_pos_kp.values.tolist()]

            li_li_kp = [[kp for pos, kp in li_pos_kp]for li_pos_kp in li_li_pos_kp]

            li_kp = ['<|kp|> ' + '<|kp|> '.join(li_kp) for li_kp in li_li_kp]
            

            self.np_keyphrase_lens = np.array([self.tokenizer.encode(kp,
                                                                     add_special_tokens=False,
                                                                     truncation=False,
                                                                     padding='do_not_pad',
                                                                     return_tensors=None).__len__() for kp in li_kp])

            dict_cached_order = {'np_textlens': self.np_textlens,
                                 'np_rstlens': self.np_rstlens,
                                 'np_keyphrase_lens': self.np_keyphrase_lens}

            pickle.dump(dict_cached_order, open(fp_cached_order, "wb"))

        # # These ones below are used during the RSTTokenizers encode_input function to pad rst and kp
        # #v1 We initialize the rst/kp  lengths as the max lengths of our training set
        # # In the Sampler, we change the max length to that of its pre-prescribed batch
        # self.rst_len = [tokenizer.max_len_rst]*self.__len__()
        # self.key_phrase_len = [tokenizer.max_len_kp]*self.__len__()

        # v2 We initialize the rst/kp lengths as the actual length of each entry
        # In the Sampler, we change the max length to that of its pre-prescribed batch
        self.rst_len = copy.deepcopy(self.np_rstlens)
        self.key_phrase_len = copy.deepcopy(self.np_keyphrase_lens)

        self.data = self.data.to_dict('records')

    def __len__(self):
        # return (self.line_end - self.line_start)
        return len(self.data)

    def __getitem__(self, index):

        rst_rels, rst_ns, rst_pos, li_kp, li_kprstpos, utterance, dict_pos_edu = self.getitem_extract_datum(
            index)

        if self.inference == True:

            utterance_prompt = ' '.join(utterance.split(' ')[:1])

            encoded = self.tokenizer.encode_input(rst_rel=rst_rels, rst_ns=rst_ns, rst_pos=rst_pos,
                                                  li_kp=li_kp,
                                                  li_kprstpos=li_kprstpos,
                                                  utterance_prompt=utterance_prompt,
                                                  dict_pos_edu=dict_pos_edu,
                                                  max_len_rst= min( self.rst_len[index], self.tokenizer.max_len_rst),
                                                  max_len_kp= min( self.key_phrase_len[index], self.tokenizer.max_len_kp) )

            encoded['orig_rst_rels'] = rst_rels
            encoded['orig_rst_ns'] = rst_ns
            encoded['orig_rst_pos'] = rst_pos

            encoded['orig_utt'] = utterance
            encoded['orig_key_phrase'] = li_kp

            encoded['orig_dict_pos_edu'] = dict_pos_edu
            encoded['orig_li_kprstpos'] = li_kprstpos

        elif self.inference == False:

            encoded = self.tokenizer.encode_input(
                rst_rels, rst_ns, rst_pos,
                li_kp=li_kp,
                li_kprstpos=li_kprstpos,
                utterance=utterance,
                dict_pos_edu=dict_pos_edu,
                max_len_rst= min( self.rst_len[index], self.tokenizer.max_len_rst),
                max_len_kp= min( self.key_phrase_len[index], self.tokenizer.max_len_kp)
            )

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
        # double sort used for some reason??
        sorted_order_rst = [i[0] for i in sorted(enumerate(rst_pos), key=lambda x: (
            RSTTokenizer.edukp_pos_sort_function(x[1]), x[1]),)]

        rst_rels = [rst_rels[idx] for idx in sorted_order_rst]
        rst_ns = [rst_ns[idx] for idx in sorted_order_rst]
        rst_pos = [rst_pos[idx] for idx in sorted_order_rst]
        # endregion

        # Key phrase scores
        li_pos_kp = json.loads(datum['li_pos_kp'])

        if len(li_pos_kp) > 0:
            li_pos_kp = sorted(
                li_pos_kp, key=lambda pos_kp: RSTTokenizer.edukp_pos_sort_function(int(pos_kp[0])))
            li_kprstpos, li_kp = zip(*li_pos_kp)
            li_kprstpos = [int(pos) for pos in li_kprstpos]

        else:
            li_kp = []
            li_kprstpos = []

        # Utterance
        utterance = ujson.loads(datum['txt_preproc'])

        #pos and edus
        dict_pos_edu = json.loads(datum['dict_pos_edu']) 

        return rst_rels, rst_ns, rst_pos, li_kp, li_kprstpos, utterance, dict_pos_edu

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
    training_module = RSTGPT2_TrainingModule.instatiate_training_module(
        tparams, mparams)
    trainer, training_module = RSTGPT2_TrainingModule.instatiate_trainer(
        tparams, tb_logger, training_module, tparams['debugging'] )

    RSTGPT2_TrainingModule.start(trainer, tparams, training_module, mparams)


if __name__ == '__main__':

    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)

    # add model specific args
    mparams = RSTGPT2.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = RSTGPT2_TrainingModule.parse_train_specific_args(parent_parser)

    if tparams.mode == "test":
        assert tparams.gpus in [0, 1]

    if tparams.gpus not in [0, 1]:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] =  '6535' #'65502'

    try:
        main(vars(tparams), vars(mparams))
    except Exception:
        print(traceback.format_exc())

# CUDA_VISIBLE_DEVICES=1,2 python3 train_RSTGPT.py --batch_size 12 --version 1 --workers 12 --gpus 2 --tag "RSTGPT with aligned attention and regularisation" --max_len_utt 270 --max_len_rst 36 --max_len_kp 64 --rst_tree_aligned_attention 1 --rst_segment_method segbot
# CUDA_VISIBLE_DEVICES=1,2 python3 train_RSTGPT.py --batch_size 19 --version 2 --workers 12 --gpus 2 --tag "RSTGPT without aligned attention and regularisation" --max_len_utt 270 --max_len_rst 36 --max_len_kp 64 --rst_tree_aligned_attention 0 --rst_segment_method segbot

# python3 train_RSTGPT.py --tpu_cores 2 --tpu_nodes 1 --batch_size 12 --version 31 --workers 12  --tag "RSTGPT2 with aligned attention, regularisation and unlikelihood loss" --max_len_utt 270 --max_len_rst 36 --max_len_kp 64 --rst_tree_aligned_attention 1 --rst_segment_method segbot --ull_loss_tkn --prev_context_len 0 --val_check_interval 0.33
# python3 train_RSTGPT.py --tpu_cores 2 --tpu_nodes 1  --batch_size 12 --version 31 --workers 12 --tag "RSTGPT2 with regularisation and unlikelihood loss" --max_len_utt 270 --max_len_rst 36 --max_len_kp 64 --rst_tree_aligned_attention 0 --rst_segment_method segbot --ull_loss_tkn --prev_context_len 0 --val_check_interval 0.33