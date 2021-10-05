import os

# from torch.optim import lr_scheduler


os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ['TOKENIZERS_PARALLELISM'] = "true"

from collections import OrderedDict
import argparse
import bisect
import copy
import glob
import inspect
import json
import pickle
import random
import traceback
import types
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import (Any, Dict, Iterator, List, Optional, TypeVar, Tuple, Union)

import einops
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

import torch.distributed as dist
import torch.nn as nn
import transformers
from transformers import BartForConditionalGeneration
import ujson
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin
from pytorch_lightning.utilities.distributed import _get_rank
from sklearn import preprocessing as sklp
from torch.nn import CrossEntropyLoss
from torch.utils.data import Sampler 
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from transformers import (BartConfig, BartTokenizerFast)
from transformers.modeling_outputs import (BaseModelOutput, ModelOutput,
                                           Seq2SeqLMOutput, Seq2SeqModelOutput)
from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration, shift_tokens_right)
from transformers.optimization import AdafactorSchedule
from transformers.tokenization_utils_base import AddedToken
import string
import utils_nlg_v3 as utils
from utils_nlg_v3 import EmbeddingRstPos, mpatch_save_model, RstModelMixin

import torch_optimizer as optim
import functools
import operator

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

from DockerImages.feng_hirst_rst_parser.src import parser_wrapper3
from DockerImages.feng_hirst_rst_parser.src.parse2 import DiscourseParser

# patched method
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    ndim = mask.ndim
    if ndim == 2:
        bsz, src_len = mask.size()

        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(
            bsz, 1, tgt_len, src_len).to(dtype)

    elif ndim == 3:
        bsz, tgt_len_, src_len = mask.size()
        
        expanded_mask = mask[:, None, :, :].expand(
            bsz, 1, tgt_len_, src_len).to(dtype)

    else:
        raise ValueError("Encoder Attention mask should have three dimensions Decoder Attention mask should have two dimensions")
        
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

transformers.models.bart.modeling_bart._expand_mask = _expand_mask

# Monkey Patch for the BartModel Encoder forward - to prevent the automatic addition of positional encoding
def bart_encoder_forward(
    self,
    input_ids=None,
    attention_mask=None,
    head_mask=None,
    inputs_embeds=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    ):
    r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
            provide it.
            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
            for details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
            into associated vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
            returned tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
            for more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    elif inputs_embeds is None:
        raise ValueError("Inputs embeds must be defined")
    # elif input_ids is not None:
    #     input_shape = input_ids.size()
    #     input_ids = input_ids.view(-1, input_shape[-1])

    # else:
    #     raise ValueError("You have to specify either input_ids or inputs_embeds")

    # if inputs_embeds is None:
    #     raise ValueError("Inputs embeds must be defined")
        # inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        # embed_pos = self.embed_positions(input_shape)

    hidden_states = inputs_embeds  
    hidden_states = self.layernorm_embedding(hidden_states)
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.dropout, training=self.training)

    # expand attention_mask
    if attention_mask is None:
        raise ValueError("Attention mask must be defined")

    # [bsz,tgt_seq_len, src_seq_len ] -> [bsz, 1, tgt_seq_len, src_seq_len]
    attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        assert head_mask.size()[0] == (
            len(self.layers)
        ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):  # skip the layer
            layer_outputs = (None, None)
        else:
            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    (head_mask[idx] if head_mask is not None else None),
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(
                        head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
    )

# Monkey Patch for BartModel forward - to allow a seperate cross-attention mask to be passed in as a argument
def bart_forward(self,
                 input_ids=None, attention_mask=None,
                 decoder_input_ids=None,
                 decoder_attention_mask=None,
                 decoder_cross_attention_mask=None,
                 head_mask=None,
                 decoder_head_mask=None,
                 cross_attn_head_mask=None,
                 encoder_outputs=None,
                 past_key_values=None,
                 inputs_embeds=None,
                 decoder_inputs_embeds=None,

                            decoder_context_rstpos=None,
                            decoder_edu_rstpos=None,
                 use_cache=None,
                 output_attentions=None,
                 output_hidden_states=None,
                 return_dict=None,
                 ):

    # different to other models, Bart automatically creates decoder_input_ids from
    # input_ids if no decoder_input_ids are provided
    if decoder_input_ids is None and decoder_inputs_embeds is None:
        decoder_input_ids = shift_tokens_right(
            input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if encoder_outputs is None:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(
                encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(
                encoder_outputs) > 2 else None,
        )

    # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        # encoder_hidden_states=None, #encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        encoder_attention_mask=decoder_cross_attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values,
        inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if not return_dict:
        return decoder_outputs + encoder_outputs

    return Seq2SeqModelOutput(
        last_hidden_state=decoder_outputs.last_hidden_state,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )

class RSTBart_Config(BartConfig):

    def __init__(self,
                 base_model_name='facebook/bart-base',
                 model_name="RSTBart",
                 scale_grad_by_freq=True,
                 max_len_rst=28,
                 max_len_key_phrase=30,
                 max_len_utt=256,
                 rst_tree_aligned_attention=False,
                 max_rst_pos=4094,
                 **kwargs):

        super().__init__(**kwargs)

        self.base_model_name = base_model_name
        self.model_name = model_name
        self.scale_grad_by_freq = scale_grad_by_freq
        self.max_len_utt = max_len_utt
        self.max_len_rst = max_len_rst
        self.max_len_key_phrase = max_len_key_phrase
        self.rst_tree_aligned_attention = rst_tree_aligned_attention
        self.rst_rel_li = ['Attribution',
                           'Background', 'Cause', 'Comparison', 'Condition',
                           'Contrast', 'Elaboration', 'Enablement', 'Evaluation',
                           'Explanation', 'Joint', 'Manner-Means', 'Topic-Comment',
                           'Summary', 'Temporal', 'Topic-Change', 'same-unit', 'textual-organization']
        self.rst_ns_li = ['NN', 'NS', 'SN']
        self.max_rst_pos = max_rst_pos
        self.rst_added_tokens = 2
        #self.force_bos_token_to_be_generated = True
        self.forced_bos_token_id=self.bos_token_id

        self.vocab_size = self.vocab_size + self.rst_added_tokens
        
class RSTBart(BartForConditionalGeneration, RstModelMixin):

    def __init__(self,
                 config: RSTBart_Config):

        super().__init__(config)

        self.base_model_name = config.base_model_name
        self.model_name = config.model_name
        self.scale_grad_by_freq = config.scale_grad_by_freq
        self.max_len_rst = config.max_len_rst
        self.max_len_key_phrase = config.max_len_key_phrase
        self.max_len_utt = config.max_len_utt
        self.rst_tree_aligned_attention = config.rst_tree_aligned_attention
        self.model.forward = types.MethodType(bart_forward, self.model)
        self.model.encoder.forward = types.MethodType(
            bart_encoder_forward, self.model.encoder)
                
        self.embed_rst_rels = torch.nn.Embedding( len(self.config.rst_rel_li)+1,
                                                 self.config.d_model, padding_idx=len(
                                                     self.config.rst_rel_li),
                                                 scale_grad_by_freq=self.scale_grad_by_freq)
        self.embed_rst_rels.weight.data.normal_(
            mean=0.0, std=self.config.init_std)

        self.embed_rst_ns = torch.nn.Embedding(len(self.config.rst_ns_li)+1,
                                               self.config.d_model, padding_idx=len(
                                                   self.config.rst_ns_li),
                                               scale_grad_by_freq=self.scale_grad_by_freq)
        self.embed_rst_ns.weight.data.normal_(mean=0.0, std=self.config.init_std)

        self.embed_rst_pos = EmbeddingRstPos(max_rst_index=self.config.max_rst_pos,
                                             max_rst_level=RSTTokenizer.node_level(
                                                 self.config.max_rst_pos),
                                                rst_encoding_ndim=self.config.d_model,
                                                init_val=0.05,
                                                std =self.config.init_std )

        self.loss_fct = CrossEntropyLoss()
 
        self.generation_params = {
                                            'early_stopping': True,
                                            'do_sample':True, 
                                            'top_k':50, 
                                            'top_p':0.95, 
                                            'no_repeat_ngram_size': 2,
                                            'min_length': 5, 'max_length': 190 }
    def forward(self,
                rst_start_token_id=None,
                rst_rel=None,
                rst_ns=None,
                rst_pos=None,
                key_phrase_ids=None,
                li_kprstpos=None,

                position_ids_kp=None,
                attention_mask=None,

                decoder_input_ids=None,
                labels=None,

                input_ids=None,
                decoder_attention_mask=None,
                decoder_cross_attention_mask=None,
                
                decoder_context_rstpos=None,
                decoder_edu_rstpos=None,
                curr_edu_pos=None,

                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if encoder_outputs == None and inputs_embeds == None:
            inputs_embeds = self.embed(
                rst_start_token_id,
                rst_rel,
                rst_ns,
                rst_pos,
                key_phrase_ids,
                li_kprstpos,
                position_ids_kp)

        transformer_outputs = self.model(input_ids=None,
                             attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids,
                             encoder_outputs=encoder_outputs,
                             decoder_attention_mask=decoder_attention_mask,
                             decoder_cross_attention_mask=decoder_cross_attention_mask,
                             head_mask=head_mask,
                             decoder_head_mask=decoder_head_mask,
                             cross_attn_head_mask=cross_attn_head_mask,
                             past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds,
                             decoder_inputs_embeds=decoder_inputs_embeds,
                            decoder_context_rstpos=decoder_context_rstpos,
                            decoder_edu_rstpos=decoder_edu_rstpos,
                             use_cache=use_cache,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states,
                             return_dict=return_dict
                             )

        lm_logits = self.lm_head(transformer_outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            # masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            
            
            lm_logits = lm_logits.contiguous()
            labels = labels.contiguous()
            masked_lm_loss = self.loss_fct( lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        output =  Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            decoder_hidden_states=transformer_outputs.decoder_hidden_states,
            decoder_attentions=transformer_outputs.decoder_attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            encoder_last_hidden_state=transformer_outputs.encoder_last_hidden_state,
            encoder_hidden_states=transformer_outputs.encoder_hidden_states,
            encoder_attentions=transformer_outputs.encoder_attentions,
        )

        output['curr_edu_pos'] = curr_edu_pos
        output['context_rstpos'] = decoder_context_rstpos
        output['edu_rstpos'] = decoder_edu_rstpos

        return output

    def embed(
        self,
        rst_start_token_id,
        rst_rel,
        rst_ns,
        rst_pos,
        key_phrase_ids,
        li_kprstpos,
        position_ids_kp
     ):
        # RST context embedding
        rst_start_token_embed = self.model.encoder.embed_tokens(rst_start_token_id)
        rst_rel_embed = self.embed_rst_rels(rst_rel) 
        rst_ns_embed = self.embed_rst_ns(rst_ns)
        rst_pos_embed = self.embed_rst_pos(rst_pos) 
        rst_embed = ( rst_rel_embed + rst_ns_embed + rst_pos_embed ) 

        # Key Phrase context embedding
        topics_phrase_embed = self.model.encoder.embed_tokens( key_phrase_ids) 
        topics_rst_pos_embed = self.embed_rst_pos(li_kprstpos) 
        topics_embed = topics_rst_pos_embed + topics_phrase_embed

        inputs_embeds = torch.cat([
            rst_start_token_embed,
            rst_embed,
            topics_embed,
        ], axis=-2) * self.model.encoder.embed_scale

        # Position Embedding
        position_embed_kp = super(
                            type(self.model.encoder.embed_positions), 
                            self.model.encoder.embed_positions).forward(position_ids_kp + self.model.encoder.embed_positions.offset)
        bs,_,dim = position_embed_kp.shape
        position_embed = torch.cat(  [ position_embed_kp.new_zeros( [bs, rst_embed.shape[1]+1 ,dim] ) , position_embed_kp], axis=1 )

        inputs_embeds = inputs_embeds + position_embed

        return inputs_embeds

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        decoder_cross_attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_context_rstpos = None,
        decoder_edu_rstpos = None,
        **kwargs
        ):
        # cut decoder_input_ids if past is used
        if self.rst_tree_aligned_attention:
            curr_edu_pos = self.get_curr_edu_pos(decoder_input_ids, decoder_edu_rstpos)

            if past is not None:
                # calculating the new cross attention mask

                decoder_input_ids = decoder_input_ids[:, -1:]
                
                
                decoder_cross_attention_mask = self.RSTTokenizer.prepare_cross_attention_mask( context_rst_pos=decoder_context_rstpos,
                                                                                                past = None, utt_len = decoder_input_ids.shape[1],
                                                                                                utterance_ids = decoder_input_ids,
                                                                                                curr_edu_pos = curr_edu_pos )


            else:
                # calculating the new cross attention mask
                if decoder_input_ids.shape[1]!= decoder_cross_attention_mask.shape[-2]:
                    decoder_cross_attention_mask = self.RSTTokenizer.prepare_cross_attention_mask( context_rst_pos=decoder_context_rstpos,
                                                                                                past = decoder_cross_attention_mask,
                                                                                                utt_len = decoder_input_ids.shape[1],
                                                                                                utterance_ids = decoder_input_ids,
                                                                                                curr_edu_pos = curr_edu_pos )

        else:        
            if past is not None and use_cache==True:
                decoder_input_ids = decoder_input_ids[:, -1:]
                decoder_cross_attention_mask=decoder_cross_attention_mask[:, -1:, : ]
            elif past is None:
                decoder_cross_attention_mask = decoder_cross_attention_mask
                decoder_input_ids = decoder_input_ids
            

        
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_cross_attention_mask":decoder_cross_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            # change this to avoid caching (presumably for debugging)
            "use_cache": use_cache,
        }

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
     ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            embed_kwargs = {
                k:model_kwargs.pop(k) for k in list(model_kwargs.keys())
                    if k in ['rst_start_token_id',
                                'rst_rel',
                                'rst_ns',
                                'rst_pos',
                                'key_phrase_ids',
                                'li_kprstpos',
                                'position_ids_kp']
            }
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
            }
            encoder_kwargs['inputs_embeds'] = self.embed(**embed_kwargs)
            
            model_kwargs["encoder_outputs"] = encoder(
                input_ids, return_dict=True, **encoder_kwargs)
        return model_kwargs

    
    @classmethod
    def load_model(cls, model_name="RSTBart", model_version=None, mparams_new={}, device="cuda:0"):

        if model_version != None:
            #load from a pretrained RSTBart
            checkpoint = RSTBart_TrainingModule.get_ckpt_file(
                f'./models/{model_name}/version_{model_version}/checkpoints')

            mparams = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                'base_model_name', 'model_name', 'max_len_key_phrase',
                'max_len_rst', 'max_len_utt',
                'scale_grad_by_freq','rst_tree_aligned_attention']}

            # overriding with new keys
            for key, value in mparams_new.items():
                mparams[key] = value

            mconfig = RSTBart_Config.from_pretrained(mparams['base_model_name'], **mparams)

            # Loading Training Module
            training_module = RSTBart_TrainingModule(
                mconfig, mode='inference')
            training_module.load_state_dict(checkpoint['state_dict'])

            model = training_module.model
            tok = training_module.RSTTokenizer

            # Deleting checkpoints to free up GPU space
            del checkpoint
            torch.cuda.empty_cache()

            # if torch.cuda.is_available():
            if device != 'cpu' and torch.cuda.is_available():
                model = model.to(device)

            return model, tok
                
        else:
            raise ValueError("At least one of model_version or mconfig must not be None ")
    
    def on_train_end(self):

        # Saving Model using the pytorch method.
        # This allows relaoding using from_pretrained
        self.save_pretrained(
            f"./models_pt/{self.config.model_name}/version_{self.config.version}/")

    @staticmethod
    def parse_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--base_model_name',
                            default='facebook/bart-base', required=False)
        parser.add_argument('--model_name', default='RSTBart', required=False)
        parser.add_argument('--max_len_utt', type=int, default=250)
        parser.add_argument('--max_len_rst', type=int, default=30)
        parser.add_argument('--max_len_key_phrase', type=int, default=40)
        parser.add_argument('--scale_grad_by_freq', type=lambda x: bool(int(x)), default=False,
                            help="Inverse the gradients to the emebdding layers based on the occurence of each index in the minibatch ")
        parser.add_argument('--rst_tree_aligned_attention', type=lambda x: bool(int(x)), default=False)
        mparams = parser.parse_known_args()[0]
        return mparams

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if model_kwargs.get("decoder_cross_attention_mask") is not None:
            model_kwargs["decoder_cross_attention_mask"] =  model_kwargs["decoder_cross_attention_mask"].index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        
        return input_ids, model_kwargs

class RSTTokenizer(BartTokenizerFast, utils.EffeciencyMixin, utils.RstTokenizerMixin):
    rst_tree_aligned_attention = False  

    # Setting up RST2

    rst_rel_li = ['Attribution',
                  'Background', 'Cause', 'Comparison', 'Condition',
                  'Contrast', 'Elaboration', 'Enablement', 'Evaluation',
                  'Explanation', 'Joint', 'Manner-Means', 'Topic-Comment',
                  'Summary', 'Temporal', 'Topic-Change', 'same-unit', 'textual-organization']  # Add this to savable config

    rst_rel_labeler = sklp.LabelEncoder()
    rst_rel_labeler.fit(rst_rel_li)

    rst_ns_li = ['NN', 'NS', 'SN']
    rst_ns_labeler = sklp.LabelEncoder()
    rst_ns_labeler.fit(rst_ns_li)

    max_rst_pos = 4094

    # Setting up context lengths
    max_len_rst = 12
    max_len_key_phrase = 24
    max_len_utt = 1024
    max_rst_pos = 4094

    special_token_count = 2

    rst_start_token = "<rst>"
    keyphrase_start_token = "<kp>"

    def encode_input(self, rst_rel, rst_ns, rst_pos, li_kp, li_kprstpos,
                     utterance=None, utterance_prompt=None, dict_pos_edu=None,
                     max_len_rst=None,
                     exclude_from_output=[], device=None):
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
            utterance = utterance.lstrip( string.punctuation )

        rst_rel, rst_ns, rst_pos, rst_pad_len = self.encode_rst(
            rst_rel, rst_ns, rst_pos, max_len_rst)
        key_phrase_ids, li_kprstpos, ta_tokens_pos, kp_phrase_lens = self.encode_keyphrase(
            li_kp, li_kprstpos)

        decoder_input_ids, labels, utt_len = self.encode_utterance(utterance, utterance_prompt, context_len=1 + rst_rel.shape[-1] + key_phrase_ids.shape[-1] )

        r_len = 1 + rst_rel.shape[-1]
        rt_len = r_len + key_phrase_ids.shape[-1]

        # Building position ids for key phrase
        position_ids_kp = torch.cat(
            [torch.arange(tpl, dtype=torch.long) for tpl in kp_phrase_lens])

        # Building Encoder Attention Mask
            # prepending 0 to rst_pos in order to factor in 
        attention_mask = self.prepare_encoder_attention_mask(
            r_len, rt_len, ta_tokens_pos, kp_phrase_lens, rst_pos= torch.cat( [ rst_pos[0:1], rst_pos]),
            li_kprstpos=li_kprstpos)

        

        decoder_cross_attention_mask = self.prepare_cross_attention_mask(dict_pos_edu, torch.cat([ rst_pos[0:1], rst_pos]),
                                                                         li_kprstpos, utt_len, rt_len,
                                                                         utterance_ids = decoder_input_ids,
                                                                         training=True
                                                                         )
        

        attention_mask = self.prepare_attention_mask_handle_padding(attention_mask,
                            r_len, rst_pad_len, max_len_rst)
        
        decoder_cross_attention_mask = self.prepare_attention_mask_handle_padding(decoder_cross_attention_mask,
                            r_len, rst_pad_len, max_len_rst, cross_attn=True)

        output =  {'rst_start_token_id': self.rst_start_token_id,
                
                'rst_rel': rst_rel, 'rst_ns': rst_ns, 'rst_pos': rst_pos,

                'key_phrase_ids': key_phrase_ids.contiguous(),
                'li_kprstpos': li_kprstpos.contiguous(),

                'position_ids_kp': position_ids_kp.contiguous(),

                'attention_mask': attention_mask,
                'labels': labels,

                'decoder_input_ids': decoder_input_ids,
                'decoder_cross_attention_mask': decoder_cross_attention_mask,
                
                }
        
        if self.rst_tree_aligned_attention:
            output['decoder_context_rstpos']= torch.cat( [ rst_pos[0:1], rst_pos, li_kprstpos ] )
            
            dec_rst_pos = [ self.clamp_values(np.array(int(key)),utils.MAX_LONG_VALUE).item(0)  for key in dict_pos_edu.keys()]
            
            output['decoder_edu_rstpos'] = torch.tensor( sorted( dec_rst_pos
                                                                , key=RSTTokenizer.edukp_pos_sort_function ) , 
                                                dtype=torch.long )
        
        if device != None:
            for key in output:
                if output[key] != None:
                    output[key] = output[key].to(device).unsqueeze(0)
        
        for key in exclude_from_output:
            output.pop(key,None)
                

        return output

    def encode_utterance(self, utterance=None, utterance_prompt=None, context_len=None):

        # Creating labels/targets
        # Creating labels/targets
        if utterance_prompt != None:
            # utterance_prompt = self.bos_token + utterance_prompt
            utterance_prompt = self.eos_token + self.bos_token + utterance_prompt

            utt_prompt_tok_ids = self.encode(
                utterance_prompt, 
                add_special_tokens=False,
                # add_special_tokens=True,

                return_attention_mask=False,
                padding='do_not_pad',
                        truncation=False,
                max_length=self.max_len_utt,
                return_tensors='pt',
                return_length=False,
            )[0]

            decoder_input_ids = utt_prompt_tok_ids.contiguous()
            labels = None
            utt_len = decoder_input_ids.shape[-1]
                
        if utterance != None:
            utterance = self.bos_token + utterance + self.eos_token + self.pad_token
            labels = self.encode(
                utterance, 
               add_special_tokens=False,
                # add_special_tokens=True,
                return_attention_mask=False,
                padding='do_not_pad',
                        truncation=True,
                max_length=self.max_len_utt,
                return_tensors='pt',
                return_length=False,
            )[0]

            decoder_input_ids = shift_tokens_right(labels.unsqueeze(0), 
                                    self.pad_token_id ,
                                    decoder_start_token_id=self.eos_token_id)[0]
            utt_len = decoder_input_ids.shape[-1]

        return decoder_input_ids, labels, utt_len

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

        # tnsr_pos = torch.LongTensor(self.clamp_values(
        #     np.array(rst_pos), utils.MAX_LONG_VALUE))  
        
        tnsr_pos = torch.LongTensor( [RSTTokenizer.clamp_value(val) for val in rst_pos] )

        # padding ns and pos
        # The ns and pos embedding layer uses the index value 0 as a padding index
        # For this index the vector is initialized to zer0 and as such never updates
        len_ = tnsr_rels.shape[0]

        if variable_padding_size != None:

            pad_len = min( variable_padding_size, self.max_len_rst) -1

            if len_ > pad_len:
                tnsr_rels   = tnsr_rels[:pad_len]
                tnsr_ns     = tnsr_ns[:pad_len]
                tnsr_pos    = tnsr_pos[:pad_len]
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

            elif len_ < self.max_len_rst -1 :
                diff = self.max_len_rst -1 - len_
                tnsr_rels = torch.nn.functional.pad(
                    tnsr_rels, (0, diff), value=self.pad_values['rst_rel'])
                tnsr_ns = torch.nn.functional.pad(
                    tnsr_ns, (0, diff), value=self.pad_values['rst_ns'])
                tnsr_pos = torch.nn.functional.pad(
                    tnsr_pos, (0, diff), value=self.pad_values['rst_pos'])
            else:
                diff = 0
        
        return tnsr_rels, tnsr_ns, tnsr_pos, diff

    def encode_keyphrase(self, topics, li_kprstpos):
        """[summary]

            Args:
                topics ([type]): [list of topics (phrases or words)]
                topics_score ([type]): [list of float scores for each topic relevancy]

            Raises:
                Exception: [description]

            Returns:
                [type]: [description]
        """
        if len(topics)!=0:
            max_len = self.max_len_key_phrase

            str_topics = ''.join(['<kp> '+topic for topic in topics])
            dict_encoding = self(str_topics, add_special_tokens=False,
                                return_attention_mask=False,
                                truncation=True,
                                padding='do_not_pad',
                                return_tensors='np',
                                max_length=max_len,
                                return_special_tokens_mask=False,
                                return_length=True)
            topic_phrases = dict_encoding['input_ids'][0]

            # Repeating each score in the case where the score is allocated to a phrase topic which is broken down into constituent words
            # e.g. topics - ["fast car", "motorbike", "long rail road"], scores = [0.9, 0.4, 0.2] -> scores = [0.9, 0.9, 0.9, 0.4, 0.4, 0.2, 0.2, 0.2, 0.2]
            # have to do it after tokenization due to bytepair encoding
            # get index of where <kp> tokens occur
            kp_idxs = np.where(
                topic_phrases == self.keyphrase_start_token_id_np )[0]

            topic_phrases = torch.LongTensor(topic_phrases)

            # filtering out idxs if index is larger than padding value
            kp_idxs = kp_idxs[kp_idxs < max_len]

            # get difference in index position between <kp> tag n and <kp> tag n+1 ( for final tag use difference between tag and end of list)
            kp_phrase_lens = np.diff(kp_idxs, append=topic_phrases.numel())

            # copies each score phrase_len times to cover that phrase and handles case where there is no phrase
            _ = [[pos]*phrase_len for pos,
                          phrase_len in zip(li_kprstpos, kp_phrase_lens)]
            li_kprstpos = functools.reduce(operator.iconcat, _, [])

            tnsr_rst_pos = torch.LongTensor( [ RSTTokenizer.clamp_value( val ) for val in li_kprstpos ] )
        else:
            topic_phrases = torch.LongTensor([])
            tnsr_rst_pos = torch.LongTensor([])
            kp_idxs = np.array([])
            kp_phrase_lens = np.array([])

        return topic_phrases, tnsr_rst_pos, kp_idxs, kp_phrase_lens

    def prepare_encoder_attention_mask(self, r_len, rt_len, ta_tokens_pos, kp_phrase_lens, rst_pos=None, li_kprstpos=None):

        if self.rst_tree_aligned_attention == False:
            attention_mask = torch.zeros([rt_len, rt_len])

            # Change so key_phrases only attend to themselves and the rst start token
            # RST Tree only attends to itself  as well
            
            #attention_mask[r_len:, r_len:] = 0
            attention_mask[r_len:, r_len: ] = 0
            attention_mask[ 1:r_len, r_len: ] = 0

            #rst section attend to itself
            attention_mask[ :r_len, :r_len ] = 1
            
            #kp section attend to rst section
            attention_mask[ r_len:rt_len, :rt_len ] = 1

            # Second each topic phrase has causal masking on the tokens within the topic phrase
            # use ta_tokens_pos
            # essentially for idx i, i+1
            for ta_idx, phrase_len in zip(ta_tokens_pos, kp_phrase_lens):
                s_idx = r_len+ta_idx
                e_idx = s_idx+phrase_len
                attention_mask[s_idx:e_idx, s_idx:e_idx] = \
                    attention_mask.new_ones([phrase_len, phrase_len])
                            
        else:
            # Detecting which node each context should attend to in  O(n)
            # pos should be ordered based on left to right along an imaginary tree
            # so first re-order in terms of depth (which is just re-ordering by value )
            # Then starting from the end of the list find the direct tree of nodes to the parent
            # Store each pos and parent in a dictionary of keys=child, value=parent
            dict_rstpos_parents = torch.nn.ParameterDict()

            for pos in torch.cat((rst_pos, li_kprstpos)):
                
                if pos not in dict_rstpos_parents:
                    dict_rstpos_parents[str(pos)] = torch.nn.parameter.Parameter( torch.tensor(
                        RSTTokenizer.seq_from_root_to_edu_pos(pos) + [int(pos)] , dtype=torch.long), requires_grad=False )

            # Creating vector indicating which positions attend to which other positions
            all_pos = torch.cat((rst_pos, li_kprstpos))
            li_tens_pos = []
            for pos in all_pos:

                li_parent_tree = dict_rstpos_parents[str(pos)]

                pos_tree_aligned_attn = (
                    all_pos[..., None] == li_parent_tree).any(-1).squeeze()  # Creates a boolean vector indicating where model can attend

                li_tens_pos.append(pos_tree_aligned_attn)

            attention_mask = torch.stack(li_tens_pos, dim=0).to(torch.float)

        return attention_mask

    def prepare_cross_attention_mask(self, 
                                    dict_pos_edu=None, rst_pos=None, li_kprstpos=None,
                                     utt_len=None, rt_len=None, prev_mask=None, curr_edu_pos=None, context_rst_pos=None,
                                     utterance_ids=None,
                                    training=True,
                                     ):

        if self.rst_tree_aligned_attention:
            #training
            if prev_mask == None and curr_edu_pos==None:
                li_attn_vectors = []
                dict_rstpos_parents = torch.nn.ParameterDict()
                
                all_pos = torch.cat((rst_pos, li_kprstpos))

                li_pos_edu_idslen_ids = sorted([[str(self.clamp_values(np.array(int(pos)), utils.MAX_LONG_VALUE).item(0)), edu, None, None] for pos, edu in dict_pos_edu.items()],
                                               key=lambda x: RSTTokenizer.edukp_pos_sort_function(
                    int(x[0])))

                # Adding special tokens to edu to mirror the encoded utterance
                li_pos_edu_idslen_ids[0][1] =  self.eos_token + self.bos_token + li_pos_edu_idslen_ids[0][1]
                
                if training:
                    li_pos_edu_idslen_ids[-1][1] =  li_pos_edu_idslen_ids[-1][1] + self.eos_token
                
                
                # Find the tokenized length of each edu
                # And get the seqeunce of parents for each edu position
                # Note this ignores the length of the start and end token
                for idx in range(len(li_pos_edu_idslen_ids)):

                    if idx != 0:
                        li_pos_edu_idslen_ids[idx][1] = " " + \
                            li_pos_edu_idslen_ids[idx][1]


                    li_pos_edu_idslen_ids[idx][3] = self.encode( li_pos_edu_idslen_ids[idx][1], add_special_tokens=False )

                    li_pos_edu_idslen_ids[idx][2] = len(
                        li_pos_edu_idslen_ids[idx][3])

                    
                    pos = li_pos_edu_idslen_ids[idx][0]
                    
                    if pos not in dict_rstpos_parents:
                        dict_rstpos_parents[pos] = torch.nn.parameter.Parameter( torch.tensor(
                            RSTTokenizer.seq_from_root_to_edu_pos(int(pos)) + [int(pos)] , dtype=torch.long), requires_grad=False )

                # region EDU tokenization may be different from text tokenization due to the RST parser
                # evening up the tokenization lengths
                _len = sum(item[2] for item in li_pos_edu_idslen_ids)
                if _len != utt_len and training:

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

                # Then cycle through these lengths, and create the masking over the encoder
                    # Each edu should only attend to the rst and keyp phrases with pos in its parents pos
                    # Remember to create a 3 extras, for the starting eos and bos token and the ending eos token
                for pos, edu_txt, edu_txt_len, edu_ids in li_pos_edu_idslen_ids:

                    li_parent_tree = dict_rstpos_parents[pos]

                    pos_tree_aligned_attn = (
                        all_pos[..., None] == li_parent_tree).any(-1).squeeze() 
                    
                    # Repeating by tokenized length of EDU
                    pos_tree_aligned_attn = einops.repeat(pos_tree_aligned_attn,
                        'd -> l d', l=edu_txt_len )

                    li_attn_vectors.append(pos_tree_aligned_attn)

                # need to transpose since _expand_mask requires masks to be bsz, 1, tgt_len, src_len
                attention_mask = torch.cat(li_attn_vectors, dim=0).to(torch.float)
                
                if attention_mask.shape[0] >utt_len:
                    attention_mask = attention_mask[:utt_len, :]
            
            #generating
            else:
                all_pos = context_rst_pos
                
                li_batch_new_attn = []
                for pos in curr_edu_pos: 
                
                    li_parent_tree = torch.tensor( RSTTokenizer.seq_from_root_to_edu_pos(pos.item()) + [pos.item()], device=prev_mask.device )
                    
                    pos_tree_aligned_attn = (
                            all_pos[..., None] == li_parent_tree).any(-1).squeeze()

                    li_batch_new_attn.append(pos_tree_aligned_attn.unsqueeze(0))
                
                # batch_new_attn = torch.cat(li_batch_new_attn, axis=0)
                
                attention_mask = torch.stack(
                    li_batch_new_attn, dim=0).float()  # shape( bs, 1 , context )


                # appending to new attention_mask if it exists otherwise just return the attention
                prev_mask = torch.nn.functional.pad(prev_mask, (0, 1), value=0)

                attention_mask = torch.cat([prev_mask, attention_mask], axis=1)
                        
        else:
            #training
            if prev_mask == None:
                # NOTE: padding in rst section must be handled later
                attention_mask = torch.ones((utt_len, rt_len))
            #generation
            else:
                dims = prev_mask.shape()
                attention_mask = torch.cat(
                    [prev_mask[:, -1:, :], prev_mask.new_ones([dims[0], 1, 1])], axis=-1)

        return attention_mask

    def prepare_attention_mask_handle_padding(self, attention_mask,
                                                r_len=None, rst_pad_len=0, max_len_rst=None,
                                                rt_len=None, kp_pad_len=0, kp_max_len=None,
                                                cross_attn =False ):

        if max_len_rst != None and rst_pad_len != 0:
            attention_mask[:, r_len-rst_pad_len:r_len] = 0
            if not cross_attn:
                attention_mask[r_len-rst_pad_len:r_len, :] = 0

        if kp_max_len != None and kp_pad_len != 0:
            attention_mask[:, rt_len-kp_pad_len:rt_len] = 0
            if not cross_attn:
                attention_mask[rt_len-kp_pad_len:rt_len, :] = 0

        return attention_mask

    @classmethod
    def from_pretrained(cls,
                        dir_tokenizer="./tokenizers/RSTBart",
                        base_tokenizer_name="facebook/bart-base",
                        rst_params={},
                        **kwargs):  # max_len_rst, max_len_key_phrase, max_rst_depth, max_len_utt, max_rst_pos

        if os.path.exists(dir_tokenizer):
            tokenizer = super(RSTTokenizer, cls).from_pretrained(
                dir_tokenizer, local_files_only=True, **kwargs)

        else:

            at_rst_start = AddedToken(cls.rst_start_token, lstrip=False, rstrip=False) if isinstance(
                cls.rst_start_token, str) else cls.rst_start_token
            at_topic_start = AddedToken(cls.keyphrase_start_token, lstrip=False, rstrip=False) if isinstance(
                cls.keyphrase_start_token, str) else cls.keyphrase_start_token
            additional_special_tokens = [at_rst_start, at_topic_start]

            cls = super(RSTTokenizer, cls).from_pretrained(base_tokenizer_name,
                                                           additional_special_tokens=additional_special_tokens)

            cls.save_pretrained(dir_tokenizer)
            tokenizer = cls

        tokenizer.rst_start_token_id        = tokenizer.encode( tokenizer.rst_start_token, return_tensors="pt", add_special_tokens=False)[0]
        tokenizer.keyphrase_start_token_id  = tokenizer.encode( tokenizer.keyphrase_start_token, return_tensors="pt", add_special_tokens=False)[0]
        tokenizer.keyphrase_start_token_id_np = tokenizer.keyphrase_start_token_id.numpy()

        for k, v in rst_params.items():
            setattr(cls, k, v)

        return tokenizer

class RSTBart_TrainingModule(pl.LightningModule):

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
                 mode='train_new',
                 tag='',
                 batching_style='effecient',
                 **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.gpus = gpus
        self.mode = mode
        self.workers = workers
        self.batching_style = batching_style
        self.RSTTokenizer = RSTTokenizer.from_pretrained(f"./tokenizers/{mconfig.model_name}",
                                                         base_tokenizer_name=mconfig.base_model_name,
                                                         rst_params={name: getattr(mconfig, name) for name in ['max_len_rst',
                                                                                                               'max_len_key_phrase',
                                                                                                               'max_rst_depth',
                                                                                                               'max_len_utt', 
                                                                                                               'max_rst_pos',
                                                                                                               'max_rst_pos',
                                                                                                               'rst_tree_aligned_attention'] if hasattr(mconfig, name)
                                                                     }
                                                         )

        mconfig.vocab_size = mconfig.vocab_size-2 
        self.model = RSTBart.from_pretrained( mconfig.base_model_name, config=mconfig )
        mconfig.vocab_size = mconfig.vocab_size+2
        self.model.config.vocab_size = mconfig.vocab_size
        self.model.resize_token_embeddings(self.model.config.vocab_size)


        
        self.pad_values = {'rst_start_token': mconfig.pad_token_id,
                           'rst_rel': self.model.embed_rst_rels.padding_idx,
                           'rst_ns': self.model.embed_rst_ns.padding_idx,
                           'rst_pos': self.model.embed_rst_pos.padding_idx,

                           'key_phrase_ids': mconfig.pad_token_id,
                           'li_kprstpos': self.model.embed_rst_pos.padding_idx,

                           'position_ids_kp': mconfig.pad_token_id,

                           'utt_tok_ids': mconfig.pad_token_id,
                           'attention_mask': 0.0,

                           'labels': self.model.loss_fct.ignore_index,
                           'decoder_input_ids': mconfig.pad_token_id,
                           'decoder_cross_attention_mask': 0.0,

                            'decoder_edu_rstpos': -1,
                                
                            'decoder_context_rstpos': -1
                           }
        
        self.RSTTokenizer.pad_values = self.pad_values

        self.pad_maxlens = {
            'rst_start_token': 1,
            'rst_rel': mconfig.max_len_rst-1,
            'rst_ns': mconfig.max_len_rst-1,
            'rst_pos': mconfig.max_len_rst-1,

            'key_phrase_ids': mconfig.max_len_key_phrase,
            'li_kprstpos': mconfig.max_len_key_phrase,

            'utt_tok_ids': mconfig.max_len_utt if mconfig.max_len_utt else self.config.max_position_embeddings,
            'labels': mconfig.max_len_utt if mconfig.max_len_utt else self.config.max_position_embeddings,
            'decoder_input_ids': mconfig.max_len_utt if mconfig.max_len_utt else self.config.max_position_embeddings,

            'attention_mask': mconfig.max_len_rst + mconfig.max_len_key_phrase,  # axis:max_length
            'decoder_cross_attention_mask': [ mconfig.max_len_utt , mconfig.max_len_rst+mconfig.max_len_key_phrase ] , #max_lens in both 2d dimensions

            'position_ids_kp': mconfig.max_len_key_phrase,
            
            'decoder_edu_rstpos': mconfig.max_rst_pos // 2,
            'decoder_context_rstpos':mconfig.max_len_rst + mconfig.max_len_key_phrase

        }
        self.RSTTokenizer.pad_maxlens = self.pad_maxlens
        
        if mconfig.rst_tree_aligned_attention:
            self.model.RSTTokenizer = self.RSTTokenizer

        if self.mode in ['train_new', 'train_cont', 'test']:
            self.dir_data = utils.get_path(dir_data)
            self.accumulate_grad_batches = accumulate_grad_batches
            self.tag = tag

            self.dg = DataLoaderGenerator(self.dir_data,  self.batch_size, self.RSTTokenizer,
                                 workers=self.workers, mode=self.mode, gpus=self.gpus,
                                 pad_maxlens=self.pad_maxlens, pad_values=self.pad_values,
                                 batching_style=self.batching_style
                                 )

            if self.mode == "test":
                self.create_data_loaders(['test'])
            else:
                self.create_data_loaders(['train', 'val', 'inference'] )
                self.inference_samples = list(islice(self.inference_dl, 3))
                del self.inference_dl

        if self.mode in ['train_new', 'train_cont']:
            self.max_epochs = max_epochs
            self.warmup_proportion = warmup_proportion
            self.learning_rate = learning_rate

            train_params_to_save = self.return_params()
            mparams_to_save = {param: getattr(mconfig, param) for param in list(filter(
                lambda p: p not in ['self','kwargs'], list(inspect.signature(RSTBart_Config.__init__).parameters.keys()) ))}

            self.hparams.update({**train_params_to_save, **mparams_to_save})
            pl.core.saving.save_hparams_to_yaml(os.path.join(os.path.dirname(
                kwargs['dir_checkpoints']), "hparams.yaml"), self.hparams)

            bad_words = ["<rst>", "<kp>"]
            bad_words_ids = [self.RSTTokenizer.encode(
                bad_word,) for bad_word in bad_words]
            bad_words_ids = bad_words_ids + \
                [self.RSTTokenizer.encode(bad_word) for bad_word in bad_words]
            bad_words_ids = bad_words_ids 


            
            # self.trainer.logger.add_hparams( vars(mconfig) )

        if self.mode in ['inference']:
            self.eval()
            self.freeze()

    @staticmethod
    def parse_train_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=True, allow_abbrev=False)
        parser.add_argument('--dir_data', default="./dataset_v3_2",
                            help="Relative directory path for datafiles")
        parser.add_argument('--model_dir', default="./models/")
        parser.add_argument('--max_epochs', default=8, type=int)
        parser.add_argument('--accumulate_grad_batches', default=1, type=int)
        parser.add_argument('--batch_size', default=20, type=int)
        parser.add_argument('--batching_style', default='effecient', type=str, choices=['effecient','standard'])

        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--workers', default=16, type=int)  # TODO: change to 6
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--mode', default='train_new', type=str,
                            choices=['train_new', 'train_cont', 'test', 'inference'])
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

        if tparams['mode'] in ["train_new"]:            
            mconfig = RSTBart_Config.from_pretrained(mparams['base_model_name'], **mparams)

            training_module = RSTBart_TrainingModule(mconfig, **tparams)

        elif tparams['mode'] in ["train_cont", "inference"]:

            checkpoint = RSTBart_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            # restore/update param files from the checkpoint
            if "hyper_parameters" in checkpoint.keys() and tparams['override'] == False:
                tparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                     'learning_rate', 'precision', 'splits', 'tag']})

                mparams.update({k: v for k, v in checkpoint['hyper_parameters'].items() if k in [
                    'base_model_name', 'model_name', 'max_len_utt','max_len_rst','max_len_key_phrase',
                    'scale_grad_by_freq','rst_tree_aligned_attention' ]})

                mparams = mparams

            else:
                print("param files not found utilsing default or user entered params\n")

            mconfig = RSTBart_Config.from_pretrained(mparams['base_model_name'], **mparams)

            # Restore/update Training Module
            training_module = RSTBart_TrainingModule(mconfig, **tparams)

            training_module.load_state_dict(checkpoint['state_dict'])

        elif tparams['mode'] in ["test"]:

            checkpoint = RSTBart_TrainingModule.get_ckpt_file(
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
            training_module = RSTBart_TrainingModule(
                **tparams, mparams=mparams)
            training_module.load_state_dict(checkpoint['state_dict'])

        else:
            raise ValueError(
                "tparams['mode'] must be in range [train_new, train_cont, test, inference]")

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

        # checkpoint_callback._save_model = types.MethodType(
        #     utils.monkey_save_model, checkpoint_callback)  # monkey patch

        # checkpoint_callback._save_model = mpatch_save_model(checkpoint_callback._save_model)

        checkpoint_callback._save_model = types.MethodType(
            mpatch_save_model(checkpoint_callback._save_model), checkpoint_callback)  #

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience = 40,       
            verbose=False,
            mode='min'
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(early_stop_callback)

        if tparams['gpus'] in [0, 1]:
            trainer_vars = {}
        else:

            trainer_vars = {    'accelerator': 'ddp',
                            'plugins': DeepSpeedPlugin(stage=1, 
                                                        contiguous_gradients=True,
                                                         ) 
                            # 'plugins' : DDPPlugin(find_unused_parameters=True)
                            }

        if tparams['mode'] in ["train_new"]:

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    #progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                                                    default_root_dir=tparams['dir_checkpoints'],
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks,
                                                    val_check_interval=0.05,
                                                    limit_val_batches = 0.25,
                                                    reload_dataloaders_every_n_epochs=1,
                                                    num_sanity_val_steps=2,
                                                    replace_sampler_ddp=False,
                                                    **trainer_vars,
                                                    )

        elif tparams['mode'] in ["train_cont", "inference"]:
            # restoring checkpoint
            checkpoint = RSTBart_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    logger=tb_logger,
                                                    precision=tparams['precision'],
                                                    callbacks=callbacks, 
                                                    val_check_interval=0.05,
                                                    limit_val_batches = 0.25,
                                                    reload_dataloaders_every_n_epochs=1,
                                                    num_sanity_val_steps=0,
                                                    replace_sampler_ddp=False,

                                                    **trainer_vars,
                                                    )

            # load callback states
            trainer.on_load_checkpoint(checkpoint)
            
            try:
                trainer.global_step = checkpoint['global_step']
                trainer.current_epoch = checkpoint['epoch'] - 1
            except Exception:
                trainer.fit_loop.global_step = checkpoint['global_step']
                trainer.fit_loop.current_epoch = checkpoint['epoch'] -1

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

            # restoring checkpoint
            checkpoint = RSTBart_TrainingModule.get_ckpt_file(
                tparams['dir_checkpoints'])

            training_module.load_state_dict(checkpoint['state_dict'])

            trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**tparams),
                                                    progress_bar_refresh_rate=tparams['accumulate_grad_batches'],
                                                    check_val_every_n_epoch=1,
                                                    checkpoint_callback=False,
                                                    logger=tb_logger,
                                                    log_every_n_steps=1,
                                                    precision=tparams['precision'],
                                                    )

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

        if tparams['mode'] in ['train_new', 'train_cont']:
            trainer.fit(training_module)

        if tparams['mode'] in ["test"]:

            checkpoint = RSTBart_TrainingModule.get_ckpt_file(
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

        input_ = batch
        output = self.forward(input_)
        loss = output.loss

        loss_key = f"{step_name}_loss"
        output = {}

        if step_name == 'train':
            output["loss"] = loss

        else:
            str_loss_key = loss_key
            output[str_loss_key] = loss

        return output
    
    def step_end(self, output, step_name):
        
        if step_name == "train":
            loss_key = "loss"
            loss = output[loss_key].mean()
            on_step = True
            on_epoch = False
        else:
            loss_key =  f"{step_name}_loss"
            loss = output[loss_key].mean()
            on_step = False
            on_epoch = True

        self.log(loss_key, loss, logger=True, on_step=on_step, on_epoch=on_epoch, sync_dist=True )

    def training_step(self, batch, batch_idx):
        output = self.step(batch, "train")
        return output

    def validation_step(self, batch, batch_idx):
        output = self.step(batch, "val")
        return output

    def test_step(self, batch, batch_idx):
        output = self.step(batch, "test")
        return output
    
    def training_step_end(self, output ):
        return self.step_end(output, "train")
    
    def validation_step_end(self, output):
        return self.step_end(output, "val")

    def training_epoch_end(self, outputs):
        self.epoch_end_log(outputs, "train")

    def validation_epoch_end(self, outputs: List[dict]):
        self.epoch_end_log(outputs, "val")

    def test_epoch_end(self, outputs: List[dict]):
        self.epoch_end_log(outputs, "test")

    def epoch_end_log(self, outputs, step_name):

        if step_name == "train":
            pass
        else:
            loss = torch.stack([x[f"{step_name}_loss"]for x in outputs]).mean()
            
            self.log(f"{step_name}_loss", loss, logger=True, prog_bar=True, sync_dist=True)
        
        if step_name == "val" and _get_rank() == 0:
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

                    df = pd.DataFrame(columns=['epoch', 'rst_rels', 'topics', 'utterance',
                                            'dict_pos_edu', 'li_kprstpos',

                                            'rst_ns',
                                            'rst_pos',
                                            ])

                    rst_rels = encoded_input.pop('orig_rst_rels')
                    rst_ns = encoded_input.pop('orig_rst_ns')
                    rst_pos = encoded_input.pop('orig_rst_pos')

                    topics = encoded_input.pop('orig_key_phrase')
                    utterance = encoded_input.pop('orig_utt')
                    dict_pos_edu = encoded_input.pop('orig_dict_pos_edu')

                    orig_li_kprstpos = encoded_input.pop('orig_li_kprstpos')

                    datum = {
                        'epoch': -1,

                        'rst_rels': ', '.join(rst_rels),
                        'rst_ns': ', '.join(rst_ns),
                        'rst_pos': rst_pos,

                        "topics": ', '.join(topics),
                        "utterance": utterance,
                        "dict_pos_edu": json.dumps(dict_pos_edu),

                        "li_kprstpos": json.dumps(orig_li_kprstpos),
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
                # encoded_input.pop('labels', None)

                for k in list(encoded_input.keys()):
                    encoded_input[k] = encoded_input[k].to(self.model.device )

                with torch.no_grad():
                    output = self.model.generate(
                        None, use_cache=True, **encoded_input, **self.model.generation_params)    
                    output = output[0]
                
                decoded_text = self.RSTTokenizer.decode(output,
                                                    skip_special_tokens=False)
                datum = {
                    'epoch': self.current_epoch,
                    'rst_rels': '',
                    'topics': '',
                    'utterance': json.dumps(decoded_text),
                    'dict_pos_edu': '',
                    'li_kprstpos': '',
                    'rst_ns': '',
                    'rst_pos': ''
                }

                pd.DataFrame.from_records([datum]).to_csv(fp, index=False, mode='a', header=False)
                # Saving to file
        
        else:
            pass
        
    def create_data_loaders(self, modes ):

        if 'train' in modes:
            self.train_dl = self.dg.prepare_dataloader(
                split_name='train')
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

        return self.dg.prepare_dataloader(
                split_name='train')

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

        optimizer = optim.Adafactor(self.model.parameters(), scale_parameter=True, 
                        relative_step=True, warmup_init=True, lr=None )


        lr_scheduler = AdafactorSchedule(optimizer)


        return { 'optimizer':optimizer, "lr_scheduler": lr_scheduler, "interval": "step", "monitor": "val_loss"}    
    
    def return_params(self):
        params = {}
        keys = ['batch_size', 'accumulate_grad_batches', 'learning_rate', 'max_epochs', 'dir_data'
                'tag']

        params = {
            k: self.__dict__[k] for k in keys if k in self.__dict__.keys()
        }

        return params

class DataLoaderGenerator():
    """Handles the creation of dataloaders for a train, val and test set
    """

    def __init__(self, dir_data, batch_size,
                 tokenizer,
                 workers=0, mode='train_new',
                 gpus=1,
                 pad_values={},
                 pad_maxlens={},
                 batching_style='effecient',
                 **kwargs):

        self.dir_data = dir_data
        self.tokenizer = tokenizer
        self.splits = {'train': 0.6, 'val': 0.2, 'test': 0.2}

        self.batch_size = batch_size
        self.batching_style = batching_style
        self.workers = workers
        self.mode = mode
        self.gpus = gpus
        self.pad_values = pad_values
        self.pad_maxlens = pad_maxlens

    def prepare_dataloader(self,
                           split_name='train'):
        """Prepares a dataloader given a directory of data for NLG language module
            # The current method takes a percentage of data from each subdirectory
            Args:
                dir_dset ([type]): [description]
        """
        dir_data = self.dir_data

        # getting all files from all different subreddits/types of conversation
        #debugging
        fns = glob.glob(os.path.join(utils.get_path(dir_data), "*", "*"))
        fns = [fn for fn in fns if os.path.split(
            fn)[-1] != "lock" and "dict_len" not in fn][:10]
                
        # getting number of utterances records in each file
        files_sizes = [int(fn[-10:]) for fn in fns]

        # defining starting line and total lines to use for dataset
        
        if split_name == 'train':
            line_starts = [0]*len(files_sizes)
            line_ends = [ls+int(fs*self.splits['train'])
                         for ls, fs in zip(line_starts, files_sizes)]
            inference = False
            bs = self.batch_size
            shuffle=True
            collate_fn = lambda batch: self.tokenizer.default_collate_pad(batch)


        elif split_name == 'val':
            line_starts = [int(fs*self.splits['train']) for fs in files_sizes]
            line_ends = [ls+int(fs*self.splits['val'])
                         for ls, fs in zip(line_starts, files_sizes)]
            shuffle = True
            inference = False
            bs = self.batch_size
            collate_fn = lambda batch: self.tokenizer.default_collate_pad(batch)


        elif split_name == 'test':
            line_starts = [int(fs*(1-self.splits['test']))
                           for fs in files_sizes]
            shuffle = False
            line_ends = files_sizes
            inference = False
            bs = self.batch_size
            collate_fn = lambda batch: self.tokenizer.default_collate_pad(batch)
            

        elif split_name == 'inference':
            line_starts = [int(fs*(1-self.splits['test']))
                           for fs in files_sizes]
            line_ends = files_sizes
            sampler = None
            inference = True
            shuffle=False
            bs = 1
            collate_fn = lambda batch: self.tokenizer.default_collate_pad(batch)

        li_dsets = [SingleDataset(_f, self.tokenizer, line_start, line_end, inference )
                    for _f, line_start, line_end in zip(fns, line_starts, line_ends)]

        concat_dset = torch.utils.data.ConcatDataset(li_dsets)

        if self.gpus <= 1 and split_name not in ['inference', 'test']:
            sampler = SizedOrdered_Sampler(concat_dset, bs, shuffle=shuffle, batching_style=self.batching_style)
        elif self.batching_style=='effecient' and self.gpus > 1 and split_name not in ['inference', 'test' ]:
            sampler = SizedOrdered_DistributedSampler(concat_dset, bs, shuffle=shuffle, gpus=self.gpus)
        else:
            sampler = None

        dataloader = torch.utils.data.DataLoader(concat_dset, 
                                                batch_size= bs,
                                                 num_workers=self.workers,
                                                 sampler=sampler,
                                                 pin_memory=False,
                                                 collate_fn=collate_fn)

                                                 
        return dataloader

# concat basically makes all entries one long list of sequential indexes
# sampler creates a randomised index list to sample from list above
# In smapler, we can access concat dataset and each individual dataset
# In each dataset add a list of the rst_len_pad

class SingleDataset(torch.utils.data.Dataset):
    """creates a dataloader given a directory of text files each containing a conversation

        create a custom index which sorts the entries by their length
    """
    def __init__(self, file_path, tokenizer, line_start, line_end, inference,**kwargs):
        self.fp = file_path
        self.tokenizer = tokenizer
        self.line_start = line_start
        self.line_end = line_end
        self.inference = inference

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
            file_path), f"bart_dict_lens_{line_start}_to_{line_end}.pkl")

        # if os.path.exists( fp_cached_order):
        #     os.remove(fp_cached_order)

    
        if os.path.exists(fp_cached_order):
            dict_cached_order = pickle.load(open(fp_cached_order, "rb"))
            self.np_textlens = dict_cached_order['np_textlens']
            self.np_rstlens = dict_cached_order['np_rstlens']
            self.np_keyphrase_lens = dict_cached_order['np_keyphrase_lens']

        else:
            # len of text
            # self.np_textlens = self.data.txt_preproc.str.len().to_numpy() #.argsort()

            self.np_textlens = np.stack(
                [self.tokenizer.encode(ujson.loads(txt), return_tensors='np', add_special_tokens=False,
                                    truncation=False, padding='do_not_pad').size for txt in self.data.txt_preproc.values.tolist()]
            )
            # len of rst
            self.np_rstlens = np.array(
                [1 + len(json.loads(rst)) for rst in self.data.rst.values.tolist()])

            # len of keyphrase
            li_li_pos_kp = [json.loads(
                li_pos_kp) for li_pos_kp in self.data.li_pos_kp.values.tolist()]

            li_li_kp = [[kp for pos, kp in li_pos_kp]
                        for li_pos_kp in li_li_pos_kp]

            li_kp = [''.join(['<|kp|> ' + kp for kp in li_kp])
                     for li_kp in li_li_kp]

            self.np_keyphrase_lens = np.array([self.tokenizer.encode(kp,
                                                                     add_special_tokens=False,
                                                                     truncation=False,
                                                                     padding='do_not_pad',
                                                                     return_tensors=None).__len__() for kp in li_kp])

            dict_cached_order = {'np_textlens': self.np_textlens,
                                'np_rstlens': self.np_rstlens,
                                'np_keyphrase_lens': self.np_keyphrase_lens}

            pickle.dump(dict_cached_order, open(fp_cached_order, "wb"))

        # Initialise rst_lens to actuallengths
        self.rst_len = self.np_rstlens
        
        self.data = self.data.to_dict('records')

    def __len__(self):
        #return (self.line_end - self.line_start)
        return len( self.data )

    def __getitem__(self, index):

        rst_rels, rst_ns, rst_pos, li_kp, li_kprstpos, utterance, dict_pos_edu = self.getitem_extract_datum(
            index)

        if self.inference == True:

            utterance_prompt = ' '.join(utterance.split(' ')[:2])

            encoded = self.tokenizer.encode_input(rst_rel=rst_rels, rst_ns=rst_ns, rst_pos=rst_pos,
                                                  li_kp=li_kp,
                                                  li_kprstpos=li_kprstpos,
                                                  utterance_prompt=utterance_prompt,
                                                  max_len_rst=self.rst_len[index],
                                                  dict_pos_edu=dict_pos_edu)

            encoded['orig_rst_rels'] = rst_rels
            encoded['orig_rst_ns'] = rst_ns
            encoded['orig_rst_pos'] = rst_pos

            encoded['orig_utt'] = utterance
            encoded['orig_key_phrase'] = li_kp

            encoded['orig_dict_pos_edu'] = dict_pos_edu
            encoded['orig_li_kprstpos'] = li_kprstpos

        elif self.inference==False:

            encoded = self.tokenizer.encode_input(
                rst_rels, rst_ns, rst_pos,
                li_kp=li_kp,
                li_kprstpos=li_kprstpos,
                utterance=utterance,
                dict_pos_edu=dict_pos_edu,
                max_len_rst=self.rst_len[index],
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
        sorted_order = [i[0] for i in sorted(enumerate(rst_pos), key=lambda x: (
            RSTTokenizer.edukp_pos_sort_function(x[1]), x[1]),)]
        rst_rels = [rst_rels[idx] for idx in sorted_order]
        rst_ns = [rst_ns[idx] for idx in sorted_order]
        rst_pos = [rst_pos[idx] for idx in sorted_order]
        # endregion

        # Key phrase scores
        li_pos_kp = json.loads(datum['li_pos_kp'] )
        if len(li_pos_kp)>0:
            # top 3 important prhases from utterance
            li_pos_kp = sorted( li_pos_kp, key=lambda pos_kp: RSTTokenizer.edukp_pos_sort_function(int(pos_kp[0])) )
            li_kprstpos, li_kp = zip(*li_pos_kp)
            li_kprstpos = tuple(int(pos) for pos in li_kprstpos)
        else:
            li_kp = []
            li_kprstpos = []

        # Utterance
        utterance = ujson.loads(datum['txt_preproc'])

        #pos and edus
        dict_pos_edu = json.loads(datum['dict_pos_edu'])        
   
        return rst_rels, rst_ns, rst_pos, li_kp, li_kprstpos, utterance, dict_pos_edu

class SizedOrdered_Sampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.
    #TODO; add this to pytorch. Sampler to sort nlp datasets by size
    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, batch_size, shuffle, batching_style='effecient' ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size

                #v1
        if batching_style=='standard':
            self.li_chunked_ordered_lens = list(range(len(self.data_source)))
            random.shuffle(self.li_chunked_ordered_lens)
        
        else:
            np_txt_lens = np.concatenate(
                [ds.np_textlens for ds in self.data_source.datasets]).flatten()
            np_rst_lens = np.concatenate(
                [ds.np_rstlens for ds in self.data_source.datasets]).flatten()
            np_key_phrase_lens = np.concatenate(
                [ds.np_keyphrase_lens for ds in self.data_source.datasets]).flatten()

            # Indices are sorted in order of 1.tokenized txt length, key_phrase_length then rst length
            np_ordered_lens = np.lexsort(
                (np_rst_lens, np_key_phrase_lens, np_txt_lens))
            # We Randomly re-arrange them in batches of batch size
        

            li_chunked_lens = [np_ordered_lens[idx:idx+batch_size]
                                for idx in range(0, np_ordered_lens.size - batch_size, batch_size)]

            if shuffle:
                random.shuffle(li_chunked_lens)

            # Getting max sizes for rst in each chunk
            self.li_chunk_rst_len = [
                np.take(np_rst_lens, idxs).max() for idxs in li_chunked_lens]

            self.li_chunked_ordered_lens = np.concatenate(li_chunked_lens).tolist()


            # iterating through chunk_idx, data_idxs enumerate(self.li_chunked):
            for chunk_idx, data_idxs in enumerate(li_chunked_lens):
                rst_len = self.li_chunk_rst_len[chunk_idx]

                for data_idx in data_idxs:
                    dataset_idx = bisect.bisect_right(
                        self.data_source.cumulative_sizes, data_idx)

                    if dataset_idx == 0:
                        sample_idx = data_idx
                    else:
                        sample_idx = data_idx - \
                            self.data_source.cumulative_sizes[dataset_idx - 1]

                    self.data_source.datasets[dataset_idx].rst_len[sample_idx] = rst_len

    def __iter__(self):
        return iter(self.li_chunked_ordered_lens)

    def __len__(self) -> int:
        if self.batch_size != -1:
            return len(self.data_source)
        else:
            return len( self.li_chunked_ordered_lens )

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
            >>> for epoch in rDange(start_epoch, n_epochs):
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

        self.batch_size = batch_size

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            #num_replicas = dist.get_world_size()
            num_replicas = gpus
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
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

        # self.num_samples
        #self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

        # new code
        #self.dataset = dataset
        self.data_source = dataset
        np_txt_lens = np.concatenate(
            [ds.np_textlens for ds in self.data_source.datasets]).flatten()
        np_rst_lens = np.concatenate(
            [ds.np_rstlens for ds in self.data_source.datasets]).flatten()
        np_key_phrase_lens = np.concatenate(
            [ds.np_keyphrase_lens for ds in self.data_source.datasets]).flatten()

        # Indices are sorted in order of the text lens of records in the datasets
        np_ordered_lens = np.lexsort(
            (np_rst_lens, np_key_phrase_lens, np_txt_lens))

        # We Randomly re-arrange them in batches of batch size
        li_chunked_lens = [np_ordered_lens[idx:idx+batch_size]
                           for idx in range(0, np_ordered_lens.size-batch_size, batch_size)]

        # Divide into n sublists,
        # Each sublist at index i, contains the indices for process at rank i
        # Each sublist at index i, is a list non flatten indices. Each index represents items in the dataset

        li_li_chunked_lens = [
            [li_chunked_lens[(self.num_replicas*idx)+_rank]
             for idx in range(len(li_chunked_lens)//self.num_replicas)]
                for _rank in range(self.num_replicas)]

        # shuffle each processes subllist in the same order to optimize paralel training
        _ = list(zip(*li_li_chunked_lens))

        if shuffle:
            random.shuffle(_)
        
        # unpacking into worker size length list 
        li_li_chunked_lens = list(zip(*_))

        # Getting max sizes for rst and key_phrase in each chunk
        self.li_li_chunk_rst_len = [ [np.take(np_rst_lens, idxs).max() for idxs in li_chunked_lens ] 
                                            for li_chunked_lens in li_li_chunked_lens ]
        
        self.li_li_chunked_ordered_lens = [ np.concatenate(li_chunked_lens).tolist() for li_chunked_lens in li_li_chunked_lens ]
        
        for (li_chunked_lens, li_chunk_rst_len) in zip( li_li_chunked_lens, self.li_li_chunk_rst_len ) :
            # iterating through chunk_idx, data_idxs enumerate(self.li_chunked):
            
            for chunk_idx, data_idxs in enumerate(li_chunked_lens):
                rst_len = li_chunk_rst_len[chunk_idx]

                for data_idx in data_idxs:
                    dataset_idx = bisect.bisect_right(
                        self.data_source.cumulative_sizes, data_idx)

                    if dataset_idx == 0:
                        sample_idx = data_idx
                    else:
                        sample_idx = data_idx - \
                            self.data_source.cumulative_sizes[dataset_idx - 1]

                    self.data_source.datasets[dataset_idx].rst_len[sample_idx] = rst_len
                
    def __iter__(self) -> Iterator[T_co]:

        return iter(self.li_li_chunked_ordered_lens[self.rank])

    def __len__(self) -> int:
        if self.batch_size != -1:
            return len(self.data_source)
        else:
            return len( self.li_li_chunked_ordered_lens[0] )

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
        save_dir=os.path.abspath(tparams['model_dir']),
        name=mparams['model_name'],
        version=tparams['version'])

    tparams['version'] = tb_logger.version

    tparams['dir_checkpoints'] = os.path.join(
        tparams['model_dir'], mparams['model_name'], f"version_{tparams['version']}", 'checkpoints')

    os.makedirs(tparams['dir_checkpoints'], exist_ok=True)

    # initiating training loop
    training_module = RSTBart_TrainingModule.instatiate_training_module(
        tparams, mparams)
    trainer, training_module = RSTBart_TrainingModule.instatiate_trainer(
        tparams, tb_logger, training_module)
    RSTBart_TrainingModule.start(trainer, tparams, training_module, mparams)

if __name__ == '__main__':

    parent_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)

    # add model specific args
    mparams = RSTBart.parse_model_specific_args(parent_parser)

    # add the trainer specific args
    tparams = RSTBart_TrainingModule.parse_train_specific_args(parent_parser)

    if tparams.mode == "test":
        assert tparams.gpus in [0, 1]

    if tparams.gpus not in [0, 1]:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '65502'

    try:
        main(vars(tparams), vars(mparams))
    except Exception:
        print(traceback.format_exc())

# CUDA_VISIBLE_DEVICES=1 python3 train_RSTBart.py --batch_size 32 --version 1  --precision 16 --mode train_new --workers 6 --rst_tree_aligned_attention 0 --scale_grad_by_freq 1 --max_epochs 12 --gpus 1 --max_len_utt 190 --max_len_rst 28 --max_len_key_phrase 40 --tag "RSTBart with normal attention"
# CUDA_VISIBLE_DEVICES=1 python3 train_RSTBart.py --batch_size 32 --version 1  --precision 16 --mode train_new --workers 6 --rst_tree_aligned_attention 1 --scale_grad_by_freq 1 --max_epochs 12 --gpus 1 --max_len_utt 190 --max_len_rst 28 --max_len_key_phrase 40 --tag "RSTBart with normal attention"
