# Extracted from https://github.com/allenai/comet-atomic-2020/blob/master/models/comet_atomic2020_bart/utils.py
  
import itertools
import json
import linecache
from multiprocessing import Value
import os
import pickle
import warnings
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List
import nltk

import sacrebleu
import bert_score

dirname = os.path.dirname(__file__)
from copy import deepcopy
from argparse import Namespace
import numpy as np
import torch
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu, metrics
from torch import nn
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
#from transformers.models.bart import BaseModelOutput
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.generation_utils import *
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_convert
from torch._six import string_classes

from typing import Optional, Callable, Union, Optional, List, Iterable
import collections

import random 
import regex as re
np_str_obj_array_pattern = re.compile(r'[SaUO]')


huggingface_names = {'bart_base': "facebook/bart-base"}

def get_path(_path,_dir=False):

    if os.path.isabs(_path) == False:
        _path = os.path.join(dirname, _path)
    
    _path = os.path.realpath(_path)
    
    if _dir:
        os.makedirs(_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(_path), exist_ok=True)

    return _path

def load_pretrained_transformer( model_name='bart', transformer=True, 
                                    tokenizer=False):
    _dir_transformer = os.path.join( get_path("./models"), model_name )
    exists = os.path.isdir(_dir_transformer)
    output = {}
    
    if exists == False:    
        model_tokenizer = AutoTokenizer.from_pretrained(huggingface_names[model_name])
        model = BartForConditionalGeneration.from_pretrained(huggingface_names[model_name] )
        
        model_tokenizer.save_pretrained(_dir_transformer)
        model.save_pretrained(_dir_transformer)

    if tokenizer == True:
        output['tokenizer'] = AutoTokenizer.from_pretrained(_dir_transformer)

    if transformer == True:
        output['transformer'] = BartForConditionalGeneration.from_pretrained(_dir_transformer)
    
    return output

def load_base_tokenizer( model_name,
                            dir_tokenizer,
                            base_tokenizer_name,
                            output_version="phrase"
                            ):
    
    if os.path.isdir(dir_tokenizer):
        base_tokenizer = AutoTokenizer.from_pretrained(dir_tokenizer, use_fast=False)

    # retreiving base tokenizer from online or from local distillgpt2
    else:
        dir_transformer = os.path.join("./models", base_tokenizer_name)
        exists = os.path.isdir(dir_transformer)            

        if exists==True:
            base_tokenizer = AutoTokenizer.from_pretrained(dir_transformer, use_fast=False)
            config = AutoConfig.from_pretrained(dir_transformer)

        elif exists==False:
            base_tokenizer = AutoTokenizer.from_pretrained(huggingface_names['base_tokenizer_name'],use_fast=False)
            config = AutoConfig.from_pretrained(huggingface_names['base_tokenizer_name'])

            os.makedirs(dir_tokenizer)
                
            base_tokenizer.init_kwargs['name_or_path'] = dir_tokenizer
                #base_tokenizer.init_kwargs['special_tokens_map_file'] = os.path.join(dir_tokenizer,"special_tokens_map.json")
                
            base_tokenizer.save_pretrained(dir_tokenizer)
            config.save_pretrained(dir_tokenizer)
    
    return base_tokenizer

def BART_encoder_forward(
    self,
    input_ids=None,
    attention_mask=None,
    head_mask=None,
    inputs_embeds=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **kwargs
    ):
    r"""
    MONKEY_PATCHED: Allowed for position_ids and position_embeds to be passed in. Before the position ids were auto generated

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
        head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the heas is **masked**.

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
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

    # HACK
    #embed_pos = self.embed_positions(input_shape)
                                # HACK
    hidden_states = inputs_embeds #+ embed_pos
    hidden_states = self.layernorm_embedding(hidden_states)
    hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

    # expand attention_mask
    if attention_mask is not None:
        
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask_2(attention_mask, inputs_embeds.dtype)
        
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
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
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

def BART_forward(
    #region All the same
    self,
    input_ids=None,
    attention_mask=None,
    decoder_input_ids=None,
    decoder_attention_mask=None,
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
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )
    #endregion
    # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
    # NOTE : change here
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_outputs[0],
                #NOTE: Akanni Hack. 1st adapt mask for decoder, then expand it for your code
            #    we feed a non generic 3 dim (bs, tgt_len, src_len ) attn_mask to encoder.
            #   IN source code The decoder uses the 2dim encoded_attn_mask to decide where to attnd
            #   Our 3dim mask is not interoperable with this system.
            #   So we reduce our 3dim mask to a 2dim mask (bs, src_len) after it has been used by the encoded
            #   This 2dim encoded_attn mask will be 1 for every non pad position and 0 otherwise
                # This is achieved by putting 1 for every src_len position that has at least one other position attending to it
                    # Since this indicates it is not a mask token
        encoder_attention_mask=torch.where( attention_mask.sum(dim=-1)==0,0,1),
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

def transform_patch( self, val ):
    encoded = self.transform(val)
    encoded = encoded + self.starting_idx
    return encoded

def inverse_transform_patch(self, val):
    if type(val) in [ np.array, torch.Tensor]:
        val = val.tolist()
    
    val = [v - self.starting_idx for v in val ]    
    val = [v for v in val if v< len(self.classes_) ]
    decoded = self.inverse_transform( val )
    
    return decoded

#region Mixin Classes to override
def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        
        tti  = kwargs['tail_treepos_ids']

        decoder_inputs_embeds = self.model.shared( decoder_input_ids ) + \
            self.comerst().embedding_rst_pos( tti.repeat(1, decoder_input_ids.shape[-1] ) ) 


        decoder_inputs_embeds = decoder_inputs_embeds * self.model.decoder.embed_scale

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            #"decoder_input_ids": decoder_input_ids,
            "decoder_inputs_embeds":decoder_inputs_embeds,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

def greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = None,
    **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
    r"""
    Generates sequences for models with a language modeling head using greedy decoding.
    Parameters:...
    Return:...
    Examples::...
    """
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
    stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    cur_len = input_ids.shape[-1]

    this_peer_finished = False  # used by synced_gpus only
    while True:

        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            )
        

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                )

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        
        # NOTE: new input_id willl have to be put through custom embedding layer
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        cur_len = cur_len + 1

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids
#endregion 

# region Used during training
def default_collate_pad(batch, pad_values=None):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)


    if isinstance(elem, torch.Tensor):
                   
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)

        return torch.stack(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate_pad([torch.as_tensor(b) for b in batch], pad_values)

        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        dict_output = {}
        for key in elem:
            li_ = [d[key] for d in batch if d[key]!=None]

            #it = iter(batch)
            if len(li_)>0:
                elem_size = len(li_[0])
            else:
                elem_size = 0

            if not all(len(elem_) == elem_size for elem_ in li_):
                # raise RuntimeError('each element in list of batch should be of equal size')
                # it = iter(batch)
                largest_seq = max( len(elem_) for elem_ in li_ ) 
                
                
                if li_[0].dim() == 1:
                    padded_li = pad_sequence(li_, batch_first=True, padding_value=pad_values[key] ) 
                    #unstacking
                    li_ = torch.unbind(padded_li, 0)
                
                    #handling 2d attention mask
                elif li_[0].dim() == 2:
                    
                    for idx in range(len(li_)):
                        elem_ = li_[idx]
                        missing_dims = largest_seq - len(elem_)
                        if missing_dims > 0:
                            # adding missing_dims paddings to dim 1 which reflects masking the new padding tokens
                            # adding paddings value 0 - to dim 0 which reflects the 

                            elem_ = torch.nn.functional.pad( elem_, (0, missing_dims, 0, missing_dims), 
                                mode='constant', value=0.0 )
                            li_[idx] = elem_
                            


            dict_output[key] = default_collate_pad( li_, pad_values )    
        return dict_output

    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate_pad(samples,pad_values) for samples in zip(*batch)))

    raise TypeError(default_collate_err_msg_format.format(elem_type))
             
def get_path(_path,_dir=False):

    if os.path.isabs(_path) == False:
        _path = os.path.join(dirname, _path)
    
    _path = os.path.realpath(_path)
    
    if _dir:
        os.makedirs(_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(_path), exist_ok=True)

    return _path

def _expand_mask_2(mask, dtype, tgt_len = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, tgt_len, src_len = mask.size()

    expanded_mask = mask[:, None, :, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

def monkey_save_model(self, trainer, filepath: str):
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

def freeze_params(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False
# endregion

def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)

def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]

# region  COMERST Scores
def calculate_bleu_score(output_lns, refs_lns, order=4, **kwargs) -> float:
    
    """Uses sacrebleu's corpus_bleu implementation."""
    args = Namespace(
        smooth_method=kwargs.get('smooth_method', 'exp'), 
        smooth_value=None, 
        force=False,
        short=False,
        lc=kwargs.get('lc',True),
        tokenize=kwargs.get('tokenize', sacrebleu.tokenizers.DEFAULT_TOKENIZER ) )
    
    #Hack - changing the default order
    metrics.BLEU.NGRAM_ORDER = order
    sacrebleu.metrics.BLEU.extract_ngrams.__defaults__ = (1, order)
    
    metric = metrics.BLEU(args)
        
    li_scores = []

    for pred, ref in zip(output_lns, refs_lns):
        
        score = metric.corpus_score(
            pred, ref, 
            use_effective_order=kwargs.get('use_effective_order',True) ).score
        
        li_scores.append(score)

    avg_score = sum(li_scores)/len(li_scores)

    #returning to normal
    metrics.BLEU.NGRAM_ORDER = 4
    sacrebleu.metrics.BLEU.extract_ngrams.__defaults__ = (1, 4)
    
    return round( avg_score, 3) 

ROUGE_KEYS = ["rougeL"]
def calculate_rouge(output_lns: List[str], reference_lns: List[str], use_stemmer=True) -> Dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: round( v.mid.fmeasure, 3) for k, v in result.items()}

def calculate_meteor(output_lns: List[str], reference_lns: List[str], use_stemmer=True) -> float:
    
    li_scores =  []
    
    for pred, ref in zip(output_lns, reference_lns):
        score = nltk.translate.meteor_score.meteor_score([ref], pred) 
        li_scores.append(score)
    
    score = sum(li_scores)/len(li_scores)

    return round(score, 3)

def calculate_bertscore(output_lns: List[str], reference_lns: List[str],
                            model_type="roberta-large-mnli",
                            device="cuda:0",
                            pre_existing_bscore=None,
                            delete_ater_gen=True,
                            return_bscore=False,
                            batch_size=2 ) -> Dict:
    
    assert not (delete_ater_gen and return_bscore)

    if pre_existing_bscore != None:
        bscore_model = pre_existing_bscore
    else:
        bscore_model = bert_score.BERTScorer(
            model_type = model_type,
            device = device )

    prec, rec, f1 = bscore_model.score(output_lns, reference_lns,
                        verbose=False, batch_size=batch_size,
                        return_hash=False )
    
    prec = torch.mean( prec ).item()
    rec = torch.mean( rec ).item()
    f1 = torch.mean( f1 ).item()

    output = ( prec, rec, f1,  )

    if delete_ater_gen:
        bscore_model._model.to('cpu')
        del bscore_model._model
        del bscore_model
    
    if return_bscore:
        output = ( output,  tuple([bscore_model]) )
    
    return output

# endregion    


#region Dataprocessing
comet_tail_to_ignore1 = [ "none", "NONE" ,np.nan, '?',
        'l', '3', 't', 'Y', 'X', 'g', '0', 'F', 'q', 'v', '`', 'h', 's', 'a', 'n', 'c','1',
        'd', 'e', 'i', 'ok', 'no', 'xx', 'NO','na', 'aF', 'N/', 'to', 'sd', 'up', 'it',
        'Hi', 'tv', 'Na', 'me', 'be',
        'iv', 'cd', 'co', 'st', 'us', 'or', '4h',
        'oz', 'fl', 'in', 'rv','uk', 'do', 'mb', 'li', 'ai', 'g4', 'vd',
        'go', 'ex', 'c9', '21', 'el', '2h', 'ox', 'on',
        'q\\', 'ge', 'ru', 'th', 'TV', 'ID', 'Id', 'HR', 'sw', 'CD', 'ii']

comet_tail_to_ignore2 = [np.nan,
        'q', '`',
        '?', 'v', 'to', 'ok', 'NO', 'na', 'no',
        'go', 'tv', 'TV', 'do', 'ar', 're', 'it', 'PC', 'me']

comet_tail_to_ignore3 = [np.nan,
        'o', 'B', 'a', 'u',
        'r', 'Y', 'q', 'ok', 'na', 'NO', 'no', 'aC', 'to', 'in',
        'st', 'up', 'do', 'go', 'be', 'un', 'tv', 'TV', 'ID', 'ox', 'CD']

comet_tail_to_ignore = list( set( comet_tail_to_ignore1 + comet_tail_to_ignore2 + comet_tail_to_ignore3 ) )
#endregion