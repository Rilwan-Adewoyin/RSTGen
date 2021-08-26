from itertools import compress
import os
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, GPT2LMHeadModel
dirname = os.path.dirname(__file__)
from datetime import date

import regex as re
import torch
from torch import nn
from typing import Optional, Callable, Union, Optional, List, Iterable, Tuple
import numpy as np
import copy
from transformers.generation_logits_process import (
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from functools import lru_cache, reduce

import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_convert
from torch._six import string_classes
import collections

import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate_err_msg_format
from transformers.generation_beam_search import BeamHypotheses

#region loading and saving
def get_path(_path,_dir=False):

    if os.path.isabs(_path) == False:
        _path = os.path.join(dirname, _path)
    
    _path = os.path.realpath(_path)
    
    if _dir:
        os.makedirs(_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(_path), exist_ok=True)

    return _path

def load_pretrained_transformer( model_name='bert-base-cased', transformer=True, 
                                    tokenizer=False):
    _dir_transformer = os.path.join( get_path("./models"), model_name )
    exists = os.path.isdir(_dir_transformer)
    output = {}

    if exists == False:    
        model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        #model = AutoModel.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        model_tokenizer.save_pretrained(_dir_transformer)
        model.save_pretrained(_dir_transformer)

    if tokenizer == True:
        output['tokenizer'] = AutoTokenizer.from_pretrained(_dir_transformer)

    if transformer == True:
        output['transformer'] = GPT2LMHeadModel.from_pretrained(_dir_transformer)
    
    return output

def load_pretrained_tokenizer_local( model_name='NLG'):

    _dir_tknzr = os.path.join( get_path("./models",_dir=True), model_name )
    exists = os.path.isdir(_dir_tknzr)

    if exists == True:    
        if model_name == "NLG":
            tknzr =  AutoTokenizer.from_pretrained(_dir_tknzr)
            res = tknzr
        else:
            raise ValueError()
    else:
        res = False
    
    return res


def save_version_params(t_params, m_params, version_code="DaNet_v000"):
    dated_trained = date.today().strftime("%d-%m-%Y")
    
    t_params.date_trained = dated_trained
    m_params.date_trained = dated_trained

    _dir_version = get_path(f"./models/{version_code}/",_dir=True)
    
    tp_fp = os.path.join(_dir_version,'tparam.json')
    mp_fp = os.path.join(_dir_version,'mparam.json')

    json.dump( vars(t_params), open(tp_fp,"w") )
    json.dump( vars(m_params), open(mp_fp,"w") )

    return True    

#endregion 

#region Monkey Patches the save module
def monkey_save_model(self, trainer, filepath: str):
    #TODO: suggest this change on github pytorch lightning 
    # in debugging, track when we save checkpoints
    trainer.dev_debugger.track_checkpointing_history(filepath)

    # make paths
    if trainer.is_global_zero:
        self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)

    # delegate the saving to the trainer
    if self.save_function is not None:
        self.save_function(filepath, self.save_weights_only)
    
    self.to_yaml()
#endregion

#region RST helper


tree_order = {
    ():0,
    
    (0,):1,(1,):2,
    
    (0,0):3,(0,1):4,(1,0):5,(1,1):6,
    
    (0,0,0):7, (0,0,1):8,(0,1,0):9, (0,1,1):10,
    (1,0,0):11, (1,0,1):12, (1,1,0):13, (1,1,1):14,
    
    (0,0,0,0):15, (0,0,0,1):16, (0,0,1,0):17, (0,0,1,1):18,
    (0,1,0,0):19, (0,1,0,1):20, (0,1,1,0):21, (0,1,1,1):22,
    (1,0,0,0):23, (1,0,0,1):24, (1,0,1,0):25, (1,0,1,1):26,
    (1,1,0,0):27, (1,1,0,1):28, (1,1,1,0):29, (1,1,1,1):30,
    
    }

# function which returns position in tree based on the binary tuple indicating left,rights down a tree
def tree_order_func(tuple_pos):
    
    pos = 0
    for binary_left_right in tuple_pos:
        pos = 2*pos + 2**binary_left_right
    
    return pos


rst_rel_li = ['Attribution',
                'Background','Cause','Comparison','Condition',
                'Contrast','Elaboration','Enablement','Evaluation',
                'Explanation','Joint','Manner-Means','Topic-Comment',
                'Summary','Temporal','Topic-Change','same-unit','textual-organization'] #Add this to savable config
#endregion

#region RST info Encoding

MAX_LONG_VALUE = torch.iinfo(torch.long).max

class RstTokenizerMixin():
  
    
    @staticmethod
    @lru_cache()
    def edukp_pos_sort_function(edukp_pos: int):
        # We use a sorting function to know tree leftright order of edukp_pos
            # sort_function
            # from root_pos find the sequence of left/rights down the tree to each edukp_pos
            # Then use the 1/2, 1/4 method to calculate edukpos float representtion on flat line
            # Then retun this float
            # NOTE: intuition -> imageine binary tree is collapsed to a flatline. root=0 , left/right from parent= +/- 0.5^n

        li_leftright_seq = RstTokenizerMixin.left_right_seq_from_root_to_edu_pos(edukp_pos) 
        
        # Now calculate the flattened position using the sequence of left and rights
        _ = {'L':-1, 'R':+1}
        li_flattened_pos_contributions = [  _[direction]*(0.5**(idx+1)) for idx,direction in enumerate(li_leftright_seq)  ]
        flattened_pos = sum(li_flattened_pos_contributions)

        return flattened_pos

    @staticmethod
    @lru_cache()
    def node_level(node_pos):
        val = math.floor( math.log( node_pos+1 , 2 ) )
        
        return val

    @staticmethod
    @lru_cache()
    def left_right_seq_from_root_to_edu_pos( edukp_pos: int):
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

    def clamp_values(self, x, max):

        #clamps values in a tree method where the parent tree nodes is the evel
            # to reduce to
        # we use this since the rst positions in our tree are often too large 
        # for torch.long to handle
        while x.max() >= max:
            x = np.where( x<max, x, np.floor_divide(x-1,2) )                    
        return x.astype( int )

class EmbeddingRstPos(nn.Module, RstTokenizerMixin):
    def __init__(self, max_rst_index=62, max_rst_level=8, rst_encoding_ndim=768,init_val=0.5
                    ):
        super(EmbeddingRstPos, self).__init__()

        self.max_rst_index = max_rst_index
        self.max_rst_level = max_rst_level
        self.left_right_seq_from_root_to_edu_pos = EmbeddingRstPos.left_right_seq_from_root_to_edu_pos

        self.init_val = init_val
        self.fixed_rst_encoding = self.make_rst_encoding( )
        self.ffd = torch.nn.Linear(self.max_rst_level, rst_encoding_ndim, bias=False )
        
        self.padding_idx = self.fixed_rst_encoding.padding_idx
        
        
    def forward(self, x ):
        while x.max() >= self.max_rst_index:
            x = torch.where( x>=self.max_rst_index, torch.ceil( (x-2)/2 ).long(), x )
   

        x = self.fixed_rst_encoding(x)
        x = self.ffd( x )
        return x
    
    def make_rst_encoding(self):
        
        embedding_weight = torch.zeros( 
                                (self.max_rst_index, self.max_rst_level ),
                                dtype = torch.float )
        
        # zero index embedding vector
        zero_embedding = np.zeros( [self.max_rst_level] )

        split_dir_numb = {'L':-self.init_val, 'R':self.init_val}
        
        # for each embedding
        for idx in range(self.max_rst_index):
            
            idx_embedding = copy.deepcopy( zero_embedding )
            
            # Determine the sequence of lefts and rights to reach node    
            left_rights_from_root_to_pos = EmbeddingRstPos.left_right_seq_from_root_to_edu_pos( idx )
            
            # Convert sequence of LRs to a sequence of -1 and 1s and 0s
            for idx1, val in enumerate(left_rights_from_root_to_pos):
                idx_embedding[idx1] = split_dir_numb[val]

            # set this as the new embedding
            embedding_weight[idx] = torch.FloatTensor( idx_embedding )

        fixed_rst_encoding = torch.nn.Embedding.from_pretrained( embedding_weight ,
                                    freeze=True, padding_idx=self.max_rst_index-1 )

        return fixed_rst_encoding

#endregion

#region dataloading

class EffeciencyMixin():
    
    def compress_padding( self,
        li_input_ids, li_pad_token_ids, input_embeds, *args):
        """ First for each datum remove all padding due to the head parts
            Then use pad sequence to ensure they are all the same elnght"""
        
        """Remove columns that are populated exclusively by pad_token_id"""
        
        li_keep_column_mask = [ input_ids.ne(pad_token_ids) for input_ids, pad_token_ids in zip(li_input_ids, li_pad_token_ids) ]
        
        keep_column_mask = torch.cat(li_keep_column_mask, axis=-1)

        input_embeds = self.compress_padding_inner(input_embeds, 1, keep_column_mask)

        res = tuple()
        for tens, compress_dim in args:
            compressed_tens = self.compress_padding_inner(tens, compress_dim, keep_column_mask)  
            res = res + (compressed_tens, )
        
        return (input_embeds, ) + res 

    def compress_padding_inner( self, tensor_, compress_dims, keep_column_mask ):
        li_subtens = tensor_.unbind(dim=0)
        
        if compress_dims == 1:
            li_subtens = [ subtens[keep_column_mask[idx2]] for idx2, subtens in enumerate(li_subtens) ]
            batched_padded_subtens = pad_sequence(li_subtens, batch_first=True, padding_value=0.0) #this tensor only hass padingg at the end
        
        elif compress_dims == 2:
            max_len = keep_column_mask.sum(axis=1).max()
            li_subtens = [ subtens[ keep_column_mask[idx2], :][: , keep_column_mask[idx2] ] 
                for idx2, subtens in enumerate(li_subtens) ]
            li_padded_subtens = [ torch.nn.functional.pad( tens, (0, max_len-tens.shape[0] , 0, max_len-tens.shape[1]), value=0.0 ) 
                                for tens in li_subtens]
            batched_padded_subtens = torch.stack(li_padded_subtens)

        return batched_padded_subtens

    def default_collate_pad(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size


        """

        pad_values = self.pad_values
        pad_maxlens = self.pad_maxlens
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
                        
                        if largest_seq>pad_maxlens.get(key):
                            for idx in range(len(li_)):
                                if len(li_[idx])>pad_maxlens.get(key):
                                    li_[idx] = li_[idx][:pad_maxlens.get(key)]
                        
                        padded_li = pad_sequence(li_, batch_first=True, padding_value=pad_values.get(key,0) ) 
                        #unstacking
                        li_ = torch.unbind(padded_li, 0)
                    
                        #handling 2d attention mask
                    elif li_[0].dim() == 2:
                        
                        for idx in range(len(li_)):
                            elem_ = li_[idx]
                            
                            missing_dims = min( largest_seq, pad_maxlens.get(key) )  - len(elem_)

                            if missing_dims > 0:
                                # adding missing_dims paddings to dim 1 which reflects masking the new padding tokens
                                # adding paddings value 0 - to dim 0 which reflects the 

                                elem_ = torch.nn.functional.pad( elem_, (0, missing_dims, 0, missing_dims), 
                                    mode='constant', value=0.0 )
                                                    
                            elif missing_dims < 0:
                                elem_ = elem_[ :missing_dims, :missing_dims ]
                            
                            li_[idx] = elem_

                dict_output[key] = self.default_collate_pad( li_ )    
            return dict_output

        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate_pad(samples,pad_values) for samples in zip(*batch)))
        
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.default_collate_pad(samples) for samples in transposed]
        
        raise TypeError(default_collate_err_msg_format.format(elem_type))

    
    

#endregion

#region Generation Mixins

    # region Huggingface v4.7.0
    # endregion

    #region Huggingface v4.2.0
class GenerationMixin42_gpt:
    "GenerationMixin following transformers v4.2 generation methodology"
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

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                decoder_start_token_id is not None
            ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

            # get encoder and store encoder outputs
            encoder = self.get_encoder()

            encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)
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


# region overriden methods GPT-2 
def forward_gpt(
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
# endregion

#endregion