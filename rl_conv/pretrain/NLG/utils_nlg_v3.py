import os
import json
from transformers import AutoTokenizer, GPT2LMHeadModel
dirname = os.path.dirname(__file__)
from datetime import date

from itertools import (combinations, combinations_with_replacement, cycle,
                       islice, permutations)
import random
import regex as re
import torch
from pytorch_lightning.callbacks import Callback

from torch import nn
from typing import Optional, Callable, Union, Optional, List, Iterable, Tuple
import numpy as np
import copy

from functools import lru_cache
import nltk
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_convert
from torch._six import string_classes
import collections
from itertools import groupby

import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate_err_msg_format
from math import floor
import regex as re

pattern_punctuation_space = re.compile(r'\s([?.!";:#_](?:\s|$))')
pattern_capitalize_after_punct = re.compile(r"(\A\w)|"+                  # start of string
             "(?<!\.\w)([\.?!] )\w|"+     # after a ?/!/. and a space, 
                                          # but not after an acronym
             "\w(?:\.\w)|"+               # start/middle of acronym
             "(?<=\w\.)\w",               # end of acronym
             )
pattern_apostrophe = re.compile(r"\b\s+'\b")
pattern_brackets_rm_space = re.compile('\(\s*(.*?)\s*\)')

import pytextrank
import en_core_web_sm
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    #install using python -m spacy download en_core_web_sm
nlp = en_core_web_sm.load()

from spacy.language import Language

try:
    nlp.add_pipe("textrank", last=True)
    
except Exception as e:

    @Language.component("textrank")
    def textrank(doc):
        tr = pytextrank.TextRank()
        doc = tr.PipelineComponent(doc)
        return doc

    nlp.add_pipe("textrank", last=True)


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
    # if self.save_function is not None:
        # self.save_function(filepath, self.save_weights_only)
        trainer.save_checkpoint(filepath)
    
        self.to_yaml()


class SaveModelCallBack(Callback):

    def on_train_end(self, trainer, pl_module):

        # Saving Model using the pytorch method.
        # This allows relaoding using from_pretrained
        os.makedirs(
            f"./models_pt/{pl_module.model_name}/version_{trainer.logger.version}/", exist_ok=True)

        pl_module.model.save_pretrained(
            f"./models_pt/{pl_module.model_name}/version_{trainer.logger.version}/")
#endregion

def mpatch_save_model(func):

    def inner(self, *args):

        func(*args)

        self.to_yaml()

    return inner


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

#region RST Framework Class
class RstModelMixin():

    def get_curr_edu_pos(self, decoder_input_ids, edu_rstpos):
        """
        li_edu_rst_pos: the list of possible edus positions for each text in this batch
        """
        
        # Decode the output ids to get a text
        li_gen_text = [self.RSTTokenizer.decode(
            ids, skip_special_tokens=True) for ids in torch.unbind( decoder_input_ids,0) ]

        # run edu splitter on text generated so far
        # li_textwedutoken = parser_wrapper3.main( json_li_li_utterances= ujson.dumps([li_gen_text]),
        #                                     skip_parsing=True, redirect_output=True)

        li_textwedutoken = self.rst_parser.parse_li_utterances(li_gen_text)
        # calculating edu of current text
        li_edu_count = [
            ' '.join(li_words[:-1]).count('EDU_BREAK')+1 for li_words in li_textwedutoken]

        # removing any padding
        edu_rstpos = [tens if not (-1 in tens) else tens[: (tens == -1).nonzero(
            as_tuple=True)[0]] for tens in edu_rstpos][0]

        # selecting approapriate edu value.
        curr_edu_pos = [edu_rstpos[min(edu_rstpos.numel()-1, max(0, edu_count-1))]
                        for edu_count in li_edu_count]

        return curr_edu_pos 

#endregion

#region RST TOkenizer

MAX_LONG_VALUE = torch.iinfo(torch.long).max

class RstTokenizerMixin():
  
    
    @staticmethod
    @lru_cache(maxsize=4096)
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
    @lru_cache(maxsize=4096)
    def node_level(node_pos):
        val = math.floor( math.log( node_pos+1 , 2 ) )
        
        return val

    @staticmethod
    @lru_cache(maxsize=4096)
    def left_right_seq_from_root_to_edu_pos( edukp_pos: int):
        # from root_pos find the sequence of left/rights down the tree to each edukp_pos

        li_leftright_seq = [] #sequence of left-rights to get from the root to the edukp_pos

        while edukp_pos>0:
            parent_pos = RstTokenizerMixin.parent_node(edukp_pos) #(parent_pos-1 )/2
            # child node is left child node if (child_node_pos-1 /2)==int
            # child node is right child node if (child_node_pos-1 /2)=/int
            if edukp_pos%2==0:
                child_position_rel_to_parent = 'R'
            else:
                child_position_rel_to_parent = 'L'
            
            li_leftright_seq = [child_position_rel_to_parent] + li_leftright_seq

            parent_pos = math.floor(parent_pos)
            edukp_pos = parent_pos
        
        if edukp_pos<0:
            edukp_pos = 0
            
        return li_leftright_seq

    @staticmethod
    @lru_cache(maxsize=4096)
    def seq_from_root_to_edu_pos( edukp_pos: int):
        # from root_pos find the sequence of left/rights down the tree to each edukp_pos

        parent_pos = edukp_pos
        li_seq = [] #sequence of left-rights to get from the root to the edukp_pos

        while parent_pos>0:
            parent_pos = RstTokenizerMixin.parent_node(parent_pos) #(parent_pos-1 )/2
            # child node is left child node if (child_node_pos-1 /2)==int
            # child node is right child node if (child_node_pos-1 /2)=/int

            
            li_seq = [parent_pos] + li_seq

            parent_pos = math.floor(parent_pos)
        
        return li_seq

    @staticmethod
    @lru_cache(maxsize=4096)
    def parent_node(edukp_pos: int ):
        
        if edukp_pos > 0:
            parent_pos = (edukp_pos-1 )/2
        else:
            parent_pos = 0
                    
        parent_pos = math.floor(parent_pos)  
            
        return parent_pos

    def clamp_values(self, x, max):

        #clamps values in a tree method where the parent tree nodes is the evel
            # to reduce to
        # we use this since the rst positions in our tree are often too large 
        # for torch.long to handle

        while x.max() >= max:
            x = np.where( x<max, x, np.floor_divide(x-1,2) )     

        return x.astype( int )
    
    @lru_cache(maxsize=4096)
    def clamp_value( pos ):
        while pos >= MAX_LONG_VALUE:
            pos = RstTokenizerMixin.parent_node(pos)
        return pos


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



class EmbeddingRstPos(nn.Module, RstTokenizerMixin):
    def __init__(self, max_rst_index=62, max_rst_level=8, rst_encoding_ndim=768,
                    init_val=0.05, std=0.02):
        super(EmbeddingRstPos, self).__init__()

        self.max_rst_index = max_rst_index
        self.max_rst_level = max_rst_level
        self.left_right_seq_from_root_to_edu_pos = EmbeddingRstPos.left_right_seq_from_root_to_edu_pos

        self.init_val = init_val
        self.fixed_rst_encoding = self.make_rst_encoding( )
        self.ffd = torch.nn.Linear(self.max_rst_level, rst_encoding_ndim, bias=True )
        self.ffd.weight.data.normal_(mean=0.0, std=std)
        
        self.padding_idx = self.fixed_rst_encoding.padding_idx
        
        
    def forward(self, x ):
        if x.numel()==0:
            return x
            
        while x.max() >= self.max_rst_index:
            x = torch.where( x>=self.max_rst_index, torch.ceil( (x-2)/2 ).long(), x )
   

        x = self.fixed_rst_encoding(x)
        x = self.ffd( x )
        x = torch.nn.functional.gelu(x)
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

    def default_collate_pad(self, batch): #, pad_values, pad_maxlens):
        r"""Puts each data field into a tensor with outer dimension batch size
        """

        if len(batch)==1 :
            # case: inference
            elem = batch[0]
            elem.pop('labels',None)
            for key in elem:
                if "orig" not in key:
                    elem[key] = elem[key].unsqueeze(0)
            return elem
            
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
                if len(li_)<=1:
                    continue
                if len(li_)>1:
                    elem_size = len(li_[0])
                else:
                    elem_size = 0
                
                if key != 'decoder_cross_attention_mask':
                    bool_size_check = all(len(elem_) == elem_size for elem_ in li_) 
                else: 
                    bool_size_check = all( len(elem_) == elem_size for elem_ in li_) and all( elem_.shape[1] == li_[0].shape[1] for elem_ in li_) 
                
                if not bool_size_check :
                    # raise RuntimeError('each element in list of batch should be of equal size')
                    # it = iter(batch)
                    largest_seq = max( len(elem_) for elem_ in li_ )
                    
                    try:
                        
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
                                
                                if key != 'decoder_cross_attention_mask':

                                    missing_dims = min( largest_seq, pad_maxlens.get(key) )  - len(elem_)

                                    if missing_dims > 0:
                                        # adding missing_dims paddings to dim 1 which reflects masking the new padding tokens
                                        # adding paddings value 0 - to dim 0 which reflects the 

                                        elem_ = torch.nn.functional.pad( elem_, (0, missing_dims, 0, missing_dims), 
                                            mode='constant', value=0.0 )
                                                            
                                    elif missing_dims < 0:
                                        elem_ = elem_[ :missing_dims, :missing_dims ]
                                
                                else:
                                    largest_seq_2 = max( elem_.shape[1] for elem_ in li_ )
                                    
                                    missing_dims1 = min( largest_seq, pad_maxlens.get(key)[0] )  - len(elem_)
                                    missing_dims2 = min( largest_seq_2, pad_maxlens.get(key)[1] ) - elem_.shape[1]

                                    if missing_dims1 > 0:
                                        # adding missing_dims paddings to dim 1 which reflects masking the new padding tokens
                                        # adding paddings value 0 - to dim 0 which reflects the 

                                        elem_ = torch.nn.functional.pad( elem_, (0, 0, 0, missing_dims1), 
                                            mode='constant', value=pad_values.get(key) )
                                    
                                    elif missing_dims1 < 0:
                                        elem_ = elem_[ :missing_dims1, : ]

                                    if missing_dims2 > 0:
                                        # adding missing_dims paddings to dim 1 which reflects masking the new padding tokens
                                        # adding paddings value 0 - to dim 0 which reflects the 

                                        elem_ = torch.nn.functional.pad( elem_, (0, missing_dims2, 0, 0), 
                                            mode='constant', value=pad_values.get(key) )
                                    
                                    elif missing_dims2 < 0:
                                        elem_ = elem_[ :, :missing_dims2 ]

                                li_[idx] = elem_
                    
                    except Exception as e:
                        print(key)
                        raise e 
                    
                dict_output[key] = self.default_collate_pad( li_ )    
            return dict_output

        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate_pad(samples) for samples in zip(*batch)))
        
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.default_collate_pad(samples ) for samples in transposed]
        
        raise TypeError(default_collate_err_msg_format.format(elem_type))

#endregion

#region data making

def edu_fixer(li_textwedutoken, li_text):
        
    li_li_edus = [ list( split(text_wedutoken,"EDU_BREAK") )[:-1] for text_wedutoken in li_textwedutoken ]
    
    for li_edutext in li_li_edus:
        for idx2,elem in enumerate(li_edutext):
            elem.reverse() #reversing list of words in an edu
            it = enumerate(elem)
            edu_len = len(elem) 
            elem_new =  [next(it)[1]+str_ if ( idx!=edu_len-1 and (str_[0] == "'" or str_ in ["n't", ".", "?", "!", ",", "[", "]" ]) ) else str_ for idx,str_ in it]
            elem_new.reverse()

            li_edutext[idx2] = elem_new

    # for each utterance, merge list of words into one text
    li_li_edus = [ [ ' '.join( edus ) for edus in li_edus ] for li_edus in li_li_edus ]
    
    # Fixing:
        # random spaces in words due to splitting at apostrophes such as isn 't
        # random spaces due to splitting at forward slash
        # random spaces due to converting brakcests to - LRB - and - RRB - codes
    li_li_edus = [ [edutxt.replace(" n't", "n't").replace(" / ", "/").replace(" '", "'").replace("- LRB -", "(").replace("- RRB -", ")").replace("-LRB-", "(").replace("-RRB-", ")")
                     if edutxt not in origtext else edutxt for edutxt in li_edutext ] for li_edutext, origtext in zip( li_li_edus, li_text) ]
    
    #outer re.sub does that space inbetween brakcets/
    li_li_edus = [ [ re.sub('\[\s*(.*?)\s*\]', r'[\1]', re.sub( pattern_punctuation_space, r"'", edutxt)) for edutxt in li_edutext] for li_edutext in  li_li_edus ]
    for idx in range(len(li_li_edus)):
        li_edus = li_li_edus[idx]
        li_edus =  [ re.sub(pattern_brackets_rm_space, r'(\1)', edu_text) for edu_text in li_edus ]
        li_edus =  [ re.sub(pattern_punctuation_space, r'\1', edu_text) for edu_text in li_edus ]

    return li_li_edus

def split(sequence, sep):
    chunk = []
    for val in sequence:
        if val == sep:
            yield chunk
            chunk = []
        else:
            chunk.append(val)
    yield chunk

def non_parseable_remover(li_dict_rsttext):
    print("Start non parseable remover")
    if len(li_dict_rsttext) == 0:
        return li_dict_rsttext

    # check if the edus in dict_pos_edu and the normal utterance have a specific level of coverage
    # if they do then continue
    # if they don't then pop li_edus
    for idx in reversed(list( range(len(li_dict_rsttext)))):

        edu_text = re.sub(pattern_brackets_rm_space, r'(\1)', " ".join( li_dict_rsttext[idx]['li_edus'] ) )
        edu_text = re.sub( pattern_punctuation_space, r'\1', edu_text )
        len_edu_text = len(  edu_text.split(' ') )

        len_text = len( li_dict_rsttext[idx]['txt_preproc'].split(' ') )

        if len_edu_text<min( len_text*0.85, len_text-1) or len_edu_text>max( 1.15*len_text, 1+len_text):
            
            li_dict_rsttext.pop(idx)

            # #pop dict_pos_edu and li_edus
            # li_dict_rsttext[idx].pop('dict_pos_edu')
            # li_dict_rsttext[idx].pop('li_edus')
            # li_dict_rsttext[idx].pop('rst')
            # # raise NotImplementedError("need to replace the rst tree for these cases since the text was truncated") 


            #making dict_pos_edu and li_edus again
    
    print("End non parseable remover")
    return li_dict_rsttext

def position_edus(li_dict_rsttext):
    #Creates dict_pos_edu
    if len(li_dict_rsttext) == 0:
        return li_dict_rsttext
         
    for idx in range(len(li_dict_rsttext)):
        
        if 'dict_pos_edu' in li_dict_rsttext[idx]:
            continue            

        li_rst_pos = [ rst_node['pos'] for rst_node in li_dict_rsttext[idx]['rst'] ]
        
        li_child_pos =  sum( [ find_child_edus(pos, li_rst_pos ) for pos in li_rst_pos ], [] )

        # sorting the child_pos by their rst order
        li_child_pos = sorted( li_child_pos, key= lambda pos: RstTokenizerMixin.edukp_pos_sort_function(pos) )

        li_edus = li_dict_rsttext[idx].pop('li_edus')

        dict_pos_edu = { edu_pos:edu for edu_pos, edu in zip( li_child_pos, li_edus ) }
        
        li_dict_rsttext[idx]['dict_pos_edu'] = dict_pos_edu
    
    return li_dict_rsttext


def _parse_trees(li_strtree):
    
    #parses tree into an nltk object
    li_subtrees = []

    # Parsing a list of subtrees in the utterance tree li_strtree
    for idx, pt_str in enumerate(li_strtree):
        try:
            if pt_str in ['', None]: raise ValueError
            _ = nltk.tree.Tree.fromstring(pt_str, brackets="{}")
        except ValueError:
            _ = None
            pass
        li_subtrees.append(_)
    
    return li_subtrees


def _tree_to_rst_code(_tree):
    """Converst RST Tree to rst code used in NLG model

        Args:
            method (int, optional): [description]. Defaults to 1.
        
        Return:
            if method==0:
                Three lists zipped together
                List 1 Represents A Flattened version of the rst relations in the RST tree
                List 2 the nuclearity/satellite couple type e.g. N-N or NS
                List 3 The position in a binary tree of max depth 5

                #TODO: possibly figure out some way to normalize this vector
    """

    # Getting List 1 and 2
    li_rels_ns = []
    
    for depth in range( _tree.height(),1,-1 ):
        
        # sublist of rst relation and nuclearity tag
        subli_rels_ns = [  re.findall(r'[a-zA-Z\-]+' ,sub_tree._label)  for sub_tree in _tree.subtrees() if sub_tree.height()==depth  ]
        subli_rels_ns = [ [ _li[0], ''.join(_li[1:]).lstrip('unit') ] for _li in subli_rels_ns ]

        li_rels_ns.extend(subli_rels_ns)

    # Getting List 3
        #getting position of all non leave
    tree_pos = _tree.treepositions()
    leaves_pos = _tree.treepositions('leaves')
    pos_xleaves = list(set(tree_pos) - set(leaves_pos)) #unordered
    pos_xleaves = [  tuple(x if x<2 else 1 for x in _tuple ) for _tuple in pos_xleaves]        #binarizing ( correcting any number above 1 to 1)
        # reording pos_xleaves to breadfirst ordering
    #li_bintreepos = sorted([ utils_nlg.tree_order.get(x,-1) for x in pos_xleaves])
    li_bintreepos = sorted( [tree_order_func(x) for x in pos_xleaves] )

    # Zipping List 1 2 and 3
    li_dict_rels_ns_bintreepos = [  {'rel':rels_ns[0], 'ns':rels_ns[1], 'pos': bintreepos } for rels_ns,bintreepos in zip(li_rels_ns,li_bintreepos) if bintreepos!=-1 ]

    return li_dict_rels_ns_bintreepos

def _tree_to_li_du(_tree, li_results=None):
    ### Takes an RST encoded tree and extracts mutually exclusive discourse units
        # that cover the whole span of the tree. This method uses recursive operations 
        # and updates an th li_results object inplace

    direct_childs = len(_tree)
    li_child = [ _tree[idx] for idx in range(direct_childs) ]

    # Formatting children that arent trees, but are one word, by Combining consecutive one word children into an utterance
    groups = []
    keys = []
    for k, g in groupby(li_child, type):  #grouping by type= str and nltk.tree.Tree
        groups.append(list(g))      # Store group iterator as a list
        keys.append(k)

    _ = [ [' '.join(group)] if key==str else group for group,key in zip(groups,keys) ] #concatenating the string groups
    li_child = sum( _, [] )
    direct_childs = len(li_child)
    
    # Parsing children to strings
    li_du_str = [ __parse_leaves(child.leaves()) if type(child)==nltk.tree.Tree else __parse_leaves(child) for child in li_child ]
    
    if(li_results == None):
        li_results = []
    
    #if tree has two subnodes
    for idx in range(direct_childs):
        
        # If child was a string always add to list since it cant be broken down furhter
        if type(li_child[idx]) == str:
            li_results.append(li_du_str[idx])
            continue

        # otherwise segment to sentence
        li_segmented_utt = sent_detector.tokenize(li_du_str[idx])

        #If over two sentences long then perform the method again
        if len(li_segmented_utt) <= 2:            
            li_results.append(li_du_str[idx])
            
        elif len(li_segmented_utt) > 2 :
            #try:
            _tree_to_li_du(li_child[idx], li_results ) 
    
    return li_results

def __parse_leaves(tree_leaves ):
   #     """tree_leaves is list of subsections of an annotated discourse unit
   #     ['_!Three','new', 'issues', 'begin', 'trading', 'on', 'the',
   #  'New', 'York', 'Stock', 'Exchange', 'today', ',!_', '_!and', 'one',
   #  'began', 'trading', 'on', 'the', 'Nasdaq/National', 'Market', 'System',
   #  'last', 'week', '.', '<P>!_'
    
   if type(tree_leaves) == list:
      tree_leaves = ' '.join(tree_leaves)

   #removing tree labels
   _str = re.sub('(_\!|<P>|\!_|<s>)',"", tree_leaves )
   # removing whitespace preceeding a punctuation
   _str2 = re.sub('(\s){1,2}([,.!?\\-\'])',r'\2',_str )

   _str3 = re.sub('  ',' ',_str2).strip()

   return _str3


def find_child_edus(pos_parentnode, li_rst_pos):
        #returns the pos of any child elements of a parent node(rst) that are edus
               
        li_child_pos = [2*pos_parentnode+1, 2*pos_parentnode+2 ]

        li_child_edu_pos = [ pos for pos in li_child_pos if pos not in li_rst_pos]

        return li_child_edu_pos 


#endregion

#Debugging tools


class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


