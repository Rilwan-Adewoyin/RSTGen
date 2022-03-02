from argparse import ArgumentError
import os
import json

from transformers.generation_utils import GenerationMixin

dirname = os.path.dirname(__file__)
from datetime import date

from itertools import (combinations, combinations_with_replacement, 
                       cycle, islice, permutations)
import random
import regex as re
import torch
from pytorch_lightning.callbacks import Callback

from torch import negative, nn

from typing import (Any, Dict, Iterator, List, Optional, OrderedDict, TypeVar)

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
from pytorch_lightning.utilities.distributed import _get_rank
import torch.distributed as dist

from torch.utils.data._utils.collate import default_collate_err_msg_format
from math import floor
import regex as re
from scipy.linalg import toeplitz
import bisect
T_co = TypeVar('T_co', covariant=True)

pattern_capitalize_after_punct = re.compile(r"(\A\w)|"+                  # start of string
                "(?<!\.\w)([\.?!] )\w|"+     # after a ?/!/. and a space, 
                                            # but not after an acronym
                "\w(?:\.\w)|"+               # start/middle of acronym
                "(?<=\w\.)\w",               # end of acronym
                )
pattern_apostrophe = re.compile(r"\b\s+'\b")
pattern_brackets_rm_space = re.compile('\(\s*(.*?)\s*\)')
pattern_punctuation_space = re.compile(r'\s([?.!";:#_](?:\s|$))')
import pytextrank
import en_core_web_sm
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    #install using python -m spacy download en_core_web_sm
nlp = en_core_web_sm.load()

from spacy.language import Language
from transformers.generation_logits_process import LogitsProcessor, _get_generated_ngrams
from torch.utils.data import Sampler, Dataset
from typing import (Any, Dict, Iterator, List, Optional, OrderedDict, TypeVar)
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
def get_path(_path,_dir=False, relative=True):

    if os.path.isabs(_path) == False:
        _path = os.path.join(dirname, _path)
    
    if relative:
        _path = os.path.realpath(_path)
    
    try:
        if _dir:
            os.makedirs(_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(_path), exist_ok=True)
    except PermissionError as e:
        print("Insufficient permission to create directory", _path)
        pass

    return _path

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

    # segmentation helpers
    cue_phrases = ['above all',
            'accordingly',
            'actually',
            'admittedly',
            'after',
            'after',
            'after all',
            'after that',
            'afterwards',
            'again',
            'all in all',
            'all the same',
            'also',
            'alternatively',
            'although',
            'always assuming that',
            'and',
            'and / or',
            'anyway',
            'as',
            'as a consequence',
            'as a corollary',
            'as a result',
            'as long as',
            'as soon as',
            'as well',
            'at any rate',
            'at first',
            'at first blush',
            'at first sight',
            'at first view',
            'at the moment when',
            'at the outset',
            'at the same time',
            'because',
            'before',
            'before',
            'but',
            'by comparison',
            'by contrast',
            'by the same token',
            'by the way',
            'certainly',
            'clearly',
            'consequently',
            'conversely',
            'correspondingly',
            'despite that',
            'despite the fact that',
            'earlier',
            'either',
            'else',
            'equally',
            'essentially, then',
            'even',
            'even so',
            'even then',
            'eventually',
            'every time',
            'except',
            'except insofar as',
            'finally',
            'first',
            'first of all',
            'firstly',
            'for',
            'for a start',
            'for example',
            'for instance',
            'for one thing',
            'for the simple reason',
            'for this reason',
            'further',
            'furthermore',
            'given that',
            'hence',
            'however',
            'if',
            'if ever',
            'if not',
            'if only',
            'if so',
            'in a different vein',
            'in actual fact',
            'in addition',
            'in any case',
            'in case',
            'in conclusion',
            'in contrast',
            'in fact',
            'initially',
            'in other words',
            'in particular',
            'in short',
            'in spite of that',
            'in sum',
            'in that case',
            'in the beginning',
            'in the case of',
            'in the end',
            'in the first place',
            'in the meantime',
            'in this way',
            'in turn',
            'inasmuch as',
            'incidentally',
            'indeed',
            'instead',
            'it follows that',
            'it might appear that',
            'it might seem that',
            'just as',
            'last',
            'lastly',
            'later',
            'let us assume',
            'likewise',
            'meanwhile',
            'merely',
            'merely because',
            'more',
            'moreover',
            'most',
            'much later',
            'much sooner',
            'naturally',
            'neither is it the case',
            'nevertheless',
            'next',
            'no doubt',
            'nonetheless',
            'not',
            'not because',
            'not only',
            'not that',
            'notably',
            'notwithstanding that',
            'notwithstanding that ,',
            'now',
            'now that',
            'obviously',
            'of course',
            'on condition that',
            'on one hand',
            'on one side',
            'on the assumption that',
            'on the contrary',
            'on the grounds that',
            'on the one hand',
            'on the one side',
            'on the other hand',
            'on the other side',
            'once',
            'once again',
            'once more',
            'or',
            'or else',
            'otherwise',
            'overall',
            'plainly',
            'presumably because',
            'previously',
            'provided that',
            'providing that',
            'put another way',
            'rather',
            'reason',
            'reciprocally',
            'regardless of that',
            'second',
            'secondly',
            'similarly',
            'simply because',
            'simultaneously',
            'since',
            'so',
            'so that',
            'specifically',
            'still',
            'subsequently',
            'such that',
            'summarising',
            'summarizing',
            'summing up',
            'suppose',
            'suppose that',
            'supposing that',
            'sure enough',
            'surely',
            'that is',
            'that is to say',
            'the fact is that',
            'the more often',
            'then',
            'then again',
            'thereafter',
            'thereby',
            'therefore',
            'think',
            'third',
            'thirdly',
            'this time',
            'though',
            'thus',
            'to be sure',
            'to begin with',
            'to conclude',
            'to start with',
            'to sum up',
            'to summarise',
            'to summarize',
            'to take an example',
            'to the degree that',
            'to the extent that',
            'too',
            'true',
            'ultimately',
            'undoubtedly',
            'unless',
            'until',
            'we might say',
            'what is more',
            'when',
            'whenever',
            'where',
            'whereas',
            'wherein',
            'wherever',
            'while',
            'yet']

    li_adding = 'also, moreover, furthermore, additionally, besides, in addition'.split(', ')
    li_comparing = 'similarly, likewise, in the same way'.split(', ')
    li_generalizing = 'on the whole, in general, broadly speaking, as a rule, in most cases'
    li_showing_cause_effect = 'therefore, thus, consequently, hence, as a result'
    li_contrasting = 'however, although, whereas, despite this fact, on one hand, on the other hand, on the contrary, still, nonetheless, instead, alternatively, in contrast, but'.split(', ')
    li_sequencing = "firstly, at first, first of all, in the first place, to begin with, in the beginning, once upon a time, secondly, thirdly, subsequently, earlier, meanwhile, later, afterwards, what's more, for a start".split(', ')
    li_emphasizing = 'above all, specially, in particular, specifically, as a matter of fact, more importantly'.split(', ')
    li_repeating = 'again and again, over and over, once again, as stated'.split(', ')
    li_examples = 'for example, for instance, such as, namely, in other words'.split(', ')
    li_concluding = 'in conclusion, finally, to sum it up, in the end, lastly, in short, eventually'.split(', ')
    li_temporal = 'when, finally, so, instead, later, after, eventually, at last'.split(', ')

    discourse_markers = {'adding':li_adding,
                        'comparing':li_comparing,
                        'generalizing':li_generalizing,
                        'showing_cause_effect':li_showing_cause_effect,
                        'contrasting':li_contrasting,
                        'sequencing':li_sequencing,
                        'emphasizing':li_emphasizing,
                        'repeating':li_repeating,
                        'examples':li_examples,
                        'concluding':li_concluding,
                        'temporal':li_temporal
                        }
    punct_end_edu = []
    punct_end_edu.extend("!),.:;?")
    punct_start_edu = []
    punct_start_edu.extend("(")

    def get_curr_edu_pos(self, decoder_input_ids, edu_rstpos, li_gen_text=None):
        """
        li_edu_rst_pos: the list of possible edus positions for each text in this batch
        """
        
        num_hypos = decoder_input_ids.shape[0]
        num_beams = num_hypos // edu_rstpos.shape[0]
             
        # Updating record of currently generated texts
        if li_gen_text == None:
            # Creating whole string for input_ids
            li_gen_text = [ self.tokenizer.decode(ids, skip_special_tokens=True) for ids in torch.unbind( decoder_input_ids,0) ]
        
        else:
            # Generating the newest word for each text  and appending it to text 
            li_new_text = [ self.tokenizer.decode( ids[-1:], skip_special_tokens=True) for ids in torch.unbind(decoder_input_ids, 0 ) ]
            li_gen_text = [ gen_text+new_text for gen_text,new_text in zip(li_gen_text, li_new_text) ]
                            
        # Checking whether or not to use previous edu_pos or new edu_pos
        # Initially we assume old edu_positions are used
        # First we check if generated_text[idx] had its segmented updated in the previous round. 
        if not hasattr(self, 'consecutive_steps_prev_edu_used'):
            self.consecutive_steps_prev_edu_used = [0]*num_hypos
        if not hasattr(self, 'prev_edu_pos'):
            self.prev_edu_pos = [ decoder_input_ids.new_tensor( [] ) for idx in range(num_hypos) ]
            curr_edu_pos = [None]* num_hypos
        else:
            # curr_edu_pos = [self.prev_edu_pos[idx][-1:] for idx in range(num_hypos)]
            curr_edu_pos = [self.prev_edu_pos[idx][-1] for idx in range(num_hypos)]

        for idx in range(num_hypos):
            
            # Automated Checks to Increase edu pos                                
            #CASE: first generated token and no context utterance
            if self.prev_edu_pos[0].numel() == 0 and decoder_input_ids.shape[1] == 1: 
                # curr_edu_pos = [ edu_rstpos[idx//num_beams][:1]  for idx in range(num_hypos) ]
                curr_edu_pos = [ edu_rstpos[idx//num_beams][0]  for idx in range(num_hypos) ]
                self.consecutive_steps_prev_edu_used = [1]*num_hypos
                break
            
            #CASE: first generated token and context utterance
            elif self.prev_edu_pos[0].numel() == 0 and decoder_input_ids.shape[1] > 1: 
                #We Segment all texts together
                if self.rst_segment_method == "fenghirst":
                    raise NotImplementedError
                elif self.rst_segment_method == "segbot":
                    # use segmenter to count up to equivalent edu
                    li_segmented_text = self.segmenter.segment_li_utterances(li_gen_text)
                    
                    li_edu_count = [ len(seg_text) for seg_text in li_segmented_text ]

                    edu_rstpos = [tens if not (-1 in tens) else tens[: (tens == -1).nonzero(
                        as_tuple=True)[0][0]] for tens in edu_rstpos]
                                        
                    pos = [ min(edu_rstpos[idx//num_beams].numel()-1, max(0, edu_count-1)) for idx, edu_count in enumerate(li_edu_count) ]

                    # curr_edu_pos = [ edu_rstpos[idx//num_beams][ pos:pos+1 ] for idx, pos in enumerate(pos) ]
                    curr_edu_pos = [ edu_rstpos[idx//num_beams][ pos ] for idx, pos in enumerate(pos) ]
                else:
                    raise ValueError
                
                self.consecutive_steps_prev_edu_used = [1]*num_hypos
                break

            #CASE: Check we have not use previous edu too many times
            elif self.consecutive_steps_prev_edu_used[idx] >= 3:
                # We do not allow segmentation checks for two consecutive rounds                
                self.consecutive_steps_prev_edu_used[idx] = 0
                curr_edu_pos[idx] = None
                
            #Case: END PUNCTUATION is previosly generated token - we advance to next EDU  
            elif li_gen_text[idx][ -1:] in self.punct_end_edu:

                # bool_use_old_edu_pos[idx] = True
                new_rst_pos_idx = (edu_rstpos[idx//num_beams] == self.prev_edu_pos[idx][-1]).nonzero(as_tuple=True)[0][-1] + 1
                
                # Ensuring index is not out of edu_rstpos         
                temp = edu_rstpos[idx//num_beams]
                edu_rstpos_nopad = temp if not (-1 in temp) else temp[:(temp == -1).nonzero(
                    as_tuple=True)[0][0]]
                new_rst_pos_idx = min( len(edu_rstpos_nopad)-1, new_rst_pos_idx )
                
                # Preventing EDU rstpos prediction from going backwards 
                # print(new_rst_pos_idx) 
                # print(edu_rstpos_nopad[new_rst_pos_idx].tolist())
                # print(self.prev_edu_pos[idx][-1].tolist())
                new_rst_pos = max(  edu_rstpos_nopad[new_rst_pos_idx].tolist(), self.prev_edu_pos[idx][-1].tolist(), key=RstTokenizerMixin.edukp_pos_sort_function  )
                new_rst_pos = torch.tensor( [new_rst_pos], device=edu_rstpos_nopad.device , dtype=torch.long)
                
                # curr_edu_pos[idx] = new_rst_pos
                curr_edu_pos[idx] = new_rst_pos[0]
                
                # If EDU START PUNCTUATION occurs we advance to the next edu if the previous edu is the same as the previous previous edu
                self.consecutive_steps_prev_edu_used[idx]+= 1

            #Case: START PUNCTUATION is previosly generated token - we check  that edu_rstpos has changed between prev two tokens
            elif li_gen_text[idx][ -1:] in self.punct_start_edu:               
                
                edu_correctly_changed = ( self.prev_edu_pos[idx][-1] == self.prev_edu_pos[idx][-2] )
                
                if edu_correctly_changed == False:
                    
                    # bool_use_old_edu_pos[idx] = True
                    new_rst_pos_idx = edu_rstpos[idx//num_beams].tolist().index( self.prev_edu_pos[idx][-1] ) + 1

                    # Ensuring index is not out of edu_rstpos         
                    temp = edu_rstpos[idx//num_beams]
                    edu_rstpos_nopad = temp if not (-1 in temp) else temp[:(temp == -1).nonzero(
                        as_tuple=True)[0][0]]
                    new_rst_pos_idx = min( len(edu_rstpos_nopad)-1, new_rst_pos_idx )
                    
                    # Preventing EDU rstpos prediction from going backwards  
                    new_rst_pos = max( [ edu_rstpos_nopad[new_rst_pos_idx].tolist(), self.prev_edu_pos[idx][-1].tolist()  ]  , key=RstTokenizerMixin.edukp_pos_sort_function  )
                    new_rst_pos = torch.tensor( [new_rst_pos], device=edu_rstpos_nopad.device , dtype=torch.long)
                    
                    # curr_edu_pos[idx] = new_rst_pos
                    curr_edu_pos[idx] = new_rst_pos[0]

                    self.consecutive_steps_prev_edu_used[idx]+= 1

                else:
                    curr_edu_pos[idx] = None
                    self.consecutive_steps_prev_edu_used[idx] = 0
                    
                # Cue Phrase / Discourse marker check
                # cp or dm can be more than one token. 
                # for i<n, n= max cue_phrase,dm length
                # If the previous i generated tokens match a cp,dm or length i then we will check for a new edu
            
            #CASE: Check if dm marker / cue phrase occurs. If so then do segment check
            else:
                #If dm marker occurs then do we set curr_edu_pos[idx] to 0. This indicates we should do a segment check
                input_ids = decoder_input_ids[idx]
                
                for idx1 in range(1, min(len(self.dict_len_phrasetok), len(input_ids) )+1 ):
                    if ( input_ids[-idx1:] == torch.stack(self.dict_len_phrasetok[idx1],axis=0) ).all(axis=-1).any():
                        curr_edu_pos[idx] = None
                    else:
                        self.consecutive_steps_prev_edu_used[idx]+= 1
                        
        
 
        # Perform Segmentation using Segmenter  on selected indexes
        if not all(curr_edu_pos):

            if self.rst_segment_method == "fenghirst":
                raise NotImplementedError
                idxs_to_update = [ idx for idx,bool_ in enumerate(bool_use_old_edu_pos) if bool_==False ]
                li_gen_text_filtr = [ gen_text for idx,gen_text in enumerate(li_gen_text) if idx in idxs_to_update]
                
                li_textwedutoken = self.segmenter.parse_li_utterances(li_gen_text_filtr)
                # calculating edu of current text
                li_edu_count = [
                    ' '.join(li_words[:-1]).count('EDU_BREAK')+1 for li_words in li_textwedutoken]

                # removing any paddingo
                edu_rstpos = [tens if not (-1 in tens) else tens[: (tens == -1).nonzero(
                    as_tuple=True)[0]] for tens in edu_rstpos][0]

                # selecting approapriate edu value.
                curr_edu_pos = [edu_rstpos[min(edu_rstpos.numel()-1, max(0, edu_count-1))]
                                for edu_count in li_edu_count]
                
                # Adding back to original list
                _ = [ curr_edu_pos[idx].pop(0) if idx in idxs_to_update else prev_edu_pos[idx] for idx in range(len(li_gen_text))  ]
                
                # Ensuring each curr_edu_pos is at least as large as the prev_edu_pos
                if prev_edu_pos != None:
                    _ = [ max( [ curr_pos, prev_pos] , key= RstTokenizerMixin.edukp_pos_sort_function ) 
                                    for curr_pos, prev_pos in zip(curr_edu_pos, prev_edu_pos)  ]
                curr_edu_pos = _
                
            elif self.rst_segment_method == "segbot":

                # idxs_to_update = [ idx for idx,bool_ in enumerate(bool_use_old_edu_pos) if bool_==False ]
                idxs_to_update = [ idx for idx,val in enumerate(curr_edu_pos) if val==None ]

                li_gen_text_filtr = [ gen_text for idx,gen_text in enumerate(li_gen_text) if idx in idxs_to_update]
                
                li_segmented_text = self.segmenter.segment_li_utterances(li_gen_text_filtr)
                
                li_edu_count = [ len(seg_text) for seg_text in li_segmented_text ]

                # removing any padding
                edu_rstpos = [tens if not (-1 in tens) else tens[: (tens == -1).nonzero(
                    as_tuple=True)[0][0]] for tens in edu_rstpos]
                            
                updated_curr_edu_pos = [ edu_rstpos[idx//num_beams][min(edu_rstpos[idx//num_beams].numel()-1, max(0, edu_count-1))]
                                for idx,edu_count in enumerate(li_edu_count) ]
                
                # check the updated curr_edu_pos aren't going backwards
                updated_curr_edu_pos = [  max(  edu_pos.tolist(), self.prev_edu_pos[idxs_to_update[idx]][-1].tolist() , key=RstTokenizerMixin.edukp_pos_sort_function  ) for idx, edu_pos in enumerate(updated_curr_edu_pos) ]
                for idx in range(len(updated_curr_edu_pos)):
                    updated_curr_edu_pos[idx] =  torch.tensor( [updated_curr_edu_pos[idx]], device=decoder_input_ids.device, dtype=torch.long)

                # Adding back to original list
                for i, idx in enumerate(idxs_to_update):
                    # curr_edu_pos[idx] = updated_curr_edu_pos[i]
                    curr_edu_pos[idx] = updated_curr_edu_pos[i][0]


        # Appending curr_edu_pos to the self.prev_edu_pos
        self.prev_edu_pos = [ torch.cat( [ prev, curr.reshape(1) ] ) for prev,curr in zip( self.prev_edu_pos, curr_edu_pos ) ]
        
        # print(curr_edu_pos)
        
        return curr_edu_pos, li_gen_text

    def _get_logits_processor(self, *args, **kwargs ):
        processor = GenerationMixin._get_logits_processor(self, *args, **kwargs )
        
        kp_logitsproc = KeyPhraseNoRepeatLogitsProcessor(self.gen_key_phrase_ids, 
                                                         self.tokenizer.keyphrase_start_token_id[0],
                                                         self.tokenizer.eos_token_id,
                                                         tokenizer=self.tokenizer)
        # nonlocal logits_processor
        processor.append( kp_logitsproc )
                
        return processor
    
    def discrete_embedding_dropout(self, input_token, emb_tbl, dropoute, scale=None):
        
        # discrete input dropout
        # 1. generate binary mark
        # 2. mask embedding matrix
        # 3. perform normal lookup of embeddings (some of which are dropped out)
        if dropoute and emb_tbl.weight.requires_grad and self.training :
            # 1. generate binary mark
            mask = emb_tbl.weight.data.new(emb_tbl.weight.size(0), 1).bernoulli_(1 - dropoute).bool()
            # 2. mask embedding matrix
            masked_embed_weight = mask * emb_tbl.weight * (1.0/(1 - dropoute))
            # I think this is important so you do not divide by 0 during normalization?
            masked_embed_weight.masked_fill_(mask.eq(0), 1e-12)
        else:
            masked_embed_weight = emb_tbl.weight

        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

        # 3. perform normal lookup of embeddings (some of which are dropped out)
        X = torch.nn.functional.embedding(input_token, masked_embed_weight,
                emb_tbl.padding_idx, emb_tbl.max_norm, emb_tbl.norm_type,
                emb_tbl.scale_grad_by_freq, emb_tbl.sparse)
        return X

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
                elif len(li_)>1:
                    elem_size = len(li_[0])

                bool_size_check = all(len(elem_) == elem_size for elem_ in li_)
                if key == 'decoder_cross_attention_mask':
                    bool_size_check = bool_size_check and all( elem_.shape[1] == li_[0].shape[1] for elem_ in li_)
                
                if not bool_size_check :
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
                                        elem_ = torch.nn.functional.pad( elem_, (0, missing_dims, 0, missing_dims), mode='constant', value=0.0 )
                                                            
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
                                a = 1
                    
                    except Exception as e:
                        print(key)
                        raise e
                    
                dict_output[key] = self.default_collate_pad( li_ )    
                
            return dict_output

        elif isinstance(elem, tuple) and hasattr(elem, '_fields'): # namedtuple
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

class SizedOrderedBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.prepare_ds()
                
    
    def __iter__(self) -> Iterator[List[int]]:
        return iter( self.li_chunked_idxs )


    def __len__(self) -> int:
        return len(self.li_chunked_idxs)
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        if self.shuffle:
            self.prepare_ds()        

    def prepare_ds(self):
        # Sorting and (maybe) shuffling
        np_txt_lens = np.concatenate(
            [ds.np_textlens for ds in self.data_source.datasets]).flatten()
        np_rst_lens = np.concatenate(
            [ds.np_rstlens for ds in self.data_source.datasets]).flatten()
        np_key_phrase_lens = np.concatenate(
            [ds.np_keyphrase_lens for ds in self.data_source.datasets]).flatten()

        tuple_factors = (np_rst_lens, np_key_phrase_lens, np_txt_lens)
        if self.shuffle:
            random_idxs = np.random.random( np_txt_lens.size )
            tuple_factors = (random_idxs, )+ tuple_factors
        np_ordered_idxs = np.lexsort(tuple_factors)


        # Handing drop last
        if self.drop_last:
            rem_records = np_txt_lens.size % self.batch_size
            li_ordered_idxs = np_ordered_idxs.tolist()
            for idx in range(rem_records):
                li_ordered_idxs.pop( random.randint(0,len(li_ordered_idxs)-1) )
            np_ordered_idxs = np.array(li_ordered_idxs)

        # We Randomly re-arrange them in batches of batch size
        self.li_chunked_idxs = [np_ordered_idxs[idx:idx+self.batch_size]
                            for idx in range(0, np_ordered_idxs.size - self.batch_size, self.batch_size)]

        if self.shuffle:
            random.shuffle(self.li_chunked_idxs)

        # Getting max sizes for rst in each chunk
        self.li_chunk_rst_len = [
            np.take(np_rst_lens, idxs).max() for idxs in self.li_chunked_idxs]

        self.li_chunk_key_phrase_len = [
            np.take(np_key_phrase_lens, idxs).max() for idxs in self.li_chunked_idxs]
        

        # Updating Chunk sizes
        for chunk_idx, data_idxs in enumerate(self.li_chunked_idxs):

            for data_idx in data_idxs:
                dataset_idx = bisect.bisect_right(
                    self.data_source.cumulative_sizes, data_idx)

                if dataset_idx == 0:
                    sample_idx = data_idx
                else:
                    sample_idx = data_idx - \
                        self.data_source.cumulative_sizes[dataset_idx - 1]

                self.data_source.datasets[dataset_idx].rst_len[sample_idx] = self.li_chunk_rst_len[chunk_idx]
                self.data_source.datasets[dataset_idx].key_phrase_len[sample_idx] = self.li_chunk_key_phrase_len[chunk_idx]

class SizedOrderedDistributedBatchSampler(Sampler[List[int]]):
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

    def __init__(self,
                data_source: Dataset, batch_size: int,
                 drop_last:bool,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int]= None,
                 seed: int = 0,
                 shuffle: bool = False,
                 units: Optional[int] = None,
                 nodes:Optional[int] = None
                 ) -> None:

        # Either using gpu set or tpu set up
        # proc_type = "gpu"*(gpus==None and num_nodes==None) + "tpu"*(tpus==None and tpu_cores==None)
        # if proc_type == "gpu":
        #     units = gpus
        #     nodes = num_nodes
        # elif proc_type == "tpu":
        #     units = tpus
        #     nodes = tpu_cores
        # else:
        #     raise ArgumentError("User must either have tpu or gpu set up")

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            try:
                num_replicas = dist.get_world_size()
            except Exception as e :
                num_replicas = units*nodes
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = _get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        # normal code
        self.num_replicas = num_replicas
        self.rank = rank

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_source = data_source
        self.shuffle = shuffle

        self.seed = seed
        self.epoch = 0

        self.prepare_ds()
       
    def __iter__(self) -> Iterator[T_co]:
        
        # for idx in range( len(self) ):
        #     batch = self.li_li_chunked_idxs[self.rank][idx]
        #     yield batch

        return iter( self.li_li_chunked_idxs[self.rank] )
    
    def __len__(self) -> int:
        
        return len(self.li_li_chunked_idxs[self.rank] )

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        if self.shuffle:
            self.prepare_ds()


    def prepare_ds(self):
        # new code
        np_txt_lens = np.concatenate(
            [ds.np_textlens for ds in self.data_source.datasets]).flatten()
        np_rst_lens = np.concatenate(
            [ds.np_rstlens for ds in self.data_source.datasets]).flatten()
        np_key_phrase_lens = np.concatenate(
            [ds.np_keyphrase_lens for ds in self.data_source.datasets]).flatten()

        g = torch.Generator()
        g.manual_seed(self.seed+self.epoch)

        # Sorting and (maybe) shuffling
        tuple_factors = (np_rst_lens, np_key_phrase_lens, np_txt_lens)

        if self.shuffle:
            random_idxs= list(range(np_txt_lens.size))
            random.Random(self.seed+self.epoch).shuffle(random_idxs) 
            tuple_factors = (random_idxs, )+ tuple_factors
        np_ordered_idxs = np.lexsort(tuple_factors)

       # Handing drop_last
        if self.drop_last:
            rem_records = np_txt_lens.size % (self.batch_size*self.num_replicas)
            li_ordered_idxs = np_ordered_idxs.tolist()
            for idx in range(rem_records):
                li_ordered_idxs.pop( random.Random(self.seed+self.epoch).randint(0,len(li_ordered_idxs)-1) )
            np_ordered_idxs = np.array(li_ordered_idxs)


        # We Randomly re-arrange them in batches of batch size
        li_chunked_idxs = [np_ordered_idxs[idx:idx+self.batch_size]
                           for idx in range(0, np_ordered_idxs.size-self.batch_size, self.batch_size)]

        # Divide into n sublists,
        # Each sublist at index i, contains the indices for process at rank i
        # Each sublist at index i, represents similar sized items in the dataset
        li_li_chunked_idxs = [
            [li_chunked_idxs[(self.num_replicas*idx)+_rank]
             for idx in range( len(li_chunked_idxs)//self.num_replicas ) ]
            for _rank in range(self.num_replicas)]

        # shuffle each processes subllist in the same order to optimize paralel training
        if self.shuffle:
            _ = list(zip(*li_li_chunked_idxs))
            random.Random(self.seed+self.epoch).shuffle(_)
            # unpacking into worker size length list
            li_li_chunked_idxs = list(zip(*_))

        self.li_li_chunked_idxs = li_li_chunked_idxs

        # Getting max sizes for rst and key_phrase in each chunk
        li_li_chunk_rst_len = [[np.take(np_rst_lens, idxs).max() for idxs in li_chunked_idxs]
                                    for li_chunked_idxs in li_li_chunked_idxs]
        li_li_chunk_key_phrase_len = [[
            np.take(np_key_phrase_lens, idxs).max()
            for idxs in li_chunked_idxs] for li_chunked_idxs in li_li_chunked_idxs]
        
                    
        #Updating the max_len_rst and max_len_keyphrase 
        for (li_chunked_idxs, li_chunk_rst_len, li_chunk_key_phrase_len) in zip(li_li_chunked_idxs, li_li_chunk_rst_len, li_li_chunk_key_phrase_len):
            # iterating through chunk_idx, data_idxs enumerate(self.li_chunked):

            for chunk_idx, data_idxs in enumerate(li_chunked_idxs):

                for data_idx in data_idxs:
                    dataset_idx = bisect.bisect_right(
                        self.data_source.cumulative_sizes, data_idx)

                    if dataset_idx == 0:
                        sample_idx = data_idx
                    else:
                        sample_idx = data_idx - \
                            self.data_source.cumulative_sizes[dataset_idx - 1]

                    self.data_source.datasets[dataset_idx].rst_len[sample_idx] = li_chunk_rst_len[chunk_idx]
                    self.data_source.datasets[dataset_idx].key_phrase_len[sample_idx] = li_chunk_key_phrase_len[chunk_idx]

#endregion

#region data making

def non_parseable_remover(li_dict_rsttext):
    # print("Start non parseable remover")
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
    
    # print("End non parseable remover")
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

#endregion

#region generation

class KeyPhraseNoRepeatLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` that enforces no repetition of encoder input ids n-grams for the decoder ids.
    See `ParlAI <https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/torch_generator_agent.py#L1350>`__.
    Args:
        encoder_ngram_size (:obj:`int`):
            All ngrams of size :obj:`ngram_size` can only occur within the encoder input ids.
        encoder_input_ids (:obj:`int`):
            The encoder_input_ids that should not be repeated within the decoder ids.
        key_phrase_ids (:list : list tensors) a list of tensors on the same device as 
    """

    def __init__(self, key_phrase_ids, key_phrase_start_id, eos_token_id,**kwargs):
        self.batch_size = key_phrase_ids.shape[0]  
        unbatched_key_phrase_ids = torch.unbind(key_phrase_ids, dim=0)
        # self.tok = kwargs.get('tokenizer',None)
        
        unbatched_kpstart_pos = [ 
                                  torch.cat( [
                                                ( kps ==key_phrase_start_id ).nonzero(as_tuple=False),
                                                ( kps ==eos_token_id ).nonzero(as_tuple=False) 
                                             ] #key phrase may be padding with eos tokens
                                            ).squeeze(-1)
                                            for kps in unbatched_key_phrase_ids 
                                ] 
        
        self.unbatched_key_phrase_ids  = [ [  kpids[start_idx+1:end_idx ] for start_idx,end_idx in zip( kpstart_pos, kpstart_pos[1:]) 
                                            if not(
                                                    kpids[start_idx]==eos_token_id and 
                                                    kpids[end_idx] ==eos_token_id )  ] 
                                        
                                        for kpids, kpstart_pos in zip( unbatched_key_phrase_ids, unbatched_kpstart_pos) ]
                # splitting on the <kp> or  <|kp|> token
        self.max_ngram_size = max( len(ids) for ids in self.unbatched_key_phrase_ids )     
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        # B x num_beams
        num_hypos = scores.shape[0]
        num_beams = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1]
        
        #Here we create a temporary kp_ngrams_mentioned
            # We only block kps if they have been mentioned in the text
            # This should be a list of dictionaries
            # One for each batch size
            # each dictionary contains keys=key_phrase excl final token values=list of final tokens of key_phrases 

        generated_kps_ids = []
        for idx in range(num_hypos):
            ngrams = collections.defaultdict(list)
            
            for kp_ids in self.unbatched_key_phrase_ids[idx//num_beams]:
                
                if any( torch.equal(kp_ids,input_ids[idx][s:s+len(kp_ids)]) for s in range(0,len(input_ids[idx])-len(kp_ids)+1) ):
                    ngrams[ tuple(kp_ids[:-1].tolist()) ].extend( kp_ids[-1:].tolist() )
                    
            generated_kps_ids.append(ngrams)
        
        #Here we check for banned tokens for every ngram size  
        banned_batch_tokens = [
                sum([ 
                        _get_generated_ngrams(generated_kps_ids[hypo_idx], input_ids[hypo_idx], ngram_size, cur_len)
                            for ngram_size in range(1, self.max_ngram_size)]    
                    ,[])
                    for hypo_idx in range(num_hypos)
                    ]
        
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores

#endregion

# region Degenerate Losses Mixin
class DegenerateLossMixin():


    def _compute_token_level_unlikelihood_loss(self, tgt_log_probs, target_tokens, kp_cands, tkn_pad_idx=-100, step_i=0):
        #https://github.com/BorealisAI/keyphrase-generation/blob/c65c45938f6056b590e89fbc3186ebe8eb1c136a/keyphrase_generation/models/copy_seq2seq_attn.py#L462
        
        """
        Calculate the token level unlikelihood loss
        - At each time step, we look at all the words in the the ground truth from the previous time steps
        - which forms the negative list at that time step
        - loss is calculated by penalizing the probability of predicting these words present the negative candidate list
        """
        #HERE
        #TODO: 
            # Code can be kept the same for Context
            # But for keywords, we can't simply look at the last n tokens since key phrases have <kp> tokens in between
            # M1: pass kp_cands to this method.
            # Furthermore, must remember to mask over the RST section. 
                # One quick way to do this is based on the target_tokens
                # the RST section responds to any section that has -100 as the index

        #TODO: Might be best to have key phrase tokens simply passed in, ignoring the <kp>

        # tgt_log_probs is [batch_size , max_tgt_len , tgt_vocab_size]
        # Collapse it to a 2d tensor:
        # to get tensor of dim [(batch_size * max_tgt_len) x tgt_vocab_size]
        batch_size, max_tgt_len, tgt_vocab_size = tgt_log_probs.size()
        tgt_log_probs = tgt_log_probs.view(-1, tgt_log_probs.size(-1))
        
        with torch.no_grad():
            # The first timestep is just the @START@ token, which is not included in the likelihoods
            # i.e., we do not ask the model to predict the @START@ token
            # target = target_tokens[:, (step_i+1):] #TODO: check this sequence hasn't already been shifted
            target= target_tokens

            # Get the context candidates
            ctx_cands = target.unsqueeze(1).expand(
                target.size(0), target.size(1), target.size(1))

            # removing all negative indexes from target
            ctx_cands = ctx_cands.masked_fill( ctx_cands<0, tkn_pad_idx )

            # get the lower triangular matrix
            # ctx_cands = (ctx_cands.tril(-1) + tkn_pad_idx)
            ctx_cands = ctx_cands.tril(-1) + torch.full_like(ctx_cands,tkn_pad_idx).triu()

            # what is the point of these 2 lines (?)
            # -- taken from https://github.com/facebookresearch/unlikelihood_training/blob/master/custom/candidate_penalty_ce_loss.py
            # NOTE: this line below assumes padding index si 1
            # ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
            # ctx_cands = ctx_cands.tril(-1) + ctx_cands_

            # Don't include the target for that timestep as a negative target
            # i.e., remove it if it was a part of the candidate list
            ctx_cands = ctx_cands.masked_fill(
                ctx_cands == target.unsqueeze(2), tkn_pad_idx)

            if self.config.prev_context_len > 0:
                
                # to consider only a pre-specified previous context size
                # len(mask_generator) == max_tgt_len
                mask_generator = [0] + [1]*self.config.prev_context_len + \
                    [0]*(max_tgt_len-self.config.prev_context_len-1)
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.toeplitz.html
                mask = toeplitz(mask_generator, np.zeros_like(mask_generator))
                # with the above mask, only the prev n words in the context are considered
                mask = torch.tensor(mask, dtype=ctx_cands.dtype,
                                    requires_grad=False, device=ctx_cands.device)
                # create the same mask for the entire batch
                mask = mask.unsqueeze(0).expand(
                    batch_size, max_tgt_len, max_tgt_len)

                # incorporate the previous-N-context-only mask
                ctx_cands = mask * ctx_cands

            # reshape to [(batch_size * max_tgt_len), max_tgt_len]
            ctx_cands = ctx_cands.view(-1, ctx_cands.size(-1))

            # get a zero matrix of the same size size as tgt_log_probs ---> [(batch_size * max_tgt_len), tgt_vocab_size]
            # for each row, fill 1s in the positions that correspond to the candidate token ids
            # negative_targets ---> is a k-hot vector with the token idx of the candidates as 1
            negative_targets = torch.zeros_like(
                tgt_log_probs).scatter_(1, ctx_cands, 1)
            
            # adjusting for tokens in the keyphrase set of tokens
            #Don't include to kp_cands if it is acc the target at this timestep
            kp_cands = kp_cands.unsqueeze(1).expand( 
                kp_cands.size(0), target.size(1), kp_cands.size(1))
            
            kp_cands = kp_cands.masked_fill(
                kp_cands == target.unsqueeze(2), tkn_pad_idx)

            kp_cands = kp_cands.view(-1, kp_cands.size(-1))

            negative_targets.scatter_(1, kp_cands, 1)

            #ensuring pad token is not included in negative targets
            negative_targets[:, tkn_pad_idx]= 0

        # - compute loss
        # tgt_log_probs refer to log of probabilities, exp() of it gives the actual prob. values
        # [(batch_size * max_tgt_len) x tgt_vocab_size]
        one_minus_probs = torch.clamp((1.0 - tgt_log_probs.exp()), min=1e-6)

        # only keep the probabilities at the negative token indices
        # [(batch_size * max_tgt_len) x tgt_vocab_size]
        loss = -torch.log(one_minus_probs)*negative_targets

        loss = loss.sum(-1)/negative_targets.sum(-1) # Summing over each training datum
        loss = loss.reshape(batch_size, max_tgt_len)
        loss = loss.sum(-1).mean() #Sum within each batch then average across the batch

        return loss

#endregion
