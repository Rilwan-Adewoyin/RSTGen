import sys, os

import traceback

import numpy
import os
from itertools import groupby

import argparse
import utils_nlg
import random

import math
import itertools
import pandas as pd

import json

import csv
import pickle
import time

import copy 

from filelock import Timeout, FileLock

import regex as re
import multiprocessing as mp
import distutils

import html

from unidecode import unidecode
batches_completed = 0
dict_args = {}

from data_setup_keyphrase_v2 import edu_fixer

import warnings
import contextlib

import spacy

# Docker Images Parser and RST tree labeller
mp1 = os.path.abspath(os.path.join('..'))
mp2 = "../DockerImages/feng_hirst_rst_parser"
mp3 = "../DockerImages/feng_hirst_rst_parser/src"
mp4 = "../DockerImages/feng_hirst_rst_parser/model"
modules_paths = [mp1, mp2, mp3, mp4]

for path_ in modules_paths:
    if path_ not in sys.path:
        sys.path.append(path_)

from DockerImages.feng_hirst_rst_parser.src import parser_wrapper3

from data_setup import _tree_to_rst_code, _parse_trees

#import contextualSpellCheck
# nlp = spacy.load("en_core_web_sm") 
# contextualSpellCheck.add_to_pipe(nlp)
# components = ["parser","ner",'contextual spellchecker']
# removing non needed componenets
# for name in nlp.pipe_names:
#     if name not in components:
#         nlp.remove_pipe(name)

import ujson

def main(   
            **kwargs):
    """[summary]

    Args:
        danet_vname ([type]): [description]
        batch_process_size (int, optional): [description]. Defaults to 200.
    """
    
    #region  Setup    


    dset_path = "./dataset_paraphrase/para-nmt-5m-processed.txt"
    
    if not os.path.exists("./dataset_paraphrase/para-edu-len-filtered.csv"):
        df_para = pd.read_csv(dset_path, sep="\t",names=["source","target"], nrows=2000)
        li_para_record = df_para.to_dict('records')
        del df_para

        # filtering out sentences for those that are 3 to 5 length in EDUs
        mp_count = 6

        with mp.Pool(mp_count) as pool:

            cs =   math.ceil( len(li_para_record) / (mp_count) )

            res = pool.map( edu_length_filter, _chunks(li_para_record, cs)  )


            li_li_bool_edulen_filt = list(res)
            li_bool_edulen_filt = sum(li_li_bool_edulen_filt, [])
        
        li_para_record_filt = [para_record for idx, para_record
                                in enumerate(li_para_record) if li_bool_edulen_filt[idx]==True ]
        
        
        df_para_filt = pd.DataFrame(li_para_record_filt)
        df_para_filt.to_csv("./dataset_paraphrase/para-edu-len-filtered.csv",index=False)
    
    else:
        df_para_filt = pd.read_csv("./dataset_paraphrase/para-edu-len-filtered.csv")

    li_para_record = df_para_filt.to_dict('records')
    del df_para_filt

    # Now Annotating these filtered paraphrasing with their RST structure
    mp_count = 6
    with mp.Pool(mp_count) as pool:
        cs =   math.ceil( len(li_para_record) / (mp_count) )
        res = pool.map( rst_tree_annotator, _chunks(li_para_record, cs) )
        li_paired_rst_structure = list(res)
        #li_paired_rst_structure = sum(li_paired_rst_structure,[])
    
    li_source_rst_structure = sum( [ pair[0] for pair in li_paired_rst_structure], [] )
    li_target_rst_structure = sum( [ pair[1] for pair in li_paired_rst_structure], [] )
    li_para_record =  sum( [ pair[2] for pair in li_paired_rst_structure], [] )

    #li_source_rst_structure, li_target_rst_structure = li_paired_rst_structure
    #li_source_rst_structure = sum(li_source_rst_structure,[])
    #li_target_rst_structure = sum(li_target_rst_structure,[])
    
    #li_source_rst_structure, li_target_rst_structure = list(map(list, zip(*li_paired_rst_structure)))

    df_para_filt = pd.DataFrame(li_para_record)
    df_para_filt['source_rst'] = li_source_rst_structure
    df_para_filt['target_rst'] = li_target_rst_structure

    df_para_filt.to_csv("./dataset_paraphrase/para-edu-len-filtered-with-annotation.csv",index=False)

    print("Finished")

def rst_tree_annotator(li_para_record_):

    li_text_source = [ record['source'] for record in li_para_record_ ]
    li_text_target = [ record['target'] for record in li_para_record_ ]

    li_li_source_target = [li_text_source, li_text_target]

    li_li_unparsed_tree = parser_wrapper3.main( json_li_li_utterances= json.dumps(li_li_source_target), 
                                                skip_parsing=False, redirect_output=True)

    li_unparsed_tree_source, li_unparsed_tree_target = li_li_unparsed_tree
    li_subtrees_source = _parse_trees(li_unparsed_tree_source)
    li_subtrees_target = _parse_trees(li_unparsed_tree_target)
    
    li_rst_dict_source = [ _tree_to_rst_code(_tree) if _tree!=None else None for _tree in li_subtrees_source ]
    li_rst_dict_target = [ _tree_to_rst_code(_tree) if _tree!=None else None for _tree in li_subtrees_target ]

    idxs_to_drop = [ idx for idx, rst_code in enumerate(li_rst_dict_source) if rst_code == [{'ns': 'a', 'pos': 0, 'rel': 'n'}] ]

    for idx in reversed(idxs_to_drop):
        li_rst_dict_source.pop(idx)
        li_rst_dict_target.pop(idx)
        li_para_record_.pop(idx)


    return [li_rst_dict_source, li_rst_dict_target, li_para_record_]


def edu_length_filter(li_para_record_):
    
        
    li_text = [ record['source'] for record in li_para_record_ ]
        
            # returns list of words for each utterance with edu tokens place between different edu segments
    li_textwedutoken = parser_wrapper3.main( json_li_li_utterances= json.dumps([li_text]), 
                                                skip_parsing=True, redirect_output=True)
    
    #TODO: Parser wrapper seperates at apostrophes
    # corrects any formatting errors caused by the segmenter
    li_li_edus = edu_fixer( li_textwedutoken )

    # for each utterance, merge list of words into one text
    li_li_edus = [ [ ' '.join( edus ) for edus in li_edus ] for li_edus in li_li_edus ]
    
    # Fixing:
        # random spaces in words due to splitting at apostrophes such as isn 't
        # random spaces due to splitting at forward slash
        # random spaces due to converting brakcests to - LRB - and - RRB - codes
    li_li_edus = [ [edutxt.replace(" n't", "n't").replace(" / ", "/").replace(" '", "'").replace("- LRB -", "(").replace("- RRB -", ")").replace("-LRB-", "(").replace("-RRB-", ")")
                     if edutxt not in origtext else edutxt for edutxt in li_edutext ] for li_edutext, origtext in zip( li_li_edus, li_text) ]
    
    li_bool_edulen_filt = [ len(li_edus)>=3 and len(li_edus)<=6 for li_edus in li_li_edus  ]
    
    return li_bool_edulen_filt 

def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
 
if __name__ == '__main__':
    #region parser
   main()


#python3 data_setup_cskg.py -bps 600 -sb -1 --mp_count 6
