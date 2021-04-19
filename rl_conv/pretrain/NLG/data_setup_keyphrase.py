import sys, os

import traceback

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


import numpy
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from itertools import groupby

import argparse
import utils_nlg
import random

import pytorch_lightning as pl
import emoji
from transformers import AutoTokenizer, AutoModel
import docker
import math
import itertools
import pandas as pd
import nltk
nltk.download('stopwords')

import rake_nltk
import json
import pytextrank 
import spacy
import en_core_web_sm
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    #install using python -m spacy download en_core_web_sm

import pke

import csv
import pickle
import time
import torch

from filelock import Timeout, FileLock


from utils import get_best_ckpt_path
from utils import get_path

import regex as re
import multiprocessing as mp
import distutils

import html

from unidecode import unidecode
batches_completed = 0
dict_args = {}


import warnings
import contextlib

import requests
from urllib3.exceptions import InsecureRequestWarning

# Docker Images Parser and RST tree labeller
mp1 = os.path.abspath(os.path.join('..'))
mp2 = "../DockerImages/feng_hirst_rst_parser"
mp3 = "../DockerImages/feng_hirst_rst_parser/src"
mp4 = "../DockerImages/feng_hirst_rst_parser/model"
modules_paths = [mp1, mp2, mp3, mp4]

for mp in modules_paths:
    if mp not in sys.path:
        sys.path.append(mp)

from DockerImages.feng_hirst_rst_parser.src import parser_wrapper3

old_merge_environment_settings = requests.Session.merge_environment_settings

def main(   batch_process_size=20,
            rst_method="feng-hirst",
            mp_count=5,
            start_batch=0,
            end_batch = 0,
            **kwargs):
    """[summary]

    Args:
        danet_vname ([type]): [description]
        batch_process_size (int, optional): [description]. Defaults to 200.
        rst_method (str, optional): [description]. Defaults to "feng-hirst".
    """
    
    #region  Setup    
    # setting up model for Da prediction
    
    #Creating Save directory
    dir_save_dataset = utils_nlg.get_path("./dataset_keyphrase/",_dir=True)
    
    # setting up subreddit data
    
    #with no_ssl_verification():
    dirs_rst_conv = "./dataset_v2"
    li_subreddit_names = list( filter( lambda name: name!="last_batch_record", os.listdir( self.dirs_rst_conv ) ) )

    dict_subreddit_fp = {subreddit: [fp for fp in glob.glob(os.path.join(dirs_rst_conv,subreddit,"*")) if os.path.split(fn)[-1]!="lock"  ]  for subreddit in  li_subreddit_names }

    for subreddit, li_fp in dict_subreddit_fp.items():
        
        dset_source = pd.from_csv( li_fp, columns=['rst','txt_preproc','subreddit'] )

        total_batch_count = math.ceil(len(dset_source)/batch_process_size)

        # Optionally auto-resuming from last completed batch
        if start_batch == -1:

            # checking if df_records exists and if a column for this subreddit exists
            fn = os.path.join(dir_save_dataset,'last_batch_record')
            _bool_file_check = os.path.exists( fn )

            auto_fnd_failed = lambda : print("User choose auto-resume from last recorded batch.\
                But no last records exists, so initialising from batch 0")

            if not _bool_file_check: #s if file exist
                start_batch = 0
                auto_fnd_failed()
                
            else: #if file does not exists
                df_records = pd.from_csv( fn, index_col = "subreddit" )
                _bool_record_check = reddit_dataset_version in df_records.index.tolist()

                if not _bool_record_check:
                    start_batch = 0
                
                else:
                    start_batch = int( df_records.loc[ subreddit, 'last_batch' ] ) + 1
                    batch_process_size = int( df_records.loc[subreddit, 'batch_process_size'] )
            
        # selecting sub batch to operate on if applicable
        if end_batch != 0:
            dset_source = dset_source.iloc[ : end_batch*batch_process_size] 
        if start_batch != 0:
            dset_source = dset_source.iloc[ start_batch*batch_process_size: ]

        # endregion
        timer = Timer()

        #region operating in batches
        global batches_completed
        batches_completed = start_batch

        while len(dset_source) > 0 :
            
            batch_li_dict_utt =  dset_source.iloc[:batch_process_size].to_records()
            
            print(f"\nOperating on batch {batches_completed} of {total_batch_count}")


            #region Segmentating the text into EDUs
            timer.start()
            

            with mp.Pool(mp_count) as pool:
                res = pool.starmap( edu_segmenter, zip( _chunks(batch_li_dict_utt, batch_process_size//mp_count ) , contaier_ids  ) )
            
            batch_li_li_edusegment = list( res ) 
            batch_li_li_edusegment = sum(batch_li_li_edusegment, [])
            batch_li_li_edusegment = [ li for li in batch_li_li_edusegment if li !=[] ]
            
            assert len( batch_li_li_edusegment) == len(batch_li_dict_utt):
             
            for dict_ , li_edusegments in zip(batch_li_dict_utt, batch_li_li_edusegment ):
                dict_['li_edus'] = li_edusegments

            timer.end("EDU Segmentation")
            #endregion

            #region  Key phrase for Each EDU
            timer.start()
            with mp.Pool(mp_count) as pool:
                res = pool.map( _topic, _chunks(batch_li_li_edusegment, batch_process_size//mp_count) )

            batch_li_keyphrase = list( res ) 
            batch_li_keyphrase = sum(batch_li_keyphrase, [])

            for dict_ , li_keyphrases in zip(batch_li_dict_utt, batch_li_keyphrase ):
                dict_['li_keyphrases'] = li_keyphrases

            timer.end("Key Phrase Extraction")
            #endregion

            #region Saving Batches
            timer.start()

                # format = subreddit/convo_code
            _save_data(batch_li_dict_utt, dir_save_dataset, batches_completed, batch_process_size, title_only )

            dset_source = dset_source[batch_process_size:]
            batches_completed += 1
            timer.end("Saving")

            #end region    

        print(f"Finished at batch {batches_completed}")        
    


def edu_segmenter( li_dict_rsttext ):
    """[summary]

    Args:
        li_dict_rsttext ([type]): [list of records each containing text with rst code annotated]

    Returns:
        [type]: [description]
    """
    
    li_text = [ _dict['txt_preprocess'] for _dict in li_dict_rsttext ]

    # returns list of words for each utterance with edu tokens place between different edu segments
    li_textwedutoken = parser_wrapper3.main( json_li_li_utterances= json.dumps([li_text]), 
                                                skip_parsing=True, redirect_output=True)
    
    # corrects any formatting errors caused by the segmenter
    li_li_edus = edu_fixer( li_textwedutoken )

    # for each utterance, merge list of words into one text
    li_li_edus = [ [ ' '.join( edus ) for edus in li_edus ] for li_edus in li_li_edus ]
    
    return li_li_edus


# edu processing 
def split(sequence, sep):
    chunk = []
    for val in sequence:
        if val == sep:
            yield chunk
            chunk = []
        else:
            chunk.append(val)
    yield chunk
# edu processing 
def edu_fixer(li_textwedutoken ):
        
    li_li_edutext = [ list( split(text_wedutoken,"EDU_BREAK") )[:-1] for text_wedutoken in li_textwedutoken ]
    
    for li_edutext in li_li_edutext:
        for idx2,elem in enumerate(li_edutext):
            elem.reverse() #reversing list of words in an edu
            it = enumerate(elem)
            edu_len = len(elem) 
            elem_new =  [next(it)[1]+str_ if ( idx!=edu_len-1 and (str_[0] == "'" or str_ in ["n't", ".", "?", "!", "," ]) ) else str_ for idx,str_ in it]
            elem_new.reverse()

            li_edutext[idx2] = elem_new
    
    return li_li_edutext

def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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

def _topic(li_li_edusegment):
    
    li_li_edukp = []

    for i in range(len(li_li_edusegment)):

        li_edusegment = li_li_edusegment[i]

        #TODO: here evaluate different ways to extract 'key phrase' from short chunks
        li_textankkw = [ 'topic_textrank':_textrank_extractor(thread_utterance['txt_preproc'])}
                for thread_utterance in li_thread_utterances]
        
        li_li_edukp.append( li_rakekw_textankkw )
        
    return li_li_edukp


def _textrank_extractor(str_utterance, lowest_score=0.0):
    # Add the below to docker file



        extractor = pke.unsupervised.TextRank()
        extractor.load_document(input=str_utterance, language='en')
        
        pos = {'NOUN', 'PROPN', 'ADJ', 'VERB','AUX','ADP','NUM','SYM'}
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(window=4,
                                  pos=pos, normalized=True
                                  )
        
    # 5. get the 10-highest scored candidates as keyphrases
    li_kp_score = extractor.get_n_best(n=10)

    # converting from list of tuples to list of list
    li_kp_score = [ list(kp_score) for kp_score in li_kp_score]

    if len(li_ranked_kws) == 0:
        li_ranked_kws = [["",0.0]]

    return li_ranked_kws
        
def _save_data(li_dict_utt, dir_save_dataset, last_batch_operated_on=0,
                batch_process_size=120, title_only=False):
    
    # Split list of utterances by the subreddit name
    # Then for each sublist
        # Get directory save name
        # Get the last saved csv file in directory
        # (fn-format = file_number_utterances in file )
            # then append more lines
            
    # Grouping utterances by the subreddit 
    grouped_li_utterances = [ ( k, list(g)) for k,g in itertools.groupby(li_dict_utt, lambda _dict: _dict['subreddit'] ) ]
        #a list of tuples; elem0: subreddit name elem1: list of convos for that subreddit
    
    for subreddit, _li_utterances in grouped_li_utterances:
        _li_utterances = [ { str(k):json.dumps(v) for k,v in dict_thread.items() } for dict_thread in _li_utterances ]

        if title_only == True:
            subreddit = subreddit+ "_title_only"
        
            
        subreddit_dir = utils_nlg.get_path( os.path.join(dir_save_dataset,subreddit), _dir=True  )  
               
        lock_path = os.path.join(subreddit_dir,"lock") 
        lock = FileLock(lock_path, timeout=60)

        with lock:
            #unlimited batch_siz
            files_ = [ fp for fp in os.listdir(subreddit_dir) if os.path.split(fp)[1] != "lock" ]
            if len(files_)>0:
                fn = files_[0]
            else:
                fn = "0000_0000000000"           
                with open( os.path.join(subreddit_dir,fn),"a+",newline=None,encoding='utf-8') as _f:
                    dict_writer = csv.DictWriter(_f,fieldnames=list(_li_utterances[0].keys() ) )
                    dict_writer.writeheader()
                    pass
            
            curr_len = int(fn[-10:])
            new_len = curr_len + len(_li_utterances)

            old_fp = os.path.join(subreddit_dir,fn)
            new_fp = os.path.join(subreddit_dir,f"{fn[:4]}_{new_len:010d}")
            
            keys = _li_utterances[0].keys()


            df_ = pd.read_csv(old_fp)
            df_ = df_.append( _li_utterances, ignore_index=True, sort=False)
            df_.to_csv( new_fp, index=False)

            if os.path.exists(old_fp) and old_fp!=new_fp:
                os.remove(old_fp)
            

            # Updating record of last batch operated on for each subreddit
            new_record = { 'batch_process_size':batch_process_size, 'last_batch':last_batch_operated_on }
            if os.path.exists( os.path.join(dir_save_dataset,'last_batch_record') ):
                df_records = pd.read_csv( os.path.join(dir_save_dataset,'last_batch_record'), index_col = "subreddit" )
            else:
                df_records = pd.DataFrame( columns=['last_batch','batch_process_size'] )
                df_records.index.names = ['subreddit']

            #df_records = df_records.append(new_record, ignore_index=True)

            for k,v in new_record.items():
                df_records.loc[ subreddit, [k] ] =  v

            df_records.to_csv( os.path.join(dir_save_dataset,'last_batch_record'), index_label='subreddit' )

          
class Timer():
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    #def __call__(self,_segment_name=None):
    def start(self):
        
        self.start_time = time.time()
        self.timing = True
        
    def end(self, segment_name):
        self.end_time = time.time()
        time_taken = self.end_time - self.start_time
        print(f"\t{segment_name} segment: {time_taken} secs")
            
if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False) 
    
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)

    parser.add_argument('--danet_vname', default="DaNet_v009",
        help="Version name of the DaNet model to use for dialogue act classifier ",
        type=str )
    
    parser.add_argument('-bps','--batch_process_size', default=120,
        help='',type=int)        
   

    parser.add_argument('--rst_method', default="feng-hirst",
        choices=['feng-hirst','akanni-rst'], type=str)

    parser.add_argument('--mp_count', default=6,
        type=int)
    
    parser.add_argument('-sb','--start_batch', default=0, type=int, help="batch of data to start from. Pass \
                                    in -1 for this batch to be autodetermined from records" )

    parser.add_argument('-eb','--end_batch', default=0, type=int, help="Final batch to finish on. Set to 0 to run until end")

    parser.add_argument('-ad','--annotate_da',default=True, type=lambda x: bool(int(x)) )


    parser.add_argument('-ar','--annotate_rst',default=True, type=lambda x: bool(int(x)) ) 

    parser.add_argument('-at','--annotate_topic',default=True, type=lambda x: bool(int(x)) )

    parser.add_argument('-rdv','--reddit_dataset_version',default='small', type=str, choices=[
                                                                        "small",
                                                                        "CasualConversation",
                                                                        "relationship_advice",
                                                                        "interestingasfuck",
                                                                        "science",

                                                                        "changemyview",
                                                                        "PoliticalDiscussion",
                                                                        "DebateReligion",
                                                                        "PersonalFinance",
                                                                        "TrueGaming",
                                                                        "Relationships",

                                                                        "WritingPrompts",

                                                                        "Music",
                                                                        "worldnews",
                                                                        "todayilearned",
                                                                        "movies",
                                                                        "Showerthoughts",
                                                                        "askscience",
                                                                        "LifeProTips",
                                                                        "explainlikeimfive",
                                                                        "DIY",
                                                                        "sports",
                                                                        "anime",
                                                                        "IamA"])
    parser.add_argument('-to','--title_only', default=False, type=lambda x: bool(int(x)) )

    args = parser.parse_args()
    
    dict_args = vars(args)

    completed = False
    
    while completed == False:
        try:
            main( **dict_args )
            completed = True
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            dict_args['start_batch'] = batches_completed + 1
            
        finally :
            # cmd = "docker stop $(docker ps -aq) > /dev/null 2>&1 & docker rm $(docker ps -aq) > /dev/null 2>&1 & docker rmi $(docker images -a -q) > /dev/null 2>&1"
            
            # cmd = docker stop $(docker ps -aq) & docker rm $(docker ps -aq) & yes | docker image prune  & docker rmi $(docker images -a -q)
            # os.system(cmd)
            # time.sleep(3)
            # os.system(cmd)
            # time.sleep(3)
            pass


#python3 data_setup.py -bps 60 -ad 0 -rdv CasualConversation -sb -1 --mp_count 6 

# python3 data_setup.py -bps 60 -ad 0 -rdv changemyview -sb 0 --mp_count 2
#python3 data_setup.py -bps 60 -ad 0 -rdv PoliticalDiscussion -sb 0 --mp_count 2

# python3 data_setup.py -bps 10 -ad 0 -rdv small -sb -1 --mp_count 6 

# python3 data_setup.py -bps 10 -ad 0 -rdv WritingPrompts -sb 1 --mp_count 6 -to 1
