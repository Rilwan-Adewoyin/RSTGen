import sys, os

import traceback


import math
import numpy
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from itertools import groupby

import argparse
import utils_nlg
import random

import math
import itertools
import pandas as pd
import nltk
nltk.download('stopwords')
import glob

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

import regex as re
import multiprocessing as mp
import distutils

import html
import ujson

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

for path_ in modules_paths:
    if path_ not in sys.path:
        sys.path.append(path_)

from DockerImages.feng_hirst_rst_parser.src import parser_wrapper3

old_merge_environment_settings = requests.Session.merge_environment_settings

def main(   batch_process_size=20,
            mp_count=4,
            resume_progress=False,
            **kwargs):
    """[summary]

    Args:
        batch_process_size (int, optional): [description]. Defaults to 20.
    """
    
    #region  Setup    
    # setting up model for Da prediction
    
    #Creating Save directory
    dir_save_dataset = utils_nlg.get_path("./dataset_keyphrase/",_dir=True)
    
    # setting up subreddit data
    
    #with no_ssl_verification():
    dirs_rst_conv = "./dataset_v2/reddit_large_annotated/"
    li_subreddit_names = list( filter( lambda name: name!="last_batch_record", os.listdir( dirs_rst_conv ) ) )

    dict_subreddit_fp = {subreddit: [fp for fp in glob.glob(os.path.join(dirs_rst_conv,subreddit,"*")) if os.path.split(fp)[-1]!="lock"  ]  for subreddit in  li_subreddit_names }

    li_subreddit_fp = [ (subreddit,li_fp) for subreddit,li_fp in  dict_subreddit_fp.items() ]

    #for subreddit, li_fp in dict_subreddit_fp.items():
    for subreddit, li_fp in li_subreddit_fp:
        
        print(f"\nOperating on Subreddit: {subreddit}. {li_subreddit_fp.index((subreddit,li_fp))} of {len(li_subreddit_fp)}")
        
        #Should only be one file in li_fp
        #assert len(li_fp) == 1
        fp = li_fp[0]

        dset_source = pd.read_csv( fp, usecols=['rst','txt_preproc','subreddit'] )
        
        total_batch_count = math.ceil(len(dset_source)/batch_process_size)

        # Optionally auto-resuming from last completed batch
        if resume_progress == True:

            # checking if df_records exists and if a column for this subreddit exists
            fn = os.path.join(dir_save_dataset,'last_batch_record')
            _bool_file_check = os.path.exists( fn )

            auto_fnd_failed = lambda : print("User choose auto-resume from last recorded batch.\
                But no last records exists, so initialising from batch 0")

            if not _bool_file_check: #s if file exist
                start_batch = 0
                auto_fnd_failed()
                
            else: #if file does not exists
                df_records = pd.read_csv( fn, index_col = "subreddit" )
                _bool_record_check = subreddit in df_records.index.tolist()

                if not _bool_record_check:
                    start_batch = 0
                
                else:
                    start_batch = int( df_records.loc[ subreddit, 'last_batch' ] ) + 1
                    batch_process_size = int( df_records.loc[subreddit, 'batch_process_size'] )
        
                print(f"Skipping forwards {start_batch} batches")
                dset_source = dset_source[start_batch*batch_process_size:]

        else:
            start_batch = 0
        # endregion

        timer = Timer()

        #region operating in batches
        global batches_completed
        batches_completed = start_batch

        while len(dset_source) > 0 :
            
            batch_li_dict_utt =  dset_source.iloc[:batch_process_size].to_dict('records')
            
            # decoding json encoded text and # removing entries which don't have have rst of length over 1 
            _batch_li_dict_utt = []
            for idx in range(len(batch_li_dict_utt)):
                
                rst_ = ujson.loads(batch_li_dict_utt[idx]['rst'])  
                
                if len(rst_)>1:
                    # batch_li_dict_utt[idx]['rst'] = rst_
                    # batch_li_dict_utt[idx]['txt_preproc'] = ujson.loads(batch_li_dict_utt[idx]['txt_preproc'])                 
                    # batch_li_dict_utt[idx]['subreddit'] = ujson.loads(batch_li_dict_utt[idx]['subreddit'])  

                    _batch_li_dict_utt.append( {'rst':rst_, 'txt_preproc':ujson.loads(batch_li_dict_utt[idx]['txt_preproc']),
                         'subreddit':ujson.loads(batch_li_dict_utt[idx]['subreddit']) } )
                else:
                    pass
            
            batch_li_dict_utt =  _batch_li_dict_utt

                
            print(f"\n\tOperating on batch {batches_completed+1} of {total_batch_count}")

            #region mp.imap EDU and Key Phrase Extraction
            timer.start()
            
            with mp.Pool(mp_count) as pool:
                
                #res = pool.map(edu_segmenter, _chunks(batch_li_dict_utt, math.ceil(batch_process_size/mp_count)), chunksize=1 )

                cs =  math.ceil(batch_process_size/ (mp_count*3) )
                res = pool.imap(edu_segmenter, _chunks(batch_li_dict_utt, cs) )
                                        
                res = pool.imap( _key_phrase, res )

                try:
                    batch_li_dict_posname_likpscore, batch_li_li_edusegment = zip(*list(res))
                
                except TypeError as e:
                    batch_li_dict_posname_likpscore = []
                    batch_li_li_edusegment = []

            batch_li_dict_posname_likpscore = sum(batch_li_dict_posname_likpscore, [])
            batch_li_li_edusegment = sum(batch_li_li_edusegment, [])

            timer.end("\t\tEDU parsing and Key Phrase Extraction")

            for dict_ , li_edusegments, li_kp_score in zip(batch_li_dict_utt, batch_li_li_edusegment, batch_li_dict_posname_likpscore ):
                dict_['li_edus'] = li_edusegments
                dict_['li_dict_posname_likpscore'] = li_kp_score
            #endregion

            #region Saving Batches
            timer.start()

            # format = subreddit/convo_code
            _save_data(batch_li_dict_utt, dir_save_dataset, batches_completed, batch_process_size )

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
    
    li_text = [ _dict['txt_preproc'] for _dict in li_dict_rsttext ]

    # returns list of words for each utterance with edu tokens place between different edu segments
    li_textwedutoken = parser_wrapper3.main( json_li_li_utterances= json.dumps([li_text]), 
                                                skip_parsing=True, redirect_output=True)
    
    # corrects any formatting errors caused by the segmenter
    li_li_edus = edu_fixer( li_textwedutoken )

    # for each utterance, merge list of words into one text
    li_li_edus = [ [ ' '.join( edus ) for edus in li_edus ] for li_edus in li_li_edus ]
    
    li_li_edus = [ [edutxt.replace(" n't", "n't").replace(" / ", "/").replace(" '", "'") if edutxt not in origtext else edutxt for edutxt in li_edutext ] for li_edutext, origtext in zip( li_li_edus, li_text) ]

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


def _key_phrase(li_li_edusegment):
    # input is a list of different utterances, each utterance is represented as a list of its edus
    # Here we extract the key phrase of an EDU
        # Our system has four possible sugesstions for each EDU s1,s2,s3,s4
        # s_i is accepted if it is below a certain length relative to the original text, otherwise we consider s_(i+1)
    
    #making spacy model #spedding up textrank processing
    spacy_kwargs = {'disable': ['ner', 'textcat', 'parser']}
    spacy_model = spacy.load('en_core_web_sm', **spacy_kwargs)
    spacy_model.add_pipe('sentencizer')
    extractor = pke.unsupervised.TextRank()

    # Defining Pos tags for each EDU. We prefer to use results from pos3 to pos0 in that order
    pos0 = {'NOUN',  'PROPN' ,'ADJ','VERB','PRON', 'SYM','ADV','DET',        'PART','X'}
    pos1 = {'NOUN',  'PROPN' ,'ADJ', 'VERB',     'SYM','ADP','AUX',        'PART','X'}
    dict_pos = { 'pos1': pos1, 'pos0':pos0}

    li_dict_posname_likpscore = []

    for i in range(len(li_li_edusegment)):

        li_edusegment = li_li_edusegment[i]

        dict_posname_likpscore = [ _key_phrase_extractor(edusegment, extractor=extractor, spacy_model=spacy_model, dict_pos=dict_pos)             
                            for edusegment in li_edusegment ]
        
        
        li_dict_posname_likpscore.append( dict_posname_likpscore )
        
    return li_dict_posname_likpscore, li_li_edusegment


def _key_phrase_extractor(str_utterance, extractor, spacy_model=None, dict_pos=[]):
    #returns the first key phrase that is under half the original text length
    
    dict_posname_likpscore = {}

    for pos_name, pos in dict_pos.items():

        # extracting kp and score
        extractor.reset()
        extractor.load_document(input=str_utterance, language='en',normalization=None, spacy_model=spacy_model)
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(window=2, normalized=True, pos=pos)
        
        li_kp_score = extractor.get_n_best(n=2, redundancy_removal=False)
        
        li_kp_score = [ [kp_score[0].replace(" n't", "n't").replace(" / ", "/").replace(" '", "'") if kp_score[0] not in str_utterance else kp_score[0], kp_score[1] ] for kp_score in li_kp_score ]

        # converting from list of tuples to list of list
        #li_kp_score = [ list(kp_score) for kp_score in li_kp_score]
        
        if len(li_kp_score) == 0:
            li_kp_score = [["",0.0]]
        
                
        dict_posname_likpscore[pos_name] = li_kp_score
    
    return dict_posname_likpscore

def _save_data(li_dict_utt, dir_save_dataset, last_batch_operated_on=0,
                batch_process_size=120):
    
    # Split list of utterances by the subreddit name
    # Then for each sublist
        # Get directory save name
        # Get the last saved csv file in directory
        # (fn-format = file_number_utterances in file )
            # then append more lines
            
    # Grouping utterances by the subreddit 
    grouped_li_dict_utt = [ ( k, list(g)) for k,g in itertools.groupby(li_dict_utt, lambda _dict: _dict['subreddit'] ) ]
        #a list of tuples; elem0: subreddit name elem1: list of convos for that subreddit
    
    for subreddit, _li_dict_utt in grouped_li_dict_utt:
        _li_dict_utt = [ { str(k):json.dumps(v) for k,v in dict_thread.items() } for dict_thread in _li_dict_utt ]
                   
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
                    dict_writer = csv.DictWriter(_f,fieldnames=list(_li_dict_utt[0].keys() ) )
                    dict_writer.writeheader()
                    pass
            
            curr_len = int(fn[-10:])
            new_len = curr_len + len(_li_dict_utt)

            old_fp = os.path.join(subreddit_dir,fn)
            new_fp = os.path.join(subreddit_dir,f"{fn[:4]}_{new_len:010d}")
            
            df_ = pd.read_csv(old_fp)
            df_ = df_.append( _li_dict_utt, ignore_index=True, sort=False)
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
    
    parser.add_argument('-bps','--batch_process_size', default=12,
                             help='',type=int)        
   
    parser.add_argument('--mp_count', default=3, type=int)
    
    parser.add_argument('-rp','--resume_progress', default=0, type=lambda x: bool(int(x)), 
                        help="whether or not to resume from last operated on file" )


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
            dict_args['resume_progress'] = True
            
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