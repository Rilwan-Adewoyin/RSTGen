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

from filelock import Timeout, FileLock

import regex as re
import multiprocessing as mp
import distutils

import html

from unidecode import unidecode
batches_completed = 0
dict_args = {}


import warnings
import contextlib

import spacy
#import contextualSpellCheck
# nlp = spacy.load("en_core_web_sm") 
# contextualSpellCheck.add_to_pipe(nlp)
# components = ["parser","ner",'contextual spellchecker']
# removing non needed componenets
# for name in nlp.pipe_names:
#     if name not in components:
#         nlp.remove_pipe(name)

import ujson

def main(   batch_process_size=20,
            mp_count=5,
            start_batch=-1,
            end_batch = 0,
            **kwargs):
    """[summary]

    Args:
        danet_vname ([type]): [description]
        batch_process_size (int, optional): [description]. Defaults to 200.
    """
    
    #region  Setup    
    
    #Creating Save directory
    dir_save_dataset = utils_nlg.get_path("./dataset_atomic2020/",_dir=True)
    
    dset_names = [ "train", "dev", "test" ]
    dict_dsetname_fp = {
        dset_name: os.path.join( dir_save_dataset,dset_name+".tsv" ) for dset_name in dset_names
    }

    for dsetname, fp in dict_dsetname_fp.items():
        print(f"Preprocessing {dsetname} dataset")

        df_data = pd.read_csv( fp, names=['head','relation','tail'], sep="\t" )

        total_batch_count = math.ceil(len(df_data)/batch_process_size)

        # Optionally auto-resuming from last completed batch
        if start_batch == -1:

            # checking if df_records exists and if a column for this dataset exists
            fn = os.path.join(dir_save_dataset,'last_batch_record')
            _bool_file_check = os.path.exists( fn )
            
            if not _bool_file_check: #s if file does exist
                start_batch = 0
                print("User choose auto-resume from last recorded batch.\
                        But no last records file exists, so initialising from batch 0")
                
            else: #if file does exists
                df_records = pd.read_csv( fn, index='dsetname' )
                
                _bool_record_check = dsetname in df_records.index.tolist()

                if not _bool_record_check:
                    print(f"User choose auto-resume from last recorded batch.\
                            Last records file exists, but there is not record for {dsetname}.tsv, \
                            so initialising from 0")
                    start_batch = 0
                
                else:
                    start_batch = int( df_records.loc[ dsetname, 'last_batch' ] ) + 1
                    batch_process_size = int( df_records.loc[dsetname, 'batch_process_size'] )
            
        # selecting sub batch to operate on if applicable
        if end_batch != 0:
            df_data = df_data.iloc[ : end_batch*batch_process_size] 
        if start_batch != 0:
            df_data = df_data.iloc[ start_batch*batch_process_size: ]

        # endregion
        timer = Timer()

        #region operating on batches
        global batches_completed
        batches_completed = start_batch

        while len(df_data) > 0 :
            
            batch_li_kgtriple =  df_data.iloc[:batch_process_size].to_dict('records')
            
            print(f"\nOperating on batch {batches_completed} of {total_batch_count}")

            #region Filtering out invalid datums
            timer.start()
            
            with mp.Pool(mp_count) as pool:
                res = pool.map( kgtriplet_filter, _chunks(batch_li_kgtriple, batch_process_size//mp_count )  )
            
            batch_li_kgtriple = list( res ) 
            batch_li_kgtriple = sum(batch_li_kgtriple, [])
                                     
            timer.end("Filtering triplets")
            #endregion

            #region  Editing the remaining datums
            timer.start()
            with mp.Pool(mp_count) as pool:
                res = pool.map( kgtriplet_fixer , _chunks(batch_li_kgtriple, batch_process_size//mp_count) )

            batch_li_kgtriple = list( res ) 
            batch_li_kgtriple = sum(batch_li_kgtriple, [])

            timer.end("Fixing remaining triplets")
            #endregion

            #region Saving Batches
            timer.start()

                # format = subreddit/convo_code
            if len(batch_li_kgtriple)>0:
                _save_data(batch_li_kgtriple, dir_save_dataset, dsetname , batches_completed, batch_process_size)

            df_data = df_data[batch_process_size:]
            batches_completed += 1
            timer.end("Saving")

            #end region    

        print(f"Completed {dset_name}: Finished at batch {batches_completed}")        
    

def kgtriplet_filter( li_kgtriple ):
    """[summary]

    Args:
        li_dict_rsttext ([type]): [list of records each containing text with rst code annotated]

    Returns:
        [type]: [description]
    """

    
    # Each filter check returns True for valid entries.
    # We remove values which return False

    bad_heads = ["PersonX makes PersonY's laws","PersonX makes PersonY's mom"]
    
    def check(head, rel, tail):
        
        # filter: tail less than two letters
        bool_2ltr = len(tail)>2
        if bool_2ltr == False:
            return False
        
        # filter: checking the value of tail
        bool_illegal_words = tail.lower() not in ["none"]
        if bool_illegal_words == False:
            return False
        
        # filter: remove triplet where tail only has punctuation :[check any is alphanumeric]
        bool_any_alphanum = any( i.isalnum() for i in iter(tail) )
        if bool_any_alphanum == False:
            return False
        
        # filter: remove lines that have _{1,5} in head/tail if the relation is not IsFilledBy
        bool_check_gap = (rel == "isFilledBy") or ("__" not in head + " " + tail) 
        if bool_check_gap == False:
            return False

        # filter: remove silly heads. These heads are only a problem for the relations that begin with o or x
        bool_check_good_head = (rel in ['isAfter', 'isBefore', 'HinderedBy']) or  (head not in bad_heads)
        if bool_check_good_head== False:
            return False
        
        return True


    for idx in range(len(li_kgtriple)-1,-1,-1):

        head = li_kgtriple[idx]['head']
        relation = li_kgtriple[idx]['relation']
        tail = li_kgtriple[idx]['tail']

        bool_check = check(head, relation, tail)

        if bool_check == False:
            li_kgtriple.pop(idx)
    
    return li_kgtriple


def kgtriplet_fixer(li_kgtriple):
    """[Removes triplets according to the following scheme]

    Args:
        li_kgtriple ([type]): [description]
    """
    def fullstop_remove(text):
        text = text[:-1] if text[-1] == "." else text 
        return text

    def insert_mask_token(text):
        text =  ' '.join( [ "<mask>" if "_" in word else word for word in text.split(' ')] )
        return text


    for idx in range(len(li_kgtriple)):

        head = li_kgtriple[idx]['head']
        relation = li_kgtriple[idx]['relation']
        tail = li_kgtriple[idx]['tail']

        #removing full stops
        head = fullstop_remove(head)
        relation = fullstop_remove(relation)
        tail = fullstop_remove(tail)

        #correcting capitlization schemes
        head = head.lower()
        relation = relation.lower()
        tail = tail.lower()
        
        #TODO: add a contextual spell check which may improve data quality
        # head = nlp(head)._.outcome_spellCheck
        # tail = nlp(tail)._.outcome_spellCheck

        # if relation is isFilledBy replacing _____ with <|mask|>
        if relation == "isFilledBy":
            head = insert_mask_token(head)
            tail = insert_mask_token(tail)


    return li_kgtriple


def split(sequence, sep):
    chunk = []
    for val in sequence:
        if val == sep:
            yield chunk
            chunk = []
        else:
            chunk.append(val)
    yield chunk

def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

        
def _save_data(li_kgtriple, dir_save_dataset, dsetname ,last_batch_operated_on=0,
                batch_process_size=120 ) :
    

    # Get directory save name
    # Get the last saved csv file in directory
    # (fn-format = file_number_utterances in file )
        # then append more lines
                
    _li_kgtriple = [ { key: ujson.dumps(value) for key,value in  kgtriple.items() }for kgtriple in li_kgtriple ]
    
    utils_nlg.get_path(os.path.join(dir_save_dataset,"locks"),_dir=True)
    lock_path = os.path.join(dir_save_dataset,"locks",f"{dsetname}_lock")

    lock = FileLock(lock_path, timeout=120)

    with lock:
        #Saving to a file that can have unlimited save size        
        fn = f"{dsetname}_v2.csv"
        fp =       os.path.join(dir_save_dataset, fn)   
        
        if not os.path.exists(fp):
            with open( fp ,"a+",newline=None,encoding='utf-8') as _f:
                dict_writer = csv.DictWriter(_f,fieldnames=list(_li_kgtriple[0].keys() ) )
                dict_writer.writeheader()
                pass
            
        df = pd.read_csv(fp)
        df = df.append( _li_kgtriple, ignore_index=True, sort=False)
        df.to_csv( fp, index=False)

        # Updating record of last batch operated on for each subreddit
        new_record = { 'batch_process_size':batch_process_size, 'last_batch':last_batch_operated_on }
        
        if os.path.exists( os.path.join(dir_save_dataset,'last_batch_record') ):
            df_records = pd.read_csv( os.path.join(dir_save_dataset,'last_batch_record'), index_col='dsetname' )
        else:
            df_records = pd.DataFrame( columns=['last_batch','batch_process_size' ] )
            df_records.index.name = "dsetname"
        

        for k,v in new_record.items():
            df_records.loc[ dsetname, [k] ] =  v

        df_records.to_csv( os.path.join(dir_save_dataset,'last_batch_record'), )

          
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
    
    parser.add_argument('-bps','--batch_process_size', default=120,
        help='',type=int)        
   
    parser.add_argument('--mp_count', default=1, type=int)
    
    parser.add_argument('-sb','--start_batch', default=-1, type=int, help="batch of data to start from. Pass \
                                    in -1 for this batch to be autodetermined from records" )

    parser.add_argument('-eb','--end_batch', default=0, type=int, help="Final batch to finish on. Set to 0 to run until end")
    
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
            time.sleep(5)
        finally:
            pass


#python3 data_setup_cskg.py -bps 60 -sb -1 --mp_count 6
