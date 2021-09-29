#This script removes specific datums
# these datums are ones wherein the rst_parser ignores specific chunks of text. This is ususally do to excessively bad grammar, (lack of punctuation),
# The rst parser during preprocessing ( heuristic_sentence_splitting line 151 prepprocesser2.py ) ignores specific lines from the text if it can't break it into a sentence

import sys, os

import traceback
import string 
import math
from difflib import SequenceMatcher
from operator import itemgetter


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from itertools import groupby

import argparse
import utils_nlg

import math
import itertools
import pandas as pd
import contextlib
import glob
from collections import defaultdict

import copy 
import json 
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

import csv

import time

from filelock import Timeout, FileLock

import multiprocessing as mp

import ujson

#batches_completed = 0
batches_completed = {}
subreddit = None

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
from DockerImages.feng_hirst_rst_parser.src.parse2 import DiscourseParser

from data_setup import _tree_to_rst_code, _parse_trees
from utils_nlg_v3 import RstTokenizerMixin 

# with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
#     rst_parser = DiscourseParser(verbose=False, skip_parsing=True,
#                     global_features=True)

# Only take dsets with 5 or more RST chunks since we want to learn transitions between chunks of 3 EDUs

def main( batch_process_size=20,
            mp_count=4,
            resume_progress=False,
            subset_no=0,
                min_rst_len = 1,
            **kwargs):
    """[summary]

        Args:
        batch_process_size (int, optional): [description]. Defaults to 20.
        subreddit_names (str, optional): List of subreddit names to filter through
    """
    orig_batch_process_size = copy.deepcopy( batch_process_size )
    #region  Setup 
    dict_subredditnames_sets = {
                #batch warwick desktop
        1: ["DebateReligion","AdviceAnimals","AmItheAsshole","Android","anime","apple","AskMen","atheism","australia","aww","baseball","Bitcoin","books","buildapc","business","canada","cars"],
                #batch dullducks
        2:["fantasyfootball","Fitness","Frugal","funny","Games","gaming","gifs","gonewild","Guildwars2","guns","hiphopheads","IAmA","last_batch_record","leagueoflegends","Libertarian","LifeProTips","magicTCG","MakeupAddiction","malefashionadvice","Marvel","MensRights","Minecraft","MMA","motorcycles","MovieDetails","movies","Music","Naruto","nba","news","nfl","NoFap","offbeat","OkCupid","photography","pics","pokemon","pokemontrades","POLITIC","PoliticalDiscussion","politics","programming","Random_Acts_Of_Amazon","relationship_advice","relationships","rupaulsdragrace","science","sex","ShingekiNoKyojin","singapore","skyrim","soccer","SquaredCircle","starcraft","technology","techsupport","teenagers","tf2","tifu","todayilearned","travel","trees","TwoXChromosomes","unitedkingdom","videos","worldnews","wow","WritingPrompts","WTF"],
                #batch enigma
        3: ["Diablo","DotA2","Drugs","Economics","electronic_cigarette","explainlikeimfive","AskReddit","askscience","AskWomen","asoiaf","CasualConversation","CFB","changemyview","Christianity","conspiracy","cringe","cringepics","dayz"]
        # 1:["australia"]
    }

    if subset_no == 0:
        subreddit_names = sum( list( dict_subredditnames_sets.values() ) , [] )
    else:
        subreddit_names = dict_subredditnames_sets[subset_no]
    
    #Creating Save directory
    dir_save_dataset = utils_nlg.get_path("./dataset_v3_2/",_dir=True)
    
    # setting up source subreddit data
    dirs_rst_conv = "./dataset_v3_1/"
    li_subreddit_names = list( filter( lambda name: name!="last_batch_record", os.listdir( dirs_rst_conv ) ) )

    dict_subreddit_fp = {subreddit: [fp for fp in glob.glob(os.path.join(dirs_rst_conv,subreddit,"*")) if os.path.split(fp)[-1]!="lock"  ]  for subreddit in  li_subreddit_names }

        #filtering subreddits
    if len(subreddit_names)==0:
        li_subreddit_fp = [ (subreddit,[ fp for fp in li_fp if "dict_lens" not in fp]) for subreddit,li_fp in  dict_subreddit_fp.items() ]
    else:
        li_subreddit_fp = [ (subreddit,[ fp for fp in li_fp if "dict_lens" not in fp]) for subreddit,li_fp in  dict_subreddit_fp.items() if (subreddit in subreddit_names and "dict_lens" not in li_fp) ]

    global subreddit
    global batches_completed
    for subreddit, li_fp in li_subreddit_fp:
        
        print(f"\nOperating on Subreddit: {subreddit}. {li_subreddit_fp.index((subreddit,li_fp))} of {len(li_subreddit_fp)}")
        
        #Should only be one file in li_fp
        fp = li_fp[0]

        dset_source = pd.read_csv( fp, usecols=['rst','txt_preproc','subreddit','subreddit','dict_pos_edu','li_pos_kp'], )
        
        
        # region Optionally auto-resuming from last completed batch
        if resume_progress == True:

            # checking if df_records exists and if a column for this subreddit exists
            fn = os.path.join(dir_save_dataset,'last_batch_record')
            _bool_file_check = os.path.exists( fn )


            if not _bool_file_check: #if file does not exist
                batches_completed[subreddit]= 0
                batch_process_size = orig_batch_process_size
                total_batch_count = math.ceil(len(dset_source)/batch_process_size)
                print("\tUser choose auto-resume from last recorded batch.\
                But no last records exists, so initialising from batch 0")
                
            else: #if file does exists
                df_records = pd.read_csv( fn, index_col = "subreddit" )
                _bool_record_check = subreddit in df_records.index.tolist()

                if not _bool_record_check:
                    batches_completed[subreddit] = 0
                    batch_process_size = orig_batch_process_size
                    total_batch_count = math.ceil(len(dset_source)/batch_process_size)

                
                else:
                    batches_completed[subreddit] = int( df_records.loc[ subreddit, 'last_batch' ] ) + 1
                    batch_process_size = int( df_records.loc[subreddit, 'batch_process_size'] )
                    total_batch_count = math.ceil(len(dset_source)/batch_process_size)

                    if batches_completed[subreddit] >= total_batch_count:
                        continue
        
                print(f"Skipping forwards {batches_completed[subreddit]} batches")
                dset_source = dset_source[batches_completed[subreddit]*batch_process_size:]

        else:
            total_batch_count = math.ceil(len(dset_source)/batch_process_size)
            batches_completed[subreddit] = 0
        # endregion

        timer = Timer()

        #region operating in batches
        
        while len(dset_source) > 0 :
            
            batch_li_dict_utt =  dset_source.iloc[:batch_process_size].to_dict('records')
            
            # decoding json encoded text and # removing entries which don't have have rst of length over 1 
            
            for idx in range(len(batch_li_dict_utt)-1, -1, -1):
                
                rst_ = ujson.loads(batch_li_dict_utt[idx]['rst'])  
                txt_prepoc = ujson.loads(batch_li_dict_utt[idx]['txt_preproc'])
                srdt  = ujson.loads(batch_li_dict_utt[idx]['subreddit'])
                dict_pos_edu = json.loads( batch_li_dict_utt[idx]['dict_pos_edu'] )
                li_pos_kp = json.loads( batch_li_dict_utt[idx]['li_pos_kp'] )


                batch_li_dict_utt[idx] =  {'rst':rst_, 'txt_preproc':txt_prepoc,
                        'subreddit': srdt, 'dict_pos_edu':dict_pos_edu,
                        'li_pos_kp':li_pos_kp } 

            
            print(f"\n\tOperating on batch {batches_completed[subreddit]+1} of {total_batch_count}")

            
            timer.start()
            
            with mp.Pool(mp_count) as pool:
                
                cs =  max( 3, math.ceil( batch_process_size / (mp_count) ) )
                
                res = pool.imap( edu_segmenter, _chunks(batch_li_dict_utt, cs), cs )

                res = pool.imap( dict_pos_edu_remover, res)

                # res = pool.imap( li_edu_adder, res)

                # res = pool.imap( position_edus, res)

                # res = pool.imap( position_kp, res)
                              
                batch_li_dict_utt = list(res)
                
                batch_li_dict_utt = sum(batch_li_dict_utt, [])

            timer.end("\t\tEDU parsing, RST correction and EDU position labelling")

            #region Saving Batches
            timer.start()

            # format = subreddit/convo_code
            if len(batch_li_dict_utt) > 0:
                _save_data(batch_li_dict_utt, dir_save_dataset, batches_completed[subreddit], batch_process_size )

            dset_source = dset_source[batch_process_size:]
            batches_completed[subreddit] += 1
            timer.end("Saving")

            #end region    
        
        # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        #     rst_parser.unload()

        print(f"Finished at batch {batches_completed[subreddit] }")        

def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# region edu processing     
def edu_segmenter(li_dict_rsttext):
    """[summary]

        Args:
            li_dict_rsttext ([type]): [list of records each containing text with rst code annotated]

        Returns:
            [type]: [description]
    """
    if len(li_dict_rsttext) == 0:
        return li_dict_rsttext

    # in data_setup_v3 the edus in dict_pos_edu were in the correct order. However, the pos were incorrectly ordered.
    for idx in range(len(li_dict_rsttext)):
        li_pos_edu = [ [key, val] for key,val in li_dict_rsttext[idx]['dict_pos_edu'].items() ]
        li_pos_edu = sorted( li_pos_edu, key= lambda item: RstTokenizerMixin.edukp_pos_sort_function(int(item[0]) ) )
        li_dict_rsttext[idx]['li_edus']  = [ item[1] for item in li_pos_edu]

    return li_dict_rsttext


def dict_pos_edu_remover(li_dict_rsttext):
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

        if len_edu_text<min( len_text*0.9, len_text-1) or len_edu_text>max( 1.1*len_text, 1+len_text):
            
            li_dict_rsttext.pop(idx)

            # #pop dict_pos_edu and li_edus
            # li_dict_rsttext[idx].pop('dict_pos_edu')
            # li_dict_rsttext[idx].pop('li_edus')
            # li_dict_rsttext[idx].pop('rst')
            # # raise NotImplementedError("need to replace the rst tree for these cases since the text was truncated") 


            #making dict_pos_edu and li_edus again

    return li_dict_rsttext

def li_edu_adder( li_dict_rsttext):
    
    if len(li_dict_rsttext) == 0:
        return li_dict_rsttext
    
    # the subset of texts that have to be fixed
    li_idx, li_text = zip( *[ [ idx, _dict['txt_preproc'] ] for idx, _dict in enumerate(li_dict_rsttext) if 'li_edus' not in _dict ] )

    if len(li_text) == 0:
        return li_dict_rsttext

    # returns list of words for each utterance with edu tokens place between different edu segments
    # li_textwedutoken = parser_wrapper3.main( json_li_li_utterances= json.dumps([li_text]), 
    #                                             skip_parsing=True, redirect_output=True)

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        rst_parser = DiscourseParser(verbose=False, skip_parsing=True,
                        global_features=True)

        li_textwedutoken = rst_parser.parse_li_utterances(  li_text )

        rst_parser.unload()

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
    
    #outer re.sub does that space inbetween brakcets/
    li_li_edus = [ [ re.sub('\[\s*(.*?)\s*\]', r'[\1]', re.sub( pattern_punctuation_space, r"'", edutxt)) for edutxt in li_edutext] for li_edutext in  li_li_edus ]
    for idx in range(len(li_li_edus)):
        li_edus = li_li_edus[idx]
        li_edus =  [ re.sub(pattern_brackets_rm_space, r'(\1)', edu_text) for edu_text in li_edus ]
        li_edus =  [ re.sub(pattern_punctuation_space, r'\1', edu_text) for edu_text in li_edus ]

    #debugging check there aren't spaces between puntuation and word before
    for idx_, idx in enumerate(li_idx):
        li_dict_rsttext[idx]['li_edus'] = li_li_edus[idx_]

    return li_dict_rsttext

def edu_fixer(li_textwedutoken):
        
    li_li_edutext = [ list( split(text_wedutoken,"EDU_BREAK") )[:-1] for text_wedutoken in li_textwedutoken ]
    
    for li_edutext in li_li_edutext:
        for idx2,elem in enumerate(li_edutext):
            elem.reverse() #reversing list of words in an edu
            it = enumerate(elem)
            edu_len = len(elem) 
            elem_new =  [next(it)[1]+str_ if ( idx!=edu_len-1 and (str_[0] == "'" or str_ in ["n't", ".", "?", "!", ",", "[", "]" ]) ) else str_ for idx,str_ in it]
            elem_new.reverse()

            li_edutext[idx2] = elem_new
    
    return li_li_edutext


def split(sequence, sep):
    chunk = []
    for val in sequence:
        if val == sep:
            yield chunk
            chunk = []
        else:
            chunk.append(val)
    yield chunk



#endregion

def position_edus(li_dict_rsttext):
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

def find_child_edus(pos_parentnode, li_rst_pos):
        #returns the pos of any child elements of a parent node(rst) that are edus
               
        li_child_pos = [2*pos_parentnode+1, 2*pos_parentnode+2 ]

        li_child_edu_pos = [ pos for pos in li_child_pos if pos not in li_rst_pos]

        return li_child_edu_pos 

def position_kp(li_dict_rsttext):
    # For Each Keyphrase we now add information which edu posiiton it occurs in on the RST Tree
    if len(li_dict_rsttext) == 0:
        return li_dict_rsttext

    for idx in range(len(li_dict_rsttext)):
        
        key_phrases = [ item[1] for item in li_dict_rsttext[idx]['li_pos_kp'] ]  # a list of lists. each sublists holds pos and keyphrase
        li_dict_rsttext[idx].pop('li_pos_kp')
        li_dict_rsttext[idx]['li_pos_kp']= []
        dict_pos_edu = li_dict_rsttext[idx]['dict_pos_edu']
        
        for kp in key_phrases:          

            # kp can spans two different EDUs. So finding length, in words, of longest common substring
            li_pos_coveragecount = []
            for pos, edu in dict_pos_edu.items():
                kp_split = kp.split()
                edu_split = [ w for w in edu.translate(str.maketrans('', '', string.punctuation)).split() if w!= " "]

                match = SequenceMatcher(None, kp_split, edu_split).find_longest_match(0, len(kp_split) ,0, len(edu_split) )
                li_pos_coveragecount.append( [pos, match.size] )
            
            kp_pos, coverage_count = max( li_pos_coveragecount, key=itemgetter(1))
            
            li_dict_rsttext[idx]['li_pos_kp'].append( [kp_pos,kp] ) 
            
    return li_dict_rsttext

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
                fn = "0000000000"           
                with open( os.path.join(subreddit_dir,fn),"a+",newline=None,encoding='utf-8') as _f:
                    dict_writer = csv.DictWriter(_f,fieldnames=list(_li_dict_utt[0].keys() ) )
                    dict_writer.writeheader()
                    pass
            
            curr_len = int(fn[-10:])
            new_len = curr_len + len(_li_dict_utt)

            old_fp = os.path.join(subreddit_dir,fn)
            new_fp = os.path.join(subreddit_dir,f"{new_len:010d}")
            
            #df_ = pd.read_csv(old_fp)
            #df_ = df_.append( _li_dict_utt, ignore_index=True, sort=False)
            #df_.to_csv( new_fp, index=False)
            pd.DataFrame(_li_dict_utt).to_csv(old_fp, mode='a', header=False, index=False)
            os.rename(old_fp,new_fp)

            # if os.path.exists(old_fp) and old_fp!=new_fp:
            #     os.remove(old_fp)
                
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
    
    parser.add_argument('-bps','--batch_process_size', default=1600,
                             help='',type=int)        
   
    parser.add_argument('--mp_count', default=16, type=int)
    
    parser.add_argument('--subset_no', default=0, type=int)
    
    parser.add_argument('-rp','--resume_progress', default=True, type=lambda x: bool(int(eval(x))) if type(x)==str else bool(int(x)), 
                        help="whether or not to resume from last operated on file" )

    args = parser.parse_args()
    
    dict_args = vars(args)

    completed = False
    
    while completed == False:
        try:
            main( **dict_args )
            completed = True
        except Exception as e:

            print(subreddit)
            print(e)
            print(traceback.format_exc())

            dir_save_dataset = utils_nlg.get_path("./dataset_v3_2/",_dir=True)
            if os.path.exists(os.path.join(dir_save_dataset,'last_batch_record')) :
                df_records = pd.read_csv( os.path.join(dir_save_dataset,'last_batch_record'), index_col = "subreddit" )
                df_records.loc[ subreddit, ['last_batch'] ] =  batches_completed[subreddit] + 2
                df_records.to_csv( os.path.join(dir_save_dataset,'last_batch_record'), index_label='subreddit' )
            

            dict_args['resume_progress'] = True
            
        finally :
            # cmd = "docker stop $(docker ps -aq) > /dev/null 2>&1 & docker rm $(docker ps -aq) > /dev/null 2>&1 & docker rmi $(docker images -a -q) > /dev/null 2>&1"
            
            # cmd = docker stop $(docker ps -aq) & docker rm $(docker ps -aq) & yes | docker image prune  & docker rmi $(docker images -a -q)
            # os.system(cmd)
            # time.sleep(3)
            # os.system(cmd)
            # time.sleep(3)
            pass

# python3 data_setup_v3.py -bps 240 -rp 1  --mp_count 6  --subset_no 1
# python3 data_setup_v3.py -bps 320 -rp 1  --mp_count 6  --subset_no 2
# python3 data_setup_v3.py -bps 240 -rp 1  --mp_count 8  --subset_no 3
