import sys, os

import traceback

import math
from types import MethodType
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
#nltk.download('stopwords')
import glob
from collections import defaultdict


import json 
import spacy
import en_core_web_sm
import pke
from pke.data_structures import Candidate

import regex as re
pattern_punctuation_space = re.compile(r'\s([?.!"](?:\s|$))')
pattern_capitalize_after_punct = re.compile(r"(\A\w)|"+                  # start of string
             "(?<!\.\w)([\.?!] )\w|"+     # after a ?/!/. and a space, 
                                          # but not after an acronym
             "\w(?:\.\w)|"+               # start/middle of acronym
             "(?<=\w\.)\w",               # end of acronym
             )
 

# sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
# #making spacy model #spedding up textrank processing
# spacy_kwargs = {'disable': ['ner', 'textcat', 'parser']}
# spacy_model = spacy.load('en_core_web_sm', **spacy_kwargs)
# exceptions = spacy_model.Defaults.tokenizer_exceptions
# filtered_exceptions = {k:v for k,v in exceptions.items() if "'" not in k}
# spacy_model.tokenizer = spacy.tokenizer.Tokenizer(spacy_model.vocab, rules = filtered_exceptions)
# spacy_model.add_pipe('sentencizer')
# extractor = pke.unsupervised.TextRank()
#     #install using python -m spacy download en_core_web_sm

# def reset(self):
#     self.sentences = []
#     self.candidates = defaultdict(Candidate)
#     self.weights = {}
#     #self.graph = nx.Graph()
#     self.graph.clear()

# extractor.reset = MethodType(reset, extractor)

import csv
import pickle
import time
import torch

from filelock import Timeout, FileLock


import multiprocessing as mp

import ujson

#batches_completed = 0
batches_completed = {}
subreddit = None

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

from data_setup import _tree_to_rst_code, _parse_trees

# Only take dsets with 5 or more RST chunks since we want to learn transitions between chunks of 3 EDUs


def main(   batch_process_size=20,
            mp_count=4,
            resume_progress=False,
            #subreddit_names = [],
                #batch du11ducks
            subreddit_names = ["AdviceAnimals","AmItheAsshole","Android","anime","apple","AskMen","AskReddit","askscience","AskWomen","asoiaf","atheism","australia","aww","baseball","Bitcoin","books","buildapc","business","canada","cars","CasualConversation","CFB","changemyview","Christianity","conspiracy","cringe","cringepics","dayz","DebateReligion","Diablo","DotA2","Drugs","Economics","electronic_cigarette","explainlikeimfive"],
                
                #batch enigma
            #subreddit_names = ["fantasyfootball","Fitness","Frugal","funny","Games","gaming","gifs","gonewild","Guildwars2","guns","hiphopheads","IAmA","last_batch_record","leagueoflegends","Libertarian","LifeProTips","magicTCG","MakeupAddiction","malefashionadvice","Marvel","MensRights","Minecraft","MMA","motorcycles","MovieDetails","movies","Music","Naruto","nba","news","nfl","NoFap","offbeat","OkCupid","photography"],
        
                #batch ... 
            #subreddit_names = ["pics","pokemon","pokemontrades","POLITIC","PoliticalDiscussion","politics","programming","Random_Acts_Of_Amazon","relationship_advice","relationships","rupaulsdragrace","science","sex","ShingekiNoKyojin","singapore","skyrim","soccer","SquaredCircle","starcraft","technology","techsupport","teenagers","tf2","tifu","todayilearned","travel","trees","TwoXChromosomes","unitedkingdom","videos","worldnews","wow","WritingPrompts","WTF"],
                min_rst_len = 6,
            **kwargs):
    """[summary]

    Args:
        batch_process_size (int, optional): [description]. Defaults to 20.
        subreddit_names (str, optional): List of subreddit names to filter through
    """
    
    #region  Setup    
        
    #Creating Save directory
    dir_save_dataset = utils_nlg.get_path("./dataset_keyphrase_v2/",_dir=True)
    
    # setting up subreddit data
    dirs_rst_conv = "./dataset_v2/reddit_large_annotated/"
    li_subreddit_names = list( filter( lambda name: name!="last_batch_record", os.listdir( dirs_rst_conv ) ) )

    dict_subreddit_fp = {subreddit: [fp for fp in glob.glob(os.path.join(dirs_rst_conv,subreddit,"*")) if os.path.split(fp)[-1]!="lock"  ]  for subreddit in  li_subreddit_names }

        #filtering subreddits
    if len(subreddit_names)==0:
        li_subreddit_fp = [ (subreddit,li_fp) for subreddit,li_fp in  dict_subreddit_fp.items() ]
    else:
        li_subreddit_fp = [ (subreddit,li_fp) for subreddit,li_fp in  dict_subreddit_fp.items() if subreddit in subreddit_names ]

    global subreddit
    global batches_completed
    for subreddit, li_fp in li_subreddit_fp:
        
        print(f"\nOperating on Subreddit: {subreddit}. {li_subreddit_fp.index((subreddit,li_fp))} of {len(li_subreddit_fp)}")
        
        #Should only be one file in li_fp
        #assert len(li_fp) == 1
        fp = li_fp[0]

        dset_source = pd.read_csv( fp, usecols=['rst','txt_preproc','subreddit'] )
        
        total_batch_count = math.ceil(len(dset_source)/batch_process_size)

        # region Optionally auto-resuming from last completed batch
        if resume_progress == True:

            # checking if df_records exists and if a column for this subreddit exists
            fn = os.path.join(dir_save_dataset,'last_batch_record')
            _bool_file_check = os.path.exists( fn )

            auto_fnd_failed = lambda : print("User choose auto-resume from last recorded batch.\
                But no last records exists, so initialising from batch 0")

            if not _bool_file_check: #if file does not exist
                batches_completed[subreddit]= 0
                auto_fnd_failed()
                
            else: #if file does exists
                df_records = pd.read_csv( fn, index_col = "subreddit" )
                _bool_record_check = subreddit in df_records.index.tolist()

                if not _bool_record_check:
                    batches_completed[subreddit] = 0
                
                else:
                    batches_completed[subreddit] = int( df_records.loc[ subreddit, 'last_batch' ] ) + 1
                    batch_process_size = int( df_records.loc[subreddit, 'batch_process_size'] )
                    total_batch_count = math.ceil(len(dset_source)/batch_process_size)

                    if batches_completed[subreddit] >= total_batch_count:
                        continue
        
                print(f"Skipping forwards {batches_completed[subreddit]} batches")
                dset_source = dset_source[batches_completed[subreddit]*batch_process_size:]

        else:
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
                batch_li_dict_utt[idx] =  {'rst':rst_, 'txt_preproc':txt_prepoc,
                        'subreddit': srdt } 

            
            print(f"\n\tOperating on batch {batches_completed[subreddit]+1} of {total_batch_count}")

            #region mp.imap EDU and Key Phrase Extraction
            timer.start()
            
            with mp.Pool(mp_count) as pool:
                
                cs =  max( 3, math.ceil( batch_process_size / (mp_count) ) )
                
                res = pool.imap( filtering_out_records, _chunks(batch_li_dict_utt, cs)  )

                res = pool.imap( processing_txt, res )

                res = pool.imap( edu_segmenter, res )

                res = pool.imap( check_full_rst, res )

                res = pool.imap( position_edus, res)
                              
                batch_li_dict_utt = list(res)
                batch_li_dict_utt = sum(batch_li_dict_utt, [])

            timer.end("\t\tEDU parsing, RST correction and EDU position labelling")

            #endregion

            #region Saving Batches
            timer.start()

            # format = subreddit/convo_code
            if len(batch_li_dict_utt) > 0:
                _save_data(batch_li_dict_utt, dir_save_dataset, batches_completed[subreddit], batch_process_size )

            dset_source = dset_source[batch_process_size:]
            batches_completed[subreddit] += 1
            timer.end("Saving")

            #end region    

        print(f"Finished at batch {batches_completed[subreddit] }")        

def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def filtering_out_records(li_dict_rsttext, min_rst_len=5):
    # Notice we are aiming to parse long texts only with 3 or more EDUs
    
    for idx in range( len(li_dict_rsttext)-1 ,-1 , -1 ):

        dict_rsttext = li_dict_rsttext[idx]
        rst_ = dict_rsttext['rst']

        # removing entry if too short
        if len(rst_)<=min_rst_len:
            li_dict_rsttext.pop(idx)
            continue

        # removing entry if text contains another post encapsulated in it, e.g. like a qouted reply
            # in reddit '>' is used to indicate a qouted text
        txt_preproc = dict_rsttext['txt_preproc']
        #TODO: noticed that there are still /n in the txt_preproc datasets
        if ">" in txt_preproc:
            li_dict_rsttext.pop(idx)
            continue
    
    return li_dict_rsttext



def processing_txt(li_dict_rsttext):
    # Preventing errors in rst tree parsing
    # Ensuring text starts with a capital letter
    # ensuring there arent spaces between punctuation and preceeding character
    
    for idx in range( len(li_dict_rsttext)-1 ,-1 , -1 ):

        dict_rsttext = li_dict_rsttext[idx]
        txt_preproc = dict_rsttext['txt_preproc']

        # capitalizing starting letter
        if not txt_preproc[:1].isupper():
            txt_preproc = txt_preproc[:1].capitalize() + txt_preproc[1:]

        # removin spaces between punctuation and preceeding word
        txt_preproc = re.sub(pattern_punctuation_space, r'\1', txt_preproc)

        # Adding correct capitalization to dataset
        txt_preproc = re.sub(pattern_capitalize_after_punct,               # end of acronym
             lambda x: x.group().upper(), 
             txt_preproc)


        li_dict_rsttext[idx]['txt_preproc'] = txt_preproc
    return li_dict_rsttext


# region edu processing     
def edu_segmenter( li_dict_rsttext ):
    """[summary]

        Args:
            li_dict_rsttext ([type]): [list of records each containing text with rst code annotated]

        Returns:
            [type]: [description]
    """
    if len(li_dict_rsttext) == 0:
        return li_dict_rsttext
         
    li_text = [ _dict['txt_preproc'] for _dict in li_dict_rsttext ]

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
    #TODO: try to stop these changes being made in parser_wrapper
    li_li_edus = [ [edutxt.replace(" n't", "n't").replace(" / ", "/").replace(" '", "'").replace("- LRB -", "(").replace("- RRB -", ")").replace("-LRB-", "(").replace("-RRB-", ")")
                     if edutxt not in origtext else edutxt for edutxt in li_edutext ] for li_edutext, origtext in zip( li_li_edus, li_text) ]

    for idx in range(len(li_dict_rsttext)):
        li_dict_rsttext[idx]['li_edus'] = li_li_edus[idx]

    return li_dict_rsttext


def edu_fixer(li_textwedutoken ):
        
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

def check_full_rst(li_dict_rsttext):

    if len(li_dict_rsttext) == 0:
        return li_dict_rsttext
         
    #If the extracted RST tree (from dataset_v2) has been truncated, we extract the full RST tree 
    
    li_edus_len = [ len(_dict['li_edus']) for _dict in li_dict_rsttext ]
    li_rst_len = [ len(_dict['rst']) for _dict in li_dict_rsttext ]

    #idxs of datums which need to have their rst values reparsed
    idxs_w_shrtnd_rst = [idx for idx in range(len(li_edus_len)) if li_edus_len[idx] != li_rst_len[idx]+1  ]

    if len(idxs_w_shrtnd_rst) == 0:
        return li_dict_rsttext

    li_text = [ li_dict_rsttext[idx]['txt_preproc'] for idx in idxs_w_shrtnd_rst ]

    li_li_unparsed_tree = parser_wrapper3.main( json_li_li_utterances= json.dumps([li_text]), 
                                                skip_parsing=False, redirect_output=True)
    li_unparsed_tree = sum( li_li_unparsed_tree, [] )
    li_subtrees = _parse_trees(li_unparsed_tree)
    
    li_rst_dict = [ _tree_to_rst_code(_tree) if _tree!=None else None for _tree in li_subtrees ]

    # Attaching the new rst codes to the dataset
        # and removing trees which could not be parsed
    for idx1, idx2 in reversed(list(enumerate(idxs_w_shrtnd_rst))):
        if li_rst_dict[idx1] == None:
            li_dict_rsttext.pop(idx2)
        else:
            li_dict_rsttext[idx2]['rst'] = li_rst_dict[idx1]

    


    return li_dict_rsttext


def position_edus(li_dict_rsttext):
    if len(li_dict_rsttext) == 0:
        return li_dict_rsttext
         
    for idx in range(len(li_dict_rsttext)):

        li_rst_pos = [ rst_node['pos'] for rst_node in li_dict_rsttext[idx]['rst'] ]
        li_child_pos =  sum( [ find_child_edus(pos, li_rst_pos ) for pos in li_rst_pos ], [] )

        li_edu = li_dict_rsttext[idx].pop('li_edus')

        dict_pos_edu = { edu_pos:edu for edu_pos, edu in zip( li_child_pos, li_edu ) }
        
        li_dict_rsttext[idx]['dict_pos_edu'] = dict_pos_edu
    
    return li_dict_rsttext

def find_child_edus(pos_parentnode, li_rst_pos):
        #returns the pos of any child elements of a parent node(rst) that are edus
               
        li_child_pos = [2*pos_parentnode+1, 2*pos_parentnode+2 ]

        li_child_edu_pos = [ pos for pos in li_child_pos if pos not in li_rst_pos]

        return li_child_edu_pos 

# region key_phrase_extraction
def _key_phrase(li_li_edusegment):
    # input is a list of different utterances, each utterance is represented as a list of its edus
    # Here we extract the key phrase of an EDU
        # Our system has four possible sugesstions for each EDU s1,s2,s3,s4
        # s_i is accepted if it is below a certain length relative to the original text, otherwise we consider s_(i+1)
    
    global extractor
    
    pos_tags = {'NOUN',  'PROPN' ,'ADJ','VERB','PRON','SYM','ADV','DET', 'PUNCT',             'PART'}
    
    li_li_kpscore = []

    for i in range(len(li_li_edusegment)):

        li_edusegment = li_li_edusegment[i]

        li_kp_score = [ _key_phrase_extractor(edusegment, extractor=extractor, spacy_model=spacy_model, pos_tags=pos_tags)             
                            for edusegment in li_edusegment ]
        
        
        li_li_kpscore.append( li_kp_score )
        
    return li_li_kpscore, li_li_edusegment

def _key_phrase_extractor(str_utterance, extractor, spacy_model, pos_tags):
    #returns the first key phrase that is under half the original text length
    
    # extracting kp and score
    extractor.reset()
    extractor.load_document(input=str_utterance, language='en',normalization=None, spacy_model=spacy_model)
    extractor.candidate_selection(pos=pos_tags)
    extractor.candidate_weighting(window=3, normalized=True, pos=pos_tags)
    
    li_kp_score = extractor.get_n_best(n=3, redundancy_removal=False)
    
    if len(li_kp_score) == 0:
        li_kp_score = [ ["",0.0] ]

    li_kp_score = [ (kp_shortener(kp_score[0], spacy_model), round(kp_score[1],4) ) for kp_score in li_kp_score ]

    #add kp_formating
    
    return li_kp_score

def kp_shortener( txt, spacy_model, pos_to_remove = ['ADV','DET','X'], tags_to_remove=['VBZ']  ):
    

    doc = spacy_model(txt)
    
    txt = ' '.join( [ tkn.text for tkn in doc if ( tkn.pos_ not in pos_to_remove) and (tkn.tag_ not in tags_to_remove)  ] )
           
    return txt
# endregion

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
    
    parser.add_argument('-bps','--batch_process_size', default=3,
                             help='',type=int)        
   
    parser.add_argument('--mp_count', default=4, type=int)
    
    parser.add_argument('-rp','--resume_progress', default=True, type=lambda x: bool(int(x)), 
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
            batches_completed[subreddit] = batches_completed[subreddit] + 2
            dict_args['resume_progress'] = True
            
        finally :
            # cmd = "docker stop $(docker ps -aq) > /dev/null 2>&1 & docker rm $(docker ps -aq) > /dev/null 2>&1 & docker rmi $(docker images -a -q) > /dev/null 2>&1"
            
            # cmd = docker stop $(docker ps -aq) & docker rm $(docker ps -aq) & yes | docker image prune  & docker rmi $(docker images -a -q)
            # os.system(cmd)
            # time.sleep(3)
            # os.system(cmd)
            # time.sleep(3)
            pass



# python3 data_setup_keyphrase2.py -bps 60 -rp 1  --mp_count 8  
