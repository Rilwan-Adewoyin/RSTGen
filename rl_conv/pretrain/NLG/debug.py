import pandas as pd
import gc
import copy
import glob
import os
import torch
from os.path import dirname, basename
import copy
from pandarallel import pandarallel

from collections import Counter, OrderedDict
import random
import pickle
import math
import matplotlib.pyplot as pl
import utils_nlg
from train_nlg import NLG, TrainingModule
import json
import itertools

## Imports for A1) Controlling Text Length
import os
import sys

pandarallel.initialize(nb_workers=4, verbose=0)
# Docker Images Parser and RST tree labeller


#from DockerImages.feng_hirst_rst_parser.src import parser_wrapper3

import nltk
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
## ------------------------ ##

## --- Imports for the attention viewing code
from bertviz import head_view


## -- Imports e_coherency
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import CTRLTokenizer, CTRLLMHeadModel

# --- Imports d
#%matplotlib inline
import regex as re
import sklearn as skl
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


class Data_Sampler():
    """
        This Provides the ability to sample from RST structure trees
        
        Can sample RST structure by multiple factors
        
            Factor 1 - Subreddit
            Factor 2 - Length of RST, by number of nodes
            Factor 3 - Contains a specific number of a certain RST relations
                        
    """
    def __init__(self, 
                 dir_dsets="./dataset_v2/reddit_large_annotated",
                rst_restore_dir="./RST_sampler"):
        
        self.dir_dsets = dir_dsets
        self.load_subreddit_fps()
        self.rst_restore_dir = rst_restore_dir
        
        self.restore_queries()
        
    def restore_queries(self):
        self.query_fp = os.path.join( self.rst_restore_dir, 'dict_querynumb_encodedquery.json' ) 
        
        if os.path.exists( self.query_fp ):
            self.dict_querynumb_encodedquery = json.load( open(self.query_fp, 'r') )
        else:
            os.makedirs( self.rst_restore_dir, exist_ok=True)
            os.makedirs( os.path.join( self.rst_restore_dir, "saved_dsets") , exist_ok=True)
            self.dict_querynumb_encodedquery = {}
                        
    def load_subreddit_fps(self):
        """ loads a list of fns sorted by subreddit name
            
            Use a dictionary with lists to hold the fns
        """
        
        #Finding subreddit names
        self.dict_subreddit_fps = { basename(dirname(path)):path for path in glob.glob(os.path.join(self.dir_dsets,"*","*")) if os.path.split(path)[-1] != "lock"  }
        #print(self.dict_subreddit_fps)
    
    def sample(
                self,
               sample_count= 10,
                li_subreddit=[], 
               li_RST_len=[],
               dict_RSTrel_counts={},
                rst_rel_count_mode='greaterequal_than'):

        dict_subreddit_fps = copy.deepcopy(self.dict_subreddit_fps)

        # filtering
        df = self.filter_from_fps(  dict_subreddit_fps,
                 li_subreddit,
                 li_RST_len,
                 dict_RSTrel_counts,
                rst_only=False,
                rst_rel_count_mode=rst_rel_count_mode) 
        
        # Sampling
        if sample_count>=0:
            adj_sample_count = min(sample_count, len(df) )
            
            li_sample = random.sample(df.values.tolist(), adj_sample_count)
        else:
            li_sample = df.values.tolist()
        
        return li_sample
            
    def sample_rst( self,
               sample_count= 10,
                li_subreddit=[], 
               li_RST_len=[],
               dict_RSTrel_counts={},
                  rst_rel_count_mode='greaterequal_than'):
    
        """
            sample_count: an integer representing the number of samples to draw
            li_subreddit: a list of the subreddit's from which the user wants samples to be drawn
            li_RST_len: a list of numbers - indicating which lengths to include
            dict_RSTrel_counts: a dictionary indicating filtering on the occurece of relations in the Tree
                                each key will be a relation. each value will be a number indicating the minimum number of times this occurence must appear in the RST tree
        """
        li_subreddit.sort()
        
        # Checking if query has been searched already and can be loaded from a saved pickle
        encoded_query = f"scount_{sample_count}__lisreddit_{'x'.join(li_subreddit)}__RSTlen_{'x'.join([str(x) for x in li_RST_len])}__RSTrelcounts_{json.dumps(dict_RSTrel_counts)}"
        bool_loaded = encoded_query in self.dict_querynumb_encodedquery.values()
        
        if bool_loaded == True:
            
            query_code = [k for k,v in self.dict_querynumb_encodedquery.items() if v==encoded_query ][0]
            dset_path = os.path.join(self.rst_restore_dir, "saved_dsets", f"{query_code}.pickle" )
            srs_rsts = pickle.load( open( dset_path, "rb") )
        
        else:
            
            dict_subreddit_fps = copy.deepcopy(self.dict_subreddit_fps)

            srs_rsts = self.filter_from_fps(  dict_subreddit_fps,
                     li_subreddit,
                     li_RST_len,
                     dict_RSTrel_counts,
                    rst_only=True,
                    rst_rel_count_mode=rst_rel_count_mode) 
            
            # Saving Filtered pandas series to pickle
                # Each new pickle file is simply labelled an increment value of the largest number in the directory e.g. 001.pickle. 002.piclkle e.g.
                # We then hold a dictionary which maps these filenames to the actual 
            query_code = f"{len(self.dict_querynumb_encodedquery)+1:03d}"
            
            path_ = os.path.join(self.rst_restore_dir, "saved_dsets", f"{query_code}.pickle" )
            pickle.dump(srs_rsts, open(path_,"wb"))
            
            self.dict_querynumb_encodedquery[ query_code ] =  encoded_query
            json.dump( self.dict_querynumb_encodedquery, open( self.query_fp, 'w') )
            
        # Sampling
        if sample_count>=0:
            li_sample = random.sample(srs_rsts.tolist(), sample_count)
        else:
            li_sample = srs_rsts.tolist()
        
        return li_sample, query_code       
    
    def filter_from_fps(self, 
                    dict_subreddit_fps,
                 li_subreddit,
                 li_RST_len,
                 dict_RSTrel_counts,
                    rst_only=False,
                    rst_rel_count_mode="greaterequal_than"):
        
        #subreddit type filtering
        if len(li_subreddit) != 0:
            dict_subreddit_fps = { k:v for k,v in dict_subreddit_fps.items() if k in li_subreddit }

            # Loading actual records into memory
        if rst_only:
            li_df = [pd.read_csv(v, header=0,squeeze=True, usecols=['rst'] ) for k,v in dict_subreddit_fps.items() ]
        else:
            li_df = [pd.read_csv(v, header=0, usecols=['txt_preproc','rst','topic_textrank' ] ) for k,v in dict_subreddit_fps.items() ]
                        
            # Filtering out data to only include the test set data
            file_sizes = [len(df) for df in li_df]
            test_starts = [int(0.8*val) for val in file_sizes]
            li_df = [ df.iloc[ts:] for df, ts in zip(li_df, test_starts) ]
            
            #parsing the json encoded strings
            if rst_only:
                for df_ in li_df:
                    df[['rst','topic_textrank']] =  df[['rst','topic_textrank']].applymap(func_parse)
            else:
                for df_ in li_df:
                    df_[['txt_preproc','rst','topic_textrank']] = df_[['txt_preproc','rst','topic_textrank']].applymap(lambda x: json.loads(x))
                    

        df = li_df.pop(0)
        df = df.append(li_df, ignore_index=True)
        del li_df
        

        # RST_len filtering            
        if len(li_RST_len) != 0:
            
            def func_rstlenfilt(li_rst_encoding, lengths_to_keep):
                
                bool_ = len(li_rst_encoding) in lengths_to_keep
                return bool_

            if rst_only:
                m = df.parallel_apply(func_rstlenfilt, lengths_to_keep=li_RST_len)
            else: 
                m = df['rst'].apply( lambda x: len(x) in li_RST_len )
            
            df = df[m]


        # RST relation count filtering
        if len(dict_RSTrel_counts) != 0:
            def func_rstrelcountfilt(li_rst_encoding, dict_rst_mincount ):
                rel_counter = Counter( [ dict_['rel'] for dict_ in li_rst_encoding] ) 
                
                if rst_rel_count_mode == "greaterequal_than":
                    bool_ = any( [ rel_counter[rel] >= dict_rst_mincount[rel] for rel in dict_rst_mincount.keys() ] ) 
                
                elif rst_rel_count_mode == "equal_to":
                    bool_ = any( [ rel_counter[rel] == dict_rst_mincount[rel] for rel in dict_rst_mincount.keys() ] ) 
                
                elif rst_rel_count_mode == "lessequal_than":
                    bool_ = any( [ rel_counter[rel] <= dict_rst_mincount[rel] for rel in dict_rst_mincount.keys() ] ) 
                    
                return bool_
            
            if rst_only:
                m = df.parallel_apply(func_rstrelcountfilt, dict_rst_mincount=dict_RSTrel_counts )
            else:
                m = df['rst'].apply(func_rstrelcountfilt, dict_rst_mincount=dict_RSTrel_counts )
                
            df = df[m]


        return df
        
    def rst_statistics_for_query(self,
                li_subreddit=[], 
               li_RST_len=[],
               dict_RSTrel_counts={}):
        """
            Produce RST related statistics for this query
            
            Saves to file information on :
                Tree depth distribution
                For each relation r, the distribution of r's occurences in each utterance
                The average number of times each relation occurs in the dataset
                
                                
        """

        # Gathering Sample
        li_sample,query_code = self.sample_rst( sample_count=-1,
                li_subreddit=li_subreddit, 
               li_RST_len=li_RST_len,
               dict_RSTrel_counts={})
        
        #Setting up directory to save RST statistics
        save_dir = os.path.join(self.rst_restore_dir, "RST_stats", query_code )
        os.makedirs(save_dir, exist_ok=True)
        
        # Saving the related queries that form these stats
        dict_query_details = {
            'subreddits_filter':li_subreddit, 
               'RST_len_filter':li_RST_len,
               'RSTrel_counts_filter':dict_RSTrel_counts
        }
        
        json.dump(dict_query_details, open( os.path.join(save_dir, "query_details.json"), "w") )
        
                
        # Saving Depth Distribution Chart
        li_rst_last_pos = [ [_dict['pos'] for _dict in li_][-1] for li_ in li_sample ]
        li_rst_depths = [ int( math.log( pos+2, 2) ) for pos in li_rst_last_pos  ]
                
        pl_hist = pl.hist(li_rst_depths, bins=list(range(max(li_rst_depths))), density=True )
        pl.title('RST Binary Tree Depth Distribution')
        pl.xlabel("Depth")
        pl.ylabel("Density")
        pl.savefig( os.path.join(save_dir, "rst_depth_distribution.png"),dpi=600 )
        pl.close()
        
              
        # Saving Occurence statistics for each relation 
            # First Converting each utterance into a dictionary recording the count of each rst relation occurence in a specific utterace
        li_utterance_relscntr = [ Counter([ _dict['rel'] for _dict in li_ ] ) for li_ in li_sample ]  # a list of relation counts for each utterance in sample
        
            # getting distribution for each RST relation, of the number of times an RSt relation occurs in an utterance
        li_rels = utils_nlg.rst_rel_li
        
        dict_relscntr = { k:Counter() for k in li_rels  }
        
        for utterance_relscntr in li_utterance_relscntr: #Iterating along the list of counters. Each counter holds the occurence of each RST in the corresponding utterance
                        
            for rel,count in utterance_relscntr.items(): #Iterating through the relation counter for one utterance    
                dict_relscntr[rel][count] += 1         #Adding 1, since we have observed another utterance where the relation r, occurs count c times
            
            relations_not_in_utterance = list( set(li_rels) - set(utterance_relscntr.keys()) ) #Finding list of utterances which did not occur
            for rel in relations_not_in_utterance: #Adding +1 to the zero occurence in counter
                dict_relscntr[rel][0] += 1
                       
            #Producing bar plots for each relation: number of times it appears in an utterance on the confition it occurs
        for rel, counter in dict_relscntr.items():
                #temp removing 0 values, and sorting by keys
            #temp_counter = OrderedDict( sorted( { k:v for k,v in counter.items() if k!=0 }.items() ) )
            temp_counter = { k:v for k,v in counter.items() if k!=0 }
            labels, values = zip(*sorted(temp_counter.items()))

            indexes = np.arange(len(labels))
            width = 1

            pl.bar(indexes, values, width)
            #pl.xticks(indexes + width * 0.5, labels)
            pl.xticks(indexes, labels)
            #plt.show()
            
            pl.title(f'{rel} Conditional Occurence Distribution')
            pl.xlabel("Count")
            pl.ylabel("Frequency")
            pl.savefig( os.path.join(save_dir, f"conditional_occurence_{rel}.png"), dpi=600 )
            pl.close()
            
            #Producing one bar plot representing the average number of times each occurence occurs across all utterances
        dict_rel_averagecounts = {}
        for rel, counter in dict_relscntr.items():
            sorted_items = sorted(counter.items() )
            rel_occurence_count = sum( [ k*v for k,v in sorted_items ] )
            utterance_count = sum( [v for k,v in sorted_items] )
            
            dict_rel_averagecounts[rel] = rel_occurence_count/utterance_count

        labels, values = zip(*dict_rel_averagecounts.items())

        width = 0.8
        indexes = np.arange(len(labels))
        

        pl.figure(figsize=(14,6))
        pl.bar(indexes, values, width )
        #pl.xticks(indexes + width * 0.5, labels, rotation=45)
        pl.xticks(indexes, labels, rotation=45)
        
        #plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
        #plt.show()

        pl.title(f'Average Number of each RST Relation per utterance')
        pl.xlabel("Relation")
        pl.yscale('log')
        pl.ylabel("Count")
        pl.savefig( os.path.join(save_dir, f"percentage_occurence_all_relations.png"), dpi=600, bbox_inches='tight' )
        pl.close()
              
sampler = Data_Sampler(dir_dsets="./dataset_v2/reddit_large_annotated",
                             rst_restore_dir="./RST_sampler")     


def load_model(model_name, **kwargs ):
    
    if model_name == "NLG_rt":
        model = TrainingModule.load_nlgmodel(model_name=model_name,
                                             model_version=kwargs.get('model_version'), max_input_len=kwargs.get('max_input_len')).cuda()
        tokenizer = None
        
    elif model_name == "gpt2":
        dir_transformer = os.path.join("./models",model_name)
        dir_tokenizer = dir_transformer + "_tknzr"
        
        exists = os.path.isdir(dir_transformer)
        
        if exists == False:    
            model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            
            os.makedirs(dir_tokenizer, exist_ok=True)
            os.makedirs(dir_transformer, exist_ok=True)
            
            model.save_pretrained(dir_transformer)
            tokenizer.save_pretrained(dir_tokenizer)
        else:
            model = GPT2LMHeadModel.from_pretrained(dir_transformer).cuda()
            if os.path.isdir(dir_tokenizer):
                tokenizer = GPT2Tokenizer.from_pretrained(dir_tokenizer)
            else:
                tokenizer = GPT2Tokenizer.from_pretrained(dir_transformer)
            
    elif model_name == "ctrl":
        dir_transformer = os.path.join("./models",model_name.replace("/","_"))
        dir_tokenizer = dir_transformer + "_tknzr"
        exists = os.path.isdir(dir_transformer) 
        
        if exists == False:    
            model = CTRLLMHeadModel.from_pretrained(model_name).cuda()
            tokenizer = CTRLTokenizer.from_pretrained(model_name)
            
            os.makedirs(dir_tokenizer, exist_ok=True)
            os.makedirs(dir_transformer, exist_ok=True)
            
            model.save_pretrained(dir_transformer)
            tokenizer.save_pretrained(dir_tokenizer)
        else:
            model = CTRLLMHeadModel.from_pretrained(dir_transformer).cuda()
            tokenizer = CTRLTokenizer.from_pretrained(dir_tokenizer)
    
    else:
        raise Exception("not exist")
    return model, tokenizer


# Testing new
model_version = 16
model_name = 'NLG_rt'
model_params={'model_version':model_version }
model, tokenizer = load_model(model_name, **model_params) 
        
    #generation params
bad_words = ["<|rst|>","<|ta|>",r"\n" ] 
bad_words_ids = [model.nlg_tokenizer.e2m_tokenizer.encode(bad_word, add_prefix_space=False) for bad_word in bad_words]
bad_words_ids = [model.nlg_tokenizer.e2m_tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in bad_words]
bad_words_ids = bad_words_ids + [[526], [55],[8172]]

generation_params = {'num_beams':1, 'temperature':1.2, 'repitition_penalty':2.0, 
                     'early_stopping':True, 'do_sample':False, 'no_repeat_ngram_size':3, 'bad_words_ids':bad_words_ids, 'max_length':5 } #,'min_length':4


li_datum = sampler.sample( 
    li_subreddit=['CasualConversation'],
    li_RST_len = [8],
    sample_count = 1
)


# calculate lengths of generated text
for idx,datum in enumerate(li_datum):

    #region: unpacking datum
    utterance, li_rst, li_topic_textrank = datum
    rst_rels = [ _dict['rel'] for _dict in li_rst ]
    rst_ns = [ _dict['ns'] for _dict in li_rst ]
    rst_pos = [ _dict['pos'] for _dict in li_rst ]
    topics, topics_score = zip( *li_topic_textrank ) 


    # region: Creating prediction
        #Creating utterance_context
    start_utt = ' '.join(utterance.split(' ')[:0])

        # Creating encoded input
    encoded_input = model.nlg_tokenizer.encode_v2_exda(rst_rels, rst_ns, rst_pos ,
                                                topics, topics_score, start_utt,
                                                pad_utterance=False, generate_mode=True)

        # Add batch dimension to data
    for key in ['tnsr_rst_rels', 'tnsr_rst_ns', 'tnsr_rst_pos',
                'tnsr_topics_phrase','tnsr_topics_score','tknzd_utt',
                'position_ids','token_type_ids',
                    'attn_mask','rst_start_token']:
        encoded_input[key] = torch.unsqueeze( encoded_input[key],axis=0).cuda()


    # Generating model output, given original rst structure
    with torch.no_grad():
        output = model.generate(encoded_input, **generation_params)
        
    decoded_text = model.nlg_tokenizer.e2m_tokenizer.decode(output[0],skip_special_tokens=True)
    
    print(decoded_text)
