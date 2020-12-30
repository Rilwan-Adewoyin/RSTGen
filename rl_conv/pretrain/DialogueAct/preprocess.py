# Workflow

# Split Dataset into 12 new files
# Then perform the following operations in parrallel
    # preprocess
    # add DA and RST annotations

# Then reform dataset into one file and shuffle
# Then set 75% as training set and 25% as test


import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import glob
import logging
import subprocess

import multiprocessing as mp
import psutil

import pandas as pd
import regex as re
import json
import utils
import time
import numpy as np
import traceback

import torch
import copy
import gc

from timeit import default_timer as timer
from datetime import timedelta

def main(argv=None):
    """Run the preprocessing pipeline."""
    args = parse_args(argv)

    #preprocessing train validation and test set
    
    sets = ['train','val','test']
    for _set in sets:
        preprocess(args, _set )
    

def parse_args(argv=None):
    """Parse command-line args."""

    def _positive_int(value):
        """Define a positive integer ArgumentParser type."""
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError(
                "Value must be positive, {} was passed.".format(value))
        return value

    parser = argparse.ArgumentParser()
    

    parser.add_argument(
        "--dir_mrda",
        default= utils.get_path("MRDA-Corpus/mrda_data"), type=str,
        help="The directory for MRDA data")
    
    parser.add_argument(
        "--dir_swda",
        default= utils.get_path("Switchboard-Corpus/swda_data"), type=str,
        help="The directory for MRDA data")

    parser.add_argument(
        "--dir_midas",
        default= utils.get_path("MIDAS-Corpus/midas_data"), type=str,
        help="The directory for MRDA data")

    parser.add_argument(
        "--output_dir", required=False,
        default= utils.get_path("combined_data/"),
        help="Output directory to write the dataset.")

    
    parser.add_argument(
        "--num_cores", default=psutil.cpu_count(logical = False),
        type=_positive_int,
        help="The number of cores to split preprocessing job over"
    )

    return parser.parse_known_args(argv)[0]


def preprocess(args, str_subset):
    """[summary]

    Args:
        args ([type]): [description]
        subset ([str]): Decides which datasets to preprocess options:['train','val','test']

    Raises:
        # OSError: [description]
    """
    #Obtaining dirs for dataset
    dir_mrda_dset = os.path.join(args.dir_mrda, str_subset)
    dir_swda_dset = os.path.join(args.dir_swda, str_subset)
    dir_midas_dset = os.path.join(args.dir_midas, str_subset)

    dir_dset = os.path.join(args.output_dir, str_subset)
    os.makedirs(dir_dset,exist_ok=True)
    
    #list of files to process
    li_fns_mrda = _get_fns( os.path.join(dir_mrda_dset,"*") ) # [fn for fn in glob.glob(pattern) ]
    li_fns_swda = _get_fns( os.path.join(dir_swda_dset,"*") ) # [fn for fn in glob.glob(pattern) ]
    li_fns_midas= _get_fns( os.path.join(dir_midas_dset,"*"))

    #list of files' codes already processed
    li_fns_mrda_proc = _get_fns( os.path.join(dir_dset, "MRDA_*.txt" ) )
    mrda_proc_codes = [ _fn.split('_')[-1] for _fn in li_fns_mrda_proc ]

    li_fns_swda_proc = _get_fns( os.path.join(dir_dset, "SWDB-DAMSL_*.txt" ) )
    swda_proc_codes = [ _fn.split('_')[-1] for _fn in li_fns_swda_proc ]

    li_fns_midas_proc = _get_fns( os.path.join(dir_dset, "midas_*.txt" ) )
    midas_proc_codes = [ _fn.split('_')[-1].replace('.txt',"") for _fn in li_fns_midas_proc ]
    # getting the 


    # removing fns from to process based on their suffix

    li_fns_mrda = [ fn for fn in li_fns_mrda if ( os.path.split(fn)[1] not in mrda_proc_codes ) ]
    li_fns_swda = [ fn for fn in li_fns_swda if ( os.path.split(fn)[1] not in swda_proc_codes ) ]
    li_fns_midas = [ fn for fn in li_fns_midas if ( os.path.split(fn)[1] not in midas_proc_codes ) ]

    #Removed fns that have already been procesed

    dict_da_map = json.load( open(utils.get_path("label_mapping.json"),"r") )
    
        
    dataset_names = ['MRDA','SWDB-DAMSL','MIDAS']
    li_li_fns = [li_fns_mrda, li_fns_swda, li_fns_midas ]
 
    usecols = { 'MRDA':[0,1,4] ,'SWDB-DAMSL':None, 'MIDAS': [0,1]    }
    names = { 'MRDA':['speaker','utterance','da'] ,'SWDB-DAMSL':['speaker','utterance','da'], 'MIDAS': ["utterance","da"]    }
    seps = {'MRDA':'|', "SWDB-DAMSL":"|", "MIDAS":"##" }

    paraphrase_counts = {
                        'action-directive': 10,
                        'reject': 6,
                        'summarize': 12,
                        'quotation': 12,
                        'hedge': 10,
                        'elaboration': 6} 
                        #NOTE: have to be even numbers, half half split for summarization and paraphrasing
    
    # Getting paraphraser
    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dict_transformer = utils.load_pretrained_transformer(model_name, transformer=True, tokenizer=True)
    model = dict_transformer['transformer']
    tokenizer = dict_transformer['tokenizer']
    (model).to(torch_device)

    for ds_name, li_fns in zip( dataset_names, li_li_fns):
        print(f"Operating on {ds_name} files")
        start_time = time.time()
        da_to_remove = dict_da_map[ds_name]['removed']
        
        for counter, fn in enumerate(li_fns):
            df_conv = pd.read_csv(fn, sep=seps[ds_name], header=None, dtype=str,
                            usecols=usecols[ds_name], names=names[ds_name] )
                            
            if ds_name in ['MRDA', 'SWDB-DAMSL']:
                df_conv = _annotate_dialog_act(df_conv,ds_name, dict_da_map)
                path = os.path.join(dir_dset, ds_name+"_"+os.path.split(fn)[1] )
                df_conv.to_csv( path, header=True, index=False, sep="|" )

                li_df_convs  = [df_conv]

            elif ds_name in ['MIDAS']:
                li_df_convs = _annotate_dialog_act_v2(df_conv, ds_name, dict_da_map)
                #TODO: Add paraphraser here

                for idx, df in enumerate(li_df_convs):
                    path = os.path.join(dir_dset, f"midas_{idx:05d}.txt")
                    df.to_csv( path, header=True, index=False, sep="|")
            
            # Paraphrasing (makes pphrases per file and saves them)
            response = _mk_pphrase(li_df_convs, paraphrase_counts, model, tokenizer, dir_dset, ds_name, fn )
            
            if counter+1 % 10 == 0:
                print(f"\t\t Finished {counter} of {len(li_fns)}")
        
        time_elapsed =time.time() - start_time
        print(f"Finished {ds_name} files:\t Time Elapsed {time_elapsed} \n\n")   
    return True

def _get_fns(pattern):
    li_fns = [fn for fn in glob.glob(pattern) ]
    return li_fns

def _annotate_dialog_act(df_conv, ds_name, dict_da_map):
    """ Annotation method for MRDA and SWDB"""

    def __parse_list_or_string(da):
        _li = sum( [ dict_da_map[ds_name]['MCONV'][word] for word in da.split() ], [])
        _set = set(_li)
        _set = sorted(_set)            
        _str = " ".join(_set)
        return _str

    # convert labels to full schema
    df_conv['da'] = df_conv['da'].apply( lambda da: dict_da_map[ds_name]['short2full'][da] )

    # remove repeated consecutive words in utterance
    df_conv['utterance'] = df_conv['utterance'].str.replace(r'\b([\S ]+)(\s+\1)+[\b\.\?\!]', r'\1')

    #remove da if in removed list given utterance less than 2 words and does not have a MCONV mapping
    remove = '|'.join(dict_da_map[ds_name]['removed'])
    pattern = re.compile(r'\b('+remove+r')\b', flags=re.IGNORECASE)

    df_conv['da'] = df_conv.apply( 
        lambda row: pattern.sub("", row.da) 
        if len(row.da.split())<=2  and (row.da not in list(dict_da_map[ds_name]['MCONV'].keys()))
        else row.da, axis=1)

    #compress lines together if same speaker
    df_conv['key'] = (df_conv.speaker!=df_conv.speaker.shift(1)).cumsum()
    df_conv = df_conv.groupby(  ['key','speaker'], sort=False ).agg( lambda x: ' '.join(x) ).reset_index().drop('key',axis=1)
    
    # remove whole utterance if no other dialoge act left
    df_conv = df_conv[ df_conv.da.replace(" ", "") != "" ].reset_index(drop=True)

        #read in file as a pandas dataframe and filter out coloumns to keep
    # convert das to mconv full schemes and remove repeated consecutive dialogue acts
    df_conv['da'] = df_conv['da'].apply( lambda da: __parse_list_or_string(da) )
    df_conv['utterance'] = df_conv['utterance'].str.strip(',')

    return df_conv

def _annotate_dialog_act_v2(df_conv, ds_name, dict_da_map):
    """ Annotation method for MIDAS """


    def __midas_da_pipeline(da):
        da = da.strip(";")
        da = da.replace(' ','')
        da = da.replace(";",' ')
        return da

    def __midas_prev_utterance_pipeline(utt):
        ## Adding question related punctuation
        utt = utt.strip(' ').capitalize()
        utt = __midas_utterance_punctuation(utt)
        return utt

    def __midas_new_utterance_pipeline(utt):
        #replace > by comma if not preceeded by EMPTY
        li_utt = utt.split('>')
        li_utt = [_utt.strip(' ') for _utt in li_utt]

        if li_utt[0] == "EMPTY":
            response = li_utt[1].capitalize()
        else:
            response = ', '.join(li_utt).capitalize()
        
        response = __midas_utterance_punctuation(response)
        return response
        ## Adding question related punctuation

    def __midas_utterance_punctuation(utt):
        """Determines whether utterance is a question or statement"""
        li_question_starters = ['What',"Why","Where","How","Who","When","What","Which","Is","Did","Does","Are","Can",'Have',"Would"]
        if utt.split(' ')[0] in li_question_starters:
            utt = utt + "?"
        else:
            utt = utt + "."
        return utt

        #Each row contains one utterance, with both the context and response divided by ":"
    
    df_conv[['prev_utterance','new_utterance']] = df_conv['utterance'].str.split(':', 1, expand=True)

    df_conv[['prev_utterance','new_utterance','da']] =  df_conv[['prev_utterance','new_utterance','da']].apply( lambda row: 
        ( __midas_prev_utterance_pipeline(row.prev_utterance), __midas_new_utterance_pipeline(row.new_utterance), __midas_da_pipeline(row.da) ), axis=1,result_type="expand")
    df_conv.drop('utterance',axis=1, inplace=True)
       
    # convert labels to full schema (multiple labels may occur in each line before conversion)
    to_remove = dict_da_map[ds_name]['removed']
    df_conv['da'] = df_conv['da'].apply( lambda das: ' '.join( sorted( sum( [dict_da_map[ds_name]['MCONV'][da] for da in das.split(' ') if da not in to_remove ], [] ) ) ) )

    #Remove rows with no da label
    df_conv = df_conv[ df_conv['da']!= "" ]
    
    li_df_convs = [ df_conv.iloc[idx:idx+1] for idx in range(df_conv.shape[0]) ]

    for idx in range(len(li_df_convs)):
        li_df_convs[idx] = li_df_convs[idx].append( {'new_utterance': li_df_convs[idx]['prev_utterance'].iloc[0] }, ignore_index=True )
        li_df_convs[idx] = li_df_convs[idx].iloc[::-1].drop('prev_utterance',axis=1).reindex()
        li_df_convs[idx]['speaker'] = li_df_convs[idx].index.map(str)
        li_df_convs[idx].rename( {"new_utterance":"utterance"}, axis=1,inplace=True)
        li_df_convs[idx] = li_df_convs[idx][['speaker','utterance','da']]
    
    return li_df_convs

def _mk_pphrase(li_df_conv, paraphrase_counts, model, tokenizer, dir_dset, ds_name, fn):
    """[summary]

    Args:
        li_df_conv ([type]): [description]
        paraphrase_counts ([type]): A dictionary of da acts to repeat and their associated coutns
        dir_dset is  a string mrda or swdb etc
        dset_name in train val test etc
        fn is the acc filenmae
    """

    # Check for the any row that contains they above das but does not contain "statement"
        #get their indexes

    
    # Then use the indexes to select that batch 


    # and submit for paraphrase generation
    # Then use the incremental indexes to seperate the paraphrased sentences

    # Then use the original indexes to select the preceeding utterance

    # Then return the list of two sentence paraphrased sentences.
    
    

    li_da_labels_to_pp = list(paraphrase_counts.keys())
    li_das_not_to_pp = ["backchannel","statement","other-forward-function","understanding","question","agreement"]
    li_df_pphrased = []

    for df in li_df_conv:
        

    # Check for the any row that contains they above das but does not contain "statement"
    # the lambda statment: returns true if all labels in x are in the list of das_to_pp but not in the 
    # list of das to ignore
        #1)
        idxs_to_pp = df[ df.apply(lambda row: row['da']!=None and all( (da != '') and ((da in li_da_labels_to_pp) or (da not in li_das_not_to_pp )) for da in row['da'].split(' ')  )  , axis=1) ].index.tolist()

            #  removing index 0 from list if there since cant get response to 0th index 
        idxs_to_pp = [idx for idx in idxs_to_pp if idx != 0]
        rows_to_pp = df.iloc[idxs_to_pp] #row idxs for the current utterance
        
    # Gather the amount of copies to make for each da label in rows_to_pp
        copies_to_make = [ max( paraphrase_counts[x] for x in da_label.split(' ') if x!='' )  for da_label in rows_to_pp['da'].values.tolist() ]

    # Getting paraphrase
        #current utterance
        li_li_pp_text = __get_response(model, tokenizer, rows_to_pp['utterance'].values.tolist() , copies_to_make )
        #prev utterance
        li_li_pp_text_prev = __get_response(model, tokenizer, df.iloc[np.array(idxs_to_pp)-1]['utterance'].values.tolist(), copies_to_make )

    # make a list of new dfs with the paraphrase
        li_pp_df_copy = []
        for idx, li_pp_txt, li_pp_txt_prev in zip(idxs_to_pp,  li_li_pp_text, li_li_pp_text_prev):
            if li_pp_txt==[] or li_pp_txt_prev==[]:
                continue

            df_ = df.iloc[ idx-1: idx+1 ]  #Gathering the row and prev row relating the the paraphrases
            li_pp_df_copy = [ pd.DataFrame(columns = df_.columns, data = copy.deepcopy(df_.values)) for idx2 in range(len(li_pp_txt))  ] # Creating copies
            
            #inserting the the paraphrased utterances
            for df_copy, pp_text, pp_text_prev in zip( li_pp_df_copy, li_pp_txt, li_pp_txt_prev):
                df_copy.at[1,'utterance'] = pp_text
                df_copy.at[0,'utterance'] = pp_text_prev
             
        li_df_pphrased.extend(li_pp_df_copy)

    for idx3, df2 in enumerate(li_df_pphrased):
        path = os.path.join(dir_dset, f"{ds_name}_{os.path.split(fn)[-1]}_pp_{idx3:05d}.txt")
        df2.to_csv( path, header=True, index=False, sep="|" )

    return True

def __get_response(model, tokenizer, li_input_text, li_num_return_sequences):
    
    li_tgt_text = []
    for txt, num_ret_seq in zip( li_input_text, li_num_return_sequences):
        txt_len = len(txt.split(' '))

        if txt_len > 2:
            #max_len = int(txt_len*2)
            
            batch = tokenizer.prepare_seq2seq_batch([txt], truncation=True, padding='longest', max_length=60, return_tensors="pt").to('cuda')

            translated_pp = model.generate(**batch, max_length=60, num_beams=12, num_return_sequences=num_ret_seq//2, temperature=2.1, do_sample=True, early_stopping=True) #no_repeat_ngram_size=2
            #parameters chosen after careful evaluation: sample=True produces more diverse and longer responses. This is better for evaluation on later reddit data which also tends to be longer. and this temperature circa 2 produces diverse responses too
                #The mode sample=True does not use a length penalization, so in a sense is not summarization, early_stopping=True increases variation in text
            
            translated_smrzd = model.generate(**batch, max_length=60, num_beams=8, num_return_sequences=num_ret_seq//2, do_sample=False,early_stopping=True)
            #choosing to make another few samples that are summarizations. This means that are model will be length invariant since da classes should have a mix or short and long representations
            
            tgt_text_pp = tokenizer.batch_decode( translated_pp, skip_special_tokens=True )
            tgt_text_smrzd = tokenizer.batch_decode( translated_smrzd, skip_special_tokens=True )
            tgt_text = list(set(tgt_text_pp + tgt_text_smrzd))
            
            #Adding question mark on the end if it was removed.
            ends_in_qmark = txt[-1] == "?"
            if ends_in_qmark:
                tgt_text = [ txt2.rstrip('.')+"?" if txt2[-1]!="?" else txt2 for txt2 in  tgt_text ]

            li_tgt_text.append( tgt_text )
        else:
            li_tgt_text.append([])

    return li_tgt_text

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    
    completed= False
    while completed == False:
        try:
            gc.collect()
            main()
        except Exception as e:
            print(e) 
            print("\n")
            print(traceback.print_exc())
            print("\n\nRestarting Script\n\n")
            pass

