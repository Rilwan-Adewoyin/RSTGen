# Workflow

# Split Dataset into 12 new files
# Then perform the following operations in parrallel
    # preprocess
    # add DA and RST annotations

# Then reform dataset into one file and shuffle
# Then set 75% as training set and 25% as test


import argparse
import os
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
    
    li_fns_mrda = _get_fns( os.path.join(dir_mrda_dset,"*") ) # [fn for fn in glob.glob(pattern) ]
    li_fns_swda = _get_fns( os.path.join(dir_swda_dset,"*") ) # [fn for fn in glob.glob(pattern) ]
    li_fns_midas= _get_fns( os.path.join(dir_midas_dset,"*"))

    dict_da_map = json.load( open(utils.get_path("label_mapping.json"),"r") )
    
        
    dataset_names = ['MRDA','SWDB-DAMSL','MIDAS']
    li_li_fns = [li_fns_mrda, li_fns_swda, li_fns_midas ]
 
    usecols = { 'MRDA':[0,1,4] ,'SWDB-DAMSL':None, 'MIDAS': [0,1]    }
    names = { 'MRDA':['speaker','utterance','da'] ,'SWDB-DAMSL':['speaker','utterance','da'], 'MIDAS': ["utterance","da"]    }
    seps = {'MRDA':'|', "SWDB-DAMSL":"|", "MIDAS":"##" }

    for ds_name, li_fns in zip( dataset_names, li_li_fns):
        print(f"Operating on {ds_name} files")
        start_time = time.time()
        da_to_remove = dict_da_map[ds_name]['removed']
        
        for fn in li_fns:
            df_conv = pd.read_csv(fn, sep=seps[ds_name], header=None, dtype=str,
                            usecols=usecols[ds_name], names=names[ds_name] )
                            
            if ds_name in ['MRDA', 'SWDB-DAMSL']:
                df_conv = _annotate_dialog_act(df_conv,ds_name, dict_da_map)
                path = os.path.join(dir_dset,os.path.split(fn)[1] )
                df_conv.to_csv( path, header=True, index=False, sep="|" )
            
            elif ds_name in ['MIDAS']:
                li_df_convs = _annotate_dialog_act_v2(df_conv, ds_name, dict_da_map)
                
                for idx, df in enumerate(li_df_convs):
                    path = os.path.join(dir_dset, f"midas_{idx:05d}.txt")
                    df.to_csv( path, header=True, index=False, sep="|")
        
        time_elapsed =time.time() - start_time
        print(f"Finished {ds_name} files:\t Time Elapsed {time_elapsed} \n")   
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
    df_conv['utterance'] = df_conv['utterance'].str.replace(r'\b([\S]+)(\s+\1)+\b', r'\1')

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
    df_conv = df_conv[ df_conv!= "" ]
    
    li_df_convs = [ df_conv.iloc[idx:idx+1] for idx in range(df_conv.shape[0]) ]

    for idx in range(len(li_df_convs)):
        li_df_convs[idx] = li_df_convs[idx].append( {'new_utterance': li_df_convs[idx]['prev_utterance'].iloc[0] }, ignore_index=True )
        li_df_convs[idx] = li_df_convs[idx].iloc[::-1].drop('prev_utterance',axis=1).reindex()
        li_df_convs[idx]['speaker'] = li_df_convs[idx].index.map(str)
        li_df_convs[idx].rename( {"new_utterance":"utterance"}, axis=1,inplace=True)
        li_df_convs[idx] = li_df_convs[idx][['speaker','utterance','da']]
    
    return li_df_convs




if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()