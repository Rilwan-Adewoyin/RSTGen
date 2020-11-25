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

    # parser.add_argument(
    #     "--min_length",
    #     default=9, type=_positive_int,
    #     help="The minimum length of an utterance to include.")

    # parser.add_argument(
    #     "--max_length",
    #     default=200, type=_positive_int,
    #     help="The maximum length of an utterance to include.")

    parser.add_argument(
        "--output_dir", required=False,
        default= utils.get_path("combined_data/"),
        help="Output directory to write the dataset.")

    # parser.add_argument(
    #     "--dataset_format",
    #     choices={_TF_FORMAT, _JSON_FORMAT},
    #     default="TF",
    #     help="The dataset format to write. 'TF' for serialized tensorflow "
    #          "examples in TFRecords. 'JSON' for text files with one JSON "
    #          "object per line.")

    # parser.add_argument(
    #     "--train_split", default=0.9,
    #     type=float,
    #     help="The proportion of data to put in the training set.")

    # parser.add_argument(
    #     "--num_shards_test", default=100,
    #     type=_positive_int,
    #     help="The number of shards for the test set.")

    # parser.add_argument(
    #     "--num_shards_train", default=1000,
    #     type=_positive_int,
    #     help="The number of shards for the train set.")
    
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
        subset ([type]): [description]

    Raises:
        # OSError: [description]
    """
    
    dir_mrda_dset = os.path.join(args.dir_mrda, str_subset)
    dir_swda_dset = os.path.join(args.dir_swda, str_subset)

    dir_dset = os.path.join(args.output_dir, str_subset)
    os.makedirs(dir_dset,exist_ok=True)
    
    li_fns_mrda = _get_fns( os.path.join(dir_mrda_dset,"*") ) # [fn for fn in glob.glob(pattern) ]
    li_fns_swda = _get_fns( os.path.join(dir_swda_dset,"*") ) # [fn for fn in glob.glob(pattern) ]

    dict_da_map = json.load( open(utils.get_path("label_mapping.json"),"r") )
    
        
    dataset_names = ['MRDA','SWDB-DAMSL']
    li_li_fns = [li_fns_mrda, li_fns_swda ]
    usecols = { 'MRDA':[0,1,4] ,'SWDB-DAMSL':None    }

    for name, li_fns in zip( dataset_names, li_li_fns):
        
        da_to_remove = dict_da_map[name]['removed']
        
        for fn in li_fns:
            
            df_conv = pd.read_csv(fn, sep="|", header=None, dtype=str,
                usecols=usecols[name], names=['speaker','utterance','da'] )

            df_conv = _annotate_dialog_act(df_conv,name, dict_da_map)

            path = os.path.join(dir_dset,os.path.split(fn)[1] )
            df_conv.to_csv( path, header=True, index=False, sep="|" )

    return True

def _get_fns(pattern):
    li_fns = [fn for fn in glob.glob(pattern) ]
    return li_fns

def _annotate_dialog_act(df_conv, ds_name, dict_da_map):

    # convert labels to full schema
    df_conv['da'] = df_conv['da'].apply( lambda da: dict_da_map[ds_name]['short2full'][da] )

    # remove repeated consecutive words in utterance
    df_conv['utterance'] = df_conv['utterance'].str.replace(r'\b([\w\-,\.]+)(\s+\1)+\b', r'\1')

    #remove da if in removed list, given utterance less than 5 words and does not have a MCONV mapping
    remove = '|'.join(dict_da_map[ds_name]['removed'])
    pattern = re.compile(r'\b('+remove+r')\b', flags=re.IGNORECASE)


    df_conv['da'] = df_conv.apply( 
        lambda row: pattern.sub("", row.da) 
        if len(row.da.split())<=4  and (row.da not in list(dict_da_map[ds_name]['MCONV'].keys()))
        else row.da, axis=1)

    #compress lines together if same speaker
    df_conv['key'] = (df_conv.speaker!=df_conv.speaker.shift(1)).cumsum()
    df_conv = df_conv.groupby(  ['key','speaker'], sort=False ).agg( lambda x: ' '.join(x) ).reset_index().drop('key',axis=1)
    
    #stripping final full-stop from end of sentence, depracated
    #df_conv['utterance'] = df_conv['utterance'].apply(lambda utt: utt.rstrip('.') )


    # remove whole utterance if no other dialoge act left
    df_conv = df_conv[ df_conv.da.replace(" ", "") != "" ].reset_index(drop=True)

    #
    #df_conv['da'] = df_conv['da'].apply( lambda da: " ".join( sorted( set(da.split() ) ) ) )

    def _parse_list_or_string(da):
        _li = sum( [ dict_da_map[ds_name]['MCONV'][word] for word in da.split() ], [])
        _set = set(_li)
        _set = sorted(_set)

        if "other" in _set:
            a = 1

        _str = " ".join(_set)
        return _str

        #read in file as a pandas dataframe and filter out coloumns to keep
    # convert das to mconv full schemes and remove repeated consecutive dialogue acts
    df_conv['da'] = df_conv['da'].apply( lambda da: _parse_list_or_string(da) )

    return df_conv


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()