# 1: Create a dataset that can be used by RST predictor to learn
# 2: Create a dataset that can be finetuned on using RSTGen

# Steps:
# Preprocess data: Remove Leading [txt], Remove Leading Spaces, Correct the missing gaps between it' s
# Then rst parse
# Then reform into correct text
# Then extract edu_pos info
# Then extract set of keyphrases

# FineTune this model just to use RST to predict text, no keyphrases
import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"

import string
from ast import literal_eval
import json
import argparse
import multiprocessing as mp
import math
from difflib import SequenceMatcher
from operator import itemgetter
import regex as re
import sys
import glob
import csv
import pandas as pd
import transformers
from transformers import BartTokenizer
from functools import partial
from itertools import tee
import traceback
from transformers import GPT2TokenizerFast
from filelock import Timeout, FileLock

# Docker Images Parser and RST tree labeller
mp1 = os.path.abspath(os.path.join('..'))
mp2 = "../DockerImages/feng_hirst_rst_parser"
mp3 = "../DockerImages/feng_hirst_rst_parser/src"
mp4 = "../DockerImages/feng_hirst_rst_parser/model"
modules_paths = [mp1, mp2, mp3, mp4]

for path_ in modules_paths:
    if path_ not in sys.path:
        sys.path.append(path_)
import utils_data_setup
from utils_data_setup import parse_rst_tree, textrank_extractor, position_edus, _parse_trees

from utils_nlg_v3 import non_parseable_remover, _tree_to_rst_code, _parse_trees
from DockerImages.feng_hirst_rst_parser.src.parse2 import DiscourseParser
import time

import logging
import re

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

set_global_logging_level(logging.ERROR,  ["transformers"] )
regex_wp_start = re.compile("[(WP|CW|EU|IP|CC|OT|A|TT) ]")  # remove
pattern_promptatstart = re.compile(r"\A\[[A-Z ]{2,4}\]")
pattern_writingprompts_prefix = re.compile(r"(\[[A-Z ]{2,4}\] )(.+)")

def main(batch_size=10, mp_count=1, dset_names=['train', 'valid', 'test'], start_batches=[0,0,0], end_batches=[-1,-1,-1] ):

    dir_sourcedset = "./dataset_writing_prompt/"

    dict_dname_records = {}
    for dset_name in dset_names:
        fns = glob.glob(os.path.join(dir_sourcedset, dset_name, "*") )
        fn = [fn for fn in fns if ("lock" not in fn) and ("dict_len" not in fn) ][0]
        dict_dname_records[dset_name] = pd.read_csv(fn, header=0, sep=',').to_dict('records')
        

    for idx in range(len(dset_names)):
        dset_name = dset_names[idx]
        start_batch = start_batches[idx]
        end_batch = end_batches[-1]

        if end_batch != -1:
            dict_dname_records[dset_name] = dict_dname_records[dset_name][ :batch_size*end_batch ]

        dict_dname_records[dset_name] = dict_dname_records[dset_name][batch_size*start_batch: ]
    

    for idx, dset_name in enumerate(dset_names) :
        # loading dataset
        li_records= dict_dname_records[dset_name]

        total_batch_count = int(math.ceil(len(li_records) / batch_size))
        # Operating on the dataset in batches
        batches_completed = start_batches[idx]
        while len(li_records) > 0:
            print(f"\n\t {dset_name} dset: Operating on Batch {batches_completed}: {batches_completed-start_batches[idx]} of {total_batch_count}")
            start = time.time()
            batch_li_records =  li_records[:batch_size]

            # Preprocess the reference text
            cs = max(3, math.ceil(batch_size / (mp_count)))

            # with mp.Pool(mp_count) as pool:

            #     _a =  pool.imap( preprocess, _chunks(batch_li_records, cs) )
            #     # Extract Keyphrases
            #     _b = pool.imap( key_word_parse, _a )
            #     # Extract RST structure
            #     _c = pool.imap(rst_tree_parse_records, _b) #li_edus and #full rst structure
            #     # Positioning EDUS in reference text
            #     _d = pool.imap( wrap_position_edus, _c)
            #     li_records = sum( list(_d), [] )

            li_records = []
            for li_records2 in _chunks(batch_li_records, cs):
                li_records2 = convert_(li_records2)
                li_records.append(li_records2)
            li_records = sum( li_records, [] )

            # saving
            if len(li_records)>0:
                _save_data(li_records, dset_name)

            li_records = li_records[batch_size:]
            li_records_tgt = li_records_tgt[batch_size:]
            batches_completed += 1
            end = time.time()
            iteration_time =end - start
            expected_time_left_minutes = ( (total_batch_count-batches_completed)*iteration_time )/60
            print(f"Expected {dset_name} Completion Time: {expected_time_left_minutes:.1f}" )

        print(f"Finished preparing data for {dset_name}:Batches {start_batch} to {end_batch}")

def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def convert( li_records ):
    # input
    # output [ { 'input_prompt':"", input_ , 'input_rst':parent_node  ,'target_l':left child rst nodes. "target_r":.. },  ]
    #HERE

    return li_records

def preprocess(li_records):
    
    attempts = 0
    while attempts < 100:
        attempts += 1
        try:
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
        except Exception as e:
            if attempts==100:
                raise e
            else:
                time.sleep(5)
                pass          

    for idx in reversed(range(len(li_records))):

        # joinng reference txt into one string as opposed to list of strings
        record = li_records[idx]

        prompt = record['prompt']
        reference = record['reference']

        # checking that the start of prompt is valid.
        # Removing Writing Prompts prefix codes such as [ WP ]
        valid_start = bool(re.match(pattern_promptatstart, prompt))

        if valid_start == True:
            prompt = re.sub(pattern_writingprompts_prefix, r"\2", prompt)

        # removing unicode from string and replacing with actual str
        prompt = prompt.encode('ascii', errors='ignore').decode(
            'ascii').replace("{", "").replace("}", "")
        reference = reference.encode('ascii', errors='ignore').decode(
            'ascii').replace("{", "").replace("}", "")

        # removing leading and trailing whitespace characters
        prompt = prompt.rstrip().lstrip(string.punctuation+" ")
        reference = reference.rstrip().lstrip(string.punctuation+" ")

        # reducing length to what is used in DiscoDVT experiement - done now mostly for efficacy
        prompt_maxlen = 64 #64 subwords
        reference_maxlen = 256 #512 subwords
        
        # prompt = ' '.join( prompt.split()[:prompt_maxlen] )
        # reference = ' '.join( reference.split()[:reference_maxlen] )

        prompt = tokenizer.convert_tokens_to_string( tokenizer.tokenize(prompt)[:prompt_maxlen] )
        reference = tokenizer.convert_tokens_to_string( tokenizer.tokenize(reference)[:reference_maxlen] )

        li_records[idx]['reference'] = reference
        li_records[idx]['prompt'] = prompt

    # print("Ending preprocess")
    return li_records


def key_word_parse(li_record):
    for i in range(len(li_record)):
        li_record[i]['topic_textrank'] = textrank_extractor(
            li_record[i]['reference'])
    return li_record

def rst_tree_parse_records(li_records):

    rst_parser = DiscourseParser(verbose=False, global_features=True)

    # region Parsing RST trees from reference sentences
    li_refs = [record['reference'] for record in li_records]
    li_prompt = [record['prompt'] for record in li_records]

    li_rst_dict_reference, li_li_edus_reference = parse_rst_tree(
        li_refs, rst_parser, parse_edu_segments=True, parse_rst_tree=True)

    li_rst_dict_prompt = parse_rst_tree(
        li_prompt, rst_parser, parse_edu_segments=False, parse_rst_tree=True)

    for idx in reversed( range(len(li_records))):

        if (None != li_rst_dict_reference[idx]) and (None != li_li_edus_reference[idx]) and (None != li_rst_dict_prompt[idx]) :
            li_records[idx]['rst_reference'] = li_rst_dict_reference[idx]
            li_records[idx]['li_edus_reference'] = li_li_edus_reference[idx]
            li_records[idx]['rst_prompt'] = li_rst_dict_prompt[idx]
        else:
            li_records.pop(idx)
            continue

    # print("Ending rst_tree_parse_records")
    return li_records


def wrap_position_edus(li_records):

    _l = len(li_records)

    if _l == 0:
        return li_records

    # Changing name of 'rst_reference' to 'rst' so it adhere to the method in utils_data_setup
    for idx in range(_l):
        li_records[idx]['rst'] = li_records[idx].pop('rst_reference')
        li_records[idx]['li_edus'] = li_records[idx].pop('li_edus_reference')

    li_records = position_edus(li_records)

    # Changing name of 'rst' back to 'rst_reference'
    for idx in range(_l):
        li_records[idx]['rst_reference'] = li_records[idx].pop('rst')

    return li_records


def _save_data(li_records, dset_name):

    # Split list of utterances by the subreddit name
    # Then for each sublist
    # Get directory save name
    # Get the last saved csv file in directory
    # (fn-format = file_number_utterances in file )
    # then append more lines

    try:
        # print("Starting saving")
        li_record_enc = [{str(k): (v if type(v) == str else json.dumps(
            v)) for k, v in dict_.items()} for dict_ in li_records]

        save_dir = f"./dataset_writing_prompt/{dset_name}"
        os.makedirs(save_dir, exist_ok=True)

        # Setting up fileLock so this can be run onmultiple nodes
        lock_path = os.path.join(save_dir,"file_lock") 
        lock = FileLock(lock_path, timeout=120)
        with lock:

            files_ = [fp for fp in os.listdir(save_dir) if not("lock" in fp) ]
            if len(files_) > 0:
                fn = files_[0]
            else:
                fn = "0.csv"
                with open(os.path.join(save_dir, fn), "a+", newline=None, encoding='utf-8') as _f:
                    dict_writer = csv.DictWriter(
                        _f, fieldnames=list(li_record_enc[0].keys()))
                    dict_writer.writeheader()
                    pass

            curr_len = int(fn[:-4])
            new_len = curr_len + len(li_record_enc)

            old_fp = os.path.join(save_dir, fn)
            new_fp = os.path.join(save_dir, f"{new_len}.csv")

            pd.DataFrame(li_record_enc).to_csv(
                old_fp, mode='a', header=False, index=False)
            os.rename(old_fp, new_fp)
        # print("Ending saving")

    except Exception as e:
        print(traceback.format_exc())
        raise e


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('-bs', '--batch_size', default=14*10,help='', type=int)

    parser.add_argument('--mp_count', default=14, type=int)
    
    # parser.add_argument('--dset_names', default=['train','valid', 'test'], type=eval)
    parser.add_argument('--dset_names', default=['test'], type=eval)


    parser.add_argument('--start_batches', default=[3,0,0], type=eval)
    # parser.add_argument('--start_batches', default=[0], type=eval)


    parser.add_argument('--end_batches', default=[-1,-1,-1], type=eval)
    # parser.add_argument('--end_batches', default=[100], type=eval)


    args = parser.parse_args()

    dict_args = vars(args)

    try:
        main(**dict_args)
    except Exception as e:
        print(e)
        print(traceback.format_exc())



# python3 data_setup_wp.py -bps 520 --mp_count 26


# train portions: max(1945)
#  53 - 320
# 320 - 590
# 590 - 860
# 860 - 1130
# 1130 - 1410
# 1410 - 1640
# 1640 - 1945


# val portions:
# 0 - 115
# 115 - 250

# test portions:
# 0 - 115
# 115 - 250

#srun commands
# train
# (running) srun -J t1 -p firedrake --ntasks 1 --cpus-per-task 14 --mem 20000 python3 data_setup_wp.py --dset_names "['train']"" --start_batches '[53]' --end_batches '[320]' 
# (running) srun -J t2 -p --ntasks 1 --cpus-per-task 14 --mem 30000 python3 data_setup_wp.py --dset_names "['train']" --start_batches "[320]" --end_batches "[590]"
# (running) srun -J t3 -p --ntasks 1 --cpus-per-task 14 --mem 30000 python3 data_setup_wp.py --dset_names "['train']" --start_batches "[860]" --end_batches "[1130]" 
# (running) srun -J t4 -p --ntasks 1 --cpus-per-task 14 --mem 30000 python3 data_setup_wp.py --dset_names "['train']" --start_batches "[1130]" --end_batches "[1410]"
# (running) srun -J t5 -p --ntasks 1 --cpus-per-task 14 --mem 30000 python3 data_setup_wp.py --dset_names "['train']" --start_batches "[1410]" --end_batches "[1640]"
# (running) srun -J t6 -p --ntasks 1 --cpus-per-task 14 --mem 30000 python3 data_setup_wp.py --dset_names "['train']" --start_batches "[1410]" --end_batches "[1946]"

# valid
# (running) srun -J v1 -p --ntasks 1 --cpus-per-task 14 --mem 30000 python3 data_setup_wp.py --dset_names "['valid']" --start_batches "[0]" --end_batches "[115]"


# test
# (running) srun -J ts1 -p --ntasks 1 --cpus-per-task 14 --mem 30000 python3 data_setup_wp.py --dset_names "['test']" --start_batches "[0]" --end_batches "[115]"
