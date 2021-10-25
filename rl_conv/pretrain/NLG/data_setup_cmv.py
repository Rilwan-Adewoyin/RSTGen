from ast import literal_eval
import os
import json
import argparse
import multiprocessing as mp
import math
from difflib import SequenceMatcher
from operator import itemgetter
import regex as re
import sys
import contextlib
import copy
import csv
import pandas as pd
from transformers import BartTokenizer
from functools import partial
from itertools import tee
import traceback

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

from utils_nlg_v3 import non_parseable_remover, edu_fixer, position_edus, _tree_to_rst_code, _parse_trees
from dataset_cmv.pair_repo.compute_topic_signatures import TopicSignatureConstruction

import string

#the cmv dataset is currently in a format optimized for the DYPOC model
## here we create new variants of the train, validation, test datasets for the NLG RST model
## We also create new variants of the test dataset for the PAIR Model

regex_cmv_end = re.compile( "([\.\?, ]?)[\"'\w ]{0,5}CMV([\.\?,'\" \!]{0,3})" )
regex_cmv_start = re.compile( "CMV[ ]?:[ ]?") #remove
regex_bulletpoints = re.compile( "[ ]*>[ ]*")

def main( batch_process_size = 10, mp_count=1 ):

    # conversions_map = {
    #     "train":["dyploc_rst",'dyploc_pair_rst',"dyploc_ctrl"],
    #     "val":["dyploc_rst", "dyploc_pair_rst","dyploc_ctrl"],
    #     "test":["dyploc_rst","dyploc_pair","dyploc_ctrl","dyploc_pair_rst"]
    # }
    conversions_map = {
        "train":["dyploc_ctrl"],
        "val":["dyploc_ctrl"],
        "test":["dyploc_pair"]
    }

    dir_sourcedset = "./dataset_cmv/dyploc"
    
    batch_size = batch_process_size
    
    # iterate through train, val, test
    # for dset_section in ['test','val','train']:
    for dset_section in ['test']:
        
                
        # loading dataset
        with open(os.path.join(dir_sourcedset,dset_section+".jsonl"),"r" ) as f:
            li_records = [json.loads(line) for line in f]
            total_batch_count = int( math.ceil( len(li_records) / batch_size ) )

        # Operating on the dataset in batches
        model_formats = conversions_map[dset_section]
        print(f"{model_formats}")
        batches_completed = 0
        li_records = li_records[batch_size*batches_completed:]
        print("Operating on ",dset_section," dataset")
        while len(li_records) > 0:

            batch_li_records = li_records[:batch_size]
            print(f"\n\t Batch {batches_completed} of {total_batch_count}")
            
            
            # Preprocess the reference text 
            cs =  max( 3, math.ceil( batch_size / (mp_count) ) )

            with mp.Pool(mp_count) as pool:
            
                _a =  pool.imap( preprocess , _chunks(batch_li_records, cs) )
                res_1, res_2  = tee( _a , 2)
                
                # res_dyploc_to_ctrl = pool.imap( convert_dyploc_to_ctrl, res_1  )

                # #imap paths: 
                #     # nlg: rst_tree_parse_records, dyploc_to_nlg
                #     # pair: salience_keywords_parse,  dyploc_to_pair
                #     # nlg_pair: rst_tree_parse_records, salience_keywords_parse, dyploc_to_pair_nlg
                #     # ctrl: convert_dyploc_to_ctrl

                # # setting up imap pipes
                _b = pool.imap( salience_keywords_parse, res_2)
                rrs_1, rrs_2 = tee( _b, 2 )
                
                res_dyploc_to_pair = pool.imap( convert_dyploc_to_pair, rrs_2 )
                
                # res_rst_edu = pool.imap( rst_tree_parse_records, rrs_1)
                
                # res_rst_skw_edu = pool.imap( non_parseable_remover, res_rst_edu ) # removing text with grammar so bad that it can not be parsed properly                
                # _c = pool.imap( position_edus, res_rst_skw_edu)
                # rrse_1, rrse_2 = tee(_c, 2)

                # res_dyploc_to_rst = pool.imap( convert_dyploc_to_rst, rrse_1 )
                # res_dyploc_to_pair_rst = pool.imap( convert_dyploc_to_pair_rst, rrse_2  )
                
                # getting results
                dict_li_records = {}

                # if "dyploc_ctrl" in model_formats:
                #     dict_li_records['dyploc_ctrl'] = sum( list(res_dyploc_to_ctrl), [] )  
                                    
                # if "dyploc_rst" in model_formats:
                #     dict_li_records['dyploc_rst'] = sum( list(res_dyploc_to_rst), [] )
                                                            
                if "dyploc_pair" in model_formats:
                    dict_li_records['dyploc_pair'] = sum( list(res_dyploc_to_pair), [] ) 

                # if "dyploc_pair_rst" in model_formats:
                #     dict_li_records['dyploc_pair_rst']= sum( list(res_dyploc_to_pair_rst), [] )
                
            #saving
            for m_format in model_formats :
                if m_format in dict_li_records:
                    _save_data( dict_li_records[m_format], dset_section, m_format )

            li_records = li_records[batch_size:]
            batches_completed +=1

    #  decide which models to create versions for

def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def preprocess(li_records):
    
    for idx in reversed(range(len(li_records))):

        #joinng reference txt into one string as opposed to list of strings
        record = li_records[idx]
        reference = ' '.join( record['reference_sentences'] )
        title = record['title'] 

        # remove sentences that have /u/ in them
        if "/u/" in  reference:
            li_records.pop(idx)
            continue
        
        # remove CMv from title

            # Remove CMV at end of seq
        title = re.sub( regex_cmv_end, r"\1", title)
            # Remove CMV at beginning of seq
        title = re.sub( regex_cmv_start, "", title)

        if "CMV" in  title:
            #removing record if cmv still in record
            li_records.pop(idx)
            continue

        # removing unicode from string and replacing with actual str
        reference = reference.encode('ascii',errors='ignore').decode('ascii').replace("{", "").replace("}", "")
        title = title.encode('ascii',errors='ignore').decode('ascii').replace("{", "").replace("}", "")

        #removing weird  symbols from reference
        reference = re.sub(regex_bulletpoints, "", reference)

        li_records[idx]['txt_preproc'] = reference
        li_records[idx]['title'] = title
    
    print("Ending  preprocess")
    return li_records


def salience_keywords_parse(li_records):
    
    #extract kp_set_str from dyploc context
        # get kp_set_str from actual reference text based on word salience
        # use compute_topic_signatures script
    
    tsc = TopicSignatureConstruction(li_records)
    tsc.receive_data()
    tsc.calculate_llr()
    li_records = tsc.return_records() #adds a 'kp_set_str' to each record
    print("Ending salience keywords parse")
    return li_records
    
    
def rst_tree_parse_records(li_records):

    # region Parsing RST trees from reference sentences
    li_refs = [ record['txt_preproc'] for record in li_records ]
    
    try:
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            rst_parser = DiscourseParser(verbose=False, skip_parsing=False,
                            global_features=True, segment_and_parse_tree=True)
        
            li_textwedutoken, li_unparsed_tree = rst_parser.parse_li_utterances(  li_refs )
            
            li_textwedutoken = list(li_textwedutoken)
            
            rst_parser.unload()
            
            del rst_parser

        li_subtrees = _parse_trees(li_unparsed_tree)

        li_rst_dict = [ _tree_to_rst_code(_tree) if _tree!=None else None for _tree in li_subtrees ]
    
        # Attaching the rst trees to the records
            # and removing records which could not be parsed
        for idx in reversed( range(len(li_records)) ):
            
            if li_rst_dict[idx] == None or li_textwedutoken[idx]==None :
                li_rst_dict.pop(idx)
                li_records.pop(idx)
                li_textwedutoken.pop(idx)

            elif len(li_rst_dict[idx])==1 and li_rst_dict[idx][0]['ns'] == 'a' :
                li_rst_dict.pop(idx)
                li_records.pop(idx)
                li_textwedutoken.pop(idx)

            else:
                li_records[idx]['rst'] = li_rst_dict[idx]
        
        # Now parsing the edu records 
        li_li_edus = edu_fixer( li_textwedutoken,  [ record['txt_preproc'] for record in li_records ] )
        
        for idx in range(len(li_records)):
            li_records[idx]['li_edus'] = li_li_edus[idx]

    except Exception as e:
        print(traceback.format_exc())
        raise e
    
    # print("Ending rst_tree_parse_records")
    return li_records

def convert_dyploc_to_rst( li_records ):
    print("Starting dyploc to rst")
    
    if len(li_records) == 0:
        return li_records
    try:
        li_records = copy.deepcopy(li_records)

        for idx in range(len(li_records)):
            li_records[idx].pop('kp_set_str',None)
        
        #extracting keyphrases

        for idx in reversed(range(len(li_records))):
            branch_input = li_records[idx]['branch_input']
            
            if branch_input is None:
                li_records.pop(idx)
                continue

                #extracting core concepts
            li_concepts = [ dict_['concepts'] for dict_ in branch_input if ( dict_!=None and 'concepts' in dict_ ) ]
            li_target_entity = [ dict_['target_entity'] for dict_ in branch_input if ( dict_!=None and 'target_entity' in dict_  ) ]  
            li_claims = [ dict_['claims'] for dict_ in branch_input if (dict_!=None and 'claims' in dict_)]

                #flattening
            li_concepts = sum(li_concepts, [])
            
            #handling cases where underscore is used for a target word
            li_target_entity = [ elem.replace('_'," ")  for elem in li_target_entity if elem!=None]

            li_records[idx]['dyploc_context'] = {
                'concepts':li_concepts,
                'target_entity':li_target_entity,
                'claims':li_claims,
            }

        #  position keyphrase
        for idx in range(len(li_records)):
            
            li_records[idx]['li_pos_kp'] = [] # a list of lists. each sublists holds pos and keyphrase
            li_records[idx]['li_claim'] = [] # a list of lists. each sublists holds pos and keyphrase


            dyploc_context = li_records[idx]['dyploc_context']

            for key in dyploc_context:
                dyploc_context[key] = list(set(dyploc_context[key]))

            dict_pos_edu = li_records[idx]['dict_pos_edu']
            
            for key, li_kp in dyploc_context.items():
                
                if key in ['concepts','target_entity']:
                    for kp in li_kp:
                        # FIND POS
                        # kp can span two different EDUs. So finding length, in words, of longest common substring
                        li_pos_coveragecount = []
                        for pos, edu in dict_pos_edu.items():
                            kp_split = kp.split()
                            edu_split = [ w for w in edu.translate(str.maketrans('', '', string.punctuation)).split() if w!= " "]

                            match = SequenceMatcher(None, kp_split, edu_split).find_longest_match(0, len(kp_split) ,0, len(edu_split) )
                            li_pos_coveragecount.append( [pos, match.size] )
                        
                        # If words occurs twice then the first instance is used
                        kp_pos, coverage_count = max( li_pos_coveragecount, key=itemgetter(1))
                        
                        if coverage_count > 0:
                            li_records[idx]['li_pos_kp'].append( [kp_pos, kp] ) 
                        elif coverage_count == 0:
                            #TODO: remember to map -1 to the final position (the pad position)
                            #li_records[idx]['li_pos_kp'].append( [ -1, kp] ) 
                            pass
                                
                elif key in ['claims']:
                    for kp in li_kp:
                        #li_records[idx]['li_pos_kp'].append( [-1, kp] ) 
                        li_records[idx]['li_claim'].append( kp )
        
        
        #removing unwanted info
        for idx in range(len(li_records)):
            li_records[idx]['prompt'] = li_records[idx].pop('title',None)

            li_records[idx].pop('dyploc_context',None)
            
            li_records[idx].pop('reference_sentences',None)
            li_records[idx].pop('branch_input',None)
            li_records[idx].pop('sentence_types',None)

        print("Finishing Convert dyploc to rst")
        
        return li_records

    except Exception as e:
        print(traceback.format_exc())
        raise e

def convert_dyploc_to_pair( li_records ):
    # print("Starting Convert dyploc to pair")
    
    try:
        # extract pair from the reference text
        li_records = copy.deepcopy(li_records)

        for idx in range(len(li_records)):
            li_records[idx].pop('rst',None)

        #region use kp_set_str to convert reference to template
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

        for idx in range(len(li_records)):
                
            li_records[idx]['template'] = pair_template_maker( li_records[idx]['kp_set_str'], li_records[idx]['txt_preproc'], tokenizer )

            li_records[idx]['prompt'] = li_records[idx].pop('title',None)

            #removing unwanted info
            li_records[idx].pop('reference_sentences',None)
            li_records[idx].pop('branch_input',None)
            li_records[idx].pop('sentence_types',None)
            li_records[idx].pop('li_edus',None)
            

        print("End Convert dyploc to pair") 
        return li_records
        #endregion
    
    except Exception as e:
        print(traceback.format_exc())
        raise e

def pair_template_maker(  kp_set_str, reference, tokenizer ):
    print("Starting pair template maker")

    try: 
            
        # Find the position of each tokenized word in the tokenized text
            # Must use the tokenizer used by the generation model

        li_kp_str = kp_set_str.split( ' <s>') #words must have space at the front
            
        reference_tokenized = tokenizer.encode( ' '+reference.lower().translate(str.maketrans('', '', string.punctuation)) , add_special_tokens=False )

        kp_set_str_tokenized = tokenizer.batch_encode_plus( li_kp_str,  add_special_tokens=False )['input_ids']

        li_tokens_posidx = []
        # getting position of tokenized kp in tokenized reference
        for tokens, word in zip(kp_set_str_tokenized,li_kp_str):

            start_idx = [i for i in range(0,len(reference_tokenized))
                if reference_tokenized[i:i+len(tokens)]==tokens][:1]
            
            if len(start_idx) == 1:
                li_tokens_posidx.append( [ tokens, start_idx ] )
        
        # building template
        li_tokens_posidx.sort( key=lambda sub_li: sub_li[1] ) #sort by posix
        
        template_text = []
        template_tokens = []
        mask = [None]

        #iteratively build template by place tokens and filling gaps
        # we form a template of text and tokens simultaneously
        # using template_tokens in order to correctly record positions for items in template_text
        for tokens, posidx  in li_tokens_posidx:

            next_pos = len(template_tokens)

            mask_len_before_tokens = posidx[0] - next_pos

            if mask_len_before_tokens>0:
                template_tokens.extend( [-1]*mask_len_before_tokens  )
                template_text.extend( mask*mask_len_before_tokens )
            
            # template tokens
            template_tokens.extend( tokens )
            
            # tempalate_words
            decoded_tokens = tokenizer.decode(tokens)
            li_split_text = decoded_tokens.split(' ')[1:]
            template_text.extend( li_split_text )
    
        print("Ending pair template maker")

        return template_text

    except Exception as e:
        print(traceback.format_exc())
        raise e

def convert_dyploc_to_pair_rst(li_records):    
    # print("Starting Convert dyploc to pair rst")
    
    try:
        li_records = copy.deepcopy(li_records)
        
        
        # positioning keyphrase using dict_pos_edu and kp_set_str
        for idx in reversed(range(len(li_records))):
            
            li_records[idx]['li_pos_kp'] = [] # a list of lists. each sublists holds pos and keyphrase

            if 'kp_set_str' in li_records[idx]:
                kp_set_str = li_records[idx]['kp_set_str']
            else:
                li_records.pop(idx)
                continue

            li_kp = kp_set_str.split(' <s>')

            dict_pos_edu = li_records[idx]['dict_pos_edu']
            
            
            for kp in li_kp:
                # FIND POS
                # kp can span two different EDUs. So finding length, in words, of longest common substring
                li_pos_coveragecount = []
                for pos, edu in dict_pos_edu.items():
                    kp_split = kp.split()
                    edu_split = [ w for w in edu.translate(str.maketrans('', '', string.punctuation)).split() if w!= " "]

                    match = SequenceMatcher(None, kp_split, edu_split).find_longest_match(0, len(kp_split) ,0, len(edu_split) )
                    li_pos_coveragecount.append( [pos, match.size] )
                
                # If words occurs twice then the first instance is used
                kp_pos, coverage_count = max( li_pos_coveragecount, key=itemgetter(1))
                
                if coverage_count > 0:
                    li_records[idx]['li_pos_kp'].append( [kp_pos, kp] ) 
                elif coverage_count == 0:
                    #TODO: remember to map -1 to the final position (the pad position)
                    #li_records[idx]['li_pos_kp'].append( [ -1, kp] ) 
                    pass
        
        #removing unwanted info
        for idx in range(len(li_records)):
            li_records[idx]['prompt'] = li_records[idx].pop('title', None)
            li_records[idx].pop('kp_set_str', None)
            li_records[idx].pop('reference_sentences',None)
            li_records[idx].pop('branch_input',None)
            li_records[idx].pop('sentence_types',None)
        # print("Ending Convert dyploc to pair rst")
        return li_records
    
    except Exception as e:
        print(traceback.format_exc())
        raise e

def convert_dyploc_to_ctrl( li_records ):

    if len(li_records) == 0:
        return li_records

    try:
        li_records = copy.deepcopy(li_records)
                
        for idx in reversed( range(len(li_records))):
            li_records[idx]['prompt'] = li_records[idx].pop('title',None)

            branch_input = li_records[idx]['branch_input']
            
            if branch_input is None:
                li_records.pop(idx)
                continue

                #extracting core concepts, target_entities and claims
            li_concepts = [ dict_['concepts'] for dict_ in branch_input if ( dict_!=None and 'concepts' in dict_ ) ]
            li_target_entity = [ dict_['target_entity'] for dict_ in branch_input if ( dict_!=None and 'target_entity' in dict_  ) ]  
            li_claims = [ dict_['claims'] for dict_ in branch_input if (dict_!=None and 'claims' in dict_)]

            li_concepts = sum(li_concepts, [])
            # li_claims = sum(li_claims, [])
            li_target_entity = list(set( [ x for x in li_target_entity if x is not None] ))

            #handling cases where underscore is used for a target word
            li_target_entity = [ elem.replace('_'," ")  for elem in li_target_entity]

            li_records[idx]['li_claim'] = li_claims
            li_records[idx]['li_concepts'] = li_concepts
            li_records[idx]['li_target_entity'] = li_target_entity
            
            li_records[idx].pop('reference_sentences',None)
            li_records[idx].pop('branch_input',None)
            li_records[idx].pop('sentence_types',None)    

        print("Finishing Convert Dyploc to ctrl")

        return li_records
    
    except Exception as e:
        print(traceback.format_exc())
        raise e

def _save_data(li_records, dset_section, m_format):

    # Split list of utterances by the subreddit name
    # Then for each sublist
        # Get directory save name
        # Get the last saved csv file in directory
        # (fn-format = file_number_utterances in file )
            # then append more lines
                
    try:
        # print("Starting saving")
        li_record_enc = [ { str(k):json.dumps(v) for k,v in dict_.items() } for dict_ in li_records ]
                    
        save_dir = os.path.join( "./dataset_cmv/", m_format, dset_section)
        os.makedirs(save_dir, exist_ok=True)

        
        files_ = [ fp for fp in os.listdir(save_dir) ]
        if len(files_)>0:
            fn = files_[0]
        else:       
            fn = "0.csv"           
            with open( os.path.join(save_dir,fn),"a+",newline=None,encoding='utf-8') as _f:
                dict_writer = csv.DictWriter(_f,fieldnames=list(li_record_enc[0].keys() ) )
                dict_writer.writeheader()
                pass
        
        curr_len = int(fn[:-4])
        new_len = curr_len + len(li_record_enc)

        old_fp = os.path.join(save_dir,fn)
        new_fp = os.path.join(save_dir,f"{new_len}.csv")
        
        pd.DataFrame(li_record_enc).to_csv(old_fp, mode='a', header=False, index=False)
        os.rename(old_fp, new_fp)
        print("Ending saving")
    except Exception as e:
        print(traceback.format_exc())
        raise e            

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(add_help=True)
    
    parser.add_argument('-bps','--batch_process_size', default=700,
                             help='', type=int)        
   
    parser.add_argument('--mp_count', default=12, type=int)

    args = parser.parse_args()
    
    dict_args = vars(args)

    main(**dict_args)


# python3 data_setup_cmv.py -bps 520 --mp_count 26
