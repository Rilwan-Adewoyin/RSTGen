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
import copy
import csv
import pandas as pd
from transformers import BartTokenizer
from functools import partial
from itertools import tee

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
from data_setup_v3 import edu_fixer, find_child_edus
from dataset_cmv.pair_repo.compute_topic_signatures import TopicSignatureConstruction

import string

#the cmv dataset is currently in a format optimized for the DYPOC model
## here we create new variants of the train, validation, test datasets for the NLG RST model
## We also create new variants of the test dataset for the PAIR Model

regex_cmv_end = re.compile( "([\.\?, ]?)[\"'\w ]{0,5}CMV([\.\?,'\" \!]{0,3})" )
regex_cmv_start = re.compile( "CMV[ ]?:[ ]?") #remove
regex_bulletpoints = re.compile( "[ ]*>[ ]*")

def main( batch_process_size = 10, mp_count=1 ):

    conversions_map = {
        "train":["nlg",'nlg_pair'],
        "val":["nlg", "nlg_pair"],
        "test":["nlg","pair","seq2seq","nlg_pair"]
    }


    dir_sourcedset = "./dataset_cmv/dyploc"
    
    batch_size = batch_process_size

    
    # iterate through train, val, test
    for dset_section in ['test','val','train']:
        
        
        # loading dataset
        with open(os.path.join(dir_sourcedset,dset_section+".jsonl") ) as f:
            li_records = [json.loads(line) for line in f]
            total_batch_count = len(li_records) // batch_size

        # Operating on the dataset in batches
        batches_completed = 0
        model_formats = conversions_map[dset_section]
        
        while len(li_records) > 0:

            batch_li_records = li_records[:batch_size]
            print(f"\n\tOperating on batch {batches_completed} of {total_batch_count}")

            # Preprocess the reference text 
            cs =  max( 3, math.ceil( batch_size / (mp_count) ) )

            with mp.Pool(mp_count) as pool:
            
                res = pool.imap( preprocess , _chunks(batch_li_records, cs) )
                res_1, res_2 = tee(res,2)

                #imap paths: 
                    #nlg: rst_tree_parse_records, dyploc_to_nlg
                    #pair: salience_keywords_parse,  dyloc_to_pair
                    #nlg_pair: rst_tree_parse_records, salience_keywords_parse, dyploc_to_pair_nlg
                    #seq2seq: convert_dyploc_to_seqseq

                # setting up imap pipes
                res_rst = pool.imap( rst_tree_parse_records, res_1)

                res_rst_skw = pool.imap( salience_keywords_parse, res_rst)
                rrs_1, rrs_2 = tee(res_rst_skw,2)

                res_rst_skw_edu = pool.imap(li_edu_parse_records, rrs_1)
                rrse_1, rrse_2 = tee( res_rst_skw_edu, 2)

                res_dyploc_to_nlg = pool.imap( convert_dyploc_to_nlg, rrse_1 )
                res_dyploc_to_nlgpair = pool.imap( convert_dyploc_to_nlgpair, rrse_2  )

                if dset_section in ['val','test']:
                    res_dyploc_to_pair = pool.imap( convert_dyploc_to_pair, rrs_2 )
                    res_dyploc_to_seq2seq = pool.imap( convert_dyploc_to_seqseq, res_2  )

                #getting results
                li_recs_nlg = sum( list(res_dyploc_to_nlg), [] )
                li_recs_nlgpair = sum( list(res_dyploc_to_nlgpair), [] )


                if dset_section in ['val','test']:
                    li_recs_pair = sum( list(res_dyploc_to_pair), [] )
                    li_recs_seq2seq = sum( list(res_dyploc_to_seq2seq), [] )   

            # packing results
            dict_li_records = {'nlg_pair':li_recs_nlgpair,
                    'nlg':li_recs_nlg}
            if dset_section in ['test']:
                dict_li_records['pair'] = li_recs_pair
                dict_li_records['seq2seq'] = li_recs_seq2seq        

            #saving
            for m_format in model_formats :
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

        li_records[idx]['reference'] = reference
        li_records[idx]['title'] = title
            
    return li_records

def rst_tree_parse_records(li_records):

    # region Parsing RST trees from reference sentences
    li_refs = [ record['reference'] for record in li_records ]
    
    li_li_unparsed_tree = parser_wrapper3.main( json_li_li_utterances= json.dumps([li_refs]), 
                                                skip_parsing=False, redirect_output=True)

    li_unparsed_tree = sum( li_li_unparsed_tree, [] )
    li_subtrees = _parse_trees(li_unparsed_tree)
    
    li_rst_dict = [ _tree_to_rst_code(_tree) if _tree!=None else None for _tree in li_subtrees ]

    # Attaching the rst trees to the records
        # and removing records which could not be parsed
    for idx in reversed( range(len(li_records))  ):
        if li_rst_dict[idx] == None:
            li_rst_dict.pop(idx)
            li_records.pop(idx)
        else:
            li_records[idx]['rst'] = li_rst_dict[idx]
    
    # endregion

    return li_records

def li_edu_parse_records(li_records):
    # getting edus
        # for each record
            # Divide text into edus with positions
            # then get a list of keyphrases from the context
            # get keyphrase position of each context word in target
    li_records = copy.deepcopy(li_records)

    li_refs = [ record['reference'] for record in li_records ]
    
    li_textwedutoken = parser_wrapper3.main( json_li_li_utterances= json.dumps([li_refs]), 
                                                skip_parsing=True, redirect_output=True)
        # corrects Parser wrapper seperates at apostrophes
    li_li_edus = edu_fixer( li_textwedutoken )

        # for each utterance, merge list of words into one text
    li_li_edus = [ [ ' '.join( edus ) for edus in li_edus ] for li_edus in li_li_edus ]

    
    li_li_edus = [ [edutxt.replace(" n't", "n't").replace(" / ", "/").replace(" '", "'").replace("- LRB -", "(").replace("- RRB -", ")").replace("-LRB-", "(").replace("-RRB-", ")")
                    if edutxt not in origtext else edutxt for edutxt in li_edutext ] for li_edutext, origtext in zip( li_li_edus, li_refs) ]

    for idx in range(len(li_records)):
        li_records[idx]['li_edus'] = li_li_edus[idx]

    return li_records

def salience_keywords_parse(li_records):
    #extract kp_set_str from dyploc context
        # get kp_set_str from actual reference text based on word salience
        # use compute_topic_signatures script
    tsc = TopicSignatureConstruction(li_records)
    tsc.receive_data()
    tsc.calculate_llr()
    li_records = tsc.return_records() #adds a 'kp_set_str' to each record


    return li_records

def convert_dyploc_to_nlg( li_records ):

    li_records = copy.deepcopy(li_records)

    for idx in range(len(li_records)):
        li_records[idx].pop('kp_set_str')

    
    #positioning edus
    for idx in range(len(li_records)):
        
        li_rst_pos = [ rst_node['pos'] for rst_node in li_records[idx]['rst'] ]
        li_child_pos =  sum( [ find_child_edus(pos, li_rst_pos ) for pos in li_rst_pos ], [] )

        li_edu = li_records[idx].pop('li_edus')

        dict_pos_edu = { edu_pos:edu for edu_pos, edu in zip( li_child_pos, li_edu ) }
        
        li_records[idx]['dict_pos_edu'] = dict_pos_edu
    

    #extracting keyphrases

    for idx in range(len(li_records)):
        branch_input = li_records[idx]['branch_input']

            #extracting core concepts
        li_concepts = [ dict_['concepts'] for dict_ in branch_input]
        li_target_entity = [ dict_['target_entity'] for dict_ in branch_input]  
        li_claims = [ dict_['claims'] for dict_ in branch_input if 'claims' in dict_ ]

            #flattening
        li_concepts = sum(li_concepts, [])
        #li_target_entity = sum(li_target_entity, [])

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
        li_records[idx]['prompt'] = li_records[idx].pop('title')

        li_records[idx].pop('dyploc_context')
        li_records[idx].pop('dict_pos_edu')
        
        li_records[idx].pop('reference_sentences')
        li_records[idx].pop('branch_input')
        li_records[idx].pop('sentence_types')
    
    return li_records

def convert_dyploc_to_pair( li_records ):
    # extract pair from the reference text
    li_records = copy.deepcopy(li_records)

    for idx in range(len(li_records)):
        li_records[idx].pop('rst')


    #region use kp_set_str to convert reference to template
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")


    for idx in range(len(li_records)):
            
        li_records[idx]['template'] = pair_template_maker( li_records[idx]['kp_set_str'], li_records[idx]['reference'], tokenizer )

        li_records[idx]['prompt'] = li_records[idx].pop('title')

        #removing unwanted info

        li_records[idx].pop('reference_sentences')
        li_records[idx].pop('branch_input')
        li_records[idx].pop('sentence_types')

    #endregion


    return li_records

def pair_template_maker(  kp_set_str, reference, tokenizer ):

    # Find the position of each tokenized word in the tokenized text
        # Must use the tokenizer used by the generation model

    li_kp_str = kp_set_str.split( ' <s>') #words must have space at the front
        
    reference_tokenized = tokenizer.encode( ' '+reference.lower().translate(str.maketrans('', '', string.punctuation)) , add_special_tokens=False )

    kp_set_str_tokenized = tokenizer.batch_encode_plus( li_kp_str,  , add_special_tokens=False )['input_ids']

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

    return template_text

def convert_dyploc_to_nlgpair(li_records):    
    
    li_records = copy.deepcopy(li_records)
    
    # form dict_pos_kp from kp_set_str and li_edus
    
    #positioning edus
    for idx in range(len(li_records)):
        
        li_rst_pos = [ rst_node['pos'] for rst_node in li_records[idx]['rst'] ]
        li_child_pos =  sum( [ find_child_edus(pos, li_rst_pos ) for pos in li_rst_pos ], [] )

        li_edu = li_records[idx].pop('li_edus')

        dict_pos_edu = { edu_pos:edu for edu_pos, edu in zip( li_child_pos, li_edu ) }
        
        li_records[idx]['dict_pos_edu'] = dict_pos_edu

    # positioning keyphrase using dict_pos_edu and kp_set_str
    for idx in range(len(li_records)):
        
        li_records[idx]['li_pos_kp'] = [] # a list of lists. each sublists holds pos and keyphrase

        kp_set_str = li_records[idx]['kp_set_str']

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
        li_records[idx]['prompt'] = li_records[idx].pop('title')

        li_records[idx].pop('kp_set_str')
        li_records[idx].pop('dict_pos_edu')
        
        li_records[idx].pop('reference_sentences')
        li_records[idx].pop('branch_input')
        li_records[idx].pop('sentence_types')
    
    return li_records

def convert_dyploc_to_seqseq( li_records ):
    
    li_records = copy.deepcopy(li_records)
    
    for idx in range(len(li_records)):
        li_records[idx]['prompt'] = li_records[idx].pop('title')
        
        li_records[idx].pop('reference_sentences')
        li_records[idx].pop('branch_input')
        li_records[idx].pop('sentence_types')    
    
    return li_records


def _save_data(li_records, dset_section, m_format):

    # Split list of utterances by the subreddit name
    # Then for each sublist
        # Get directory save name
        # Get the last saved csv file in directory
        # (fn-format = file_number_utterances in file )
            # then append more lines
                

    li_record_enc = [ { str(k):json.dumps(v) for k,v in dict_.items() } for dict_ in li_records ]
                
    save_dir = os.path.join( "./dataset_cmv", m_format)
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
            
if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False) 
    
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
    
    parser.add_argument('-bps','--batch_process_size', default=3,
                             help='',type=int)        
   
    parser.add_argument('--mp_count', default=1, type=int)

    args = parser.parse_args()
    
    dict_args = vars(args)

    main(**dict_args)
