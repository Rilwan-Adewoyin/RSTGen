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
import syntok.segmenter as segmenter

# Docker Images Parser and RST tree labeller
from feng_hirst_rst_parser.src import parser_wrapper3
from feng_hirst_rst_parser.src.parse2 import DiscourseParser

from rst_frameworks.utils import non_parseable_remover, edu_fixer, position_edus, _tree_to_rst_code, _parse_trees
from pair_repo.compute_topic_signatures import TopicSignatureConstruction

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
        # "train":["dyploc_ctrl"],
        # "val":["dyploc_ctrl"],
        "test":["dyploc_pair"]
    }
    # conversions_map = {
    #     # "test":["dyploc_pair",dyploc_pair_rst"],
    #     # "val":[ "dyploc_pair_rst"],
    #     # "train":['dyploc_pair_rst'],
    # }

    dir_sourcedset = ".data_files/dataset_cmv/dyploc"
    
    batch_size = batch_process_size
    
    # iterate through train, val, test
    # for dset_section in ['test','val','train']:
    for dset_section in list(conversions_map.keys()):
        
                
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
                
                # # setting up imap pipes
                _b = pool.imap( salience_keywords_parse, res_2)
                rrs_1, rrs_2 = tee( _b, 2 )
                
                res_dyploc_to_pair = pool.imap( convert_dyploc_to_pair, rrs_2 )
                
                # getting results
                dict_li_records = {}

                                                            
                if "dyploc_pair" in model_formats:
                    dict_li_records['dyploc_pair'] = sum( list(res_dyploc_to_pair), [] ) 

                
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
      
    
    # print("Ending preprocess")
    return li_records


def salience_keywords_parse(li_records):
    
    #extract kp_set_str from dyploc context
        # get kp_set_str from actual reference text based on word salience
        # use compute_topic_signatures script
        # create kp_plan_str which is the ordered version of kp_set_str
    
    tsc = TopicSignatureConstruction(li_records)
    tsc._init_totals()
    tsc.receive_data()
    tsc.calculate_llr()
    li_records = tsc.return_records() #adds a 'kp_set_str' to each record (an ordered set of kps)
    
    for idx1 in reversed(range(len(li_records))):
        utterance= li_records[idx1]['txt_preproc']
        
        if 'kp_set_str' not in li_records[idx1]:
            li_records.pop(idx1)
            continue
        
        #removing all kps less than 3 letters long
        #Moved to compute_topic_signature
        # kp_set_str = '<s>'.join([ word for word in li_records[idx1]['kp_set_str'].split('<s>') if len(word.strip())>3 ] )
        kp_set_str = li_records[idx1]['kp_set_str']
        
        li_kps = kp_set_str.split('<s>')
        li_kp_sidx_eidx = []
        
        #retrieve start and end idx for each kp in utterance
        for kp in li_kps:
            # _ = re.search(r'\b({})\b'.format(kp.strip()), utterance.lower())
            # _ = re.search(r'\b({})\b'.format(kp.strip()), utterance )
            
            _ = re.search(r'[\b]?({})\b'.format(kp), utterance ) #testing
            if _ == None:
                _ = re.search(rf'[\b]?[{string.punctuation}]?({kp})\b', utterance )
            if _ == None:
                _ = re.search(r'[\b]?({})\b'.format(kp.strip()), utterance ) #testing                
            if _ == None:
                _ = re.search(rf'[{string.punctuation}]({kp})[{string.punctuation}]', utterance ) #testing
            if _ == None:
                continue
            
            s_idx = _.start()
            e_idx = _.end()
            li_kp_sidx_eidx.append([kp, s_idx, e_idx])
        
        # retreive posiitons of sentence end markers
        li_sentence_end_pos = []
        for paragraph in segmenter.analyze(utterance):
            for sentence in paragraph:
                for token in sentence:
                    pass
                li_sentence_end_pos.append(token.offset)
        
        # Sorting list in order of sidx
        li_kp_sidx_eidx = sorted(li_kp_sidx_eidx, key=lambda subli: subli[1] )
        
        # Now create a string
        # Where we only keep sentence words that appear in the 
        # with each kp seperated by <s>
        # Also ensure that if any consecutiev kps have eidx = sidx then don't add <s>
        kp_plan_str = ''
        for idx2 in range(len(li_kp_sidx_eidx)):
            if idx2!=0:
                curr_sidx = li_kp_sidx_eidx[idx2][1]
                prev_eidx = li_kp_sidx_eidx[idx2-1][2]
                
                if curr_sidx > prev_eidx + 1 or ( curr_sidx == prev_eidx+1 and utterance[prev_eidx]=="."):
                    # kp_plan_str += " <s> "
                    kp_plan_str += " <s>" #testing
                    
                else:
                    # kp_plan_str += " "
                    kp_plan_str += "" #testing
                    
                      
            # kp_plan_str += li_kp_sidx_eidx[idx2][0].strip()
            kp_plan_str += li_kp_sidx_eidx[idx2][0] 
            
        # li_records[idx1]['kp_plan_str_1'] = kp_plan_str.strip() # <s> placed at word seperation boundaries
        li_records[idx1]['kp_plan_str_1'] = kp_plan_str # <s> placed at word seperation boundaries
                  
        #Now create a string
        # Where each sentence is divided by <s>
        # Each sentence only includes has all words not in kp removed
        kp_plan_str = ''
        for idx2 in range(len(li_kp_sidx_eidx)):
            
            #We check if the idx2 is larger than the next sentence end index in the list of sentence end indicies
            # If it is we append the  <s> and remove the next sentence end index from the list
            if li_kp_sidx_eidx[idx2][1] >= li_sentence_end_pos[0]:
                # kp_plan_str += " <s> "
                kp_plan_str += " <s>"
                li_sentence_end_pos.pop(0)
            else:                
                # kp_plan_str += " "
                if li_kp_sidx_eidx[idx2][0][0] != " ":
                    kp_plan_str += " "

            # kp_plan_str += li_kp_sidx_eidx[idx2][0].strip()
            kp_plan_str += li_kp_sidx_eidx[idx2][0]
        
        # li_records[idx1]['kp_plan_str'] = kp_plan_str.strip()
        li_records[idx1]['kp_plan_str'] = kp_plan_str
        
   
    # print("Ending salience keywords parse")
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
            li_records[idx].pop('kp_plan_str',None)
            li_records[idx].pop('kp_plan_str_1',None)

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
                
            li_records[idx]['template'] = pair_template_maker( li_records[idx]['kp_plan_str_1'], li_records[idx]['txt_preproc'], tokenizer )

            li_records[idx]['prompt'] = li_records[idx].pop('title',None)

            #removing unwanted info
            li_records[idx].pop('reference_sentences',None)
            li_records[idx].pop('branch_input',None)
            li_records[idx].pop('sentence_types',None)
            li_records[idx].pop('li_edus',None)
            

        # print("End Convert dyploc to pair") 
        return li_records
        #endregion
    
    except Exception as e:
        print(traceback.format_exc())
        raise e

def pair_template_maker(  kp_plan_str, reference, tokenizer ):
    # print("Starting pair template maker")
    """
    This uses the kp_plan_str where every word is seperated by <s> not just the sentences
    """
    
    try: 
            
        # Find the position of each tokenized word in the tokenized text
            # Must use the tokenizer used by the generation model

        li_kp_str = kp_plan_str.split(' <s>') #words must have space at the front
                   
        reference_tokenized = tokenizer.encode( reference.translate(str.maketrans('', '', string.punctuation)) , add_special_tokens=False )

        kp_plan_str_tokenized = tokenizer.batch_encode_plus( li_kp_str,  add_special_tokens=False )['input_ids']

        li_tokens_posidx = []
        # getting position of tokenized kp in tokenized reference
        for idx1, (tokens, word) in enumerate( zip(kp_plan_str_tokenized,li_kp_str) ):

            start_idx = [i for i in range(0,len(reference_tokenized))
                if reference_tokenized[i:i+len(tokens)]==tokens][:1]
            
            if len(start_idx) == 1:
                li_tokens_posidx.append( [ tokens, start_idx ] )
                        
            else:
                # adding space before first word - helps cases where first word is surrounded by punctuation
                tokens2 = tokenizer.encode( " "+li_kp_str[idx1],add_special_tokens=False)

                start_idx2 = [i for i in range(0,len(reference_tokenized))
                    if reference_tokenized[i:i+len(tokens2)]==tokens2][:1]
                
                if len(start_idx2) == 1:
                    li_tokens_posidx.append( [ tokens2, start_idx2 ] )
                    
                else:
                    # breaking up the words in the kp
                    li_subwords = word.split(' ')
                    subword_tknzed = tokenizer.batch_encode_plus( li_subwords,  add_special_tokens=False )['input_ids']
                    
                    for tokens1, word1 in zip(subword_tknzed, li_subwords):
                        start_idx1 = [i for i in range(0,len(reference_tokenized))
                            if reference_tokenized[i:i+len(tokens1)]==tokens1][:1]
                        
                        if len(start_idx) == 1:
                            li_tokens_posidx.append( [ tokens1, start_idx1 ] )
                    
        
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
            # decoded_tokens = tokenizer.decode(tokens)
            
            # li_split_text = decoded_tokens.split(' ')[1:]
            li_split_text = [ re.sub( "^ " ,"\u0120" ,tokenizer.decode(subtok)) for subtok in tokens]

            # Adding the \u0120 to start of subwords with leading space to allow interoperability with pair model
            
            template_text.extend( li_split_text )
    
        # print("Ending pair template maker")

        return template_text

    except Exception as e:
        print(traceback.format_exc())
        raise e

def convert_dyploc_to_pair_rst(li_records):    
    # print("Starting Convert dyploc to pair rst")
    
    try:
        li_records = copy.deepcopy(li_records)
        
        
        # positioning keyphrase using dict_pos_edu and kp_plan_str
        for idx in reversed(range(len(li_records))):
            
            li_records[idx]['li_pos_kp'] = [] # a list of lists. each sublists holds pos and keyphrase

            if 'kp_plan_str_1' in li_records[idx]:
                kp_plan_str = li_records[idx]['kp_plan_str_1']
                #This uses the kp_plan_str  tjat has <s> between every word not just sentences
            else:
                li_records.pop(idx)
                continue

            li_kp = kp_plan_str.split(' <s>')

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
            li_records[idx].pop('kp_plan_str',None)
            li_records[idx].pop('kp_plan_str_1',None)
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
        li_record_enc = [ { str(k):( v if type(v)==str else json.dumps(v)) for k,v in dict_.items() } for dict_ in li_records ]
                    
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
    
    parser.add_argument('-bps','--batch_process_size', default=280,
                             help='', type=int)        
   
    parser.add_argument('--mp_count', default=140, type=int)

    args = parser.parse_args()
    
    dict_args = vars(args)

    main(**dict_args)


# python3 data_setup_cmv.py -bps 520 --mp_count 26
