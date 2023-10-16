from ast import parse
import os
import pytextrank
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.language import Language

try:
    nlp.add_pipe("textrank", last=True)
    
except Exception as e:

    @Language.component("textrank")
    def textrank(doc):
        tr = pytextrank.TextRank()
        doc = tr.PipelineComponent(doc)
        return doc

    nlp.add_pipe("textrank", last=True)

from typing import List,Dict

import json
import contextlib

from feng_hirst_rst_parser.src.parse2 import DiscourseParser

import nltk
from rst_frameworks.utils import RstTokenizerMixin, tree_order_func
from pair_repo.compute_topic_signatures import TopicSignatureConstruction
import re
import string
import syntok.segmenter as segmenter
from itertools import groupby

# regions Patters
pattern_brackets_rm_space = re.compile('\(\s*(.*?)\s*\)')
pattern_punctuation_space = re.compile(r'\s([?.!";:#_](?:\s|$))')

pattern_promptatstart = re.compile(r"\A\[[A-Z ]{2,4}\]")
pattern_writingprompts_prefix = re.compile(r"(\[[A-Z ]{2,4}\] )(.+)")

pattern_deleted = re.compile("(\[deleted\]|\[removed\]|EDIT:)")
pattern_txt_emojis = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')
pattern_subreddits = re.compile(r"(/??r/)([^/]+)")
#endregion

#region KeyPhraseExtraction
def textrank_extractor(str_utterance, lowest_score=0.0):
    # Add the below to docker file
    # os.system('install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz')
    # os.system(' python -m spacy download en_core_web_sm ')
    # pytextrank using entity linking when deciding important phrases, for entity coreferencing

    doc = nlp(str_utterance)
    li_ranked_kws = [ [str(p.chunks[0]), p.rank] for p in doc._.phrases if p.rank>lowest_score ] #Take all bar the lowest score

    for li_txt_score in li_ranked_kws:
        li_txt_score[0] = li_txt_score[0].strip('"')
        li_txt_score[0] = re.sub( "(&gt;|>)", '', li_txt_score[0])
        li_txt_score[0] = re.sub(pattern_deleted, '', li_txt_score[0])
        li_txt_score[0] = re.sub(pattern_txt_emojis, '', li_txt_score[0])
        li_txt_score[0] = re.sub(pattern_subreddits, r'\2', li_txt_score[0])

    if len(li_ranked_kws) == 0:
        li_ranked_kws = [["",0.0]]

    return li_ranked_kws

# Used by the pair model when setting up the pair interpretation of dyploc dataset
def salience_keywords_parse(li_records: List[Dict[str,str]]):
    
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
        utterance= li_records[idx1]['reference']
        
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

#endregion

#region RST Tree related code
def parse_rst_tree( li_str :List[str], rst_parser=None, parse_edu_segments=False, parse_rst_tree=False ):
    """
        Returns list of RST Encodings and 
    """
    if rst_parser == None:
        rst_parser = DiscourseParser(verbose=False, global_features=False )

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):

        output = rst_parser.parse_li_utterances( li_str, 
                                                    parse_edu_segments=parse_edu_segments, 
                                                    parse_rst_tree=parse_rst_tree)
        if parse_edu_segments and parse_rst_tree:
            li_text_w_edu_token, li_unparsed_tree =output
        
        elif parse_edu_segments:
            li_text_w_edu_token = output
        
        elif parse_rst_tree:
            li_unparsed_tree = output
        

    # Parsing RST tree
    if parse_rst_tree:
        li_unparsed_tree = li_unparsed_tree
        li_subtrees = _parse_trees(li_unparsed_tree)    
        li_rst_dict = [ _tree_to_rst_code(_tree) if _tree!=None else None for _tree in li_subtrees ]

    # Getting EDU sequence
    if parse_edu_segments:
        li_li_edus = edu_fixer( li_text_w_edu_token,  li_str )
    
    
    if parse_rst_tree and parse_edu_segments:
        return li_rst_dict, li_li_edus
    elif parse_edu_segments:
        return li_li_edus
    elif parse_rst_tree:
        return li_rst_dict

def _parse_trees(li_strtree):
    
    #parses tree into an nltk object
    li_subtrees = []

    # Parsing a list of subtrees in the utterance tree li_strtree
    for idx, pt_str in enumerate(li_strtree):
        try:
            if pt_str in ['',None]: raise ValueError
            _ = nltk.tree.Tree.fromstring(pt_str, brackets="{}")
        except ValueError:
            _ = None
            pass
        li_subtrees.append(_)
    
    return li_subtrees

def _tree_to_rst_code(_tree):
    """Converst RST Tree to rst code used in NLG model

        Args:
            method (int, optional): [description]. Defaults to 1.
        
        Return:
            if method==0:
                Three lists zipped together
                List 1 Represents A Flattened version of the rst relations in the RST tree
                List 2 the nuclearity/satellite couple type e.g. N-N or NS
                List 3 The position in a binary tree of max depth 5

                #TODO: possibly figure out some way to normalize this vector
    """

    # Getting List 1 and 2
    li_rels_ns = []
    
    for depth in range( _tree.height(),1,-1 ):
        
        # sublist of rst relation and nuclearity tag
        subli_rels_ns = [  re.findall(r'[a-zA-Z\-]+' ,sub_tree._label)  for sub_tree in _tree.subtrees() if sub_tree.height()==depth  ]
        subli_rels_ns = [ [ _li[0], ''.join(_li[1:]).lstrip('unit') ] for _li in subli_rels_ns ]

        li_rels_ns.extend(subli_rels_ns)

    # Getting List 3
        #getting position of all non leave
    tree_pos = _tree.treepositions()
    leaves_pos = _tree.treepositions('leaves')
    pos_xleaves = list(set(tree_pos) - set(leaves_pos)) #unordered
    pos_xleaves = [  tuple(x if x<2 else 1 for x in _tuple ) for _tuple in pos_xleaves]        #binarizing ( correcting any number above 1 to 1)
        # reording pos_xleaves to breadfirst ordering
    #li_bintreepos = sorted([ utils_nlg.tree_order.get(x,-1) for x in pos_xleaves])
    li_bintreepos = sorted( [tree_order_func(x) for x in pos_xleaves] )

    # Zipping List 1 2 and 3
    li_dict_rels_ns_bintreepos = [  {'rel':rels_ns[0], 'ns':rels_ns[1], 'pos': bintreepos } for rels_ns,bintreepos in zip(li_rels_ns,li_bintreepos) if bintreepos!=-1 ]

    return li_dict_rels_ns_bintreepos

def _tree_to_li_du(_tree, li_results=None):
    ### Takes an RST encoded tree and extracts mutually exclusive discourse units
        # that cover the whole span of the tree. This method uses recursive operations 
        # and updates an th li_results object inplace

    direct_childs = len(_tree)
    li_child = [ _tree[idx] for idx in range(direct_childs) ]

    # Formatting children that arent trees, but are one word, by Combining consecutive one word children into an utterance
    groups = []
    keys = []
    for k, g in groupby(li_child, type):  #grouping by type= str and nltk.tree.Tree
        groups.append(list(g))      # Store group iterator as a list
        keys.append(k)

    _ = [ [' '.join(group)] if key==str else group for group,key in zip(groups,keys) ] #concatenating the string groups
    li_child = sum( _, [] )
    direct_childs = len(li_child)
    
    # Parsing children to strings
    li_du_str = [ parse_leaves(child.leaves()) if type(child)==nltk.tree.Tree else parse_leaves(child) for child in li_child ]
    
    if(li_results == None):
        li_results = []
    
    #if tree has two subnodes
    for idx in range(direct_childs):
        
        # If child was a string always add to list since it cant be broken down furhter
        if type(li_child[idx]) == str:
            li_results.append(li_du_str[idx])
            continue

        # otherwise segment to sentence
        li_segmented_utt = sent_detector.tokenize(li_du_str[idx])

        #If over two sentences long then perform the method again
        if len(li_segmented_utt) <= 2:            
            li_results.append(li_du_str[idx])
            
        elif len(li_segmented_utt) > 2 :
            #try:
            _tree_to_li_du(li_child[idx], li_results ) 
    
    return li_results

def parse_leaves(tree_leaves ):
   #     """tree_leaves is list of subsections of an annotated discourse unit
   #     ['_!Three','new', 'issues', 'begin', 'trading', 'on', 'the',
   #  'New', 'York', 'Stock', 'Exchange', 'today', ',!_', '_!and', 'one',
   #  'began', 'trading', 'on', 'the', 'Nasdaq/National', 'Market', 'System',
   #  'last', 'week', '.', '<P>!_'
    
   if type(tree_leaves) == list:
      tree_leaves = ' '.join(tree_leaves)

   #removing tree labels
   _str = re.sub('(_\!|<P>|\!_|<s>)',"", tree_leaves )
   # removing whitespace preceeding a punctuation
   _str2 = re.sub('(\s){1,2}([,.!?\\-\'])',r'\2',_str )

   _str3 = re.sub('  ',' ',_str2).strip()

   return _str3

#endregion

#region EDU related code
def edu_fixer(li_textwedutoken1, li_text1):
    
    #Temporarily remove None entries from processing
    li_textwedutoken, li_text = list( zip( *[ ( val,text)  for val,text in zip( li_textwedutoken1, li_text1) if val!= None ] ))

    li_li_edus = [ list( _split(text_wedutoken,"EDU_BREAK") )[:-1] for text_wedutoken in li_textwedutoken ]
    
    for li_edutext in li_li_edus:
        for idx2,elem in enumerate(li_edutext):
            elem.reverse() #reversing list of words in an edu
            it = enumerate(elem)
            edu_len = len(elem) 
            elem_new =  [next(it)[1]+str_ if ( idx!=edu_len-1 and (str_[0] == "'" or str_ in ["n't", ".", "?", "!", ",", "[", "]" ]) ) else str_ for idx,str_ in it]
            elem_new.reverse()

            li_edutext[idx2] = elem_new

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

    # Add None entries back in
    if len(li_li_edus)!=len(li_textwedutoken1):
        li_li_edus_with_none = [None]*len(li_textwedutoken1)
        iter_li_li_edus = iter(li_li_edus)
        
        for idx, val in enumerate(li_textwedutoken1):
            if val != None:
                li_li_edus_with_none[idx]=next(iter_li_li_edus)
        
        li_li_edus = li_li_edus_with_none

    return li_li_edus

def _split(sequence, sep):
    chunk = []
    for val in sequence:
        if val == sep:
            yield chunk
            chunk = []
        else:
            chunk.append(val)
    yield chunk

def position_edus(li_dict_rsttext):
    #Creates dict_pos_edu
    if len(li_dict_rsttext) == 0:
        return li_dict_rsttext
         
    for idx in range(len(li_dict_rsttext)):
        
        if 'dict_pos_edu' in li_dict_rsttext[idx]:
            continue            

        li_rst_pos = [ rst_node['pos'] for rst_node in li_dict_rsttext[idx]['rst'] ]
        
        li_child_pos =  sum( [ find_terminal_nodes(pos, li_rst_pos ) for pos in li_rst_pos ], [] )

        # sorting the child_pos by their rst order
        li_child_pos = sorted( li_child_pos, key= lambda pos: RstTokenizerMixin.edukp_pos_sort_function(pos) )

        li_edus = li_dict_rsttext[idx].pop('li_edus')

        dict_pos_edu = { edu_pos:edu for edu_pos, edu in zip( li_child_pos, li_edus ) }
        
        li_dict_rsttext[idx]['dict_pos_edu'] = dict_pos_edu
    
    return li_dict_rsttext

def find_terminal_nodes(pos_parentnode, li_rst_pos):
        #returns the pos of any edus child elements of the parent node(rst)
               
        li_child_pos = [2*pos_parentnode+1, 2*pos_parentnode+2 ]

        li_child_edu_pos = [ pos for pos in li_child_pos if pos not in li_rst_pos]

        return li_child_edu_pos 

#endregion

