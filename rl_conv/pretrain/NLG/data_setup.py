import sys, os
#sys.stdout = codecs.getwriter(encoding)(sys.stdout)
import traceback

import convokit
from convokit.model import ConvoKitMeta
sys.path.append(os.path.dirname(sys.path[0])) # Adds higher directory to python modules path
sys.path.append( os.path.join( os.path.dirname(sys.path[0]),"DialogueAct" ) ) 
import numpy
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import convokit
from convokit import Corpus, download
import argparse
import utils_nlg
import random
from train import TrainingModule, DaNet
import pytorch_lightning as pl
import emoji
from transformers import AutoTokenizer, AutoModel
import docker
import math
import itertools

import nltk
nltk.download('stopwords')
import rake_nltk
import json
import pytextrank
import spacy
import en_core_web_sm
    #install using python -m spacy download en_core_web_sm
nlp = en_core_web_sm.load()
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

import csv
import pickle
import time
import torch

from utils import get_best_ckpt_path
from utils import get_path

import regex as re
import multiprocessing as mp



from unidecode import unidecode

last_batch_processed = 0
dict_args = {}

pattern_hlinks =  re.compile("(\[?[\S ]*\]?)(\((https|www|ftp|http)?[\S]+\))")
pattern_hlinks2 =  re.compile("([\(]?(https|www|ftp|http){1}[\S]+)")
pattern_repword = re.compile(r'\b([\S]+)(\s+\1)+\b')
pattern_repdot = re.compile(r'[\.]{2,}')
pattern_qdab = re.compile("[\"\-\*\[\]]+")
pattern_multwspace = r'[\s]{2,}'

r1 = rake_nltk.Rake( ranking_metric=rake_nltk.Metric.DEGREE_TO_FREQUENCY_RATIO,max_length=3)

# Iterate through Conversations
        # Convert conversation to a pd.Dataframe or numpy array in the format used for Dialog Act Datasets
            # Coloumns = [Speaker, utterance, Dialog Act, RST, topics ]
        # Preprocess utterances
            # replace image links with token [Image_link], leave the [description of image_link that is in square brackers]
            # replace other links with token [link]
            # Afer utterance is empty, or only contains and [Image_link] token then remove utterance
        # Use pre-trained DA model to assign DA tag based on current utterance and previous utterance
            # Create all Dialog Acts for the whole conversation at once by passing the data in 
            # TO-DO: make sure da classifier ignores [image_link]
        # Use the pre-made RST identifier to create the RST tags for each utterance
            # Create all at once for the whole conversation
            # TO-DO: make sure rst classifier ignores [image_link]
        # Use a topic identifier to ascertain the topics of interest
        # Save Conversation to a file 
            # use the unique id of the sub-reddit
                # and the number of the conversation tree
    

def main(danet_vname,
            batch_process_size=20,
            batch_save_size=-1,
            rst_method="feng-hirst",
            mp_count=5,
            start_batch=0,
            **kwargs):
    """[summary]

    Args:
        danet_vname ([type]): [description]
        batch_process_size (int, optional): [description]. Defaults to 200.
        batch_save_size (int, optional): [description]. Defaults to -1.
        rst_method (str, optional): [description]. Defaults to "feng-hirst".
    """
    
    #region  Setup    
    # setting up model for Da prediction
    model_dir = utils_nlg.get_path(f'../DialogueAct/models/{danet_vname}')
    checkpoint_dir = f'{model_dir}/logs'

    checkpoint_path = get_best_ckpt_path(checkpoint_dir) #TOdO: need to insert changes from da branch into nlg branch
    
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # init_params = checkpoint['hyper_parameters']
    # init_params.pop('mode')
    mparams = argparse.Namespace(**json.load( open( os.path.join(model_dir,"mparam.json"),"r" ) ) )
    tparams = argparse.Namespace(**json.load( open( os.path.join(model_dir,"tparam.json"),"r" ) ) )

    danet = DaNet(**vars(mparams))
    
    danet_module = TrainingModule(mode='inference', model=danet)
    danet_module.load_state_dict(checkpoint['state_dict'] )

    danet_module.eval()
    danet_module.model.eval()
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(get_path('../DialogueAct/models/bert-base-cased') )

    # setting up docker image for rst
        #Platform dependent way to start docker
        # linux: If an error in next line, changeother "other" permission on /var/run/docker.sock to read write execute chmod o=7 /var/run/docker.sock
    try:
        client = docker.from_env(timeout=int(60*3))
    except docker.DockerException:
        os.system('chmod o=rwx var/run/docker.sock')
        client = docker.from_env(timeout=int(60*60))
    
    image_name = 'akanni96/feng-hirst-rst-parser'
    image_li = client.images.list('akanni96/feng-hirst-rst-parser')
    
        # (building)docker image
    if len(image_li)==0: 
        #Use this method during develomment
        dir_dockferfile =  utils_nlg.get_path("../DockerImages/feng-hirst-rst-parser",_dir=True)
        image = client.images.build(path=dir_dockferfile,
            nocache=True, pull=False, rm=True,forcerm=True,
            tag="akanni96/feng-hirst-rst-parser")[0]
        
    else:
        image = client.images.get(image_name)

    li_fh_container = [ client.containers.run(image, detach=True, 
        entrypoint=None,command="/bin/bash",auto_remove=True,tty=True) for x in range(mp_count) ]  

    li_fh_container_id = [container.id for container in li_fh_container ]     

    #Creating Save directory
    dir_save_dataset = utils_nlg.get_path("./dataset/reddit_small_mc",_dir=True)
    
    # setting up corpus data
    corpus = _load_data()
    li_id_dictconv  = list(corpus.conversations.items())
    total_batch_count = math.ceil(len(li_id_dictconv)/batch_process_size)
    batches_completed = 0

    if start_batch != 0:
        li_id_dictconv = li_id_dictconv[ start_batch*batch_process_size: ]
        batches_completed = start_batch
    # endregion
    
    timer = Timer()
    #region operating in batches
    
    while len(li_id_dictconv) > 0:

        batch_li_id_dictconv =  li_id_dictconv[:batch_process_size]
        batch_li_li_thread_utterances = []
        print(f"\nOperating on batch {batches_completed} of {total_batch_count}")
        last_batch_processed = batches_completed

        #region preprocessing
        timer.start()
        for id_dictconv  in batch_li_id_dictconv:
            conv_id  = id_dictconv[0]
            dictconv = id_dictconv[1]
            tree_conv_paths = dictconv.get_root_to_leaf_paths() 
            paths_count = len(tree_conv_paths)

            # Gathering utterances in thread
            li_thread_utterances = [
                {'text': utt.text, 'subreddit':utt.meta['subreddit'],
                'reply_to':utt.reply_to, 'id_utt':utt.id,
                'speaker_id':utt.speaker.id
                } for utt in dictconv.get_chronological_utterance_list()]
            
            # Preprocess each utterance -> txt_preproc
            [
                _dict.update({'txt_preproc':_preprocess(_dict['text'])}) for _dict in 
                li_thread_utterances]
            

            # Removing invalid lines in conversation
            li_thread_utterances  = [
                _dict for _dict in li_thread_utterances
                if _valid_utterance(_dict['txt_preproc']) ]

            batch_li_li_thread_utterances.append(li_thread_utterances)
        timer.end("preprocessing")
        
        

        #endregion
        
        #region DA assignment
        timer.start()
        for i, _ in enumerate(batch_li_li_thread_utterances):
            
            li_thread_utterances = batch_li_li_thread_utterances[i]

            li_utt_prevutt = [
                [ _select_utt_by_reply( _dict['reply_to'], li_thread_utterances), _dict['txt_preproc']  ]
                for _dict in li_thread_utterances
            ]

            encoded_input =  tokenizer(li_utt_prevutt, add_special_tokens=True, padding='max_length', 
                truncation=True, max_length=160, return_tensors='pt', return_token_type_ids=True)
            
            pred_da = danet_module.forward(encoded_input)
            li_li_da, li_dict_da = danet_module.format_preds(pred_da)
            
                #sequence of vectors, vector=logit score for each da class           
            #TODO consider removing dialogues where 'reply_to'==None

            [
                _dict.update({'li_da': li_da,'dict_da':dict_da }) for _dict, li_da, dict_da in 
                zip( li_thread_utterances, li_li_da, li_dict_da)
            ]

            batch_li_li_thread_utterances[i] = li_thread_utterances
        timer.end("DA")   
        #endregion

        #region Predicting the RST Tag
        timer.start()
        mp_count_rst = mp_count
        #containers = li_fh_container_id*int( (len(batch_li_li_thread_utterances)//mp_count_rst) + 1)
        contaiers =  [ li_fh_container for idx in range(int( (len(batch_li_li_thread_utterances)//mp_count_rst) + 1)) ]
        containers = sum(contaiers, [])
        with mp.Pool(mp_count_rst) as pool:

            res = pool.starmap( _rst_v2, zip( _chunks(batch_li_li_thread_utterances, batch_process_size//mp_count_rst , containers ) ) )
        batch_li_li_thread_utterances = list( res ) 
        batch_li_li_thread_utterances = sum(batch_li_li_thread_utterances, [])
        batch_li_li_thread_utterances = [ li for li in batch_li_li_thread_utterances if li !=[] ]
        timer.end("RST")

        #endregion

        #region Topic extraction
        timer.start()
        with mp.Pool(mp_count) as pool:
            res = pool.map( _topic, _chunks(batch_li_li_thread_utterances,batch_process_size//mp_count) )
        batch_li_li_thread_utterances = list( res ) 
        batch_li_li_thread_utterances = sum(batch_li_li_thread_utterances, [])
        timer.end("Topic")
        #endregion

        # region Drop keys
        for i, _ in enumerate(batch_li_li_thread_utterances):
            li_thread_utterances = batch_li_li_thread_utterances[i]
            for idx in range(len(li_thread_utterances)):
                li_thread_utterances[idx].pop('reply_to')
                li_thread_utterances[idx].pop('id_utt')
                li_thread_utterances[idx].pop('speaker_id')
            
            batch_li_li_thread_utterances[i] = li_thread_utterances
        # end region

        #region Saving Batches
        timer.start()
            # format = subreddit/convo_code
        li_utterances = list(itertools.chain.from_iterable(batch_li_li_thread_utterances))
        _save_data(li_utterances, batch_save_size, dir_save_dataset)

        li_id_dictconv = li_id_dictconv[batch_process_size:]
        batches_completed += 1
        timer.end("Saving")
        #end region    

def _load_data():
    # Donwload reddit-corpus-small if it doesnt exist
    _dir_path = utils_nlg.get_path("./dataset/reddit_small")
    if os.path.exists(_dir_path):
        use_local = True
    else:
        use_local = False
        os.makedirs(_dir_path, exist_ok=True)

    corpus = Corpus(filename=download("reddit-corpus-small", data_dir=_dir_path, use_local=use_local), merge_lines=True)
    
    return corpus

def _preprocess(text):
    # removing leading and trailing whitespace characters
    text = text.strip()
    # replacing hyperlinks with [link token]
    text = re.sub(pattern_hlinks,r"\1", text)

    # remove general hyperlinks
    text = re.sub(pattern_hlinks2,"", text)

    # replace emojis
    text = emoji.demojize( text )
        #TODO: need to remember to include emoji.emojize in the NLG decoding pipeline
        #TODO: "I'm sorry if that's how your interactions with friends go :sad_but_relieved_face:\n\nBut
                #  that's not how any of my whatsapps with my friends happened.

                #is converted to 

                #"[CLS] I'm sorry if that's how your interactions with friends go : sad _ but _ relieved _ face : But that's
                #  not how any of my whatsapps with my friends happened. [SEP]"

                #so need to have regex code to compress words between two colons (with underscores)
        
    # remove repeated words
    text = re.sub(pattern_repword, r'\1', text)

    #remove repeated periods
    text = re.sub(pattern_repdot, ".", text)

    # convert to ascii
    text = text.encode('ascii',errors='ignore').decode('ascii').replace("{", "").replace("}", "")

    #remove qoutes, dash and asterix, brackets
    text = re.sub(pattern_qdab, "", text)

    # remove multiple spaces
    text = re.sub(pattern_multwspace, ' ', text)

    return text


def _valid_utterance(txt):

    txt = txt.encode('ascii',errors='ignore').decode('ascii').replace("{", "").replace("}","")
    # Checking if someone has just drawn an image using brackets and spaces etc

    # not all space characters = not txt.ispace()
    # contains_alphabet = any( c.isalpha() for c in txt)

    # post_not_deleted = txt!="[deleted]"

    return (not txt.isspace()) and any( c.isalpha() for c in txt) and txt!="[deleted]" and txt!="removed"


def _select_utt_by_reply(reply_to_id, li_thread_utterances ):
    try:
        prev_utterance = next( _dict['txt_preproc'] for 
         _dict in li_thread_utterances 
         if _dict['id_utt'] == reply_to_id )
    except StopIteration as e:
        prev_utterance = ''
   

    return prev_utterance

def _rst(li_li_thread_utterances, fh_container_id ):
    client = docker.from_env(timeout=int(60*3))
    fh_container = client.containers.get(fh_container_id)
    new_li_li_thread_utterances = []

    for i, _ in enumerate(li_li_thread_utterances):
        
        li_thread_utterances = li_li_thread_utterances[i]
        li_utterance  = [ thread_utt['txt_preproc'] for thread_utt in li_thread_utterances ]        
        json_li_utterance = json.dumps(li_utterance)

        cmd = ['python','parser_wrapper2.py','--li_utterances', json_li_utterance]
        exit_code,output = fh_container.exec_run( cmd, stdout=True, stderr=True, stdin=False, 
                            demux=True)
        stdout, stderr = output
        #stdout = stdout.decode('utf-8')
        # a=0
        try:
            stdout_ = json.loads(stdout)
        except (TypeError, json.JSONDecodeError) as e:
            #[ dict_.update( { 'rst': None } ) for dict_ in li_thread_utterances ]
            #li_li_thread_utterances[i] = li_thread_utterances
            continue
        #li_trees = [ nltk.tree.Tree.fromstring(pt_str) for pt_str in stdout_ ] 

        li_trees = []
        for idx, pt_str in enumerate(stdout_):
            try:
                _ = nltk.tree.Tree.fromstring(pt_str, brackets="{}")
            except ValueError:
                _ = nltk.tree.Tree.fromstring(pt_str, brackets="{}")
                pass
            li_trees.append(_)

        li_rst_dict = [ _tree_to_rst_code(_tree) for _tree in li_trees ]

        [
            thread_utterance.update( {'rst':rst_dict}) for thread_utterance, rst_dict in 
            zip( li_thread_utterances, li_rst_dict)
        ]

        #li_li_thread_utterances[i] = li_thread_utterances
        new_li_li_thread_utterances.append(li_thread_utterances)
    
    return new_li_li_thread_utterances


def _rst_v2(li_li_thread_utterances, fh_container_id ):
    client = docker.from_env(timeout=int(60*3))
    fh_container = client.containers.get(fh_container_id)
    new_li_li_thread_utterances = []

    #TODO: need to get rid of the utterances that equal == "empty" (these cases are currently )
    #li_li_thread_utterances = [ [thread_utt for thread_utt in li_thread_utterances if thread_utt['txt_preproc'] != "removed"  ] for li_thread_utterances in li_li_thread_utterances]

    li_li_utterances = [ [thread_utt['txt_preproc'] for thread_utt in li_thread_utterances ] for li_thread_utterances in li_li_thread_utterances] #li of li of utts

    json_li_li_utterance = json.dumps(li_li_utterances)

    cmd = ['python','parser_wrapper3.py','--json_li_li_utterances', json_li_li_utterance]
    exit_code,output = fh_container.exec_run( cmd, stdout=True, stderr=True, stdin=False, 
                        demux=True)
    stdout, stderr = output
    
    try:
        li_strtree = json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as e:
        print(e)
        return new_li_li_thread_utterances

    for idx, str_tree in enumerate(li_strtree):
        li_thread_utterances = li_li_thread_utterances[idx]
        li_subtrees = []

        for idx, pt_str in enumerate(str_tree):
            try:
                if pt_str == '': raise ValueError
                _ = nltk.tree.Tree.fromstring(pt_str, brackets="{}")
            except ValueError:
                _ = None
                pass
            li_subtrees.append(_)

        li_rst_dict = [ _tree_to_rst_code(_tree) if _tree != None else None for _tree in li_subtrees ]

        # Keeping non erroneous utterances within a conversation - bad trees
        assert len(li_rst_dict) == len(li_thread_utterances)
        new_li_thread_utterance = []
        for idx in range(li_rst_dict):
            if li_rst_dict[idx]!=None:
                li_thread_utterances[idx].update( {'rst':li_rst_dict[idx]})
                new_li_thread_utterance.append(li_thread_utterances[idx] )
            else:
                pass

        new_li_li_thread_utterances.append(new_li_thread_utterance)
    return new_li_li_thread_utterances

def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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
    counter = 0

    for depth in range( _tree.height(),1,-1 ):
        
        subli_rels_ns = [  re.findall(r'[a-zA-Z\-]+' ,sub_tree._label)  for sub_tree in _tree.subtrees() if sub_tree.height()==depth  ]
        subli_rels_ns = [ [_li[0],''.join(_li[1:]).lstrip('unit') ] for _li in subli_rels_ns ]

        li_rels_ns.extend(subli_rels_ns)

    # Getting List 3
        #getting position of all non leave
    tree_pos = _tree.treepositions()
    leaves_pos = _tree.treepositions('leaves')
    pos_xleaves = list(set(tree_pos) - set(leaves_pos)) #unordered
    pos_xleaves = [  tuple(x if x<2 else 1 for x in _tuple ) for _tuple in pos_xleaves]        #binarizing ( correcting any number above 1 to 1)
        # reording pos_xleaves to breadfirst ordering
    li_bintreepos = sorted([ utils_nlg.tree_order.get(x,-1) for x in pos_xleaves])

    # Zipping List 1 2 and 3
    li_dict_rels_ns_bintreepos = [  {'rel':rels_ns[0], 'ns':rels_ns[1], 'pos': bintreepos } for rels_ns,bintreepos in zip(li_rels_ns,li_bintreepos) if bintreepos!=-1 ]

    return li_dict_rels_ns_bintreepos

def _topic(li_li_thread_utterances):
    for i, _ in enumerate(li_li_thread_utterances):
        li_thread_utterances = li_li_thread_utterances[i]

        li_rakekw_textankkw = [ {#'topic_rake':_rake_kw_extractor(thread_utterance['txt_preproc']),
                                    'topic_textrank':_textrank_extractor(thread_utterance['txt_preproc'])}
                for thread_utterance in li_thread_utterances]

        [
            thread_utterance.update(dict_kw) for thread_utterance, dict_kw in 
            zip( li_thread_utterances, li_rakekw_textankkw)
        ]

        li_li_thread_utterances[i] = li_thread_utterances
    return li_li_thread_utterances

def _rake_kw_extractor(str_utterance, topk=3):
    
    r1.extract_keywords_from_text(str_utterance)
    
    li_ranked_kws = r1.get_ranked_phrases_with_scores()
    
    li_ranked_kws = li_ranked_kws[:topk]

    li_ranked_kws = [ [ x[1], x[0] ] for x in li_ranked_kws ]

    return li_ranked_kws

def _textrank_extractor(str_utterance,topk=3):
    # Add the below to docker file
    # os.system(') install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz')
    # os.system python -m spacy download en_core_web_sm
    # pytextrank using entity linking when deciding important phrases, for entity coreferencing

    doc = nlp(str_utterance)
    li_ranked_kws = [ [str(p.chunks[0]), p.rank] for p in doc._.phrases ]
    return li_ranked_kws[:topk]
        
def _save_data(li_utterances, batch_save_size, dir_save_dataset):
    
    # Split list of utterances by the subreddit name
    # Then for each sublist
        # Get directory save name
        # Get the last saved csv file in directory
        # (fn-format = file_number_utterances in file )
            #if line count less than batch_save_size, then append more lines
            #otherwise make new file 

    # Grouping utterances by the subreddit 
    grouped_li_utterances = [ ( k, list(g)) for k,g in itertools.groupby(li_utterances, lambda _dict: _dict['subreddit'] ) ]
        #a list of tuples; elem0: subreddit name elem1: list of convos for that subreddit
    
    for subreddit, _li_utterances in grouped_li_utterances:
        subreddit_dir = utils_nlg.get_path( os.path.join(dir_save_dataset,subreddit), _dir=True  )  
        
        _li_utterances = [ { str(k):json.dumps(v) for k,v in dict_thread.items() } for dict_thread in _li_utterances ]
        #unlimited batch_size
        if batch_save_size < 0:
            files_ = os.listdir(subreddit_dir)
            if len(files_)>0:
                fn = files_[0]
            else:
                fn = "0000_0000000000"
                with open( os.path.join(subreddit_dir,fn),"a+",newline=None,encoding='utf-8') as _f:
                    dict_writer = csv.DictWriter(_f,fieldnames=list(_li_utterances[0].keys() ) )
                    dict_writer.writeheader()
                    pass
            
            curr_len = int(fn[-10:])
            new_len = curr_len + len(_li_utterances)

            old_fp = os.path.join(subreddit_dir,fn)
            new_fp = os.path.join(subreddit_dir,f"{fn[:4]}_{new_len:010d}")
            
            keys = _li_utterances[0].keys()
            with open(old_fp,"a+", newline=None,encoding='utf-8') as fn:
                dict_writer = csv.DictWriter(fn, keys)
                dict_writer.writerows(_li_utterances)
            
            os.rename( old_fp, new_fp )

        #limited batch save size - saving to existing file and any new files
        else:
            while len(_li_utterances)>0:
                #todo: add os.rename to files that get appended to 
                files_ = os.listdir(subreddit_dir)

                #most recent filename saved to
                last_fn = max( files_, key=int(fn[:4]) , default=f"0000_0000000000.csv")

                # checking whether file is full
                utt_count_in_last_file = int(last_fn[-10:])
                
                # extracting contents for file to add to
                max_utt_to_add = batch_save_size - utt_count_in_last_file
                
                # making newer empty file then skipping to next round of loop
                if max_utt_to_add == 0:
                    fn = f"{int(last_fn[:4])+1:04d}_0000.csv"
                    fp = os.path.join(subreddit_dir, fn )
                    with open(fp,"a+", newline='\n',encoding='utf-8') as _f:
                        pass
                    continue
                # or saving a chunk of li_utterances and moving on
                else:
                    li_utt_to_save = li_utterances[:max_utt_to_add]
                    li_utterances = li_utterances[max_utt_to_add:]
                    

                    # defining new filename
                    fp = os.path.join(subreddit_dir, max_fn )
                    keys = li_utterances[0].keys()
                    with open(fp,"a+", newline='') as _f:
                        dict_writer = csv.DictWriter(_f, keys)
                        dict_writer.writeheader()
                        dict_writer.writerows(li_utterances)
                
            
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

    parser.add_argument('--danet_vname', default="DaNet_v007",
        help="Version name of the DaNet model to use for dialogue act classifier ",
        type=str )
    
    parser.add_argument('-bps','--batch_process_size', default=120,
        help='',type=int)        

    parser.add_argument('--batch_save_size', default=-1,
        help='',type=int)        

    parser.add_argument('--rst_method', default="feng-hirst",
        choices=['feng-hirst','akanni-rst'], type=str)

    parser.add_argument('--mp_count', default=6,
        type=int)

    parser.add_argument('--start_batch', default=0, type=int)

    args = parser.parse_args()
    
    dict_args = vars(args)
    last_batch_processed = dict_args['start_batch']
    completed = False
    while completed == False:
        try:
            main( **dict_args )
            completed = True
        except Exception as e:
            # cmd = "docker stop $(docker ps -aq) & docker rm $(docker ps -aq) & docker rmi $(docker images -a -q) "
            # os.system(cmd)
            # time.sleep(5)
            # os.system(cmd)
            # time.sleep(5)
            print(e)
            print(traceback.format_exc())

            last_batch_processed = last_batch_processed + 1
            dict_args['start_batch'] = last_batch_processed
            
        finally :
            cmd = "docker stop $(docker ps -aq) & docker rm $(docker ps -aq) & docker rmi $(docker images -a -q) "
            os.system(cmd)
            time.sleep(5)
            os.system(cmd)
            time.sleep(5)

    #last bacth = 105

