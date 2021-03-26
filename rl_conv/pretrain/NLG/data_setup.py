import sys, os
#sys.stdout = codecs.getwriter(encoding)(sys.stdout)
import traceback
# os.environ['CURL_CA_BUNDLE']=""
# os.environ['REQUESTS_CA_BUNDLE'] = '' 
# os.environ['PYTHONHTTPSVERIFY'] = "0"
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import convokit
from convokit.model import ConvoKitMeta
sys.path.append(os.path.dirname(sys.path[0])) # Adds higher directory to python modules path
sys.path.append( os.path.join( os.path.dirname(sys.path[0]),"DialogueAct" ) ) 
import numpy
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from itertools import groupby
import convokit
from convokit import Corpus, download
import argparse
import utils_nlg
import random
#from train import TrainingModule, DaNet
TrainingModule=None
DaNet=None
import pytorch_lightning as pl
import emoji
from transformers import AutoTokenizer, AutoModel
import docker
import math
import itertools
import pandas as pd
import nltk
nltk.download('stopwords')
import rake_nltk
import json
import pytextrank
import spacy
import en_core_web_sm
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    #install using python -m spacy download en_core_web_sm
nlp = en_core_web_sm.load()
#tr = pytextrank.TextRank()
#nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
nlp.add_pipe("textrank", last=True)

import csv
import pickle
#import dill as pickle
import time
import torch

from filelock import Timeout, FileLock


from utils import get_best_ckpt_path
from utils import get_path

import regex as re
import multiprocessing as mp
import distutils
# import torch.multiprocessing as mp
# from torch.multiprocessing import set_start_method
import html

from unidecode import unidecode
batches_completed = 0
dict_args = {}

pattern_hlinks =  re.compile("(\[?[\S ]*\]?)(\((https|www|ftp|http)?[\S]+\))")
pattern_hlinks2 =  re.compile("([\(]?(https|www|ftp|http){1}[\S]+)")
pattern_repword = re.compile(r'\b([\S]+)(\s+\1)+\b')
pattern_reppunc = re.compile( r'([\.!?])\1+' )
pattern_qdab = re.compile("[\"\-\*\[\]]+")
pattern_multwspace = re.compile(r'[\s]{2,}')
pattern_emojis = re.compile(":[^_]+?(_){1}[\S]{2,}:") #this is the pattern for emojis from the demojize
pattern_txt_emojis = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')
pattern_amp = re.compile("&(amp){1}")
pattern_toremove = re.compile("(\[deleted\]|\[removed\]|EDIT:|Edit:)")

pattern_qoutes = re.compile("(&gt;|>)[^(\\n)]*(\\n){1,}") #removing reddit qoutes 
pattern_subreddits = re.compile(r"(/??r/)([^/]+)")

pattern_writingprompts_prefix = re.compile(r"(\[[A-Z]{2}\] )(.+)")
pattern_promptatstart = re.compile(r"\A\[[A-Z]{2}\] ")

invalidtxt1 = "**Off-Topic Discussion**: All top-level comments must be a story or poem. Reply here for other comments.\n\n#####Reminder for Writers and Readers:\n* Prompts are meant to inspire new writing.  Responses don't have to fulfill every detail.\n\n* Please remember to [be civil](https://www.reddit.com/r/WritingPrompts/wiki/rules#wiki_rule_10.3A_be_civil) in any feedback.\n\n---\n\n[](#icon-help)[^(What Is This?)](https://www.reddit.com/r/WritingPrompts/wiki/off_topic)\n[](#icon-information)[^(First Time Here?)](https://www.reddit.com/r/WritingPrompts/wiki/user_guide)\n[](#icon-exclamation)[^(Special Announcements)](https://www.reddit.com/r/WritingPrompts/wiki/announcements)\n[](#icon-comments)[^(Click For Our Chatrooms)](https://www.reddit.com/r/WritingPrompts/wiki/chat)\n"
invalidtxt2 = "Thank you for your submission! Unfortunately, it has been removed for the following reason: ^If ^you ^have ^any ^questions, ^we ^ask ^that ^you ^message ^the ^moderators submission was removed&message=I have a question regarding the removal of this submission ^directly ^for ^appeals. ^Please ^refer ^to ^our ^detailed ^rules. (CasualConversation/w/rules) ^Take ^a ^look ^at ^our ^subreddits ^directory."
# &

r1 = rake_nltk.Rake( ranking_metric=rake_nltk.Metric.DEGREE_TO_FREQUENCY_RATIO,max_length=3)

import warnings
import contextlib

import requests
from urllib3.exceptions import InsecureRequestWarning


old_merge_environment_settings = requests.Session.merge_environment_settings

# @contextlib.contextmanager
# def no_ssl_verification():
#     opened_adapters = set()

#     def merge_environment_settings(self, url, proxies, stream, verify, cert):
#         # Verification happens only once per connection so we need to close
#         # all the opened adapters once we're done. Otherwise, the effects of
#         # verify=False persist beyond the end of this context manager.
#         opened_adapters.add(self.get_adapter(url))

#         settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
#         settings['verify'] = False

#         return settings

#     requests.Session.merge_environment_settings = merge_environment_settings

#     try:
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore', InsecureRequestWarning)
#             yield
#     finally:
#         requests.Session.merge_environment_settings = old_merge_environment_settings

#         for adapter in opened_adapters:
#             try:
#                 adapter.close()
#             except:
#                 pass

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
            rst_method="feng-hirst",
            mp_count=5,
            start_batch=0,
            end_batch = 0,
            annotate_rst=True,
            annotate_da=True,
            annotate_topic= True,
            reddit_dataset_version='small',
            title_only=False,
            **kwargs):
    """[summary]

    Args:
        danet_vname ([type]): [description]
        batch_process_size (int, optional): [description]. Defaults to 200.
        rst_method (str, optional): [description]. Defaults to "feng-hirst".
    """
    
    #region  Setup    
    # setting up model for Da prediction
    
    if annotate_da == True and title_only==False: 
            #title_only forces only one line per reddit conversation, so dialogue act can not be calcualted
        model_dir = utils_nlg.get_path(f'../DialogueAct/models/{danet_vname}')
        checkpoint_dir = f'{model_dir}/logs'

        checkpoint_path = get_best_ckpt_path(checkpoint_dir) #TOdO: need to insert changes from da branch into nlg branch
        
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        

        mparams = argparse.Namespace(**json.load( open( os.path.join(model_dir,"mparam.json"),"r" ) ) )
        tparams = argparse.Namespace(**json.load( open( os.path.join(model_dir,"tparam.json"),"r" ) ) )


        danet = DaNet(**vars(mparams))
        danet_module = TrainingModule(mode='inference', model=danet)
        danet_module.load_state_dict(checkpoint['state_dict'] )
        danet_module.eval()
        torch.set_grad_enabled(False)

        tokenizer = AutoTokenizer.from_pretrained(get_path('../DialogueAct/models/bert-base-cased') )

    # setting up docker image for rst
        #Platform dependent way to start docker
        # linux: If an error in next line, changeother "other" permission on /var/run/docker.sock to read write execute : chmod o+rwx /var/run/docker.sock
    if annotate_rst == True:
        try:
            client = docker.from_env(timeout=int(60*3))
        except docker.DockerException:
            os.system('chmod o=rwx var/run/docker.sock')
            client = docker.from_env(timeout=int(60*60))
        
        image_name = 'akanni96/feng_hirst_rst_parser'
        image_li = client.images.list('akanni96/feng_hirst_rst_parser')
        
            # (building)docker image
        if len(image_li)==0: 
            #Use this method during develomment
            dir_dockferfile =  utils_nlg.get_path("../DockerImages/feng_hirst_rst_parser",_dir=True)
            image = client.images.build(path=dir_dockferfile,
                nocache=True, pull=False, rm=True,forcerm=True,
                tag="akanni96/feng_hirst_rst_parser")[0]
            
        else:
            image = client.images.get(image_name)

        li_fh_container = [ client.containers.run(image, detach=True, 
            entrypoint=None,command="/bin/bash",auto_remove=True,tty=True) for x in range(mp_count) ]  

        li_fh_container_id = [container.id for container in li_fh_container ]     

    #Creating Save directory
    if reddit_dataset_version == "small":
        dir_save_dataset = utils_nlg.get_path("./dataset_v2/reddit_small_annotated",_dir=True)
    else: 
        dir_save_dataset = utils_nlg.get_path("./dataset_v2/reddit_large_annotated",_dir=True)

    # setting up corpus data
    #with no_ssl_verification():
    corpus = _load_data(reddit_dataset_version)
    
    li_id_dictconv  = list(corpus.conversations.items())
    total_batch_count = math.ceil(len(li_id_dictconv)/batch_process_size)

    # Optionally auto-resuming from last completed batch
    if start_batch == -1:
        # checking if df_records exists and if a column for this subreddit exists
        fn = os.path.join(dir_save_dataset,'last_batch_record')
        _bool_file_check = os.path.exists( fn )

        auto_fnd_failed = lambda : print("User choose auto-resume from last recorded batch.\
            But no last records exists, so initialising from batch 0")

        if not _bool_file_check: #s if file exist
            start_batch = 0
            auto_fnd_failed()
            
        else: #if file does not exists
            df_records = pd.from_csv( fn, index_col = "subreddit" )
            _bool_record_check = reddit_dataset_version in df_records.index.tolist()

            if not _bool_record_check:
                start_batch = 0
            
            else:
                start_batch = int( df_records.loc[ subreddit, 'last_batch' ] ) + 1
                batch_process_size = int( df_records.loc[subreddit, 'batch_process_size'] )
        
    # selecting sub batch to operate on if applicable
    if end_batch != 0:
        li_id_dictconv = li_id_dictconv[ : end_batch*batch_process_size] 
    if start_batch != 0:
        li_id_dictconv = li_id_dictconv[ start_batch*batch_process_size: ]

    # endregion
    timer = Timer()

    #region operating in batches
    global batches_completed
    batches_completed = start_batch

    while len(li_id_dictconv) > 0 :
        
        batch_li_id_dictconv =  li_id_dictconv[:batch_process_size]
        batch_li_li_thread_utterances = []
        print(f"\nOperating on batch {batches_completed} of {total_batch_count}")

        #region preprocessing
        timer.start()
        for id_dictconv  in batch_li_id_dictconv:
            #conv_id  = id_dictconv[0]
            dictconv = id_dictconv[1]
            tree_conv_paths = dictconv.get_root_to_leaf_paths() 
            #paths_count = len(tree_conv_paths)

                        
            # Gathering title for conversation
            thread_utterance_title = {
                'text':dictconv.meta['title'] , 'subreddit':dictconv.meta['subreddit'],
                'reply_to':None, 'id_utt':"000",'speaker_id': 'OriginalPoster'
            } 
            li_thread_utterances = [thread_utterance_title]


            if title_only == False:
                # Gathering utterances in thread
                li_thread_replies = [
                        {'text': utt.text, 'subreddit':utt.meta['subreddit'],
                        'reply_to':(utt.reply_to or thread_utterance_title['id_utt']),
                        'id_utt':utt.id,
                        'speaker_id':utt.speaker.id}
                    for utt in dictconv.get_chronological_utterance_list()]
            
                li_thread_utterances = li_thread_utterances + li_thread_replies

            # Preprocess each utterance -> txt_preproc
            [
                _dict.update({'txt_preproc':_preprocess(_dict['text'], title_only, _dict['subreddit'] )}) for _dict in 
                li_thread_utterances]
            

            # Removing invalid lines in conversation
            li_thread_utterances  = [
                _dict for _dict in li_thread_utterances
                if _valid_utterance(_dict['txt_preproc']) ]
            
            # Removing the reply_to (id) for utterances which had their reply removed in previous step
            for _dict in li_thread_utterances:
                #Checks if there is any reply message specified
                if _select_utt_by_reply(_dict['reply_to'], li_thread_utterances) == '' :
                    _dict['reply_to'] = None

            # Only append if more than five comments in the conversation. In title only mode adjust this to 1
            min_conv_comment_count = 1 if title_only else 5

            if len(li_thread_utterances)>=min_conv_comment_count:
                batch_li_li_thread_utterances.append(li_thread_utterances)

        timer.end("preprocessing")
        #endregion
        
        #region Predicting the RST Tag
        timer.start()
        if annotate_rst == True and len(batch_li_li_thread_utterances)>0:
            mp_count_rst = mp_count
            contaier_ids =  [ li_fh_container_id for idx in range(int( (len(batch_li_li_thread_utterances)//mp_count_rst) + 1)) ]
            contaier_ids = sum(contaier_ids, [])

            with mp.Pool(mp_count_rst) as pool:
                res = pool.starmap( _rst_v2, zip( _chunks(batch_li_li_thread_utterances, batch_process_size//mp_count_rst ) , contaier_ids  ) )
            
            batch_li_li_thread_utterances = list( res ) 
            batch_li_li_thread_utterances = sum(batch_li_li_thread_utterances, [])
            
            batch_li_li_thread_utterances = [ li for li in batch_li_li_thread_utterances if li !=[] ]
            
            assert len(batch_li_li_thread_utterances ) != 0

        timer.end("RST")
        #endregion

        #region DA assignment
        timer.start()
        if annotate_da == True and annotate_rst == True and title_only==False:
            for i, _ in enumerate(batch_li_li_thread_utterances):
            
                li_thread_utterances = batch_li_li_thread_utterances[i]

                # Here, for the larger utterances (3s) I perform DA analysis on each EDU of
                    #list of prev utterance
                    #masking tokens
                li_currutt =  danet_module.model.nem( [ _dict['txt_preproc'] for _dict
                                in li_thread_utterances ] )

                li_prevutt = danet_module.model.nem([ _select_utt_by_reply( _dict['reply_to'], li_thread_utterances) for _dict
                                in li_thread_utterances ] )
                li_li_uttdu = [ _dict['dus'] for _dict in li_thread_utterances ] #list of lists of the EDU sentences for a given utterance

                

                li_dict_da = []
                li_li_da = []
                for curr_utt, prev_utt, li_uttedu in zip(li_currutt, li_prevutt, li_li_uttdu):
                    
                    #skipping preds for items with no prev_utteance
                    if prev_utt == '':
                        li_da = None
                        dict_da = None

                    else:
                        #this prev_utt is added as context  to all the edus for a given long utterance
                        if len(li_uttedu) < 2:
                            li_input = [ [prev_utt, curr_utt ] ]
                            
                        # If more than 2 EDUs we evaluate each EDU and the complete utterance
                        elif len(li_uttedu) >= 2:
                            li_input = [ [ prev_utt, uttedu] for uttedu in li_uttedu  ]

                        if len(li_uttedu) == 1 :
                            encoded_input =  tokenizer(*li_input[0], add_special_tokens=True, padding='max_length', 
                            truncation=True, max_length=512, return_tensors='pt', return_token_type_ids=True)
                        else:
                            encoded_input =  tokenizer(li_input, add_special_tokens=True, padding='max_length', 
                            truncation=True, max_length=512, return_tensors='pt', return_token_type_ids=True)
                        

                        pred_da = danet_module.forward(encoded_input)
                        pred_da = torch.sigmoid( pred_da )

                        # Reducing the pred_da to one prediction
                            # Take the maximum for each da label if
                        max_values, _ = torch.max(pred_da,axis=0)
                        pred_da = torch.where(max_values>=0.5, max_values, torch.mean( pred_da, axis=0) )

                        # Formatting predictions to atain a list and a dictionary of da scores
                        res = danet_module.format_preds( torch.unsqueeze(pred_da,axis=0), logits=False)
                        li_da = res[0][0]
                        dict_da = res[1][0]
                                
                    li_li_da.append(li_da)
                    li_dict_da.append(dict_da)
                
                    #sequence of vectors, vector=logit score for each da class           
                

                [
                    _dict.update({'li_da':li_da }) for _dict, li_da in 
                    zip( li_thread_utterances, li_li_da)
                ]

                # Removing the utterances where the reply_to is none, they were needed for DA prediction of the next utterance but should not be used as they have no context
                li_thread_utterances = [ _dict for _dict in li_thread_utterances if _dict['reply_to'] != None]

                batch_li_li_thread_utterances[i] = li_thread_utterances
            
        timer.end("DA")   
        #endregion

        #region Topic extraction
        timer.start()
        if annotate_topic == True and len(batch_li_li_thread_utterances)>0:
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
                li_thread_utterances[idx].pop('reply_to',None)
                li_thread_utterances[idx].pop('id_utt',None)
                li_thread_utterances[idx].pop('speaker_id',None)
                li_thread_utterances[idx].pop('dus',None)
            
            batch_li_li_thread_utterances[i] = li_thread_utterances
        # end region

        #region Saving Batches
        timer.start()
            # format = subreddit/convo_code
        li_utterances = list(itertools.chain.from_iterable(batch_li_li_thread_utterances))
        _save_data(li_utterances, dir_save_dataset, batches_completed, batch_process_size, title_only )

        li_id_dictconv = li_id_dictconv[batch_process_size:]
        batches_completed += 1
        timer.end("Saving")

        #end region    

    print(f"Finished at batch {batches_completed}")        
    
def _load_data(reddit_dataset_version):
    # Donwload reddit-corpus-small if it doesnt exist
    if reddit_dataset_version == 'small':
        _dir_path = utils_nlg.get_path("./dataset_v2/reddit_small")
    
    else:
        _dir_path = utils_nlg.get_path("./dataset_v2/reddit_large")

    os.makedirs(_dir_path, exist_ok=True)

    if reddit_dataset_version == 'small':

        full_path = os.path.join(_dir_path,"reddit-corpus-small")

        use_local = os.path.exists( full_path)

        corpus = Corpus(filename=download("reddit-corpus-small", data_dir=_dir_path, use_local=use_local), merge_lines=True)
    
    else:

        # _list_options = [CasualConversation,
        #                     relationship_advice,
        #                     interestingasfuck,
        #                     science,

        #                     ChangeMyView,
        #                     PoliticalDiscussion,
        #                     DebateReligion,
        #                     PersonalFinance,
        #                     TrueGaming,
        #                     Relationships,

        #                     Music,
        #                     worldnews,
        #                     todayilearned,
        #                     movies,
        #                     Showerthoughts,
        #                     askscience,
        #                     LifeProTips,
        #                     explainlikeimfive,
        #                     DIY,
        #                     sports,
        #                     anime,
        #                       "IamA"]

        subdir = f"subreddit-{reddit_dataset_version}"

        full_path = os.path.join(_dir_path,subdir)
        use_local = os.path.exists(full_path)
        print(full_path)

        #with no_ssl_verification():
        corpus = Corpus(filename=download(subdir,
                            data_dir=full_path,use_local=use_local),
                            disable_type_check=True,
                            merge_lines=False)
            
        corpus.print_summary_stats()

    return corpus

def _preprocess(text,title_only,subreddit):
    # removing invalid texts
    
    if text in [invalidtxt1, invalidtxt2]:
        text = ""

    # removing leading and trailing whitespace characters
    text = text.strip()

    # removing unwanted characters from start of text
    text = text.strip('\\')

    # replacing hyperlinks with [link token]
    text = re.sub(pattern_hlinks,r"\1", text)

    if subreddit=="WritingPrompts":
        # checking that the start of prompt is valid 
        valid_start = bool( re.match(pattern_promptatstart, text) )
        
        if title_only == True:
            if valid_start == False:
                text = ""
        
        if valid_start == True:    
            text = re.sub(pattern_writingprompts_prefix,r"\2",text)

    # remove general hyperlinks
    text = re.sub(pattern_hlinks2,"", text)

    # convert emojis to text and remove
    text = emoji.demojize( text )
        #TODO: need to remember to include emoji.emojize in the NLG decoding pipeline
        #TODO: "I'm sorry if that's how your interactions with friends go :sad_but_relieved_face:\n\nBut
                #  that's not how any of my whatsapps with my friends happened.

                #is converted to 

                #"[CLS] I'm sorry if that's how your interactions with friends go : sad _ but _ relieved _ face : But that's
                #  not how any of my whatsapps with my friends happened. [SEP]"

                #so need to have regex code to compress words between two colons (with underscores)
    text = re.sub(pattern_emojis,"",text)
    text = re.sub(pattern_txt_emojis,"", text)
    # removing reddit qoutes
    text = re.sub( pattern_qoutes, '', text )

    # removing [deleted] and [removed] from texts
    text  = re.sub( pattern_toremove,'',text)
    text = re.sub( pattern_subreddits, r'\2',text)
    
    # removing html codes
    text = html.unescape(text)

    # remove repeated words
    text = re.sub(pattern_repword, r'\1', text)

    #remove repeated periods
    text = re.sub(pattern_reppunc, r"\1", text)

    # convert to ascii
    text = text.encode('ascii',errors='ignore').decode('ascii').replace("{", "").replace("}", "")

    #remove qoutes, dash and asterix, brackets
    text = re.sub(pattern_qdab, "", text)

    # remove multiple spaces
    text = re.sub(pattern_multwspace, ' ', text)

    # Possible ading questions marks to subsentences that start with a question word
    li_segmented_txt = sent_detector.tokenize(text)
    li_format_txt = [ question_punctuation(subtxt) for subtxt in li_segmented_txt]
    text_new = ' '.join(li_format_txt)

    return text_new

def question_punctuation(utt):
    """Add question punctuation"""
    li_question_starters = ['What', "Wat" ,"Why","Where","How","Who","When","What","Which","Is","Did","Does","Are","Can",'Have',"Would"]
    if utt.split(' ')[0].capitalize() in li_question_starters:
        if utt[-1] in [".","!","?"]:
            utt = utt[:-1]
        utt = utt + "?"

    return utt

def _valid_utterance(txt):

    txt = txt.encode('ascii',errors='ignore').decode('ascii').replace("{", "").replace("}","")
    # Checking if someone has just drawn an image using brackets and spaces etc

    # not all space characters = not txt.ispace()
    # contains_alphabet = any( c.isalpha() for c in txt)

    # post_not_deleted = txt!="[deleted]"

    # checking

    return (not txt.isspace()) and any( c.isalpha() for c in txt) and txt!='"Thank you for your submission! Unfortunately, it has been removed for the following reason: ^If ^you ^have ^any ^questions, ^we ^ask ^that ^you ^message ^the ^moderators submission was removed&message=I have a question regarding the removal of this submission ^directly ^for ^appeals. ^Please ^refer ^to ^our ^detailed ^rules. (CasualConversation/w/rules) ^Take ^a ^look ^at ^our ^subreddits ^directory."'

def _select_utt_by_reply(reply_to_id, li_thread_utterances ):
    try:
        prev_utterance = next( _dict['txt_preproc'] for 
         _dict in li_thread_utterances 
         if _dict['id_utt'] == reply_to_id )
    except StopIteration as e:
        prev_utterance = ''
    
    return prev_utterance

def _rst_v2(li_li_thread_utterances, fh_container_id ):
    client = docker.from_env(timeout=int(60*3))
    fh_container = client.containers.get(fh_container_id)
    new_li_li_thread_utterances = []

    #TODO: need to get rid of the utterances that equal == "empty" (these cases are currently )
    #li_li_thread_utterances = [ [thread_utt for thread_utt in li_thread_utterances if thread_utt['txt_preproc'] != "removed"  ] for li_thread_utterances in li_li_thread_utterances]

    li_li_utterances = [ [thread_utt['txt_preproc'] for thread_utt in li_thread_utterances ] for li_thread_utterances in li_li_thread_utterances] #li of li of utts

    json_li_li_utterance = json.dumps(li_li_utterances)

    cmd = ['python','parser_wrapper3.py','--json_li_li_utterances', json_li_li_utterance, '--redirect_output', "2"]
    exit_code,output = fh_container.exec_run( cmd, stdout=True, stderr=True, stdin=False, 
                        demux=True)
    stdout, stderr = output
    
    try:
        if type(stdout) is bytes:
            stdout = str(stdout,"utf-8")

        #print("\n\n\n",stdout,"\n\n\n")
        li_strtree = json.loads( stdout )
        #li_strtree = stdout
            
        #print(len(li_strtree), "\n\n\n")

    except (TypeError, json.JSONDecodeError) as e:
        # print(f"\nError", e)
        # print("\n\n\n stderr\n", str(stderr,"utf-8"))
        # print("\n\n\nStdout\n", str(stderr,"utf-8"))
        # raise Exception
        # print(e)
        # print("here")
        # raise Exception
        return new_li_li_thread_utterances
    
    for idx, str_tree in enumerate(li_strtree):
        li_thread_utterances = li_li_thread_utterances[idx]
        li_subtrees = []

        # Parsing a list of subtrees in the utterance tree str_tree
        for idx, pt_str in enumerate(str_tree):
            try:
                if pt_str == '': raise ValueError
                _ = nltk.tree.Tree.fromstring(pt_str, brackets="{}")
            except ValueError:
                _ = None
                pass
            li_subtrees.append(_)

        li_rst_dict = [ _tree_to_rst_code(_tree) if _tree != None else None for _tree in li_subtrees ]
        
        # A list of strings
        li_dus = [ _tree_to_li_du(_tree) if _tree != None else None for _tree in li_subtrees ]

        # Keeping non erroneous utterances within a conversation - bad trees were assigned None
        assert len(li_rst_dict) == len(li_thread_utterances)
        new_li_thread_utterance = []
        for idx in range(len(li_rst_dict)):
            if li_rst_dict[idx]!=None:
                li_thread_utterances[idx].update( {'rst':li_rst_dict[idx], 'dus':li_dus[idx] } )
                new_li_thread_utterance.append(li_thread_utterances[idx] )
            else:
                pass

        new_li_li_thread_utterances.append(new_li_thread_utterance)
    return new_li_li_thread_utterances

def _da_assigning(li_li_thread_utterances:list, danet_module, tokenizer):

    new_li_li_thread_utterances = []

    for idx in range(len(li_li_thread_utterances)):

        li_thread_utterances = li_li_thread_utterances[idx]

        li_currutt =  danet_module.model.nem( [ _dict['txt_preproc'] for _dict
                        in li_thread_utterances ] )

        li_prevutt = danet_module.model.nem([ _select_utt_by_reply( _dict['reply_to'], li_thread_utterances) for _dict
                        in li_thread_utterances ] )
        li_li_uttdu = [ _dict['dus'] for _dict in li_thread_utterances ] #list of lists of the EDU sentences for a given utterance

        li_dict_da = []
        li_li_da = []

        for curr_utt, prev_utt, li_uttedu in zip(li_currutt, li_prevutt, li_li_uttdu):
            
            #skipping preds for items with no prev_utteance
            if prev_utt == '':
                li_da = None
                dict_da = None

            else:
                #this prev_utt is added as context  to all the edus for a given long utterance
                if len(li_uttedu) < 2:
                    li_input = [ [prev_utt, curr_utt ] ]
                    
                # If more than 2 EDUs we evaluate each EDU and the complete utterance
                elif len(li_uttedu) >= 2:
                    li_input = [ [ prev_utt, uttedu] for uttedu in li_uttedu  ]

                if len(li_uttedu) == 1 :
                    encoded_input =  tokenizer(*li_input[0], add_special_tokens=True, padding='max_length', 
                    truncation=True, max_length=512, return_tensors='pt', return_token_type_ids=True)
                else:
                    encoded_input =  tokenizer(li_input, add_special_tokens=True, padding='max_length', 
                    truncation=True, max_length=512, return_tensors='pt', return_token_type_ids=True)
                
                #encoded_input.to("cuda")

                pred_da = danet_module.forward(encoded_input)
                pred_da = torch.sigmoid( pred_da )

                # Reducing the pred_da to one prediction
                    # Take the maximum for each da label if
                max_values, _ = torch.max(pred_da,axis=0)
                pred_da = torch.where(max_values>=0.5, max_values, torch.mean( pred_da, axis=0) )

                # Formatting predictions to atain a list and a dictionary of da scores
                res = danet_module.format_preds( torch.unsqueeze(pred_da,axis=0), logits=False)
                li_da = res[0][0]
                dict_da = res[1][0]
                        
            li_li_da.append(li_da)
            li_dict_da.append(dict_da)
        
            #sequence of vectors, vector=logit score for each da class           
        
        [
            _dict.update({'li_da':li_da }) for _dict, li_da in 
            zip( li_thread_utterances, li_li_da)
        ]

        # Removing the utterances where the reply_to is none, they were needed for DA prediction of the next utterance but should not be used as they have no context
        li_thread_utterances = [ _dict for _dict in li_thread_utterances if _dict['reply_to'] != None]

    new_li_li_thread_utterances.append(li_thread_utterances)
    
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
    li_bintreepos = sorted([ utils_nlg.tree_order.get(x,-1) for x in pos_xleaves])

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
    li_du_str = [ __parse_leaves(child.leaves()) if type(child)==nltk.tree.Tree else __parse_leaves(child) for child in li_child ]
    
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

def __parse_leaves(tree_leaves ):
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

def _rake_kw_extractor(str_utterance, lowest_score=0.0):
    
    r1.extract_keywords_from_text(str_utterance)
    
    li_ranked_kws = r1.get_ranked_phrases_with_scores()
    
    li_ranked_kws = [ [ x[1], x[0] ] for x in li_ranked_kws if x[0]>lowest_score ]

    return li_ranked_kws

def _textrank_extractor(str_utterance, lowest_score=0.0):
    # Add the below to docker file
    # os.system('install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz')
    # os.system(' python -m spacy download en_core_web_sm ')
    # pytextrank using entity linking when deciding important phrases, for entity coreferencing

    doc = nlp(str_utterance)
    li_ranked_kws = [ [str(p.chunks[0]), p.rank] for p in doc._.phrases if p.rank>lowest_score ] #Take all bar the lowest score

    if len(li_ranked_kws) == 0:
        li_ranked_kws = [["",0.0]]

    return li_ranked_kws
        
def _save_data(li_utterances, dir_save_dataset, last_batch_operated_on=0, batch_process_size=120, title_only=False):
    
    # Split list of utterances by the subreddit name
    # Then for each sublist
        # Get directory save name
        # Get the last saved csv file in directory
        # (fn-format = file_number_utterances in file )
            # then append more lines
            
    # Grouping utterances by the subreddit 
    grouped_li_utterances = [ ( k, list(g)) for k,g in itertools.groupby(li_utterances, lambda _dict: _dict['subreddit'] ) ]
        #a list of tuples; elem0: subreddit name elem1: list of convos for that subreddit
    
    for subreddit, _li_utterances in grouped_li_utterances:
        _li_utterances = [ { str(k):json.dumps(v) for k,v in dict_thread.items() } for dict_thread in _li_utterances ]

        if title_only == True:
            subreddit = subreddit+ "_title_only"
        
            
        subreddit_dir = utils_nlg.get_path( os.path.join(dir_save_dataset,subreddit), _dir=True  )  
               
        lock_path = os.path.join(subreddit_dir,"lock") 
        lock = FileLock(lock_path, timeout=60)

        with lock:
            #unlimited batch_siz
            files_ = [ fp for fp in os.listdir(subreddit_dir) if os.path.split(fp)[1] != "lock" ]
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

            # with open(old_fp,"a+", newline=None,encoding='utf-8') as fn:
            #     dict_writer = csv.DictWriter(fn, keys, quoting=csv.QUOTE_MINIMAL,
            #         doublequote=False )
            #     dict_writer.writerows(_li_utterances)

            df_ = pd.read_csv(old_fp)
            df_ = df_.append( _li_utterances, ignore_index=True, sort=False)
            df_.to_csv( new_fp, index=False)

            if os.path.exists(old_fp) and old_fp!=new_fp:
                os.remove(old_fp)
            

            # Updating record of last batch operated on for each subreddit
            new_record = { 'batch_process_size':batch_process_size, 'last_batch':last_batch_operated_on }
            if os.path.exists( os.path.join(dir_save_dataset,'last_batch_record') ):
                df_records = pd.read_csv( os.path.join(dir_save_dataset,'last_batch_record'), index_col = "subreddit" )
            else:
                df_records = pd.DataFrame( columns=['last_batch','batch_process_size'] )
                df_records.index.names = ['subreddit']

            #df_records = df_records.append(new_record, ignore_index=True)

            for k,v in new_record.items():
                df_records.loc[ subreddit, [k] ] =  v

            df_records.to_csv( os.path.join(dir_save_dataset,'last_batch_record'), index_label='subreddit' )

          
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

    parser.add_argument('--danet_vname', default="DaNet_v009",
        help="Version name of the DaNet model to use for dialogue act classifier ",
        type=str )
    
    parser.add_argument('-bps','--batch_process_size', default=120,
        help='',type=int)        
   

    parser.add_argument('--rst_method', default="feng-hirst",
        choices=['feng-hirst','akanni-rst'], type=str)

    parser.add_argument('--mp_count', default=6,
        type=int)
    
    parser.add_argument('-sb','--start_batch', default=0, type=int, help="batch of data to start from. Pass \
                                    in -1 for this batch to be autodetermined from records" )

    parser.add_argument('-eb','--end_batch', default=0, type=int, help="Final batch to finish on. Set to 0 to run until end")

    parser.add_argument('-ad','--annotate_da',default=True, type=lambda x: bool(int(x)) )


    parser.add_argument('-ar','--annotate_rst',default=True, type=lambda x: bool(int(x)) ) 

    parser.add_argument('-at','--annotate_topic',default=True, type=lambda x: bool(int(x)) )

    parser.add_argument('-rdv','--reddit_dataset_version',default='small', type=str, choices=[
                                                                        "small",
                                                                        "CasualConversation",
                                                                        "relationship_advice",
                                                                        "interestingasfuck",
                                                                        "science",

                                                                        "changemyview",
                                                                        "PoliticalDiscussion",
                                                                        "DebateReligion",
                                                                        "PersonalFinance",
                                                                        "TrueGaming",
                                                                        "Relationships",

                                                                        "WritingPrompts",

                                                                        "Music",
                                                                        "worldnews",
                                                                        "todayilearned",
                                                                        "movies",
                                                                        "Showerthoughts",
                                                                        "askscience",
                                                                        "LifeProTips",
                                                                        "explainlikeimfive",
                                                                        "DIY",
                                                                        "sports",
                                                                        "anime",
                                                                        "IamA"])
    parser.add_argument('-to','--title_only', default=False, type=lambda x: bool(int(x)) )

    args = parser.parse_args()
    
    dict_args = vars(args)

    completed = False
    
    while completed == False:
        try:
            main( **dict_args )
            completed = True
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            dict_args['start_batch'] = batches_completed + 1
            
        finally :
            # cmd = "docker stop $(docker ps -aq) > /dev/null 2>&1 & docker rm $(docker ps -aq) > /dev/null 2>&1 & docker rmi $(docker images -a -q) > /dev/null 2>&1"
            
            # cmd = docker stop $(docker ps -aq) & docker rm $(docker ps -aq) & yes | docker image prune  & docker rmi $(docker images -a -q)
            # os.system(cmd)
            # time.sleep(3)
            # os.system(cmd)
            # time.sleep(3)
            pass


#python3 data_setup.py -bps 60 -ad 0 -rdv CasualConversation -sb -1 --mp_count 6 

# python3 data_setup.py -bps 60 -ad 0 -rdv changemyview -sb 0 --mp_count 2
#python3 data_setup.py -bps 60 -ad 0 -rdv PoliticalDiscussion -sb 0 --mp_count 2

# python3 data_setup.py -bps 10 -ad 0 -rdv small -sb -1 --mp_count 6 

# python3 data_setup.py -bps 10 -ad 0 -rdv WritingPrompts -sb 1 --mp_count 6 -to 1
