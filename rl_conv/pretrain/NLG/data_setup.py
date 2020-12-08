import sys
sys.path.append("DialogueAct") # Adds higher directory to python modules path
import numpy
import os
import convokit
from convokit import Corpus, download
import argparse
import utils
import random
from DialogueAct.train import TrainingModule
import pytorch_lightning as pl
import emoji
from transformers import AutoTokenizer, AutoModel
import docker

import itertools

import nltk
nltk.download('stopwords')
import rake_nltk

import pytextrank
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

import csv

from DialogueAct.utils import get_best_ckpt_path

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
            batch_process_size=200,
            batch_save_size=-1,
            rst_method="feng-hirst",
            **kwargs):
    """[summary]

    Args:
        danet_vname ([type]): [description]
        batch_process_size (int, optional): [description]. Defaults to 200.
        batch_save_size (int, optional): [description]. Defaults to -1.
        rst_method (str, optional): [description]. Defaults to "feng-hirst".
    """
    
    #region Setup
    corpus = _load_data()

    li_id_dictconv  = list(corpus.conversations.items())
    #li_id_dictconv = random.shuffle(li_id_dictconv )
    
    # setting up model for Da prediction
    danet_version_name =  danet_vname 
    model_dir = utils.get_path(f'../DialogueAct/models/{tparams.version_name}')
    checkpoint_dir = f'{model_dir}/logs'

    checkpoint_path = get_best_ckpt_path(checkpoint_dir) #TOdO: need to insert changes from da branch into nlg branch
    mparams = argparse.Namespace(**json.load( open( os.path.join(model_dir,"mparam.json"),"r" ) ) )
    tparams = argparse.Namespace(**json.load( open( os.path.join(model_dir,"tparam.json"),"r" ) ) )

    DaNet_module = TrainingModule.load_from_checkpoint( utils.get_path(checkpoint_path) )
    DaNet_module.eval()    
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained('../DialogueAct/models/bert-base-cased')

    os.system('docker build -t C:\\Users\Rilwa\01_Mcv\RST_FH_service\akanni-feng-hirst-service .') #TODO:change to build from a docker link
    os.system('docker run -it --rm ubuntu bash') 

    # setting up docker image for rst
    client = docker.from_env(timeout=int(60*60))
    dir_docker_images =  utils.get_path("../DockerImages",_dir=True)
    image_name = 'akanni96/feng-hirst-parser'

    image_li = client.images.list('akanni96/feng-hirst-rst-parser')[0]
    
        # (building)docker image
    if len(image_li)==0: 
        image = client.images.pull(image_name)
    else:
        image = client.images.get(image_name)
    
        #saving directory
        dir_save_dataset = utils.get_path("./dataset/reddit_small_mc")
    # endregion
    
    #region operating in batches
        # Use the mp workers at different stages, so for rst and entity selection but not da
        # TODO - Add method of incrementally saving li_thread_utterances
    while len(li_id_dictconv) > 0:
        
        batch_li_id_dictconv =  li_id_dictconv[:batch_process_size]
        batch_li_li_thread_utterances = []

        #region preprocessing
        for id_dictconv  in batch_li_id_dictconv:
            conv_id  = id_dictconv[0]
            dictconv = id_dictconv[1]
            tree_conv_paths = dictconv.get_root_to_leaf_paths() 
            paths_count = len(paths)

            # Gathering utterances in thread
            li_thread_utterances = [
                {'text': utt.text, 'subreddit':utt.subreddit,
                'reply_to':utt.reply_to, 'id_utt':utt.id,
                'speaker_id':utt.speaker.id
                } for utt in a_conv.get_chronological_utterance_list()]
            
            # Preprocess each utterance -> txt_preproc
            li_thread_utterances = [
                _dict.update({'txt_preproc':_preprocess(_dict.txt)}) for _dict in 
                li_thread_utterances]
            batch_li_li_thread_utterances.append(li_thread_utterances)
        #endregion
        
        #region DA assignment
        for i, _ in enumerate(batch_li_li_thread_utterances):
            
            li_thread_utterances = batch_li_li_thread_utterances[i]

            li_utt_prevutt = [
                [ _select_utt_by_reply( _dict['reply_to'], li_thread_utterances), _dict['txt_preproc']  ]
                for _dict in li_thread_utterances
            ]

            tknzd_seqs =  tokenizer(li_utt_prevutt)
            pred_das = DaNet_module.forward(tknzd_seqs, output_mode = "class names") 
                #sequence of vectors, vector=logit score for each da class
            pred_das = pred_das.tolist()
            

            li_thread_utterances = [
                _dict.update({'da': da }) for _dict, da in 
                zip( li_thread_utterances, pred_das)
            ]

            batch_li_li_thread_utterances[i] = li_thread_utterances
        #endregion

        #region Predicting the RST Tag
        if rst_method == "feng-hirst":
            # multiprocessing use here
            fh_container  = client.containers.run(image_name, detach=True, stream=False, socket=False)
            logs = fh_container.attach

            for i, _ in enumerate(batch_li_li_thread_utterances):
                li_thread_utterances = batch_li_li_thread_utterances[i]
                
                    # # thread_utterance = {'text':, 'subreddit':,
                    #     'reply_to':t, 'id_utt':,
                    #     'speaker_id':, 'txt_preproc':,
                    #       'da':
                    #     }

                li_utterance  = [ thread_utt.txt_preproc for thread_utt in li_thread_utterances ]
                json_li_utterance = json.dumps(li_utterance)
                
                #response = fh_container.run( entrypoint=entrypoint, command=command )
                cmd = ['python','parser_wrapper2.py','--li_utterances', json_li_utterance]
                exit_code,output = fh_container.exec_run( cmd, stdout=True, stderr=True, stdin=False, demux=True) #stream=False,  )
                stdout, stderr = output

                li_trees = [ nltk.tree.Tree.fromstring(pt_str) for pt_str in json.loads(stdout) ] 

                # Creating two versions of the li_rst_methods
                li_rst_dict = [ _tree_to_rst_code(_tree, method=0) for _tree in li_trees ]

                li_thread_utterances = [
                    _dict.update(rst_dict) for thread_utterance, rst_dict in 
                    zip( li_thread_utterances, li_rst_dicts)
                ]

                batch_li_li_thread_utterances[i] = li_thread_utterances
                # output type of this docker env: trees.parse_tree.ParseTree, Tree.fromstring(str)
                
            client.containers.prune()

        elif rst_method == "akanni":
            pass
            
        #endregion

        #region Topic extraction
        for i, _ in enumerate(batch_li_li_thread_utterances):
            li_thread_utterances = batch_li_li_thread_utterances[i]

            li_rakekw_textankkw = [ {'topic_rake':_rake_kw_extractor(thread_utterance.txt_preproc),
                                        'topic_textrank':_textrank_extractor(thread_utterance.txt_preproc)}
                    for thread_utterance in li_thread_utterances]

            li_thread_utterances = [
                _dict.update(dict_kw) for thread_utterance, dict_kw in 
                zip( li_thread_utterances, li_rakekw_textankkw)
            ]

            batch_li_li_thread_utterances[i] = li_thread_utterances
        #endregion

        #region Saving Batches
            # format = subreddit/convo_code
        _save_data(batch_li_li_thread_utterances, batch_save_size, dir_save_dataset)
        li_id_dictconv = li_id_dictconv[batch_process_size:]
        #end region    


def _load_data():
    # Donwload reddit-corpus-small if it doesnt exist
    _dir_path = utils.get_path("dataset\\reddit_small")
    if os.path.exists(_dir_path):
        use_local = True
    else:
        use_local = False
        os.makedirs(_dir_path, exist_ok=True)

    corpus = Corpus(filename=download("reddit-corpus-small", data_dir=os.path.abspath(_dir_path), use_local=use_local), merge_lines=True)
    
    return corpus

def _preprocess(txt):

    # replacing hyperlinks with [link token]
    #https://mathiasbynens.be/demo/url-regex
    # "(\[[\w\-,\. ]+\])(\((https|www|ftp)?\S+\))"
    #pattern = r"([\w]+)(\(_^(?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?!10(?:\.\d{1,3}){3})(?!127(?:\.\d{1,3}){3})(?!169\.254(?:\.\d{1,3}){2})(?!192\.168(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\x{00a1}-\x{ffff}0-9]+-?)*[a-z\x{00a1}-\x{ffff}0-9]+)(?:\.(?:[a-z\x{00a1}-\x{ffff}0-9]+-?)*[a-z\x{00a1}-\x{ffff}0-9]+)*(?:\.(?:[a-z\x{00a1}-\x{ffff}]{2,})))(?::\d{2,5})?(?:/[^\s]*)?$_iuS\))"
    pattern =  re.compile("(\[[\w\-,\. ]+\])(\((https|www|ftp)?\S+\))")
    text = re.sub(pattern,r"\1", text)
        #causes [candid pics without makeup] --> [ candid pics without makeup ]
        # correct this in output

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
    txt = txt.replace(r'\b([\w\-,\.]+)(\s+\1)+\b', r'\1')
    
    return txt

    # Get a list of utterance and its preceeding utterance

def _select_utt_by_reply(reply_to_id, li_thread_utterances ):
    prev_utterance = next(_dict['txt_preproc'] for 
        _dict in li_thread_utterances 
        if _dict['speakder_id'] == reply_to_id )
    
    return prev_utterance

def _tree_to_rst_code(_tree, method=1):
    """Converst RST Tree to rst code used in NLG model

    Args:
        method (int, optional): [description]. Defaults to 1.
    
    Return:
        if method==0:
            Two lists
            List 1 Represents A Flattened version of the relations in the RST tree
            List 2 the nuclearity/satellite couple type
            - element in first list
                - Flatten relations by concat relations left to right for each level from top to bottom
                - Then where each line ends add the text <NEW-LINE>
            - element in second list
                - list representing the ns relationship at each pos

        if method==1:
            A vector representing the counts of each relation in the response
            - A length n list, where n=number of relations possible
            - Each unit represents number of times that relation occured
            #TODO: possibly figure out some way to normalize this vector
    """


    #iter_nodes = list( nltk.utils.breadth_first(_tree) )
    #Todo: check code for filtering relation and  ns relation-type
    lli_relations_ns = [  re.findall(r'[a-zA-Z]+' ,_tree._label)  for _tree in _tree.subtrees() ]
    li_relations_ns = [ [_li[0],_li[1:]] for _li in li_relations_ns]
        
    return li_relations_ns
    
def _rake_kw_extractor(str_utterance, topk=3):
    r1 = Rake( ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO,max_length=3)
    r1.extract_keywords_from_text(str_utterance)
    
    li_ranked_kws = r1.get_ranked_phrases_with_scores()
    
    li_ranked_kws = li_ranked_kws[:topk]

    return li_ranked_kws

def _textrank_extractor(str_utterance,topk=3):
    # Add the below to docker file
    # os.system(') install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz')
    # os.system python -m spacy download en_core_web_sm

    doc = nlp(str_utterance)

    li_ranked_kws = [ (p.chunks[0], p.rank) for p in doc._.phrases ]

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
    grouped_li_utterances = [ ( k, list(g)) for k,g in groupby(li_utterances, lambda _dict: _dict['subreddit'] ) ]
        #a list of tuples; elem0: subreddit name elem1: list of convos for that subreddit
    
    
    for subreddit, _li_utterances in grouped_li_utterances:
        subreddit_dir = utils.get_path( os.join(dir_save_dataset,subreddit), _dir=True  )  
        
        
        #unlimited batch_size
        if batch_save_size < 0:
            files_ = os.listdir(subreddit_dir)
            fn = files_[0]
            curr_len = int(max_fn[4:])
            new_len = curr_len + len(li_utterances)

            fp = os.path.join(subreddit_dir,f"{max_fn[:4]}_{new_len:04d}.csv")
            keys = li_utterances[0].keys()
            with open(fp,"a+", newline='') as _f:
                dict_writer = csv.DictWriter(_f, keys)
                dict_writer.writerows(li_utterances)

        #limited batch save size - saving to existing file and any new files
        else:
            while len(_li_utterances)>0:
                
                files_ = os.listdir(subreddit_dir)

                #most recent filename saved to
                last_fn = max( files_, key=int(fn[:4]) , default=f"0000_0000.csv")

                # checking whether file is full
                utt_count_in_last_file = int(last_fn[4:])
                
                # extracting contents for file to add to
                max_utt_to_add = batch_save_size - utt_count_in_last_file
                
                # making newer empty file then skipping to next round of loop
                if max_utt_to_add == 0:
                    fn = f"{int(last_fn[:4])+1:04d}_0000.csv"
                    fp = os.path.join(subreddit_dir, fn )
                    with open(fp,"a+", newline='') as _f:
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
                
            

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False) 
    #parent_parser2 = argparse.ArgumentParser(add_help=False)    
    
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)


    parser.add_argument('--danet_vname', default="DaNetV003",
        help="Version name of the DaNet model to use for dialogue act classifier ",
        type=str )
    
    parser.add_argument('--batch_process_size', default=200,
        help='',type=int)        

    parser.add_argument('--batch_save_size', default=-1,
        help='',type=int)        

    parser.add_argument('--rst_method',default="feng-hirst",
        options=['feng-hirst','akanni-rst'], type=str)

    args = parser.parse_args()
    main( **vars(args) )