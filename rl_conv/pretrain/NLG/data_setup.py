import numpy
import os
import convokit
from convokit import Corpus, download
import argparse
import utils
import random
import train
import pytorch_lightning as pl
import emoji
from transformers import AutoTokenizer, AutoModel

def main(checkpoint_path,**kwargs):

    # Iterate through Conversations
        # Convert conversation to a pd.Dataframe or numpy array in the format used for Dialog Act Datasets
            # Coloumns = [Speaker, utterance, Dialog Act, RST, topics ]
        # Preprocess utterances
            # replace image links with token [Image_link], leave the [description of image_link that is in square brackers]
            # replace other links with token [link]
            # Afer utterance is empty, or only contains and [Image_link] token then remove utterance
        # Use pre-trained DA model to assign DA tag based on current utterance and previous utterance
            # Create all Dialog Acts for the whole conversation at once by passing the data in together
        # Use the pre-made RST identifier to create the RST tags for each utterance
            # Create all at once for the whole conversation
        # Use a topic identifier to ascertain the topics of interest
        # Save Conversation to a file 
            # use the unique id of the sub-reddit
                # and the number of the conversation tree
    
    # Then Iterate through formatted conversations
        # Open Groups of 10
            # Join 10, seperate using a marker (during training you open 1 file, then split into n files based on the marker to create n different datasets)

    corpus = _load_data()

    li_id_dictconv  = list(corpus.conversations.items())
    li_id_dictconv = random.shuffle(li_id_dictconv )
    
    # setting up model for Da prediction
    danet_version_name =  'DaNet_v003'
    model_dir = utils.get_path(f'../DialogueAct/models/{tparams.version_name}')
    checkpoint_dir = f'{model_dir}/logs'

    checkpoint_path = utils.get_best_ckpt_path(checkpoint_dir)
    mparams = argparse.Namespace(**json.load( open( os.path.join(model_dir,"mparam.json"),"r" ) ) )
    tparams = argparse.Namespace(**json.load( open( os.path.join(model_dir,"tparam.json"),"r" ) ) )

    DaNet_module = train.TrainingModule.load_from_checkpoint( utils.get_path(checkpoint_path) )
    DaNet_module.eval()    
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained('../DialogueAct/models/bert-base-cased')


    # Use the mp workers here

    for id_dictconv  in li_id_dictconv:
        conv_id  = id_dictconv[0]
        dictconv = id_dictconv[1]

        tree_conv_paths = dictconv.get_root_to_leaf_paths() 
        paths_count = len(paths)

        # Cycle through each utterance in the dictconv(thread) and
            # 1) preprocess it
            # 
            # 2) predict it's Dialog Act
                # Use the 'in response to' tag to gather the previous message
                # Skip the root dialog in each example
            # 3) Predict its RST tag
            # 4) Predict the main topics of the utterance, if any
        
        # Get list of utterances in thread
        #_keys = ['text', 'subreddit', 'reply_to', 'id' ]
        li_thread_utterances = [
            {'text': utt.text, 'subreddit':utt.subreddit,
            'reply_to':utt.subreddit, 'id_utt':utt.id,
            'speaker_id':utt.speaker.id
            } for utt in a_conv.get_chronological_utterance_list()]
        
        # Add Dialog Acts
            # Preprocess each utterance -> txt_preproc
        li_thread_utterances = [
            _dict.update({'txt_preproc':_preprocess(_dict.txt)}) for _dict in 
            li_thread_utterances
        ]
        
        #     # Tokenize each utterance -> txt_tokenize
        #     # drop preproc
        # li_thread_utterances = [
        #     _dict.update( {'text_tokenize': tokenizer(_dict.txt_preproc) } ) for _dict in 
        #     li_thread_utterances
        ]

        # Get a list of each txt_tokenize and its preceeding txt_tokenize
        # Feed to model
        def select_utt_by_reply(reply_to_id, li_thread_utterances ):
            prev_utterance = next(_dict['txt_preproc'] for _dict in li_thread_utterances 
                if _dict['speakder_id'] == reply_to_id )
            
            return prev_utterance

        li_utt_prevutt = [
            [ _dict['txt_preproc'], select_utt_by_reply( _dict['reply_to'], li_thread_utterances )  ]
            for _dict in li_thread_utterances
        ]

        #here
        _input =  tokenizer(li_utt+prevutt)
        
        DaNet_module.forward()

        # remove repeated consecutive words in utterance




            

        # Get individual conversation trees
        for conv_path in tree_conv_paths:
        li_speaker_utterance = [ {'speaker_conv':_conv.speaker.id, 
            'utterance':_conv.text, 'da':_conv.da,
            'rst': _conv.rst, 'topics':_conv.topics
            } for  _conv in conv_path ]

        df = pd.DataFrame(li_speaker_utterance)

        # Save them to files
            # format = subreddit/




def load_data():
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
    "(\[[\w\-,\. ]+\])(\((https|www|ftp)?\S+\))"
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

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False) 
    #parent_parser2 = argparse.ArgumentParser(add_help=False)    
    
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
    parser.add_argument('--checkpoint_path', default=None, help="Path to the \
        DA model checkpoint file)" )        

    args = parser.parse_args()


    #main( **vars(args) )
    main()