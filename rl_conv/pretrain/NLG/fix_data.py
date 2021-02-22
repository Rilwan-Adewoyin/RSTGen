#fix dataset script
import html
import glob
import os
import pandas as pd
import json
import regex as re
import argparse
import csv 
import nltk
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

# This script provides two functions:
# 1) Fix errors from first iteration of data creation
# 2) Created the subdataset where only sentences with n or more edus are included

#python3 fix_data.py --data_dir "./dataset/reddit_large_annotated" -fdd "./dataset/reddit_large_annotated" --fix_ds 1 --gen_long_ds 1

#region regex_patterns
#pattern_qoutes = re.compile("(&gt;|>)[^(\\n)]*(\\n){1,}") #removing reddit qoutes 
pattern_qoutes = re.compile("andgt;")
pattern_deleted = re.compile("(\[deleted\]|\[removed\]|EDIT:)")
pattern_edu = re.compile("[,\-!?:.]+")
pattern_txt_emojis = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')
pattern_subreddits = re.compile(r"(/??r/)([^/]+)")
#endregion


def main(data_dir, new_data_dir, fix_ds, gen_long_ds):
    
    # getting list of files
    glob_fns_regex = os.path.join( data_dir, "*", "*" )
    li_fps = glob.glob(glob_fns_regex)

    # creating a list of new paths for each fileos.path.normpath(path)
    #os.makedirs( new_data_dir, exist_ok=True)
    li_new_fps = [ os.path.join( new_data_dir, *os.path.normpath(fp).split(os.sep)[2:] ) for fp in li_fps ]
    gen_long_dir = data_dir+"_long" if gen_long_ds else None
    
    for fp, new_fp in zip(li_fps, li_new_fps):
        process_and_save(fp, new_fp, fix_ds, gen_long_ds, gen_long_dir)



def process_and_save( fp, new_fp, fix_ds=True, gen_long_ds=True, gen_long_dir=None ):
    
    data = pd.read_csv(fp, sep=',', header=0 )
    columns = data.columns
    labels = ["text","subreddit","txt_preproc",'rst','topic_textrank']
    
    if fix_ds == True:
        data = data.apply(process_row, axis=1, result_type='expand')
        
        #remove empty rows
        data = data[  [ not txt.isspace() and any(c.isalpha() for c in txt) for txt in data.txt_preproc]  ]

        #remove rows where the topic_textrank is length 0
        #data = data [  [len(json.loads(txt))!=0 for txt in data.topic_textrank] ]

            
        #saving fixed files
        new_fp = f"{new_fp[:-10]}{len(data):010d}"
        
        os.makedirs( os.path.dirname(new_fp), exist_ok=True)

        data.to_csv(new_fp, index=False) #, quoting=csv.QUOTE_NONE)
    
    if gen_long_ds ==True:
        
        #remove empty rows
        if not fix_ds:
            data = data[  [ not txt.isspace() and any(c.isalpha() for c in txt) for txt in data.txt_preproc]  ]
            #remove rows where the topic_textrank is length 0
            #data = data [  [len(json.loads(txt))!=0 for txt in data.topic_textrank] ]

        # saving utterances with sentences longer than 2 approx dus
        fps_long2 = os.path.join( gen_long_dir+"2", *os.path.normpath(new_fp).split(os.sep)[2:] )
        os.makedirs( os.path.dirname(fps_long2), exist_ok=True)

        #saving texts that have at least two sentences, distinguished using markers such as full stop
        #_ = [ re.split( pattern_edu, txt) for txt in data.txt_preproc.values.tolist() ]
        _ = [ sent_detector.tokenize(txt)  for txt in data.txt_preproc.values.tolist() ]
        bol_mask_long2 = [ len(split_text)>1 for split_text in _ ] #utterances with more than 2 subsections
        
        
        # saving utterances with sentences longer than 2 approx dus
        data_long2 = data[ bol_mask_long2 ]
        fps_long2 = f"{fps_long2[:-10]}{len(data_long2):010d}"
        data_long2.to_csv(fps_long2, index=False) #, quoting=csv.QUOTE_NONE)

        # saving utterances with sentences longer than 2 approx dus
        fps_long3 = os.path.join( gen_long_dir+"3", *os.path.normpath(new_fp).split(os.sep)[2:] )
        os.makedirs( os.path.dirname(fps_long3), exist_ok=True)
        
        bol_mask_long3 = [ len(split_text)>2 for split_text in _ ] #utterances with more than 3 subsections
        
        data_long3 = data[ bol_mask_long3 ]
        fps_long3 = f"{fps_long3[:-10]}{len(data_long3):010d}"
        data_long3.to_csv(fps_long3, index=False) #, quoting=csv.QUOTE_NONE)


def process_row(datum):

    #THIS RE-pROCESSES FR ERRORS THAT WERE FOUND during the first generation of text used to combined combined_data_v2
    #RST Cleaning
    datum['text'] = datum['text'].strip('"')
    try:
        rst = json.loads(datum['rst'])  
    except json.decoder.JSONDecodeError as e:
        rst = ast.literal_eval(datum['rst'] )
    
    for dict_ in rst:
        dict_['rel'] =  dict_['rel'].strip('"')
        dict_['ns'] =  dict_['ns'].strip('"')
    
    rst = json.dumps(rst)
    datum['rst'] = rst
    
    #Subreddit Cleaning
    datum['subreddit'] =  datum['subreddit'].strip('"')

    #Text preproc cleaning
    txt_preproc = datum['txt_preproc']
    txt_preproc =  txt_preproc.strip('"\'')
    txt_preproc = re.sub( pattern_qoutes, '', txt_preproc )
    txt_preproc = re.sub(pattern_deleted, '', txt_preproc)
    txt_preproc = html.unescape(txt_preproc)
    txt_preproc = re.sub(pattern_txt_emojis, '' ,txt_preproc)
    txt_preproc = re.sub( pattern_subreddits ,  r'\2', txt_preproc )
    txt_preproc = txt_preproc.strip(' \'"')
    datum['txt_preproc'] = txt_preproc

    #TextRank cleaning
    topic_textrank = json.loads(datum['topic_textrank'])
    for li_txt_score in topic_textrank:
        li_txt_score[0] = li_txt_score[0].strip('"')
        li_txt_score[0] = re.sub( "(&gt;|>)", '', li_txt_score[0])
        li_txt_score[0] = re.sub(pattern_deleted, '', li_txt_score[0])
        li_txt_score[0] = re.sub(pattern_txt_emojis, '', li_txt_score[0])
        li_txt_score[0] = re.sub(pattern_subreddits, r'\2', li_txt_score[0])
    
    if len(topic_textrank) == 0:
        entry = ['',0.0]
        topic_textrank = [ ['',0.0] ]

    datum['topic_textrank'] = json.dumps(topic_textrank)

    return datum 

    
if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False) 
    
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)

    parser.add_argument('-dd','--data_dir', default="./dataset/reddit_large_annotated",
        type=str )

    parser.add_argument('-ndd','--new_data_dir', default="./dataset/reddit_large_annotated_fixed",
        type=str )    

    parser.add_argument('--fix_ds', default=True, type=lambda x: bool(int(x)), required=False )
    parser.add_argument('--gen_long_ds', default=True, type=lambda x: bool(int(x)) , required=False )
    
    args = parser.parse_args()
    
    dict_args = vars(args)

    main( **dict_args )


