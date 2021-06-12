#fix dataset script
import glob
import os
import pandas as pd
import ujson
import argparse 
import numpy as np


# This script provides two functions:
# 1) TODO: clean text that may incude garble
#endregion


def main(data_dir):
    
    # getting list of files
    glob_fns_regex = os.path.join( data_dir, "*", "*" )
    li_fps = glob.glob(glob_fns_regex)
    li_fps = [fn for fn in li_fps if os.path.split(fn)[-1]!="lock"]
    
    for idx, fp in enumerate(li_fps):
        process_and_save(fp )
        if idx % 10 == 0:
            print(f"Operating on file {idx} of {len(li_fps)}")



def process_and_save( fp ):
    
    data = pd.read_csv(fp, sep=',', header=0 )
    columns = data.columns
    
    if 'li_edus' not in columns and 'li_dict_posname_likpscore' not in columns:
        os.remove(fp)
        return True
        
    #remove rows where pos0 is empty
    pos0_records = data['li_dict_posname_likpscore'].values

    pos0_records = [ujson.decode(val) if type(val)==str else False for val in pos0_records ]

    # Creating a filter to check all pos0's are larger than length 0
    bool_filter = [ all( len(dict_['pos0'][0][0])!=0  for dict_ in li_dict ) if li_dict!=False else False for li_dict in pos0_records  ]
    
    # remove rows
    data = data[  bool_filter  ]
        
    #saving fixed files
    new_fp = f"{fp[:-10]}{len(data):010d}"
    
    os.remove(fp)
    data.to_csv(new_fp, index=False) #, quoting=csv.QUOTE_NONE)
    

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False) 
    
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)

    parser.add_argument('-dd','--data_dir', default="./dataset_keyphrase",
        type=str )    
    args = parser.parse_args()
    
    dict_args = vars(args)

    main( **dict_args )


