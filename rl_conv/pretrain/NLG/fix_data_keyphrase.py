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
            
    #remove rows where dict_pos_edu is less than 6 length
    li_dict_pos_edu = data['dict_pos_edu'].values

    li_dict_pos_edu = [ujson.decode(val) for val in li_dict_pos_edu ]

    # Creating a filter to check all pos0's are larger than length 0
        
    bool_filter = [len(_dict)>=6 for _dict in li_dict_pos_edu]

    # remove rows
    data = data[  bool_filter  ]
        
    #saving fixed files
    new_fp = f"{fp[:-10]}{len(data):010d}"
    
    os.remove(fp)
    data.to_csv(new_fp, index=False) #, quoting=csv.QUOTE_NONE)
    

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False) 
    
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)

    parser.add_argument('-dd','--data_dir', default="./dataset_keyphrase_v2",
        type=str )    
    args = parser.parse_args()
    
    dict_args = vars(args)

    main( **dict_args )


