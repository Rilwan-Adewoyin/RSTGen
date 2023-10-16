import glob
import os
import pandas as pd
import json
import regex as re
    
# clearning change my view dataset
def main():
    dir_data = ".data/data_files/dataset_cmv/"
    li_fps = glob.glob(os.path.join( dir_data, "nlg*","train","*") ) + \
                glob.glob(os.path.join( dir_data, "nlg*","val","*") ) +\
                glob.glob(os.path.join( dir_data, "nlg*","test","*") )

    for fp in li_fps:

        df = pd.read_csv(fp)

        li_records = df.to_dict('records')
        old_len = len(li_records)

        for idx in reversed(range(len(li_records))):
            
            record_rst =  json.loads(li_records[idx]['rst'])

            if len(record_rst)==1 and record_rst[0]['ns'] == 'a' :
                li_records.pop(idx)
                continue
        
        new_len = len(li_records)

        if new_len  != old_len:
            new_fp = os.path.join( os.path.dirname(fp), f"{new_len}.csv" )
            pd.DataFrame(li_records).to_csv( new_fp , header=True, index=False)
            os.remove(fp)

    print("Done")



if __name__ == '__main__':

    main()