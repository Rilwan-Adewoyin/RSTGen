import glob
import os
import pandas as pd
import json
import ujson
import multiprocessing as mp
import traceback

def main():
    dir_data = "./dataset_v3/"
    li_fps = glob.glob(os.path.join( dir_data, "*","*") ) 

    li_fps_to_rm =  [fn for fn in li_fps if os.path.split(
            fn)[-1] == "lock" or "dict_len" in fn]
    
    for fn in li_fps_to_rm:
        os.remove(fn)

    li_fps = [fn for fn in li_fps if os.path.split(
            fn)[-1] != "lock" and "dict_len" not in fn]

    with mp.Pool(20) as p:
        res = p.map(work, li_fps, chunksize=1 )
    
    list(res)
        
    print("Done")

def work(fp):
    
    print( f"Operating on {fp.split('/')[-3] }- {fp.split('/')[-2] } ")
    df = pd.read_csv(fp, sep=',', header=0, )

    li_records = df.to_dict('records')
    old_len = len(li_records)

    for idx in reversed(range(len(li_records))):
        
        record_rst =  json.loads(li_records[idx]['rst'])

        if len(record_rst)==1:
            if len(li_records[idx]['txt_preproc'].split() ) < 6:
                li_records.pop(idx)
                continue
                    
        if len(record_rst)==0:
            li_records.pop(idx)
            continue
        
        if len(record_rst)==1 and record_rst[0]['ns'] == 'a' :
            li_records.pop(idx)
            continue
        
        li_pos_kp = json.loads( li_records[idx]['li_pos_kp'] )
        
        if len( li_pos_kp ) == 1 and len( li_pos_kp[0] ) == 0  :
            li_records.pop(idx)
            continue

        if len( li_pos_kp ) == 1 and li_pos_kp[0][1] == ''  :
            li_records.pop(idx)
            continue
        
    new_len = len(li_records)

    if new_len  != old_len:
        new_fp = os.path.join( os.path.dirname(fp), f"{new_len:010d}" )
        pd.DataFrame(li_records).to_csv( new_fp , header=True, index=False)
        os.remove(fp)
        print(f"changed {fp} to {new_fp}")
        
        
    elif fp[-4:] == ".csv":
        new_fp = fp[:-4]
        pd.DataFrame(li_records).to_csv( new_fp , header=True, index=False)
        os.remove(fp)
        

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
