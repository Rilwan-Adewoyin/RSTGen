import os
import json
from transformers import AutoTokenizer, AutoModel
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
dirname = os.path.dirname(__file__)
from datetime import date
import glob
import regex as re


def get_path(_path,_dir=False):

    if os.path.isabs(_path) == False:
        _path = os.path.join(dirname, _path)
    
    _path = os.path.realpath(_path)
    
    if _dir:
        os.makedirs(_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(_path), exist_ok=True)

    return _path


def load_pretrained_transformer( model_name='bert-base-cased', transformer=True, tokenizer=False):
    #If model name contains a forward slash then only take second half
    if "/" in model_name:
        _dir_transformer = os.path.join( get_path("./models"), model_name.split("/")[-1] )
    else:
        _dir_transformer = os.path.join( get_path("./models"), model_name )

    exists = os.path.isdir(_dir_transformer)

    output = {}

    if exists == False:   

        if model_name == "tuner007/pegasus_paraphrase":
            model_tokenizer = PegasusTokenizer.from_pretrained(model_name)
            model = PegasusForConditionalGeneration.from_pretrained(model_name)
        else:
            model_tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

        
        
        model_tokenizer.save_pretrained(_dir_transformer)
        model.save_pretrained(_dir_transformer)

    if tokenizer == True:
        output['tokenizer'] = AutoTokenizer.from_pretrained(_dir_transformer)

    if transformer == True:
        if model_name == "tuner007/pegasus_paraphrase":
            output['transformer'] = PegasusForConditionalGeneration.from_pretrained(_dir_transformer)
        else:
            output['transformer'] = AutoModel.from_pretrained(_dir_transformer)
    
    return output

def save_version_params(t_params=None, m_params=None, version_code="DaNet_v000"):
    dated_trained = date.today().strftime("%d-%m-%Y")
    _dir_version = get_path(f"./models/{version_code}/",_dir=True)

    if t_params is not None:
        t_params['date_trained'] = dated_trained
        tp_fp = os.path.join(_dir_version,'tparam.json')
        json.dump( vars(t_params), open(tp_fp,"w") )
    
    if m_params is not None:
        m_params['date_trained'] = dated_trained 
        mp_fp = os.path.join(_dir_version,'mparam.json')
        json.dump( vars(m_params), open(mp_fp,"w") )

    return True    

def get_version_name(model_name):
    _dir_models = get_path("./models")
    li_modelversions = glob.glob( os.path.join(_dir_models,model_name+"_v*") )

    li_versioncodes = [ int(modelversion[-3:]) for modelversion in li_modelversions ]
    
    new_version_code = max( li_versioncodes, default=-1 ) + 1

    new_version_name = f"{model_name}_v{str(new_version_code).zfill(3)}" 

    return new_version_name 


def get_best_ckpt_path(dir_path):
    dir_path = get_path(dir_path)

    li_files = glob.glob(os.path.join(dir_path,"*.ckpt"))

    li_ckpts = [ re.findall( r"val_loss=[\S]{5}" ,fname)[-1][-3:] for fname in li_files ]

    index = li_ckpts.index( max(li_ckpts, key=float) )

    ckpt_path = li_files[index]

    return ckpt_path