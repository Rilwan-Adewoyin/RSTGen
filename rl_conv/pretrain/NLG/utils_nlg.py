import os
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, GPT2LMHeadModel
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


def load_pretrained_transformer( model_name='bert-base-cased', transformer=True, 
                                    tokenizer=False):
    _dir_transformer = os.path.join( get_path("./models"), model_name )
    exists = os.path.isdir(_dir_transformer)
    output = {}

    if exists == False:    
        model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        #model = AutoModel.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        model_tokenizer.save_pretrained(_dir_transformer)
        model.save_pretrained(_dir_transformer)

    if tokenizer == True:
        output['tokenizer'] = AutoTokenizer.from_pretrained(_dir_transformer)

    if transformer == True:
        output['transformer'] = AutoModel.from_pretrained(_dir_transformer)
    
    return output

def load_pretrained_tokenizer_local( model_name='NLG'):

    _dir_tknzr = os.path.join( get_path("./models",_dir=True), model_name )
    exists = os.path.isdir(_dir_tknzr)

    if exists == True:    
        if model_name == "NLG":
            tknzr =  AutoTokenizer.from_pretrained(_dir_tknzr)
            res = tknzr
        else:
            raise ValueError()
    else:
        res = False
    
    return res


def save_version_params(t_params, m_params, version_code="DaNet_v000"):
    dated_trained = date.today().strftime("%d-%m-%Y")
    
    t_params.date_trained = dated_trained
    m_params.date_trained = dated_trained

    _dir_version = get_path(f"./models/{version_code}/",_dir=True)
    
    tp_fp = os.path.join(_dir_version,'tparam.json')
    mp_fp = os.path.join(_dir_version,'mparam.json')

    json.dump( vars(t_params), open(tp_fp,"w") )
    json.dump( vars(m_params), open(mp_fp,"w") )

    return True    

    