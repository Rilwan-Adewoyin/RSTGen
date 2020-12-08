import os
import json
dirname = os.path.dirname(__file__)
from datetime import date
import glob


def get_path(_path,_dir=False):
    if os.path.isabs(_path) == False:
        _path = os.path.join(dirname,_path)
    
    _path = os.path.realpath(_path)
    
    if _dir:
        os.makedirs(_path,exist_ok=True)
    
    return _path
