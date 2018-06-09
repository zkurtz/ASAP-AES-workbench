
import json
import os
import pickle

from . import conf

def data_path(s):
    return os.path.join(conf.DATA_PATH, s)

def pickle_save(item, filepath):
    '''
    :param filepath: A relative path starting from conf.DATA_PATH
    '''
    pickle.dump(item, open(data_path(filepath), "wb"))

def pickle_load(filepath):
    '''
    :param filepath: A relative path starting from conf.DATA_PATH
    '''
    return pickle.load(open(data_path(filepath), "rb"))

def json_save(item, filepath):
    '''
    :param filepath: A relative path starting from conf.DATA_PATH
    '''
    with open(filepath, 'w') as f:
        json.dump(item, f)

def json_load(filepath):
    '''
    :param filepath: A relative path starting from conf.DATA_PATH
    '''
    with open(filepath, 'r') as f:
        return json.load(f)