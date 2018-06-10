import datalineage

from . import pipeline_methods as plm

# Constants
DL = datalineage.DataLineage('../data_lineage.yaml')
FL = dl.file_lookup()

class MakeData(object):
    def __init__(self, pipeline, verbose = 0):
        '''
        :param pipeline: (python module) Should have a method with the same name as every key in `FL.keys()`
        :param verbose: (integer) TODO
        '''
        self.module
        self.verbose = verbose

    def make(self, file_key):
        '''
        :param file_key: (str) A key of `FL`
        '''
        parents = FL[file_key]['parents']
        if len(parents) == 0:
            print(file_key + ' has no parents -- we are assuming it is up-to-date!')
        for p in parents:
            dl.ensure(p)
        make_method = getattr(self.module, file_key)
        make_method()

def train_tokenized():
    dl.ensure_parents('train')
    plm.tokenize(infile=fl('train'), outfile=fl('train_tokenized.json'))

def train_tokenized_reduced():
    print('Generating data file `train_tokenized_reduced` from `train_tokenized`')
    dl.ensure('train_tokenized')