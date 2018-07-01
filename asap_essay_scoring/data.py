
import copy
import csv
import numpy as np
import os
import pandas as pd
import pdb

from . import utils

def read_raw_csv(file, target = 'domain1_score'):
    '''
    Reading the main training file is tricky. We rely on these two posts:
       - https://stackoverflow.com/a/37723241/2232265
       - https://stackoverflow.com/a/35622971/2232265
    '''
    raw = pd.read_csv(file, sep='\t', encoding='latin-1', quoting=csv.QUOTE_NONE)
    if target in raw.columns:
        raw[target] = raw[target].astype('int')
    return raw

def get_domain1_ranges():
    ''' Assuming this is correct:
    https://github.com/nusnlp/nea/blob/3673d2af408d5a5cb22d0ed6ff1cd0b25a0a53aa/nea/asap_reader.py '''
    return {
        1: (2, 12),
        2: (1, 6),
        3: (0, 3),
        4: (0, 3),
        5: (0, 4),
        6: (0, 4),
        7: (0, 30),
        8: (0, 60)
    }

class Data(object):
    ''' Standardized data format that stores basic metadata and allows selecting on rows '''

    def __init__(self, X, y = None, group = None, which_categorical = None):
        self.X = X.copy()
        self.y = None if y is None else np.copy(y)
        self.which_categorical = copy.deepcopy(which_categorical)
        self.group = None if group is None else np.copy(group)

    def select(self, idx):
        y = None if self.y is None else self.y[idx]
        group = None if self.group is None else self.group[idx]
        return Data(
            X = self.X.iloc[idx],
            y = y,
            group = group,
            which_categorical = self.which_categorical
        )

def efpath(filename):
    ''' Generate the full path to `filename` in the engineered_features directory '''
    return utils.data_path(os.path.join('engineered_features', filename))

class DataManager(object):
    def __init__(self, target, feature_types = ['len_benchmark', 'wordvec', 'docvec']):
        self.target = target
        self._load('raw')
        self.feature_types = feature_types

    def _load(self, feature_set):
        if hasattr(self, feature_set):
            return
        if feature_set == 'raw':
            f = utils.data_path('training_set_rel3.tsv')
            self.raw = read_raw_csv(f, target=self.target)
        elif feature_set == 'tokenized':
            self.tokenized = utils.json_load(efpath('tokenized.json'))
        elif feature_set == 'wordvec_features':
            self.wordvec_features = pd.read_csv(efpath("wordvec_features.csv"))
        elif feature_set == 'docvec_features':
            self.docvec_features = pd.read_csv(efpath("docvec_features.csv"))
        else:
            raise Exception('invalid feature_set')

    def len_benchmark(self):
        self._load('tokenized')
        return pd.DataFrame({
            'nchar': [len(s) for s in self.raw.essay],
            'nword': [len(doc) for doc in self.tokenized]
        })

    def wordvec(self):
        self._load('wordvec_features')
        return self.wordvec_features

    def docvec(self):
        self._load('docvec_features')
        return self.docvec_features

    def prepare_data(self, feature_types = None):
        if feature_types is not None:
            self.feature_types = feature_types
        assert len(feature_types) > 0
        self.feature_builders = [getattr(self, s) for s in self.feature_types]
        feature_dataframes = [f() for f in self.feature_builders]
        features = pd.concat(feature_dataframes, axis=1)
        return Data(X = features,
                    y = self.raw[self.target].values,
                    group = self.raw['essay_set'])
