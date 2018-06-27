
import copy
import csv
import numpy as np
import os
import pandas as pd
import pdb

from . import utils

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
    def __init__(self, target, use_embeddings=False):
        self.target = target
        self.emb = None
        if use_embeddings:
            self.emb = pd.read_csv(efpath("txt_features.csv"))
        self.files = {
            'raw': utils.data_path('training_set_rel3.tsv'),
            'tokenized': efpath('tokenized.json')
        }

    def _attach_txt_features(self, df):
        if self.emb is not None:
            return pd.concat([df, self.emb], axis=1)
        else:
            return df

    def _read_raw(self, file):
        # Reading this is tricky. The following pandas reader relies on these two posts:
        #   - https://stackoverflow.com/a/37723241/2232265
        #   - https://stackoverflow.com/a/35622971/2232265
        raw = pd.read_csv(file, sep='\t', encoding='latin-1', quoting=csv.QUOTE_NONE)
        if self.target in raw.columns:
            raw[self.target] = raw[self.target].astype('int')
        return raw

    def prepare_data(self):
        raw = self._read_raw(self.files['raw'])
        X = pd.DataFrame({
            'nchar': [len(s) for s in raw.essay],
            'nword': [len(doc) for doc in utils.json_load(self.files['tokenized'])]
        })
        return Data(X = self._attach_txt_features(X),
                    y = raw[self.target].values,
                    group = raw['essay_set'])
