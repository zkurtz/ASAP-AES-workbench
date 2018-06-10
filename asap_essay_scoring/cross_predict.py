
''' Facilitate cross-validation for an arbitrary learner.
This computes folds, fits a learner on each fold, and gathers the predictions
on each hold-out set. It does not do any evaluation of those predictions '''

import copy
import numpy as np
import pandas as pd
import sklearn

class CrossPredict(object):
    def __init__(self, data, Learner, hyperparameters = None, verbose = 0):
        '''
        :param data: list containing numeric features 'X' as pandas DataFrame and labels 'y' as numeric numpy array
        :param learner: (a learner class inheriting from BaseLearner)
        :param verbose: (integer) verbosity level
        '''
        self.data = data
        self.params = hyperparameters
        self.Learner = Learner
        self.verbose = verbose
        # Generate data folds
        np.random.seed(0)
        kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True) #Stratified
        self.folds = [
            {'train': train_index, 'test': test_index}
            for train_index, test_index in kf.split(self.data.X)
        ]
        # Keep track of indices and their relationship to essay sets
        self.keys = pd.DataFrame({
            'essay_set': data.group,
            'ref': list(range(len(data.y)))
        })

    def _train(self, idx):
        ''' Train a classifier on the data corresponding to the rows in `idx` '''
        data = self.data.select(idx)
        learner = self.Learner(params = copy.deepcopy(self.params))
        learner.train(data)
        return learner

    def train_one_fold(self, k):
        '''
        Train a separate model for each essay set

        :param k: index of fold in self.folds
        '''
        if self.verbose:
            print('training fold ' + str(k))
        trn = self.folds[k]['train']
        return {e: self._train(group['ref'].tolist())
                for e, group in self.keys.iloc[trn].groupby('essay_set')}

    def train_all_folds(self):
        self.fits = {k: self.train_one_fold(k) for k in range(len(self.folds))}

    def predict_one_fold(self, k):
        tst = self.folds[k]['test']
        fit = self.fits[k]
        return pd.concat(
            [pd.DataFrame({
                'pred': fit[e].predict(self.data.X.iloc[group['ref']]),
                'idx': group['ref'],
                'essay_set': e
            }) for e, group in self.keys.iloc[tst].groupby('essay_set')]
        )

    def predict_all_folds(self):
        preds = pd.concat([self.predict_one_fold(k) for k in range(len(self.folds))])
        return preds.sort_values('idx')