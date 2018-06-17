
''' Facilitate cross-validation for an arbitrary learner.
This computes folds, fits a learner on each fold, and gathers the predictions
on each hold-out set. It does not do any evaluation of those predictions '''

import copy
import numpy as np
import pandas as pd
import pdb
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

    def _train_and_predict(self, train_idx, test_idx):
        ''' Train a classifier on the data corresponding to the rows in `idx` '''
        train_data = self.data.select(train_idx)
        test_data = self.data.select(test_idx)
        learner = self.Learner(params = copy.deepcopy(self.params))
        learner.train(train_data)
        return pd.DataFrame({
            'pred': learner.predict(test_data.X),
            'truth': test_data.y,
            'idx': test_idx,
            'essay_set': self.keys.essay_set[test_idx]
        })

    def _train_and_predict_one_fold(self, k):
        '''
        :param k: (integer) which fold
        '''
        if self.verbose:
            print('training fold ' + str(k))
        trn = self.folds[k]['train']
        tst = self.folds[k]['test']
        # Currently we're fitting a totally separate model for each essay set, so
        #   here we iterate over the essay sets:
        trn_es = self.keys.iloc[trn].groupby('essay_set')
        tst_es = self.keys.iloc[tst].groupby('essay_set')
        essays = tst_es.groups.keys()
        assert essays == trn_es.groups.keys() # in case cv-split was terribly uneven -- ideally should stratify it
        return pd.concat([self._train_and_predict(
            trn_es.get_group(e).ref, tst_es.get_group(e).ref) for e in essays])

    def cross_predict(self):
        kr = range(len(self.folds))
        preds = pd.concat([self._train_and_predict_one_fold(k) for k in kr])
        return preds.sort_values('idx')