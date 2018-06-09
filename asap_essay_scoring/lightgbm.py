import lightgbm as lgb
import numpy as np
import pandas as pd
import pdb
import sklearn

class Lgbm(object):
    def __init__(self, data, verbose = 0):
        '''
        :param data: list containing numeric features 'X' as pandas DataFrame and labels 'y' as numeric numpy array
        :param verbose: (integer) verbosity level
        '''
        self.X = data['X']
        self.y = data['y'].astype('int')
        self.verbose = verbose
        # Generate data folds
        np.random.seed(0)
        kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True) #Stratified
        self.folds = [
            {'train': train_index, 'test': test_index}
            for train_index, test_index in kf.split(self.X)
        ]
        # Keep track of indices and their relationship to essay sets
        self.keys = pd.DataFrame({
            'essay_set': data['essay_set'],
            'ref': list(range(len(self.y)))
        })

    def _train(self, idx):
        y = self.y[idx]
        data = lgb.Dataset(self.X.iloc[idx], y)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'verbose': -1
            #'multiclass',
            #'num_class': len(np.unique(y))
        }
        bst = lgb.train(params=params,
                        train_set=data,
                        num_boost_round=1, #00,
                        verbose_eval=False)
        return bst

    def train_one_fold(self, k):
        '''
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
                'pred': fit[e].predict(self.X.iloc[group['ref']]),
                'idx': group['ref'],
                'essay_set': e
            }) for e, group in self.keys.iloc[tst].groupby('essay_set')]
        )

    def predict_all_folds(self):
        preds = pd.concat([self.predict_one_fold(k) for k in range(len(self.folds))])
        return preds.sort_values('idx')