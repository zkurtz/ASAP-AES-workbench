import copy
from abc import ABC, abstractmethod
import numpy as np
import pdb

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

from .data import get_domain1_ranges

class AbstractLearner(ABC):
    def __init__(self, params = None):
        self.params = self.default_params()
        if params is not None:
            self.params.update(copy.deepcopy(params))

    @abstractmethod
    def default_params(self):
        pass

    @abstractmethod
    def train(self, data):
        pass

    def _bound_predictions(self, preds, essay_set):
        assert len(preds) == len(essay_set)
        asap_ranges = get_domain1_ranges()
        mins = [asap_ranges[e][0] for e in essay_set]
        maxs = [asap_ranges[e][1] for e in essay_set]
        pdb.set_trace()
        unders = np.where(preds < mins)
        overs = np.where(preds > maxs)
        preds[unders] = mins[unders]
        preds[overs] = maxs[overs]
        return preds

    @abstractmethod
    def _predict(self, X):
        pass

    def predict(self, X, groups = None):
        preds = self._predict(X)
        preds = np.round(preds).astype('int')
        if groups is not None:
            assert X.shape[0] == len(groups)
            return self._bound_predictions(preds, groups)
        return preds


class Lgbm(AbstractLearner):
    def __init__(self, params = None):
        super().__init__()

    def default_params(self):
        return {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'learning_rate': 0.1,
            'verbose': -1,
            'num_boost_round': 29
        }

    def train(self, data):
        '''
        :param data: a data.Data instance
        '''
        ld = lgb.Dataset(data.X, data.y)
        params = copy.deepcopy(self.params)
        nround = params.pop('num_boost_round')
        # zz = self.bst = lgb.cv(params=params,
        #                   train_set=ld,
        #                   num_boost_round=nround,
        #                   verbose_eval=False)
        # pdb.set_trace()
        self.bst = lgb.train(params=params,
                            train_set=ld,
                            num_boost_round=nround,
                            verbose_eval=False)

    def _predict(self, X):
        return self.bst.predict(X)


class Skrf(AbstractLearner):
    ''' Wraps sklearn random forest'''
    def __init__(self, params = None):
        super().__init__()

    def default_params(self):
        return {
            'n_estimators': 100
        }

    def train(self, data):
        '''
        :param data: a data.Data instance
        '''
        self.rf = RandomForestRegressor(n_estimators=100)
        self.rf.fit(data.X.values, data.y)

    def _predict(self, X):
        return self.rf.predict(X.values)
