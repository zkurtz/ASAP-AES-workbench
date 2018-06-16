import copy

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

class Lgbm(object):
    def __init__(self, params = None):
        self.params = self.default_params()
        if params is not None:
            self.params.update(copy.deepcopy(params))

    def default_params(self):
        return {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'verbose': -1,
            'num_boost_round': 100
        }

    def train(self, data):
        '''
        :param data: a data.Data instance
        '''
        data = lgb.Dataset(data.X, data.y)
        params = copy.deepcopy(self.params)
        nround = params.pop('num_boost_round')
        self.bst = lgb.train(params=params,
                            train_set=data,
                            num_boost_round=nround,
                            verbose_eval=False)

    def predict(self, X):
        return self.bst.predict(X)

class Skrf(object):
    ''' Wraps sklearn random forest'''
    def __init__(self, params = None):
        self.params = self.default_params()
        if params is not None:
            self.params.update(copy.deepcopy(params))

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

    def predict(self, X):
        return self.rf.predict(X.values)
