import copy

import lightgbm as lgb

class Lgbm(object):
    def __init__(self, params):
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
        ''' Train a classifier on the data corresponding to the rows in `idx` '''
        data = lgb.Dataset(data.X, data.y)
        params = copy.deepcopy(self.params)
        nround = params.pop('num_boost_round')
        self.bst = lgb.train(params=params,
                            train_set=data,
                            num_boost_round=nround,
                            verbose_eval=False)

    def predict(self, X):
        return self.bst.predict(X)
