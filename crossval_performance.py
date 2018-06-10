

import numpy as np
import os
import pandas as pd
import pdb

#from asap_essay_scoring import lightgbm as lgb
from asap_essay_scoring import cross_predict
from asap_essay_scoring import data
from asap_essay_scoring import learners
from asap_essay_scoring import metrics
from asap_essay_scoring import utils

# Configure
TARGET = 'domain1_score'

# utils
def efpath(filename):
    ''' Generate the full path to `filename` in the engineered_features directory '''
    return utils.data_path(os.path.join('engineered_features', filename))

# Load training data inputs
raw = pd.read_csv(utils.data_path('training_set_rel3.tsv'), sep='\t', encoding = "ISO-8859-1")
emb = pd.read_csv(efpath("train_txt_features.csv"))
toks = utils.json_load(efpath('train_tokenized.json'))

def prepare_data():
    print('preparing data')
    X = pd.DataFrame({
        'nchar': [len(s) for s in raw.essay],
        'nword': [len(doc) for doc in toks]
    })
    X = pd.concat([X, emb], axis = 1)
    return data.Data(X = X, y = raw[TARGET].values, group = raw['essay_set'])

def test_and_evaluate_lightgbm(data):
    print('fitting lightgbm')
    cp = cross_predict.CrossPredict(data = data, Learner = learners.Lgbm)
    cp.train_all_folds()
    print('cross-validation performance:')
    preds = cp.predict_all_folds()
    preds['pred'] = np.round(preds.pred.values).astype('int')
    preds.sort_values('idx', inplace=True)
    preds['truth'] = raw[TARGET].astype('int').values
    metrics.evaluate(preds)

def length_only_lightgbm_benchmark():
    data = prepare_data()
    test_and_evaluate_lightgbm(data)

length_only_lightgbm_benchmark()

# # Assemble data
# emb['essay_set'] = df.essay_set
# emb['nchar'] = [len(s) for s in df.essay]
# lgbm = lgb.Lgbm(emb, df.domain1_score, verbose = 1)
# lgbm.train_all_folds()
# preds = lgbm.predict_all_folds()
# preds.sort_values('idx', inplace = True)
# preds['truth'] = df[TARGET]
# preds['pred'] = np.round(preds.pred.values).astype('int')
#
# kappas = [metrics.kappa(g.pred.values, g.truth.values) for e, g in preds.groupby('essay_set')]
# metrics.mean_quadratic_weighted_kappa(kappas)
