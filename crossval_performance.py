

import numpy as np
import os
import pandas as pd
import pdb

from asap_essay_scoring import cross_predict
from asap_essay_scoring import data
from asap_essay_scoring import learners
from asap_essay_scoring import metrics
from asap_essay_scoring import utils

# Configure
TARGET = 'domain1_score' # A simplification for now; ignoring `domain2_score` for essay set 2

# utils
def efpath(filename):
    ''' Generate the full path to `filename` in the engineered_features directory '''
    return utils.data_path(os.path.join('engineered_features', filename))

# Load training data inputs
raw = pd.read_csv(utils.data_path('training_set_rel3.tsv'), sep='\t', encoding = "ISO-8859-1")
raw[TARGET] = raw[TARGET].astype('int')
emb = pd.read_csv(efpath("train_txt_features.csv"))
toks = utils.json_load(efpath('train_tokenized.json'))

def prepare_data(use_embeddings=False):
    print('preparing data')
    X = pd.DataFrame({
        'nchar': [len(s) for s in raw.essay],
        'nword': [len(doc) for doc in toks]
    })
    if use_embeddings:
        X = pd.concat([X, emb], axis = 1)
    return data.Data(X = X, y = raw[TARGET].values, group = raw['essay_set'])

def evaluate(Learner, data):
    print('fitting learner')
    cp = cross_predict.CrossPredict(data = data, Learner = Learner)
    preds = cp.cross_predict()
    print('cross-validation performance:')
    preds['pred'] = np.round(preds.pred.values).astype('int') # information loss :(
    metrics.evaluate(preds)

def length_only_benchmark_lightgbm():
    data = prepare_data(use_embeddings=False)
    evaluate(learners.Lgbm, data)

def length_only_benchmark_randomforest():
    data = prepare_data(use_embeddings=False)
    evaluate(learners.Skrf, data)

print('lightgbm: ')
length_only_benchmark_lightgbm()

print('random forest: ')
length_only_benchmark_randomforest()

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
