

import numpy as np
import os
import pandas as pd
import pdb

from asap_essay_scoring import lightgbm as lgb
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

def length_only_lightgbm_benchmark(raw, toks):
    print('assemble data')
    X = pd.DataFrame({
        'nchar': [len(s) for s in raw.essay],
        'nword': [len(doc) for doc in toks]
    })
    data = {
        'X': X,
        'y': raw[TARGET].values,
        'essay_set': raw['essay_set']
    }
    print('fit lightgbm')
    lgbm = lgb.Lgbm(data, verbose=1)
    lgbm.train_all_folds()
    print('cross-validation performance:')
    preds = lgbm.predict_all_folds()
    preds['pred'] = np.round(preds.pred.values).astype('int')
    preds.sort_values('idx', inplace=True)
    preds['truth'] = raw[TARGET].astype('int').values
    kappas = [metrics.kappa(g.pred.values, g.truth.values) for e, g in preds.groupby('essay_set')]
    print(kappas)
    print(metrics.mean_quadratic_weighted_kappa(kappas))

length_only_lightgbm_benchmark(raw, toks)

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
