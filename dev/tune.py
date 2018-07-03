
# Use hyperopt to tune the lightgbm learner

import hyperopt as hpt

from asap_essay_scoring import cross_predict
from asap_essay_scoring import data
from asap_essay_scoring import learners
from asap_essay_scoring import metrics

# Configure
TARGET = 'domain1_score' # A simplification for now; ignoring `domain2_score` for essay set 2
USE_EMBEDDINGS = False

# Load training
dm = data.DataManager(target = TARGET, use_embeddings=USE_EMBEDDINGS)
training_data = dm.prepare_data()

def objective(params):
    cp = cross_predict.CrossPredict(
        data = training_data, Learner = learners.Lgbm, n_fold = 5,
        hyperparameters = ?,
        verbose = True)
    preds = cp.cross_predict()
    metrics.evaluate(preds)

space = hpt.hp.choice('a',
    [
        ('case 1', 1 + hpt.hp.lognormal('c1', 0, 1)),
        ('case 2', hpt.hp.uniform('c2', -10, 10))
    ])

best = hpt.fmin(objective, space, algo=hpt.tpe.suggest, max_evals=100)
