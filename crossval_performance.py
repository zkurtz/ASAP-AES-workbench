
import numpy as np

from asap_essay_scoring import cross_predict
from asap_essay_scoring import data
from asap_essay_scoring import learners
from asap_essay_scoring import metrics

# Configure
TARGET = 'domain1_score' # A simplification for now; ignoring `domain2_score` for essay set 2
USE_EMBEDDINGS = True

# Load training
dm = data.DataManager(target = TARGET, use_embeddings=USE_EMBEDDINGS)
training_data = dm.prepare_data()

grp8 = np.where(8 == training_data.group)
d8 = training_data.select(grp8)
df = d8.X.copy()
df['y'] = d8.y
df.y.corr(df.nchar)
df.y.corr(df.nword)

def evaluate(Learner, data):
    print('fitting learner')
    cp = cross_predict.CrossPredict(data = data, Learner = Learner)
    preds = cp.cross_predict()
    print('cross-validation performance:')
    preds['pred'] = np.round(preds.pred.values).astype('int') # information loss :(
    metrics.evaluate(preds)

print('lightgbm: ')
evaluate(learners.Lgbm, data = training_data)

# print('random forest: ')
# evaluate(learners.Skrf, data = training_data)
