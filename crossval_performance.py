from asap_essay_scoring import cross_predict
from asap_essay_scoring import data
from asap_essay_scoring import learners
from asap_essay_scoring import metrics

# Configure
TARGET = 'domain1_score'
FEATURE_TYPES = ['len_benchmark', 'wordvec', 'token']

# Load training
dm = data.DataManager(target = TARGET)
training_data = dm.prepare_data(FEATURE_TYPES)

def evaluate(Learner, data):
    print('fitting learner on cross-validation folds')
    cp = cross_predict.CrossPredict(
        data = data, Learner = Learner, n_fold = 5, verbose = True)
    preds = cp.cross_predict()
    #preds = cp.cheat()
    print('cross-validation performance:')
    metrics.evaluate(preds)

print('lightgbm: ')
evaluate(learners.Lgbm, data = training_data)

# print('random forest: ')
# evaluate(learners.Skrf, data = training_data)