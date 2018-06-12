# ASAP-AES: word2vec and more

This repo presents a solution for a Kaggle-hosted competition on automatic essay 
scoring: https://www.kaggle.com/c/asap-aes

The [performance testing log](#Performance-testing-log) documents progress over time.

## Set up

Developed on OSX + python 3.6 with no formal code tests. Install python 
packages as per environment.yml.

Download data as follows (the project code assumes this exact directory structure): 

```bash
# Download the data from kaggle
pip install kaggle
kaggle competitions download -c asap-aes

# Organize the data directory
cd ~/.kaggle/competitions/asap-aes
mkdir engineered_features
```

## Model preparation, building, testing

```bash
# Feature engineering on the training set
python featurize_training_data.py

# Fit the model: choose between one of two kinds:
#   - cross_val: estimate the out-of-sample performance
#   - build_final: build a model using all of the training data
python performance_crossval.py
TODO python build_final_classifier.py

# Compute the engineered features on the test set and test the classifier
TODO python featurize_testing_data.py
TODO python performance_final.py
```

## Performance testing log

This log tracks the progress in terms of improvements of prediction accuracy, 
including notes about which changes in methodology appeared to have
led to improvements.

Model development (including feature engineering, hyperparameter tuning, etc) relies on
cross-validation and excludes use of the test data (`test_set.tsv`) data.

Whenever I reach accuracy milestones that I can't resist 
bragging about, I take the final step of computing the performance score on the test 
set. I do this as rarely as possible, however, since each use of the test 
data (`test_set.tsv`) contaminates the training procedure by 
leaking information about the test set. 

### Cross-validation

- 2018.06.09: 
    - 0.047 LightGBM on # of words, # of characters. Definitely coder error is at fault considering 
    that the analogous benchmark scored 0.64 on the kaggle leaderboard.
    - 0.3153 restored default LightGBM 100 boosting rounds. Still, why so low?
    - 0.363 include word2vec features

- 2018.06.10:
    - 0.281 is what I get with the random forest as in 
    the [benhamner benchmark](https://github.com/benhamner/ASAP-AES/blob/master/Benchmarks/length_benchmark.py)
    ... something's off
    
### Final evaluation

- ... TODO
