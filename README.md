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
python featurize_data.py

# Cross-validate the model
python performance_crossval.py
```

## Performance testing log

This log tracks the progress in terms of improvements of prediction accuracy, 
including notes about which changes in methodology appeared to have
led to improvements.

Our test results are ultimately a bit biased upwards because our development process
consists of a series of experiments on the same training data set (even though
we are using cross validation). In an ideal scenario, we would complete
model development on a training set and then test the result on a
never-before-seen test set.

However, labels for the Kaggle-leaderboard test set (`test_set.tsv`) appear to
never have been released. For example, [these authors state](https://arxiv.org/pdf/1606.04289.pdf)

> ... the test set was released without
the gold score annotations, rendering any comparisons
futile, and we are therefore restricted in
splitting the given training set to create a new test
set.

Indeed, some published papers appear to have gotten away with reporting a
cross-validation score as their main accuracy result, including 
[this one](https://www.aclweb.org/anthology/D16-1193) and 
[this one](https://dl.acm.org/citation.cfm?id=3098160). 


- 2018.06.09: 
    - 0.047 LightGBM on # of words, # of characters. Definitely coder error is at fault considering 
    that the analogous benchmark scored 0.64 on the kaggle leaderboard.
    - 0.3153 restored default LightGBM 100 boosting rounds. Still, why so low?
    - 0.363 include word2vec features

- 2018.06.10:
    - 0.281 is what I get with the random forest as in 
    the [benhamner benchmark](https://github.com/benhamner/ASAP-AES/blob/master/Benchmarks/length_benchmark.py)
    ... something's off
    