# ASAP-AES: word2vec and more

This repo presents a solution for a 
[Kaggle-hosted competition](https://www.kaggle.com/c/asap-aes) 
on automatic essay scoring. 

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

## Organization

`featurize_data.py` and `performance_crossval.py` are the main model building and
evaluation scripts as explained below. The `notebooks` directory contains a couple of
exploratory analyses.

## Model preparation, building, testing

```bash
# Feature engineering on the training set
python featurize_data.py

# Cross-validate the model
python performance_crossval.py
```

## Performance log

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
    - 0.3153 restored default LightGBM 100 boosting rounds.
    - 0.363 include word2vec features

- 2018.06.10: 0.281 with the random forest as in [benhamner benchmark](https://github.com/benhamner/ASAP-AES/blob/master/Benchmarks/length_benchmark.py)
    whereas his leaderboard score is 0.64? Something's off! One possible explanation is that
    the testing set was systematically different -- and easier to score -- than
    the training set in the Kaggle competition. Note that https://github.com/zlliang/essaysense also
    report training-set QWK scores far below that benchmark.

- 2018.06.17: 0.381 simply by increasing the wordvec dimensionality to 100 (from 25)

- 2018.07.03: 0.387 after dropping doc2vec and adding word2vec means (in addition to percentiles).
The new "sequential euclidean differences" feature did not seem to help.


## Related work

The following are very similar projects to this one. I've ranked them very approximatley in 
descending order of sophistication:

- https://github.com/zlliang/essaysense 
- https://github.com/benhamner/ASAP-AES/blob/master/Benchmarks/length_benchmark.py