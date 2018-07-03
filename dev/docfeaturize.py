'''
Generate features on the training data

Workflow note: For development purposes, to avoid having to re-run the pipeline
from the beginning after every edit, data gets saved and reloaded between every
block of code. This allows for continuing from the point of modification by reloading
the saved version of the previous step. This is a silly hack that will soon be replaced
by some kind of file-dependency-graph-aware tool like dagger or dvc
'''

import os
import pdb

from asap_essay_scoring import pipeline as pl
from asap_essay_scoring import utils

def efpath(filename):
    ''' Generate the full path to `filename` in the engineered_features directory '''
    return utils.data_path(os.path.join('engineered_features', filename))

print('Generate document-level features from word2vec embeddings')
pl.essay_features_from_word2vec(
    word2vec_infile=efpath("vocab_embedding.csv"),
    reduced_docs_infile=efpath("tokenized_reduced.json"),
    outfile=efpath("wordvec_features.csv")
)
