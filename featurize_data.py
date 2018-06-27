'''
Generate features on the training data

Workflow note: For development purposes, to avoid having to re-run the pipeline
from the beginning after every edit, data gets saved and reloaded between every
block of code. This allows for continuing from the point of modification by reloading
the saved versio of the previous step. This is a silly hack that will soon be replaced
by some kind of file-dependency-graph-aware tool like dagger or dvc
'''

import os
import pdb

from asap_essay_scoring import pipeline as pl
from asap_essay_scoring import utils

def efpath(filename):
    ''' Generate the full path to `filename` in the engineered_features directory '''
    return utils.data_path(os.path.join('engineered_features', filename))

print('Tokenize essays')
pl.tokenize(infile=utils.data_path('training_set_rel3.tsv'),
            outfile=efpath('tokenized.json'))

print('Translate the docs to a limited vocabulary, using only the most common tokens' +
      ' and replacing all others with "infrequentista"')
pl.reduce_docs_to_smaller_vocab(
    infile=efpath('tokenized.json'),
    outfile=efpath('tokenized_reduced.json')
)

print('Fit a 20-d word2vec model on the training data')
pl.fit_word2vec(
    infile=efpath('tokenized_reduced.json'),
    outfile=efpath("vocab_embedding.csv")
)

print('Fit a doc2vec model on the training data')
pl.fit_doc2vec(
    infile=efpath('tokenized_reduced.json'),
    outfile=efpath("doc_embedding.csv")
)

print('Generate word2vec document-level features for all documents')
pl.essay_features_from_embeddings(
    reduced_docs_infile=efpath('tokenized_reduced.json'),
    word2vec_infile=efpath("vocab_embedding.csv"),
    doc2vec_infile = efpath("doc_embedding.csv"),
    outfile=efpath("txt_features.csv")
)

