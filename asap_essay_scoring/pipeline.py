'''
A collection of wrappers for other asap_essay_scoring functions to support
operating each function using file(s) input and writing the output to file(s)

These functions make lots of assumptions about the format and contents of input files.
It could make sense to break off this submodule from the main module if asap_essay_scoring
develops into a more general tool (not specific to ASAP-AES)
'''
from gensim.models.word2vec import Word2Vec
import pandas as pd
import pdb

from . import tokens
from . import utils
from . import vocab

def tokenize(infile, outfile):
    df = pd.read_csv(infile, sep='\t', encoding="ISO-8859-1")
    tk = tokens.Tokenizer()
    doc_list = tk.apply_tokenize(df.essay)
    utils.json_save(doc_list, outfile)

def reduce_docs_to_smaller_vocab(infile, outfile, target_file = None):
    '''
    Simplify a list of tokenized documents by reducing the vocabulary size
    :param infile: List of tokenized docs, the basis for the reduced vocabulary. If `target_file` is None,
    then the simplified version of `infile` is what will get written to `outfile`
    :param outfile: Where to write output
    :param target_file: If specified, this is the collection of documents that will be reduce (instead of `infile`),
    but `infile` is still the basis for the vocab
    '''
    doc_list = utils.json_load(infile)
    vc = vocab.Vocab(vocab_size=3000)
    vc.build_from_tokenized_docs(doc_list)
    if target_file is not None:
        target_docs = utils.json_load(target_file)
        reduced_docs = vc.reduce_docs(target_docs)
    else:
        reduced_docs = vc.reduce_docs(doc_list)
    utils.json_save(reduced_docs, outfile)

def fit_word2vec(infile, outfile):
    reduced_docs = utils.json_load(infile)
    # Abusing tools here slightly: Word2Vec expects a list of sentences, but we're providing
    #   a list of documents instead, pretending that each document is a single sentence. The
    #   fact that we include punctuation as tokens in our tokenization may help to preserve
    #   the sentence structure that we're otherwise ignoring
    wv = Word2Vec(reduced_docs, size = 100, iter = 25)
    vocab = list(wv.wv.vocab.keys())
    df = pd.DataFrame([wv.wv.word_vec(w) for w in vocab], index=vocab)
    df.to_csv(outfile, index = True)

def essay_features_from_word_embeddings(reduced_docs_infile, embedding_infile, outfile):
    reduced_docs = utils.json_load(reduced_docs_infile)
    embedding = pd.read_csv(embedding_infile, index_col = 0)
    dft = vocab.DocFeaturizer(vocab_embedding=embedding)
    feats = dft.featurize_corpus(reduced_docs)
    feats.to_csv(outfile, index = False)