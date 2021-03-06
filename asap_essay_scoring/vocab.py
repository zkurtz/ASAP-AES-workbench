import numpy as np
import pandas as pd
import pdb

import when_will_it_end as wwie

class Vocab(object):
    # ## vocab of size 3000 is enough to account for nearly 95% of word-instances:
    # sum(counts[:3000])/sum(counts)
    def __init__(self, vocab_size = 3000, wildcard = 'infrequentista'):
        '''
        the tokenized version of a doc
        :param vocab_size: (int) Include only this many of the most-frequent words
        :param wildcard: (int) Use this to represent andy words that fail to meet
        the `vocab_size` threshold
        '''
        self.wildcard = wildcard
        self.vocab_size = vocab_size

    def build_from_tokenized_docs(self, doc_list):
        '''
        :param doc_list: (list of lists of strings) Each inner list is a list of strings
        representing a tokenized doc
        '''
        flat_tokens = [t for toks in doc_list for t in toks]
        self.vocab_table = pd.Series(flat_tokens).value_counts()
        self.vocab_lookup = pd.DataFrame({
            'token': self.vocab_table.index[:self.vocab_size].tolist(),
        })
        self.vocab_lookup['token_mirror'] = self.vocab_lookup.token

    def doc_2_vocab(self, tokenized_doc):
        doc = pd.DataFrame({'token': tokenized_doc})
        vdoc = doc.merge(self.vocab_lookup, on='token', how='left').fillna(self.wildcard)
        return vdoc.token_mirror.tolist() #[self.BREAK]*5 +

    def reduce_docs(self, doc_list, flatten = False):
        '''
        :param doc_list: (list of lists of strings) Each inner list is a list of strings
        representing a tokenized doc
        '''
        ndocs = len(doc_list)
        dv_list = [None]*ndocs
        lpm = wwie.LoopProgressMonitor(n = ndocs)
        for k in range(ndocs):
            lpm()
            dv_list[k] = self.doc_2_vocab(doc_list[k])

        if flatten:
            return [k for lst in dv_list for k in lst]
        return dv_list


class DocFeaturizer(object):
    '''Methods to derive numerical features for a tokenized document (i.e. a list of strings)'''
    def __init__(self, vocab_embedding):
        '''
        :param vocab_embedding: (pandas.DataFrame) Index is the vocab, each row is a
        numeric vector representing an embedding for a word
        '''
        self.ve = vocab_embedding

    def doc2embedding(self, doc):
        '''
        :param doc: List of strings
        :return: (pandas.DataFrame) doc is [non-unique] index and columns are the word
        vectors defined in self.ve
        '''
        return pd.DataFrame({'order': range(len(doc))}, index=doc
            ).join(self.ve, how='left').sort_values('order').drop('order', axis=1)

    def featurize_doc(self, doc):
        emb = self.doc2embedding(doc)
        N = emb.shape[1]
        mean = emb.mean()
        mean.index = ['wv_' + str(k) + '_mean' for k in range(N)]
        qts = emb.quantile(q=[0.05, 0.95])
        pct5 = qts.iloc[0]
        pct95 = qts.iloc[1]
        pct5.index = ['wv_' + str(k) + '_pct_5' for k in range(N)]
        pct95.index = ['wv_' + str(k) + '_pct_95' for k in range(N)]
        sequential_euclidean_dist = np.mean((emb[:-1].values - emb[1:].values)**2, axis = 1)
        sed = pd.Series({
            'seq_dist_mean': np.mean(sequential_euclidean_dist),
            'seq_dist_pct_5': np.percentile(sequential_euclidean_dist, 5),
            'seq_dist_pct_95': np.percentile(sequential_euclidean_dist, 95)
        })
        return pd.concat([pct5, pct95, mean, sed])

    def featurize_corpus(self, doc_list):
        n_docs = len(doc_list)
        lpm = wwie.LoopProgressMonitor(n = n_docs)
        def fd(doc):
            lpm()
            return self.featurize_doc(doc)
        df = pd.concat([fd(doc) for doc in doc_list], axis=1).transpose()
        return df