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
        qts = emb.quantile(q=[0.05, 0.95])
        qvec = pd.concat([qts.iloc[0], qts.iloc[1]])
        return qvec.tolist() + [len(doc)]

    def featurize_corpus(self, doc_list):
        n_docs = len(doc_list)
        fl = [None]*n_docs
        lpm = wwie.LoopProgressMonitor(n = n_docs)
        for k in range(n_docs):
            lpm()
            fl[k] = self.featurize_doc(doc_list[k])
        return pd.DataFrame(fl)