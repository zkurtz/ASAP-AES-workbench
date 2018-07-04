import spacy

import when_will_it_end as wwie

ENTITIES = set(["PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT", "CAPS"])


class Tokenizer(object):
    def __init__(self):
        self.nlp = spacy.load('en', disable = ['tagger', 'parser', 'ner'])
        self.ents_lookup = {e[:4]: e for e in ENTITIES}

    def entity_processor(self, w):
        ''' Return any @<entity[K]> substitutions without the K '''
        pref = w[1:5]
        if pref in self.ents_lookup:
            return '@' + self.ents_lookup[pref]
        return w

    def word_processor(self, w):
        w = w.text
        if w[0] == '@':
            return self.entity_processor(w)
        return w.lower()

    def tokenize(self, string):
        return [self.word_processor(w) for w in self.nlp(string)]

    def apply_tokenize(self, list_of_strings):
        ns = len(list_of_strings)
        lpm = wwie.LoopProgressMonitor(n = ns)
        res = [None]*ns
        for k in range(ns):
            lpm()
            res[k] = self.tokenize(list_of_strings[k])
        return res

