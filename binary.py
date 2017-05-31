'''
Binary classifier and corresponding datasets
'''

import codecs
import os
import numpy as np
import logging
import time

from tools.validation import InnerKFoldClassifier

class BinaryClassifierEval(object):
    def __init__(self, pos, neg, seed=1111):
        self.seed = seed
        self.samples, self.labels = pos + neg, [1] * len(pos) + [0] * len(neg)
        self.n_samples = len(self.samples)

    def do_prepare(self, params, prepare):
        # prepare is given the whole text
        return prepare(params, self.samples)
        # prepare puts everything it outputs in "params" : params.word2id, params.lut etc
        # Those output will be further used by "batcher".

    def loadFile(self, fpath):
        with codecs.open(fpath, 'rb', 'latin-1') as f:
            return [line.encode('utf-8').split() for line in f.read().splitlines()]

    def run(self, batcher, params):
        enc_input = []
        # Sort to reduce padding
        sorted_corpus = sorted(zip(self.samples, self.labels), key=lambda z:(len(z[0]), z[1]))
        sorted_samples = [x for (x,y) in sorted_corpus]
        sorted_labels = [y for (x,y) in sorted_corpus]
        logging.info('Generating sentence embeddings')
        for ii in range(0, self.n_samples, params.batch_size):
            batch = sorted_samples[ii:ii + params.batch_size]
            embeddings = batcher(batch, params)
            enc_input.append(embeddings)
        enc_input = np.vstack(enc_input)
        logging.info('Generated sentence embeddings')

        config_classifier = {'nclasses':2, 'seed':self.seed, 'usepytorch':params.usepytorch}
        clf = InnerKFoldClassifier(enc_input, np.array(sorted_labels), config_classifier)
        devacc, testacc = clf.run()
        logging.debug('Dev acc : {0} Test acc : {1}\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc, 'ndev': self.n_samples, 'ntest': self.n_samples}

class CREval(BinaryClassifierEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : CR *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'custrev.pos'))
        neg = self.loadFile(os.path.join(task_path, 'custrev.neg'))
        super(self.__class__, self).__init__(pos, neg, seed)

class MREval(BinaryClassifierEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : MR *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'rt-polarity.pos'))
        neg = self.loadFile(os.path.join(task_path, 'rt-polarity.neg'))
        super(self.__class__, self).__init__(pos, neg, seed)

class SUBJEval(BinaryClassifierEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SUBJ *****\n\n')
        obj = self.loadFile(os.path.join(task_path, 'subj.objective'))
        subj = self.loadFile(os.path.join(task_path, 'subj.subjective'))
        super(self.__class__, self).__init__(obj, subj, seed)

class MPQAEval(BinaryClassifierEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : MPQA *****\n\n')
        pos = self.loadFile(os.path.join(task_path, 'mpqa.pos'))
        neg = self.loadFile(os.path.join(task_path, 'mpqa.neg'))
        super(self.__class__, self).__init__(pos, neg, seed)
