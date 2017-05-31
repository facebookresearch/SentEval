'''
TREC question-type classification
'''

import codecs
import os
import logging
import numpy as np

from tools.validation import KFoldClassifier


class TRECEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : TREC *****\n\n')
        self.seed = seed
        self.train = self.loadFile(os.path.join(task_path, 'train_5500.label'))
        self.test = self.loadFile(os.path.join(task_path, 'TREC_10.label'))
        
    def do_prepare(self, params, prepare):
        samples = self.train['X'] + self.test['X']
        return prepare(params, samples)
        
    def loadFile(self, fpath):
        trec_data = {'X':[], 'y':[]}
        tgt2idx = {'ABBR':0, 'DESC':1, 'ENTY':2, 'HUM':3, 'LOC':4, 'NUM':5}
        with open(fpath, 'rb') as f:
            for line in f:
                target, sample = line.strip().split(':', 1)
                sample = sample.split(' ', 1)[1].split()
                assert target in tgt2idx, target
                trec_data['X'].append(sample)
                trec_data['y'].append(tgt2idx[target])
        return trec_data


    def run(self, batcher, params):
        train_embeddings, test_embeddings = [], []
        
        # Sort to reduce padding
        sorted_corpus_train = sorted(zip(self.train['X'], self.train['y']), key=lambda z:(len(z[0]), z[1]))
        train_samples = [x for (x,y) in sorted_corpus_train]
        train_labels = [y for (x,y) in sorted_corpus_train]
        
        sorted_corpus_test = sorted(zip(self.test['X'], self.test['y']), key=lambda z:(len(z[0]), z[1]))
        test_samples = [x for (x,y) in sorted_corpus_test]
        test_labels = [y for (x,y) in sorted_corpus_test]
        
        # Get train embeddings
        for ii in range(0, len(train_labels), params.batch_size):
            batch = train_samples[ii:ii + params.batch_size]
            embeddings = batcher(batch, params)
            train_embeddings.append(embeddings)
        train_embeddings = np.vstack(train_embeddings)
        logging.info('Computed train embeddings')

        # Get test embeddings
        for ii in range(0, len(test_labels), params.batch_size):
            batch = test_samples[ii:ii + params.batch_size]
            embeddings = batcher(batch, params)
            test_embeddings.append(embeddings)
        test_embeddings = np.vstack(test_embeddings)
        logging.info('Computed test embeddings')
        
        config_classifier = {'nclasses':6, 'seed':self.seed, 'usepytorch':params.usepytorch}
        clf = KFoldClassifier({'X':train_embeddings, 'y':np.array(train_labels)},
                              {'X':test_embeddings, 'y':np.array(test_labels)},
                              config_classifier)
        devacc, testacc, _ = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for TREC\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc, 'ndev': len(self.train['X']), 'ntest': len(self.test['X'])}



