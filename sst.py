# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. 
#

'''
SST - binary classification
'''

import os
import logging
import numpy as np

from tools.validation import SplitClassifier

class SSTBinaryEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SST Binary classification *****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'sentiment-train'))
        dev = self.loadFile(os.path.join(task_path, 'sentiment-dev'))
        test = self.loadFile(os.path.join(task_path, 'sentiment-test'))
        self.sst_data = {'train':train, 'dev':dev, 'test':test}
        
    def do_prepare(self, params, prepare):
        samples = self.sst_data['train']['X'] + self.sst_data['dev']['X'] + \
                  self.sst_data['test']['X']
        return prepare(params, samples)
        
    def loadFile(self, fpath):
        sst_data = {'X':[], 'y':[]}
        with open(fpath, 'rb') as f:
            for line in f:
                sample = line.strip().split('\t')
                sst_data['y'].append(int(sample[1]))
                sst_data['X'].append(sample[0].split())
        return sst_data
    

    def run(self, params, batcher):
        sst_embed = {'train':{}, 'dev':{}, 'test':{}}
                      
        for key in self.sst_data:  
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_data = sorted(zip(self.sst_data[key]['X'], self.sst_data[key]['y']), key=lambda z:(len(z[0]), z[1]))
            self.sst_data[key]['X'], self.sst_data[key]['y'] = map(list, zip(*sorted_data))
           
            
            sst_embed[key]['X'] = []
            for ii in range(0, len(self.sst_data[key]['y']), params.batch_size):
                batch = self.sst_data[key]['X'][ii:ii + params.batch_size]
                embeddings = batcher(params, batch)
                sst_embed[key]['X'].append(embeddings)
            sst_embed[key]['X'] = np.vstack(sst_embed[key]['X'])
            sst_embed[key]['y'] = np.array(self.sst_data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))

        config_classifier = {'nclasses':2, 'seed':self.seed, 'usepytorch':params.usepytorch,
                            'classifier':params.classifier, 'nhid': params.nhid}
        clf = SplitClassifier(X={'train':sst_embed['train']['X'], 'valid':sst_embed['dev']['X'], 'test':sst_embed['test']['X']},
                              y={'train':sst_embed['train']['y'], 'valid':sst_embed['dev']['y'], 'test':sst_embed['test']['y']},
                              config=config_classifier)
        
        devacc, testacc = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for SST Binary classification\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc, 'ndev': len(sst_embed['dev']['X']), 'ntest': len(sst_embed['test']['X'])}
