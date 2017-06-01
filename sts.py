# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

'''
STS-2014 (unsupervised) and STS-benchmark (supervised) tasks
'''

import os
import numpy as np
import logging

from scipy.stats import spearmanr, pearsonr

from utils import cosine
from sick import SICKRelatednessEval


class STS14Eval(object):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.data = {}
        self.samples = []
        self.datasets = ['deft-forum', 'deft-news', 'headlines', 'images', 'OnWN', 'tweet-news']
        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in open(taskpath + '/STS.input.%s.txt' % dataset).read().splitlines()])
            gs_scores = [float(x) for x in open(taskpath + '/STS.gs.%s.txt' % dataset).read().splitlines()]
            sent1 = [s.split() for s in sent1]
            sent2 = [s.split() for s in sent2]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores), key=lambda z:(len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))
            
            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2
        

    def do_prepare(self, params, prepare):
        return prepare(params, self.samples)

    def run(self, batcher, params):
        results = {}
        for dataset in self.datasets:
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]
                
                # we assume that the get_batch function already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(batch1, params)
                    enc2 = batcher(batch2, params)

                    for kk in range(enc2.shape[0]):
                        sys_score = cosine(np.nan_to_num(enc1[kk]), np.nan_to_num(enc2[kk]))
                        sys_scores.append(sys_score)

            results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores), 'spearman': spearmanr(sys_scores, gs_scores)}
            logging.debug('{0} : pearson = {1}, spearman = {2}'.format(dataset, results[dataset]['pearson'], results[dataset]['spearman']))
        avg_pearson = np.mean([results[dset]['pearson'][0] for dset in results.keys()])
        avg_spearman = np.mean([results[dset]['spearman'][0] for dset in results.keys()])
        results['all'] = {'pearson': avg_pearson, 'spearman': avg_spearman}
        logging.debug('Results (all) : Pearson = {0}, Spearman = {1}\n'.format(results['all']['pearson'], results['all']['spearman']))

        return results

    
class STSBenchmarkEval(SICKRelatednessEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.sick_data = {'train':train, 'dev':dev, 'test':test}
        
    def loadFile(self, fpath):
        sick_data = {'X_A':[], 'X_B':[], 'y':[]}
        with open(fpath, 'rb') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])
        
        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data