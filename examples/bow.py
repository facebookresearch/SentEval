# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

import sys
import numpy as np

import torch
from torch.autograd import Variable

from exutil import dotdict
import data

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'
PATH_TO_GLOVE = 'glove/glove.840B.300d.txt'
                
# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


"""
Note : 

The user has to implement two functions :
    1) "batcher" : transforms a batch of sentences into sentence embeddings.
        i) takes as input a "batch", and "params".
        ii) outputs a numpy array of sentence embeddings
        iii) Your sentence encoder should be in "params" 
    2) "prepare" : sees the whole dataset, and can create a vocabulary
        i) outputs of "prepare" are stored in "params" that batcher will use.
"""


def batcher(batch, params):
    batch = [sent if sent!=[] else ['.'] for sent in batch]
    embeddings = []
    
    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            sentvec.append(params.word_vec['.'])
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings
            

def prepare(params, samples):
    _, params.word2id = data.create_dictionary(samples)
    params.word_vec = data.get_wordvec(PATH_TO_GLOVE, params.word2id)
    return


# Set params for SentEval
params_senteval = {'seed': 1111, 'task_path': PATH_TO_DATA, 'usepytorch': False, 'batch_size': 64,'verbose': 2, 'classifier':'LogReg'}
params_senteval = dotdict(params_senteval)

# set pytorch cuda device
torch.cuda.set_device(1)

if __name__ == "__main__":
    params_senteval.model = None # No model here, just for illustration purposes
    se = senteval.SentEval(batcher, prepare, params_senteval)#'MR', 'CR', 'SUBJ','MPQA', 
    transfer_tasks = ['MR', 'CR', 'SUBJ','MPQA', 'SST', 'TREC', 'SICKRelatedness',\
                      'SICKEntailment', 'MRPC', 'STS14', 'ImageAnnotation']
    results = se.eval(transfer_tasks)

    
    
    
    
    
    
