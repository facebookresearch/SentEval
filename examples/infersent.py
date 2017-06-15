# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

import sys, os
import torch
from exutil import dotdict

# Set PATHs
GLOVE_PATH = 'glove/glove.840B.300d.txt'
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'
MODEL_PATH = 'infersent.pickle'


assert os.path.isfile(MODEL_PATH), 'download infersent.pickle'
# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval    

# set gpu device
torch.cuda.set_device(1)



def batcher(batch, params):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)  
    

    
"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define transfer tasks
transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness',\
                  'SICKEntailment', 'MRPC', 'STS14']

# define senteval params
params_senteval = dotdict({'usepytorch': True,
                           'task_path': PATH_TO_DATA,
                           })

if __name__ == "__main__":
    # Load model
    params_senteval.infersent = torch.load(MODEL_PATH)
    params_senteval.infersent.set_glove_path(GLOVE_PATH)

    se = senteval.SentEval(batcher, prepare, params_senteval)
    results_transfer = se.eval(transfer_tasks)

    print results_transfer

