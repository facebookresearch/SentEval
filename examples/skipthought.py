# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

"""
Example of file to compare skipthought vectors with our InferSent model
"""

import sys
reload(sys)  
sys.setdefaultencoding('utf8')

import torch
from exutil import dotdict


# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data/'
PATH_TO_SKIPTHOUGHT = ''
assert PATH_TO_SKIPTHOUGHT != '', 'Download skipthought and 

# import skipthought and Senteval
sys.path.insert(0, PATH_TO_SKIPTHOUGHT)
sys.path.insert(0, PATH_TO_SENTEVAL)
import skipthoughts
import senteval


def batcher(batch, params):
    embeddings = skipthoughts.encode(params.encoder, [unicode(' '.join(sent), errors="ignore")\
                                     if sent!=[] else '.' for sent in batch],\
                                     verbose=False, use_eos=True)
    return embeddings

def prepare(params, samples):
    return


# Set params for SentEval
params_senteval = {'usepytorch':True,
                   'task_path':PATH_TO_DATA,
                   'batch_size':512}
params_senteval = dotdict(params_senteval)
torch.cuda.set_device(1)



if __name__ == "__main__":
    params_senteval.encoder = skipthoughts.load_model()
    se = senteval.SentEval(batcher, prepare, params_senteval)
    se.eval(['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14', 'ImageAnnotation'])

    
    
    
    
    
    
