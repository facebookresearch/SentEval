# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

"""
Example of file to compare skipthought vectors with our InferSent model
"""
import logging

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import torch
from exutil import dotdict


# Set PATHs
PATH_TO_SENTEVAL = '../senteval'
PATH_TO_DATA = '../data/senteval_data/'
PATH_TO_SKIPTHOUGHT = ''
assert PATH_TO_SKIPTHOUGHT != '', 'Download skipthought and set correct PATH'

# import skipthought and Senteval
sys.path.insert(0, PATH_TO_SKIPTHOUGHT)
sys.path.insert(0, PATH_TO_SENTEVAL)
import skipthoughts
import senteval


def prepare(params, samples):
    return

def batcher(params, batch):
    embeddings = skipthoughts.encode(params.encoder, [unicode(' '.join(sent), errors="ignore")\
                                     if sent!=[] else '.' for sent in batch],\
                                     verbose=False, use_eos=True)
    return embeddings


# Set params for SentEval
params_senteval = {'usepytorch':True,
                   'task_path':PATH_TO_DATA,
                   'batch_size':512}
params_senteval = dotdict(params_senteval)

# set gpu device
torch.cuda.set_device(1)

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    params_senteval.encoder = skipthoughts.load_model()
    se = senteval.SentEval(params_senteval, batcher, prepare)
    se.eval(['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14', 'ImageAnnotation'])







