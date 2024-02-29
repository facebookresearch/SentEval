# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging

# Set PATHs
PATH_TO_SENTEVAL = ''
PATH_TO_DATA = 'data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

from transformers import AutoTokenizer, AutoModel

name = 'roberta-base'
device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).to(device)

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [" ".join(sent) if sent != [] else ['.'] for sent in batch]
    encoded_input = tokenizer(batch, return_tensors='pt', padding=True).to(device)
    output = model(**encoded_input).last_hidden_state
    mask = encoded_input['attention_mask'].unsqueeze(2)
    embeddings = (output * mask).sum(1) / mask.sum(1)
    return embeddings.detach()


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
                        'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                        'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
