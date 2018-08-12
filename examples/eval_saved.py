# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
A script to run SentEval on pre-computed embeddings from a file.
"""

from __future__ import absolute_import, division, unicode_literals

import argparse
import json
import logging
import os
import sys

import numpy as np

# Set PATHs
PATH_TO_SENTEVAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL, 'data')

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def main():
    logging.basicConfig(format='%(asctime)s : %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('sentences',
                        help='a text file containing all SentEval sentences')
    parser.add_argument('embeddings',
                        help='a NumPy binary file containing the corresponding embeddings')
    parser.add_argument('-t', '--tasks',
                        help='a comma-separated list of tasks')
    parser.add_argument('--no-gpu', action='store_true',
                        help='do not use GPU (turn off PyTorch)')
    args = parser.parse_args()

    sent2emb = {}

    def join_sentence(sent):
        if sys.version_info < (3, 0):
            sent = [w.decode('utf-8') if isinstance(w, str) else w for w in sent]
        else:
            sent = [w.decode('utf-8') if isinstance(w, bytes) else w for w in sent]
        return ' '.join(sent)

    def prepare(params, samples):
        # Build the mapping from sentences to embeddings
        sent2emb.clear()
        samples_set = set(join_sentence(sent) for sent in samples)
        all_embeddings = np.load(args.embeddings, mmap_mode='r')
        with open(args.sentences) as f_sent:
            for i, sent in enumerate(f_sent):
                if sys.version_info < (3, 0):
                    sent = sent.decode('utf-8')
                sent = sent.rstrip('\n')
                if sent in samples_set:
                    sent2emb[sent] = all_embeddings[i]

    def batcher(params, batch):
        embeddings = np.stack(
            [sent2emb[join_sentence(sent)] for sent in batch])
        if len(embeddings.shape) != 2:
            embeddings = embeddings.reshape(len(embeddings), -1)
        assert len(embeddings.shape) == 2
        return embeddings

    params_senteval = {
        'task_path': PATH_TO_DATA, 'usepytorch': not args.no_gpu, 'kfold': 10
    }
    params_senteval['classifier'] = {
        'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5,
        'epoch_size': 4
    }

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    if args.tasks is not None:
        transfer_tasks = args.tasks.split(',')
    else:
        transfer_tasks = se.list_tasks

    results = se.eval(transfer_tasks)
    json.dump(results, sys.stdout, skipkeys=True)


if __name__ == '__main__':
    main()
