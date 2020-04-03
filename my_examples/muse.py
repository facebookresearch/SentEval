from __future__ import absolute_import, division

import sys
import logging
import tensorflow_hub as hub

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    # Load model
    params.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    return


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = params.model(batch)
    return embeddings.numpy()


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 5, 'batch_size': 128,
                   'classifier': {'nhid': 0, 'optim': 'rmsprop', 'tenacity': 3, 'epoch_size': 2}}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SICKEntailment', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment_RU', 'SST2_RU', 'SST3_RU', 'TREC_RU', 'MRPC_RU'
                      'STSBenchmark', 'SICKRelatedness',
                      'STSBenchmark_RU', 'SICKRelatedness_RU'
                     ]
    results = se.eval(transfer_tasks)
    print(results)
