from __future__ import absolute_import, division

import os
import sys
import logging
import gensim
import numpy as np

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'
# Download model from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit to folder 'word2vec'
PATH_TO_MODEL = os.path.join('word2vec', 'GoogleNews-vectors-negative300.bin')

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    # Load model
    if not os.path.exists(PATH_TO_MODEL):
        raise Exception("There are no pretrained model in \"" + PATH_TO_MODEL + "\"")

    params.model = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_MODEL, binary=True)
    return


def get_sentence_embedding(sentence, params):
    embedding = np.zeros((300,), dtype=np.float32)
    for token in sentence:
        token = token.lower()
        if token in params.model.wv.vocab.keys():
            embedding += params.model.wv[token]
    embedding = embedding / len(sentence)
    return embedding


def batcher(params, batch):
    batch = [sent if sent != [] else [''] for sent in batch]
    embeddings = [get_sentence_embedding(sentence, params) for sentence in batch]
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 5, 'batch_size': 128,
                   'classifier': {'nhid': 0, 'optim': 'rmsprop', 'tenacity': 3, 'epoch_size': 2}}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SICKEntailment', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'STSBenchmark', 'SICKRelatedness'
                      ]
    results = se.eval(transfer_tasks)
    print(results)
