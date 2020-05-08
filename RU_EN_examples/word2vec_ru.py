from __future__ import absolute_import, division

import os
import sys
import logging
import numpy as np
from wikipedia2vec import Wikipedia2Vec

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'
# Download Russian model from https://wikipedia2vec.github.io/wikipedia2vec/pretrained/ to folder 'word2vec_ru'
PATH_TO_MODEL = os.path.join('word2vec_ru', 'ruwiki_20180420_300d.pkl')

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    # Load model
    if not os.path.exists(PATH_TO_MODEL):
        raise Exception("There are no pretrained model in \"" + PATH_TO_MODEL + "\"")

    params.model = Wikipedia2Vec.load(PATH_TO_MODEL)
    return


def get_sentence_embedding(sentence, params):
    embedding = np.zeros((300,), dtype=np.float32)
    for token in sentence:
        token = token.lower()
        if params.model.dictionary.get_word(token) is not None:
            embedding += params.model.get_word_vector(token)
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
    transfer_tasks = ['SICKEntailment_RU', 'SST2_RU', 'SST3_RU', 'TREC_RU', 'MRPC_RU'
                      'STSBenchmark_RU', 'SICKRelatedness_RU'
                      ]
    results = se.eval(transfer_tasks)
    print(results)
