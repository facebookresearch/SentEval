from __future__ import absolute_import, division

import sys
import logging
from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder

# Need use tensorflow <= 1.13.2

logging.basicConfig(level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    # Load model
    params.model = ELMoEmbedder("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz")
    return


def batcher(params, batch):
    batch = [sent if sent != [] else [''] for sent in batch]
    embeddings = params.model(batch)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 5, 'batch_size': 128,
                   'classifier': {'nhid': 0, 'optim': 'rmsprop', 'tenacity': 3, 'epoch_size': 2}}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['TREC_RU', 'SICKEntailment_RU', 'SST3_RU', 'MRPC_RU', 'SST2_RU',
                      'STSBenchmark_RU', 'SICKRelatedness_RU'
                     ]
    results = se.eval(transfer_tasks)
    print(results)
