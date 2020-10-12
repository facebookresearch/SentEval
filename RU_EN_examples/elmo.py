from __future__ import absolute_import, division

import sys
import logging
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    # Load model
    params.model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=False)
    return


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = params.model(batch, signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        embeddings_array = sess.run(tf.reduce_mean(embeddings, 1))
    return embeddings_array


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
