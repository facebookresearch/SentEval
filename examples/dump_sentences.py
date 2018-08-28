# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
A script to dump all sentences (tokenized) to standard output.
"""

from __future__ import absolute_import, division, unicode_literals

import argparse
import logging
import os
import sys

# Set PATHs
PATH_TO_SENTEVAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL, 'data')

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def main():
    logging.basicConfig(format='%(asctime)s : %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--tasks",
                        help="a comma-separated list of tasks")
    args = parser.parse_args()

    def prepare(params, samples):
        for sent in samples:
            if sys.version_info < (3, 0):
                sent = [w.decode('utf-8') if isinstance(w, str) else w for w in sent]
                print(' '.join(sent).encode('utf-8'))
            else:
                sent = [w.decode('utf-8') if isinstance(w, bytes) else w for w in sent]
                print(' '.join(sent))

    def batcher(params, batch):
        # Block evaluation and continue with the next task.
        raise Done

    params_senteval = {
        'task_path': PATH_TO_DATA
    }

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    if args.tasks is not None:
        transfer_tasks = args.tasks.split(',')
    else:
        transfer_tasks = se.list_tasks

    for task in transfer_tasks:
        try:
            se.eval([task])
            raise RuntimeError(task + " not completed")
        except Done:
            pass


class Done(Exception):
    pass


if __name__ == "__main__":
    main()
