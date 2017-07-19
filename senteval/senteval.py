# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

from binary import CREval, MREval, MPQAEval, SUBJEval
from snli import SNLIEval
from trec import TRECEval
from sick import SICKRelatednessEval, SICKEntailmentEval
from mrpc import MRPCEval
from sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, \
    STSBenchmarkEval
from sst import SSTBinaryEval
from rank import ImageCaptionRetrievalEval


class SentEval(object):
    def __init__(self, params, batcher, prepare=None):
        # setting default parameters
        params.usepytorch = True if 'usepytorch' not in params else \
            params.usepytorch
        params.classifier = 'LogReg' if 'classifier' not in params else \
            params.classifier
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.batch_size = 128 if 'batch_size' not in params else \
            params.batch_size
        params.seed = 1111 if 'seed' not in params else params.seed
        params.kfold = 5 if 'kfold' not in params else params.kfold
        self.params = params

        self.batcher = batcher
        if prepare:
            self.prepare = prepare
        else:
            self.prepare = lambda x, y: None

        # sanity check
        assert params.classifier in ['LogReg', 'MLP']
        if params.classifier == 'MLP':
            assert params.nhid > 0, 'When using an MLP, \
                you need to set params.nhid>0'
        if not params.usepytorch and params.classifier == 'MLP':
            assert False, 'No MLP implemented in scikit-learn'

        self.list_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST', 'TREC', 'MRPC',
                           'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
                           'SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13',
                           'STS14', 'STS15', 'STS16']

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)
        if name == 'CR':
            self.evaluation = CREval(tpath + '/CR', seed=self.params.seed)
        elif name == 'MR':
            self.evaluation = MREval(tpath + '/MR', seed=self.params.seed)
        elif name == 'MPQA':
            self.evaluation = MPQAEval(tpath + '/MPQA', seed=self.params.seed)
        elif name == 'SUBJ':
            self.evaluation = SUBJEval(tpath + '/SUBJ', seed=self.params.seed)
        elif name == 'SST':
            self.evaluation = SSTBinaryEval(tpath + '/SST/binary',
                                            seed=self.params.seed)
        elif name == 'TREC':
            self.evaluation = TRECEval(tpath + '/TREC', seed=self.params.seed)
        elif name == 'MRPC':
            self.evaluation = MRPCEval(tpath + '/MRPC', seed=self.params.seed)
        elif name == 'SICKRelatedness':
            self.evaluation = SICKRelatednessEval(tpath + '/SICK',
                                                  seed=self.params.seed)
        elif name == 'STSBenchmark':
            self.evaluation = STSBenchmarkEval(tpath + '/STS/STSBenchmark',
                                               seed=self.params.seed)
        elif name == 'SICKEntailment':
            self.evaluation = SICKEntailmentEval(tpath + '/SICK',
                                                 seed=self.params.seed)
        elif name == 'SNLI':
            self.evaluation = SNLIEval(tpath + '/SNLI', seed=self.params.seed)
        elif name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            fpath = name + '-en-test'
            self.evaluation = eval(name + 'Eval')(tpath + '/STS/' + fpath,
                                                  seed=self.params.seed)
        elif name == 'ImageCaptionRetrieval':
            self.evaluation = ImageCaptionRetrievalEval(tpath + '/COCO',
                                                        seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
