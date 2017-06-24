# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

'''

Generic sentence evaluation scripts wrapper

'''
from binary import CREval, MREval, MPQAEval, SUBJEval
from snli import SNLIEval
from trec import TRECEval
from sick import SICKRelatednessEval, SICKEntailmentEval
from mrpc import MRPCEval
from sts import STS14Eval, STSBenchmarkEval
from sst import SSTBinaryEval
from rank import ImageAnnotationEval


class SentEval(object):
    def __init__(self, batcher, prepare, params):
        # setting default parameters
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.classifier = params.classifier or 'LogReg'
        params.nhid = params.nhid or 0
        params.batch_size = params.batch_size or 128
        params.seed = params.seed or 1111
        self.params = params
                
        self.batcher = batcher
        self.prepare = prepare
        
        # sanity check
        assert params.classifier in ['LogReg', 'MLP']
        if params.classifier == 'MLP':
            assert params.nhid>0, 'When using an MLP, you need to set params.nhid>0'
        if not params.usepytorch and params.classifier == 'MLP':
            assert False, 'No MLP implemented in scikit-learn'        
        
        self.list_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST', 'TREC', 'MRPC', 'SICKRelatedness',\
                      'SICKEntailment', 'STSBenchmark', 'STS14', 'SNLI', 'ImageAnnotation']        
        

    def eval(self, name):
        ''' evaluate on evaluation [name], either takes string or list of strings '''
        if (isinstance(name, list)):
            self.results = {x:self.eval(x) for x in name}
            return self.results

        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)
        if name == 'CR':
            self.evaluation = CREval(self.params.task_path + '/CR', seed=self.params.seed)
        elif name == 'MR':
            self.evaluation = MREval(self.params.task_path + '/MR', seed=self.params.seed)
        elif name == 'MPQA':
            self.evaluation = MPQAEval(self.params.task_path + '/MPQA', seed=self.params.seed)
        elif name == 'SUBJ':
            self.evaluation = SUBJEval(self.params.task_path + '/SUBJ', seed=self.params.seed)
        elif name == 'SST':
            self.evaluation = SSTBinaryEval(self.params.task_path + '/SST/binary', seed=self.params.seed)
        elif name == 'TREC':
            self.evaluation = TRECEval(self.params.task_path + '/TREC', seed=self.params.seed)
        elif name == 'MRPC':
            self.evaluation = MRPCEval(self.params.task_path + '/MRPC', seed=self.params.seed)
        elif name == 'SICKRelatedness':
            self.evaluation = SICKRelatednessEval(self.params.task_path + '/SICK', seed=self.params.seed)
        elif name == 'STSBenchmark':
            self.evaluation = STSBenchmarkEval(self.params.task_path + '/STS/STSBenchmark', seed=self.params.seed)
        elif name == 'SICKEntailment':
            self.evaluation = SICKEntailmentEval(self.params.task_path + '/SICK', seed=self.params.seed)
        elif name == 'SNLI':
            self.evaluation = SNLIEval(self.params.task_path + '/SNLI', seed=self.params.seed)
        elif name == 'STS14':
            self.evaluation = STS14Eval(self.params.task_path + '/STS/STS14', seed=self.params.seed)
        elif name == 'ImageAnnotation':
            self.evaluation = ImageAnnotationEval(self.params.task_path + '/COCO', seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)
        
        self.results = self.evaluation.run(self.batcher, self.params)
        
        return self.results
    
    def microavg(self, tasks, evaltype='dev'):
        assert evaltype in ['dev', 'test'], "eval type must be in ['dev', 'test']"
        metrics = ['acc', 'ntest'] if evaltype=='test' else ['devacc', 'ndev']
        micro, nsamples = 0, 0
        
        for task in tasks:
            micro += self.results[task][metrics[0]] * self.results[task][metrics[1]]
            nsamples += self.results[task][metrics[1]]
        micro /= nsamples
        
        return micro
    
    def macroavg(self, tasks, evaltype='dev'):
        assert evaltype in ['dev', 'test'], "eval type must be in ['dev', 'test']"
        metrics = ['acc'] if evaltype=='test' else ['devacc']
        macro = 0
        
        for task in tasks:
            macro += self.results[task][metrics[0]]
        macro /= len(tasks)
        
        return macro
    
    
    
    
