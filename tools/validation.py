# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

"""
Validation and classification
(train)            :  inner-kfold classifier
(train, test)      :  kfold classifier
(train, dev, test) :  split classifier

"""

import logging

import numpy as np

import sklearn
assert(sklearn.__version__>="0.18.0"), "need to update sklearn to version >= 0.18.0"
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from classifier import LogReg, MLP


# Pytorch version
class InnerKFoldClassifier(object):
    """
    (train) split classifier : InnerKfold.
    """
    def __init__(self, X, y, config):
        self.X = X
        self.y = y
        self.featdim = X.shape[1]
        self.nclasses = config['nclasses']
        self.seed = config['seed']
        self.devresults = []
        self.testresults = []
        self.usepytorch = config['usepytorch']
        self.classifier = config['classifier']
        self.nhid = config['nhid']
        self.modelname = 'sklearn-LogReg' if not config['usepytorch'] else 'pytorch-' + config['classifier']

        self.k = 10
        
    def run(self):
        logging.info('Training {0} with (inner) {1}-fold cross-validation'.format(self.modelname, self.k))

        regs = [10**t for t in range(-5,-1)] if self.usepytorch else [2**t for t in range(-2,4,1)]
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=1111)
        innerskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
        count = 0
        for train_idx, test_idx in skf.split(self.X, self.y):
            count += 1
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            scores = []
            for reg in regs:
                regscores = []
                for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
                    X_in_train, X_in_test = X_train[inner_train_idx], X_train[inner_test_idx]
                    y_in_train, y_in_test = y_train[inner_train_idx], y_train[inner_test_idx]
                    if self.usepytorch:
                        if self.classifier == 'LogReg':
                            clf = LogReg(inputdim=self.featdim, nclasses=self.nclasses, l2reg=reg, seed=self.seed)
                        elif self.classifier == 'MLP':
                            clf = MLP(inputdim=self.featdim, hiddendim=self.nhid, nclasses=self.nclasses, l2reg=reg, seed=self.seed)
                        clf.fit(X_in_train, y_in_train, validation_data=(X_in_test, y_in_test))
                    else:
                        clf = LogisticRegression(C=reg, random_state=self.seed)
                        clf.fit(X_in_train, y_in_train)
                    regscores.append(clf.score(X_in_test, y_in_test))
                scores.append(round(100*np.mean(regscores),2))
            optreg = regs[np.argmax(scores)]
            logging.info('Best param found at split {0}: l2reg = {1} with score {2}'.format(count, optreg, np.max(scores)))
            self.devresults.append(np.max(scores))

            if self.usepytorch:
                if self.classifier == 'LogReg':
                    clf = LogReg(inputdim=self.featdim, nclasses=self.nclasses, l2reg=optreg, seed=self.seed)
                elif self.classifier == 'MLP':
                    clf = MLP(inputdim=self.featdim, hiddendim=self.nhid, nclasses=self.nclasses, l2reg=optreg, seed=self.seed)
                devacc = clf.fit(X_train, y_train, validation_split=0.05)
            else:
                clf = LogisticRegression(C=optreg, random_state=self.seed)
                clf.fit(X_train, y_train)
                
            self.testresults.append(round(100*clf.score(X_test, y_test),2))
            
        devaccuracy = round(np.mean(self.devresults), 2) # TODO
        testaccuracy = round(np.mean(self.testresults), 2)
        return devaccuracy, testaccuracy


class KFoldClassifier(object):
    """
    (train, test) split classifier : cross-validation on train.
    """
    def __init__(self, train, test, config):
        self.train = train
        self.test = test
        self.featdim = self.train['X'].shape[1]
        self.nclasses = config['nclasses']
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier = config['classifier']
        self.nhid = config['nhid']
        self.modelname = 'sklearn-LogReg' if not config['usepytorch'] else 'pytorch-' + config['classifier']

        self.k = 10

    def run(self):
        # cross-validation
        logging.info('Training {0} with {1}-fold cross-validation'.format(self.modelname, self.k))
        regs = [10**t for t in range(-5,-1)] if self.usepytorch else [2**t for t in range(-1,6,1)]
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        scores = []

        for reg in regs:
            scanscores = []
            for train_idx, test_idx in skf.split(self.train['X'], self.train['y']):
                # Split data
                X_train, y_train = self.train['X'][train_idx], self.train['y'][train_idx]
                X_test, y_test = self.train['X'][test_idx], self.train['y'][test_idx]

                # Train classifier
                if self.usepytorch:
                    if self.classifier == 'LogReg':
                        clf = LogReg(inputdim=self.featdim, nclasses=self.nclasses, l2reg=reg, seed=self.seed)
                    elif self.classifier == 'MLP':
                        clf = MLP(inputdim=self.featdim, hiddendim=self.nhid, nclasses=self.nclasses, l2reg=reg, seed=self.seed)
                    clf.fit(X_train, y_train, validation_data=(X_test, y_test))
                else:
                    clf = LogisticRegression(C=reg, random_state=self.seed)
                    clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                scanscores.append(score)
            # Append mean score
            scores.append(round(100*np.mean(scanscores),2))
        
        # evaluation
        logging.info([('reg:'+str(regs[idx]), scores[idx]) for idx in range(len(scores))])
        optreg = regs[np.argmax(scores)]
        devaccuracy = np.max(scores)
        logging.info('Cross-validation : best param found is reg = {0} with score {1}'.format(optreg, devaccuracy))
        
        logging.info('Evaluating...')
        if self.usepytorch:
            if self.classifier == 'LogReg':
                clf = LogReg(inputdim = self.featdim, nclasses=self.nclasses, l2reg=optreg, seed=self.seed)
            elif self.classifier == 'MLP':
                clf = MLP(inputdim = self.featdim, hiddendim=self.nhid, nclasses=self.nclasses, l2reg=optreg, seed=self.seed)
            devacc = clf.fit(self.train['X'], self.train['y'], validation_split=0.05)
        else:
            clf = LogisticRegression(C=optreg, random_state=self.seed)
            clf.fit(self.train['X'], self.train['y'])
        yhat = clf.predict(self.test['X'])
        
        testaccuracy = clf.score(self.test['X'], self.test['y'])
        testaccuracy = round(100*testaccuracy, 2)
        
        return devaccuracy, testaccuracy, yhat
        
    
class SplitClassifier(object):
    """
    (train, valid, test) split classifier.
    """
    def __init__(self, X, y, config):
        self.X = X
        self.y = y
        self.nclasses = config['nclasses']
        self.featdim = self.X['train'].shape[1]
        self.seed = config['seed']
        self.usepytorch = config['usepytorch']
        self.classifier = config['classifier']
        self.nhid = config['nhid']
        self.cudaEfficient = False if 'cudaEfficient' not in config else config['cudaEfficient']
        self.modelname = 'sklearn-LogReg' if not config['usepytorch'] else 'pytorch-' + config['classifier']
        self.nepoches = None if 'nepoches' not in config else config['nepoches']
        self.maxepoch = None if 'maxepoch' not in config else config['maxepoch']
        
    def run(self):
        logging.info('Training {0} with standard validation..'.format(self.modelname))
        regs = [10**t for t in range(-5,-1)] if self.usepytorch else [2**t for t in range(-2,4,1)]
        scores = []
        for reg in regs:
            if self.usepytorch:
                if self.classifier == 'LogReg':
                    clf = LogReg(inputdim=self.featdim, nclasses=self.nclasses, l2reg=reg,\
                                 seed=self.seed, cudaEfficient=self.cudaEfficient)
                elif self.classifier == 'MLP':
                    clf = MLP(inputdim=self.featdim, hiddendim=self.nhid, nclasses=self.nclasses,\
                              l2reg=reg, seed=self.seed, cudaEfficient=self.cudaEfficient)
                # small hack : MultiNLI/SNLI specific
                if self.nepoches: clf.nepoches = self.nepoches
                if self.maxepoch: clf.maxepoch = self.maxepoch
                clf.fit(self.X['train'], self.y['train'], validation_data=(self.X['valid'], self.y['valid']))
            else:
                clf = LogisticRegression(C=reg, random_state=self.seed)
                clf.fit(self.X['train'], self.y['train'])
            scores.append(round(100*clf.score(self.X['valid'], self.y['valid']),2))
        logging.info([('reg:'+str(regs[idx]), scores[idx]) for idx in range(len(scores))])
        optreg = regs[np.argmax(scores)]
        devaccuracy = np.max(scores)
        logging.info('Validation : best param found is reg = {0} with score {1}'.format(optreg, devaccuracy))               
        clf = LogisticRegression(C=optreg, random_state=self.seed)
        logging.info('Evaluating...')
        if self.usepytorch:
            if self.classifier == 'LogReg':
                clf = LogReg(inputdim = self.featdim, nclasses=self.nclasses, l2reg=optreg,\
                             seed=self.seed, cudaEfficient=self.cudaEfficient)
            elif self.classifier == 'MLP':
                clf = MLP(inputdim = self.featdim, hiddendim=self.nhid, nclasses=self.nclasses,\
                          l2reg=optreg, seed=self.seed, cudaEfficient=self.cudaEfficient)
            # small hack : MultiNLI/SNLI specific
            if self.nepoches: clf.nepoches = self.nepoches
            if self.maxepoch: clf.maxepoch = self.maxepoch
            devacc = clf.fit(self.X['train'], self.y['train'], validation_data=(self.X['valid'], self.y['valid']))
        else:
            clf = LogisticRegression(C=optreg, random_state=self.seed)
            clf.fit(self.X['train'], self.y['train'])
            
        testaccuracy = clf.score(self.X['test'], self.y['test'])
        testaccuracy = round(100*testaccuracy, 2)
        return devaccuracy, testaccuracy
    
    