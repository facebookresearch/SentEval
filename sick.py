'''
SICK
'''

import codecs
import os
import logging
import numpy as np

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

from tools.relatedness import RelatednessPytorch
from tools.validation import SplitClassifier

class SICKRelatednessEval(object):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SICK-Relatedness*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.sick_data = {'train':train, 'dev':dev, 'test':test}
        
    def do_prepare(self, params, prepare):
        samples = self.sick_data['train']['X_A'] + self.sick_data['train']['X_B'] + \
                  self.sick_data['dev']['X_A'] + self.sick_data['dev']['X_B'] + \
                  self.sick_data['test']['X_A'] + self.sick_data['test']['X_B']
        return prepare(params, samples)
        
    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A':[], 'X_B':[], 'y':[]}
        with open(fpath, 'rb') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[3])
        
        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data
    
    
    def run(self, network, batcher, params):
        sick_embed = {'train':{}, 'dev':{}, 'test':{}}
                      
        for key in self.sick_data:  
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_corpus = sorted(zip(self.sick_data[key]['X_A'],
                                       self.sick_data[key]['X_B'],
                                       self.sick_data[key]['y']),
                                   key=lambda z:(len(z[0]), len(z[1]), z[2]))
            
            self.sick_data[key]['X_A'] = [x for (x,y,z) in sorted_corpus]
            self.sick_data[key]['X_B'] = [y for (x,y,z) in sorted_corpus]
            self.sick_data[key]['y'] = [z for (x,y,z) in sorted_corpus]
           
            for txt_type in ['X_A', 'X_B']:
                sick_embed[key][txt_type] = []
                for ii in range(0, len(self.sick_data[key]['y']), params.batch_size):
                    batch = self.sick_data[key][txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(network, batch, params)
                    sick_embed[key][txt_type].append(embeddings)
                sick_embed[key][txt_type] = np.vstack(sick_embed[key][txt_type])
            sick_embed[key]['y'] = np.array(self.sick_data[key]['y'])
            logging.info('Computed {0} embeddings'.format(key))
            
        # Train
        trainA = sick_embed['train']['X_A']
        trainB = sick_embed['train']['X_B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = self.encode_labels(self.sick_data['train']['y'])
        
        # Dev
        devA = sick_embed['dev']['X_A']
        devB = sick_embed['dev']['X_B']
        devF = np.c_[np.abs(devA - devB), devA * devB]
        devY = self.encode_labels(self.sick_data['dev']['y'])
        
        # Test
        testA = sick_embed['test']['X_A']
        testB = sick_embed['test']['X_B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = self.encode_labels(self.sick_data['test']['y'])
        
        config_classifier = {'seed':self.seed, 'nclasses':5}
        clf = RelatednessPytorch(train={'X':trainF, 'y':trainY},
                                    valid={'X':devF, 'y':devY},
                                    test={'X':testF, 'y':testY},
                                    devscores=self.sick_data['dev']['y'],
                                    config=config_classifier)
        
        devpr, yhat = clf.run()
        
        pr = pearsonr(yhat, self.sick_data['test']['y'])[0]
        sr = spearmanr(yhat, self.sick_data['test']['y'])[0]
        se = mean_squared_error(yhat, self.sick_data['test']['y'])
        logging.debug('Dev : Pearson {0}'.format(devpr))
        logging.debug('Test : Pearson {0} Spearman {1} MSE {2} for SICK Relatedness\n'.format(pr, sr, se))
        #import pdb
        #pdb.set_trace()
        return {'devpearson':devpr, 'pearson': pr, 'spearman':sr, 'mse':se, 'yhat':yhat, 'ndev': len(devA), 'ntest': len(testA)}
       
    def encode_labels(self, labels, nclass=5):
        """
        Label encoding from Tree LSTM paper (Tai, Socher, Manning)
        """
        Y = np.zeros((len(labels), nclass)).astype('float32')
        for j, y in enumerate(labels):
            for i in range(nclass):
                if i+1 == np.floor(y) + 1:
                    Y[j,i] = y - np.floor(y)
                if i+1 == np.floor(y):
                    Y[j,i] = np.floor(y) - y + 1
        return Y
                

class SICKEntailmentEval(SICKRelatednessEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SICK-Entailment*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.sick_data = {'train':train, 'dev':dev, 'test':test}
        
    def loadFile(self, fpath):
        label2id = {'CONTRADICTION':0, 'NEUTRAL':1, 'ENTAILMENT':2}
        skipFirstLine = True
        sick_data = {'X_A':[], 'X_B':[], 'y':[]}
        with open(fpath, 'rb') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[4])
        sick_data['y'] = [label2id[s] for s in sick_data['y']]
        return sick_data
    
    def run(self, network, batcher, params):
        sick_embed = {'train':{}, 'dev':{}, 'test':{}}
                      
        for key in self.sick_data:  
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            sorted_corpus = sorted(zip(self.sick_data[key]['X_A'],
                                       self.sick_data[key]['X_B'],
                                       self.sick_data[key]['y']),
                                   key=lambda z:(len(z[0]), len(z[1]), z[2]))
            
            self.sick_data[key]['X_A'] = [x for (x,y,z) in sorted_corpus]
            self.sick_data[key]['X_B'] = [y for (x,y,z) in sorted_corpus]
            self.sick_data[key]['y'] = [z for (x,y,z) in sorted_corpus]
            
            for txt_type in ['X_A', 'X_B']:
                sick_embed[key][txt_type] = []
                for ii in range(0, len(self.sick_data[key]['y']), params.batch_size):
                    batch = self.sick_data[key][txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(network, batch, params)
                    sick_embed[key][txt_type].append(embeddings)
                sick_embed[key][txt_type] = np.vstack(sick_embed[key][txt_type])
            logging.info('Computed {0} embeddings'.format(key))
            
        # Train
        trainA = sick_embed['train']['X_A']
        trainB = sick_embed['train']['X_B']
        trainF = np.c_[np.abs(trainA - trainB), trainA * trainB]
        trainY = np.array(self.sick_data['train']['y'])
        
        # Dev
        devA = sick_embed['dev']['X_A']
        devB = sick_embed['dev']['X_B']
        devF = np.c_[np.abs(devA - devB), devA * devB]
        devY = np.array(self.sick_data['dev']['y'])
        
        # Test
        testA = sick_embed['test']['X_A']
        testB = sick_embed['test']['X_B']
        testF = np.c_[np.abs(testA - testB), testA * testB]
        testY = np.array(self.sick_data['test']['y'])
        
        config_classifier = {'nclasses':3, 'seed':self.seed, 'usepytorch':params.usepytorch}
        clf = SplitClassifier(X={'train':trainF, 'valid':devF, 'test':testF},
                              y={'train':trainY, 'valid':devY, 'test':testY},
                              config=config_classifier)
        
        devacc, testacc = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} for SICK entailment\n'.format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc, 'ndev': len(devA), 'ntest': len(testA)}
        
    
    
