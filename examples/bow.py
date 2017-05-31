import sys
import numpy as np

import torch
from torch.autograd import Variable

from exutil import dotdict
import data

# Set PATHs
PATH_TO_SENTEVAL = '/home/aconneau/notebooks/senteval/'
PATH_TO_DATA = '/mnt/vol/gfsai-east/ai-group/users/aconneau/projects/sentence-encoding/transfer-tasks-automatic/'
PATH_TO_GLOVE = '/mnt/vol/gfsai-east/ai-group/users/aconneau/glove/glove.840B.300d.txt'
                
# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


"""
Note for users : 

You have to implement two functions :
    1) "batcher" : transforms a batch of sentences into sentence embeddings.
        i) takes as input a "batch", and "params".
        ii) 
    2) "prepare" : sees the whole dataset, and can create a vocabulary
        i) outputs of "prepare" are stored in "params" that batcher will use.
        ii) "prepare" can create a dictionary (chars/bpe/words).
        iii) Below, "prepare" prepares the lookup-table of GloVe embeddings.
"""


def batcher(batch, params):
    batch = [sent if sent!=[] else ['.'] for sent in batch]
    embeddings = []
    
    for sent in batch:
        sentvec = np.zeros(300)
        nbwords = 0
        for word in sent:
            if word in params.word_vec:
                sentvec += params.word_vec[word]
                nbwords += 1
        if nbwords == 0:
            sentvec = params.word_vec['.']
            nbwords += 1
        sentvec /= nbwords
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings
            

def prepare(params, samples):
    _, params.word2id = data.create_dictionary(samples)
    params.emb_dim = 300
    params.word_vec = data.get_wordvec(PATH_TO_GLOVE, params.word2id)
    return


# Set params for SentEval
params_senteval = {'usepytorch':True,
                   'task_path':PATH_TO_DATA,
                   'seed':1111,
                   'verbose':2, # 2: debug, 1: info, 0: warning
                   'batch_size':64}

params_senteval = dotdict(params_senteval)

torch.cuda.set_device(2)

if __name__ == "__main__":
    params_senteval.model = None # No model here, just for illustration
    se = senteval.SentEval(batcher, prepare, params_senteval)
    se.eval(['MR', 'CR', 'SUBJ','MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14', 'ImageAnnotation'])
    # se.eval(['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14', 'ImageAnnotation'])

    
    
    
    
    
    
