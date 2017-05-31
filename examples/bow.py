import sys
import numpy as np

import torch
from torch.autograd import Variable

from exutil import dotdict
import data

# Set PATHs
PATH_TO_SENTEVAL = '/home/aconneau/notebooks/senteval/'
PATH_TO_DATA = '/mnt/vol/gfsai-east/ai-group/users/aconneau/projects/sentence-encoding/transfer-tasks-automatic/'
                
# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


"""
Note for users : 

You have to implement two functions :
    1) "batcher" : transforms a batch of sentences into sentence embeddings.
        i) takes as input a network, a "batch", and "params".
        ii) 
    2) "prepare" : sees the whole dataset, and can create a vocabulary
        i) outputs of "prepare" are stored in "params" that batcher will use.
        ii) "prepare" can create a dictionary (chars/bpe/words).
        iii) Below, "prepare" prepares the lookup-table of GloVe embeddings.
"""


def batcher(network, batch, params):
    batch = [sent if sent!=[] else ['.'] for sent in batch]
    X = [torch.LongTensor([params.word2id[w] for w in s]) for s in batch]
    X, lengths = data.get_batch(X, params.word2id['<p>'])
    word_embed = params.lut.cpu()(Variable(X)).data.numpy()
    embeddings = []
    for i in range(len(batch)):
        bow = np.zeros(300)
        for j in range(lengths[i]):
            bow += word_embed[j,i,:]
        bow = bow / lengths[i]
        embeddings.append(bow)
    embeddings = np.vstack(embeddings)
    return embeddings


def prepare(params, samples):
    _, params.word2id = data.create_dictionary(samples)
    params.emb_dim = 300
    params.eos_index = params.word2id['</s>']
    params.sos_index = params.word2id['</s>']
    params.pad_index = params.word2id['<p>']
    _, params.lut, _ = data.get_lut_glove('840B.300d', params.word2id)
    params.lut.cuda()
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
    model = None # No model here
    se = senteval.SentEval(params_senteval.task_path, model, batcher, prepare, params_senteval)
    se.eval(['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14', 'ImageAnnotation'])
    # se.eval(['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14', 'ImageAnnotation'])

    
    
    
    
    
    
