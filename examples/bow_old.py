import sys, os
import torch
from torch.autograd import Variable

from data import get_batch, create_dictionary, get_lut_glove
from exutil import dotdict

# get path to SentEval
import getpass
username = getpass.getuser()
dirpath = '/home/{0}/fbsource/fbcode/experimental/deeplearning/dkiela/senteval/'.format(username)

# import SentEval
assert os.path.exists(os.path.join(dirpath, 'senteval.py'))
sys.path.insert(0, dirpath)
import senteval


"""
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
    batch = [sent if sent != []  else ['<p>'] for sent in batch]
    padidx = params.word2id['<p>']
    X = [torch.LongTensor([params.word2id[w] if w in params.word2id else padidx for w in s]) for s in batch]

    X = get_batch(X, padidx)[0]
    X = Variable(X, volatile=True).cuda()
    
    embeddings = torch.mean(params.lut(X), 0).squeeze(0)
    return embeddings.data.cpu().numpy()

def prepare(params, samples):
    params.word2id = create_dictionary(samples)[1]
    params.lut = get_lut_glove('840B.300d', params.word2id)[1]
    params.lut.cuda()
    return


params_senteval = {'usepytorch':True,
                   'task_path':'/mnt/vol/gfsai-east/ai-group/users/aconneau/projects/sentence-encoding/transfer-tasks-automatic/',
                   'seed':1111,
                   'verbose':2, # 2: debug, 1: info, 0: warning
                   'batch_size':64}
params_senteval = dotdict(params_senteval)

torch.cuda.set_device(2)


if __name__ == "__main__":
    model = None
    se = senteval.SentEval(params_senteval.task_path, model, batcher, prepare, params_senteval)
    results = se.eval(['MRPC'])
    # results = se.eval(['MR', 'CR', 'SUBJ', 'MPQA', 'MRPC', 'TREC', 'SICKRelatedness', 'SICKEntailment'])

    