import sys
reload(sys)  
sys.setdefaultencoding('utf8')

import torch
from torch.autograd import Variable

import data
from exutil import dotdict

# get path to SentEval
import getpass
username = getpass.getuser()

# Set PATHs
PATH_TO_SKIPTHOUGHT = '/home/aconneau/notebooks/sentence2vec/skipthought/'
PATH_TO_SENTEVAL = '/home/{0}/fbsource/fbcode/experimental/deeplearning/dkiela/senteval/'.format(username)
PATH_TO_DATA = '/mnt/vol/gfsai-east/ai-group/users/aconneau/projects/sentence-encoding/transfer-tasks-automatic/'


sys.path.insert(0, PATH_TO_SKIPTHOUGHT)
import skipthoughts
                
# get path to SentEval
import getpass
username = getpass.getuser()
# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def batcher(network, batch, params):
    embeddings = skipthoughts.encode(network, [unicode(' '.join(sent), errors="ignore")\
                                     if sent!=[] else '.' for sent in batch],\
                                     verbose=False, use_eos=True)
    return embeddings

def prepare(params, samples):
    return


# Set params for SentEval
params_senteval = {'usepytorch':True,
                   'task_path':PATH_TO_DATA,
                   'seed':1111,
                   'batch_size':512}
params_senteval = dotdict(params_senteval)
torch.cuda.set_device(2)



if __name__ == "__main__":
    model = skipthoughts.load_model()
    se = senteval.SentEval(params_senteval.task_path, model, batcher, prepare, params_senteval)
    #se.eval(['TREC'])
    se.eval(['MR'])
    # se.eval(['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14', 'ImageAnnotation'])

    
    
    
    
    
    
