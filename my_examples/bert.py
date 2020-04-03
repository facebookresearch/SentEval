from __future__ import absolute_import, division

import sys
import logging
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    # Initialize Multilingual BERT model
    params.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    params.model = BertModel.from_pretrained('bert-base-multilingual-cased')
    params.model.eval()
    return


def get_sentence_embedding(text, params):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = params.tokenizer.tokenize(marked_text)
    indexed_tokens = params.tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        encoded_layers, _ = params.model(tokens_tensor, segments_tensors)
    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)
    token_vecs = []
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs.append(sum_vec)
    token_vecs = encoded_layers[11][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding.numpy()


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = [get_sentence_embedding(sentence, params) for sentence in batch]
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 5, 'batch_size': 128,
                   'classifier': {'nhid': 0, 'optim': 'rmsprop', 'tenacity': 3, 'epoch_size': 2}}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['SICKEntailment', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment_RU', 'SST2_RU', 'SST3_RU', 'TREC_RU', 'MRPC_RU'
                      'STSBenchmark', 'SICKRelatedness'
                      'STSBenchmark_RU', 'SICKRelatedness_RU'
                      ]
    results = se.eval(transfer_tasks)
    print(results)
