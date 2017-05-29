import time
import numpy as np
from random import randint
import logging

import torch
import torch.nn as nn


# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>']   = 1e9 + 4
    words['</s>']  = 1e9 + 3
    words['<p>']   = 1e9 + 2
    #words['<UNK>'] = 1e9 + 1
    sorted_words = sorted(words.items(), key=lambda x: -x[1]) # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i
    
    return id2word, word2id

# Get batch
def get_batch(batch_sentences, index_pad = 1e9 + 2):
    lengths = np.array([sentence.size(0) for sentence in batch_sentences])
    batch   =  torch.LongTensor(lengths.max(), len(batch_sentences)).fill_(int(index_pad))
    for i in xrange(len(batch_sentences)):
        batch[:lengths[i], i] = torch.LongTensor(batch_sentences[i])
    return batch, lengths
    
    
    
# Get lookup-table and fill it with GloVe vectors
def get_lut_glove(glove_type, word2id):
    word_emb_dim = int(glove_type.split('.')[1].split('d')[0])
    src_embeddings = nn.Embedding(len(word2id), word_emb_dim,
                                  padding_idx=word2id['<p>']) # padding id

    glove_path = '/mnt/vol/gfsai-east/ai-group/users/aconneau/glove/'
    n_words_with_glove = 0
    last_time = time.time()
    words_found = {}
    words_not_found = []
    
    # Initializing lut with GloVe vectors.
    # Words that do not have GloVe vectors have random vectors.
    with open(glove_path + 'glove.' + glove_type + '.txt') as f:
        for line in f:
            word = line.split(' ', 1)[0]
            if word in word2id:
                glove_vect = torch.FloatTensor(list(map(float, line.split(' ', 1)[1].split(' '))))
                src_embeddings.weight.data[word2id[word]].copy_(torch.FloatTensor(glove_vect))
                n_words_with_glove += 1
                words_found[word] = ''
    
    # get words with no GloVe vectors.
    for word in word2id:
        if word not in words_found:
            words_not_found.append(word)

            
    logging.info('Creating lookup-table of GloVe embeddings ..')
    logging.info('Found ' +  str(len(words_found)) + ' words with GloVe vectors, out of ' + str(len(word2id)) + ' words in vocabulary in ' + str(round(time.time()-last_time,2)) + ' seconds.')

    rdm_idx = 0 if len(words_not_found) < 8 else randint(0, len(words_not_found) - 1 - 7)
    logging.info('Example of 7 words in the word2id dict but with no GloVe vectors : ' + str(words_not_found[rdm_idx:rdm_idx+7]))

    
    return word_emb_dim, src_embeddings.cuda(), words_not_found

