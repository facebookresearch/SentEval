# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

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

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}
    n_found = 0
    with open(path_to_vec) as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word = line.split(' ', 1)[0]
            if word in word2id:
                word_vec[word] = np.array(list(map(float, line.split(' ', 1)[1].split(' '))))
                n_found += 1
                
    logging.info('Found {0} words with word vectors, out of {1} words'.format(n_found, len(word2id)))                
    return word_vec
