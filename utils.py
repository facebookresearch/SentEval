import numpy as np
import logging
import time
from random import randint

import torch
import torch.nn as nn

def create_dictionary(sentences):
    words = {}
    for s in sentences:
        for word in s:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
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

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
