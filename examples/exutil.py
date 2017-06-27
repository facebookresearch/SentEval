# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. 
#

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Get lookup-table with GloVe vectors
def get_lut_glove(glove_type, glove_path, word2id):
    word_emb_dim = int(glove_type.split('.')[1].split('d')[0])
    src_embeddings = nn.Embedding(len(word2id), word_emb_dim, padding_idx=word2id['<p>'])
    #src_embeddings.weight.data.fill_(0)

    n_words_with_glove = 0
    last_time = time.time()
    words_found = {}
    words_not_found = []
    
    # initializing lut with GloVe vectors,
    # words that do not have GloVe vectors have random vectors.
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
            
    print 'GLOVE : Found ' +  str(len(words_found)) + ' words with GloVe vectors, out of ' +\
                                str(len(word2id)) + ' words in vocabulary'
    print 'GLOVE : Took ' + str(round(time.time()-last_time,2)) + ' seconds.'
    rdm_idx = 0 if len(words_not_found)<8 else randint(0, len(words_not_found) - 1 - 7)
    print 'GLOVE : 7 words in word2id without GloVe vectors : ' + str(words_not_found[rdm_idx:rdm_idx + 7])
    
    return word_emb_dim, src_embeddings.cuda(), words_not_found