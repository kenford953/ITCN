import argparse
import pickle

import torch
import sys
import os
import gensim

import numpy as np

from time import time


def get_embed():
    model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)

    phrases = []
    ids = []
    relation_embed = []
    with open(os.path.join(path, 'relation2id.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            phrase_id = line.strip('/')
            phrase_id = line.split(':')
            phrase_id[0] = phrase_id[0].replace("/", " ").replace('_', " ")
            phrase_id[0] = phrase_id[0].replace(".", '')
            phrases.append(phrase_id[0])
            phrase_id[1] = phrase_id[1].replace('\n', '')
            ids.append(phrase_id[1])

    phrases_embed = []
    for phrase in phrases:
        words = phrase.split(' ')
        del words[0]
        words_embed = []
        for word in words:
            if word in model:
                word_embed = model[word]
            else:
                word_embed = np.zeros(300)
            word_embed = np.expand_dims(word_embed, axis=0)
            words_embed.append(word_embed)
        words_embed = np.concatenate(words_embed, 0)
        phrase_embed = np.mean(words_embed, axis=0)
        phrase_embed = torch.FloatTensor(phrase_embed)
        phrase_embed = phrase_embed.unsqueeze(0)
        phrases_embed.append(phrase_embed)
    id_embed_dict = dict(zip(ids, phrases_embed))
    return id_embed_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get id2relation and id2entity')
    parser.add_argument("-dataset", "--dataset", default="FB15k-237-v1", help="dataset")
    parser.add_argument("--hop", type=int, default=3)
    args = parser.parse_args()

    path = "./data/{}/".format(args.dataset)
    id2embed = get_embed()
    with open(os.path.join(path, 'id2embed_word2vec.pkl'), 'wb') as f:
        pickle.dump(id2embed, f, pickle.HIGHEST_PROTOCOL)
    print("finish save the embedding of relation to id2embed_word2vec.pkl")