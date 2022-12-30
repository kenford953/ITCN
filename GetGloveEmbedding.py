import argparse
import pickle

import torch
import sys
import os

import numpy as np

from time import time


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def get_embed():
    '''start = time()
    with open('./data/glove.840B.300d.txt') as f:
        word2embed = dict(get_coefs(*line.strip().split(' ')) for line in f)

    with open('./data/glove_dict.pkl', 'wb') as f:
        pickle.dump(word2embed, f, pickle.HIGHEST_PROTOCOL)
    stop = time()
    print(str(stop - start) + "s")
    sys.exit()'''

    with open('./data/glove_dict.pkl', 'rb') as f:
        word2embed = pickle.load(f)

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
            word_embed = word2embed.get(word, np.zeros(300))
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
    parser.add_argument("-dataset", "--dataset", default="nell-v1", help="dataset")
    parser.add_argument("--hop", type=int, default=3)
    args = parser.parse_args()

    path = "./data/{}/".format(args.dataset)
    id2embed = get_embed()
    with open(os.path.join(path, 'id2embed_glove.pkl'), 'wb') as f:
        pickle.dump(id2embed, f, pickle.HIGHEST_PROTOCOL)
    print("finish save the embedding of relation to id2embed_glove.pkl")

