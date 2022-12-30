import argparse
import pickle

import torch
import sys
import os

import numpy as np

from transformers import BertModel, BertTokenizer


def get_embed():
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    embed_model = BertModel.from_pretrained(model_name)

    path = "./data/{}/".format(args.dataset)
    if args.dataset in ['nell-v1', 'nell-v2', 'nell-v3', 'nell-v4']:
        phrases = []
        ids = []
        relation_embed = []
        with open(os.path.join(path, 'relation2id.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                phrase_id = line.strip('concept:\n')
                phrase_id = phrase_id.split(':')
                phrases.append(phrase_id[0])
                ids.append(phrase_id[1])
        for phrase in phrases:
            token_id = tokenizer.encode(phrase, add_special_tokens=True)
            token_id = torch.tensor([token_id])
            with torch.no_grad():
                last_hidden_states = embed_model(token_id)[0]
                last_hidden_states = last_hidden_states.mean(1)
                relation_embed.append(last_hidden_states)
        id_embed_dict = dict(zip(ids, relation_embed))

    elif args.dataset in ['FB15k-237-v1', 'FB15k-237-v2', 'FB15k-237-v3', 'FB15k-237-v4']:
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
        for phrase in phrases:
            token_id = tokenizer.encode(phrase, add_special_tokens=True)
            token_id = torch.tensor([token_id])
            with torch.no_grad():
                last_hidden_states = embed_model(token_id)[0]
                last_hidden_states = last_hidden_states.mean(1)
                relation_embed.append(last_hidden_states)
        id_embed_dict = dict(zip(ids, relation_embed))
    return id_embed_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get id2relation and id2entity')
    parser.add_argument("-dataset", "--dataset", default="nell-v1", help="dataset")
    parser.add_argument("--hop", type=int, default=3)
    args = parser.parse_args()

    path = "./data/{}/".format(args.dataset)
    id2embed = get_embed()
    with open(os.path.join(path, 'id2embed.pkl'), 'wb') as f:
        pickle.dump(id2embed, f, pickle.HIGHEST_PROTOCOL)
    print("finish save the embedding of relation to id2embed.pkl")


