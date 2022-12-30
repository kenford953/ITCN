import argparse
import pickle

import torch
import sys
import os

import numpy as np


def get_id(filename,  is_unweigted=False, directed=True, saved_relation2id=None):
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triples_data = {}
    rows, cols, data = [], [], []
    unique_entities = set()

    ent = 0
    rel = 0

    for filename1 in filename:

        data = []
        with open(filename1) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

       # triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    return entity2id, relation2id, rel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get relation2id and entity2id')
    parser.add_argument("-dataset", "--dataset", default="nell-v1", help="dataset")
    parser.add_argument("--hop", type=int, default=3)
    args = parser.parse_args()

    path = "./data/{}/".format(args.dataset)
    entity2id, relation2id, rel = get_id([os.path.join(path, 'train.txt'),
                                          os.path.join(path, 'valid.txt'),
                                          os.path.join(path, 'train_inductive.txt')])

    fileObject1 = open(os.path.join(path, 'entity2id.txt'), 'w')
    for key, value in entity2id.items():
        fileObject1.write("{}:{}".format(key, value))
        fileObject1.write('\n')
    fileObject1.close()

    fileObject2 = open(os.path.join(path, 'relation2id.txt'), 'w')
    for key, value in relation2id.items():
        fileObject2.write("{}:{}".format(key, value))
        fileObject2.write('\n')
    fileObject2.close()
    print('please correctly edit the "relation2id.txt" file before get the relation embedding as "python GetRelationEmbedding.py"')
    '''The words of relation in nell dataset are connected. So we need to manually add blank between two words.'''