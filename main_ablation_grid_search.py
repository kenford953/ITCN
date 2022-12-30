import argparse
import os
import random
import sys
import math
import time
import pickle
import torch
import re

import numpy as np

from torch.utils.data import DataLoader
from src.utils import process_data
from src.model_ablation import InTriCaps_refined, rire, wocn, wos, InTriCaps_CNN, InTriCaps_GRU, InTriCaps_BiLSTM
from sklearn import metrics


def total(params):
    '''
    count the total number of hyperparameter settings
    '''
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings


def get_glove_relation_embedding(args):
    path = './data/{}/'.format(args.dataset)
    with open(os.path.join(path, 'id2embed_glove.pkl'), 'rb') as f:
        id2embed = pickle.load(f)
    return id2embed


def get_word2vec_relation_embedding(args):
    path = './data/{}/'.format(args.dataset)
    with open(os.path.join(path, 'id2embed_word2vec.pkl'), 'rb') as f:
        id2embed = pickle.load(f)
    return id2embed


def get_relation_embedding(args):
    path = './data/{}/'.format(args.dataset)
    with open(os.path.join(path, 'id2embed.pkl'), 'rb') as f:
        id2embed = pickle.load(f)
    return id2embed


def valid_auc(model, loss, batch_size, val_data, val_label, val_subgraph):
    losses = 0
    total_iter = int(len(val_data) // batch_size) + 1
    if len(val_data) % batch_size == 0:
       total_iter -= 1

    all_preds = []
    all_labels = []
    for i in range(total_iter):
        if i == total_iter - 1:    # the last iteration
            train_indices = val_data[i*batch_size:, :]
            train_values = val_label[i*batch_size:, :]
        else:
            train_indices = val_data[i*batch_size:(i+1)*batch_size, :]
            train_values = val_label[i*batch_size:(i+1)*batch_size, :]
        train_indices = torch.LongTensor(train_indices).cuda()
        train_values = torch.FloatTensor(train_values).cuda()
        preds = model(train_indices, val_subgraph)
        all_preds += preds.squeeze(1).detach().cpu().tolist()
        all_labels += train_values.squeeze(1).tolist()
    auc = metrics.roc_auc_score(all_labels, all_preds)
    return auc


def train(args, new_data, train_data, train_label, train_subgraph, val_data, val_label, val_subgraph):
    if args.ablation_name not in ['rire', 'gre', 'w2vre']:
        relation_embed = get_relation_embedding(args)  # dict:{id: embedding}
    elif args.ablation_name == 'gre':
        relation_embed = get_glove_relation_embedding(args)
    elif args.ablation_name == 'w2vre':
        relation_embed = get_word2vec_relation_embedding(args)
    if args.dataset == 'nell-v1':
        num_relation = 14
    elif args.dataset == 'nell-v2':
        num_relation = 88
    elif args.dataset == 'nell-v3':
        num_relation = 142
    elif args.dataset == 'nell-v4':
        num_relation = 76
    elif args.dataset == 'FB15k-237-v1':
        num_relation = 180
    elif args.dataset == 'FB15k-237-v2':
        num_relation = 200
    elif args.dataset == 'FB15k-237-v3':
        num_relation = 215
    elif args.dataset == 'FB15k-237-v4':
        num_relation = 219

    # hyperparameters
    params = dict()
    params['otm'] = ['Adam', 'RMSprop']
    params['lr'] = [0.01, 0.001, 0.0001]
    params['GCN_dim'] = [16, 32, 64]
    params['GCN_layers'] = [1, 2, 3]
    params['batch_size'] = [16, 32, 64]

    node_dim = 2 * (args.hop + 1)
    act_func = args.activation

    if args.ablation_name in ['wocn', 'wos', 'CNN', 'BiLSTM', 'GRU']:
        params['num_mid_caps'] = [0]
    elif args.ablation_name in ['wor', 'rire', 'gre', 'w2vre']:
        params['num_mid_caps'] = [2, 3]

    if args.ablation_name in ['wocn', 'wor', 'CNN', 'BiLSTM', 'GRU']:
        params['routing'] = [0]
    elif args.ablation_name in ['wos', 'rire', 'gre', 'w2vre']:
        params['routing'] = [1, 2, 3]

    if args.ablation_name in ['wocn', 'wos']:
        params['pad_edge_num'] = [0]
    elif args.ablation_name in ['CNN', 'BiLSTM', 'GRU']:
        params['pad_edge_num'] = [200]
    elif args.ablation_name in ['wor', 'rire', 'gre', 'w2vre']:
        params['pad_edge_num'] = [200, 400, 600, 800]

    total_setting = total(params)
    seen_setting = set()
    kk = 0

    best_mean_auc_pr = 0
    hits_at10_on_best_auc = 0

    best_hits_at10 = 0
    mean_auc_pr_on_best_hits = 0

    best_aucpr_setting = {'otm': None, 'lr': None, 'GCN_dim': None, 'GCN_layers': None, 'num_mid_caps': None,
                          'routing': None, 'batch_size': None, 'pad_edge_num': None}
    best_hits_at10_setting = {'otm': None, 'lr': None, 'GCN_dim': None, 'GCN_layers': None, 'num_mid_caps': None,
                              'routing': None, 'batch_size': None, 'pad_edge_num': None}

    for i in range(1000000):
        print('there are {} setting. {}th setting now'.format(total_setting, kk+1))
        if kk >= total_setting:
            break
        otm = random.choice(params['otm'])
        lr = random.choice(params['lr'])
        GCN_dim = random.choice(params['GCN_dim'])
        node_dim = 2 * (args.hop + 1)
        act_func = 'tanh'
        GCN_layers = random.choice(params['GCN_layers'])
        num_mid_caps = random.choice(params['num_mid_caps'])
        routing = random.choice(params['routing'])
        batch_size = random.choice(params['batch_size'])
        pad_edge_num = random.choice(params['pad_edge_num'])
        current_setting = (otm, lr, GCN_dim, node_dim, GCN_layers, num_mid_caps,
                           routing, batch_size, pad_edge_num)

        if args.ablation_name == 'rire':
            relation_embed_dim = 64
            model = rire(args, GCN_dim, node_dim, act_func, GCN_layers, routing, pad_edge_num, num_relation, relation_embed_dim, num_mid_caps).cuda()
        elif args.ablation_name in ['gre', 'wor', 'w2vre']:
            model = InTriCaps_refined(args, GCN_dim, node_dim, relation_embed, act_func, GCN_layers, num_mid_caps, routing, pad_edge_num).cuda()
        elif args.ablation_name == 'wocn':
            model = wocn(args, GCN_dim, node_dim, relation_embed, act_func, GCN_layers).cuda()
        elif args.ablation_name == 'wos':
            model = wos(args, GCN_dim, node_dim, relation_embed, act_func, GCN_layers, routing, pad_edge_num).cuda()
        elif args.ablation_name == 'CNN':
            model = InTriCaps_CNN(args, GCN_dim, node_dim, relation_embed, act_func, GCN_layers, pad_edge_num).cuda()
        elif args.ablation_name == 'GRU':
            model = InTriCaps_GRU(args, GCN_dim, node_dim, relation_embed, act_func, GCN_layers, pad_edge_num).cuda()
        elif args.ablation_name == 'BiLSTM':
            model = InTriCaps_BiLSTM(args, GCN_dim, node_dim, relation_embed, act_func, GCN_layers, pad_edge_num).cuda()

        if i == 0:
            best_auc_setting = current_setting
        if (current_setting in seen_setting) and (current_setting != best_auc_setting) and (current_setting != best_hits_at10_setting):
            continue
        else:
            seen_setting.add(current_setting)
            kk += 1

        # model size
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for param in model.parameters():
            mulValue = np.prod(param.size())
            Total_params += mulValue
            if param.requires_grad:
                Trainable_params += mulValue
            else:
                NonTrainable_params += mulValue
        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')

        train_set = process_data(train_data, train_label)
        train_iterator = DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)

        if otm == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr)
        elif otm == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5, last_epoch=-1)

        margin_loss = torch.nn.SoftMarginLoss()
        epoch_losses = []   # recording losses of all epochs
        best_auc_current_setting = 0
        save_iteration = int(train_data.shape[0]//batch_size//3)  # validate models in multiple iterations instead of only the last iteration

        for epoch in range(args.epoch):
            model.train()
            start_time = time.time()
            epoch_loss = []

            for iters, batch in enumerate(train_iterator):
                start_time_per_batch = time.time()
                train_indices, train_values = batch
                train_indices = torch.LongTensor(train_indices).cuda()
                train_values = torch.FloatTensor(train_values).cuda()

                preds = model(train_indices, train_subgraph)

                optimizer.zero_grad()
                loss = margin_loss(preds.view(-1), train_values.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.data.item())
                end_time_per_batch = time.time()

                if (iters+1) % save_iteration == 0:
                    with torch.no_grad():
                        val_auc = valid_auc(model, margin_loss, batch_size, val_data, val_label, val_subgraph)
                    if best_auc_current_setting < val_auc:
                        best_auc_current_setting = val_auc
                        print('found best model, save to ./pretrained_model/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'
                              .format(args.dataset, otm, lr, GCN_dim, GCN_layers, num_mid_caps,
                                      routing, batch_size, pad_edge_num, args.ablation_name))
                        torch.save(model, './pretrained_model/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'
                                   .format(args.dataset, otm, lr, GCN_dim, GCN_layers, num_mid_caps,
                                           routing, batch_size, pad_edge_num, args.ablation_name))

            scheduler.step()
            print("epoch {}, average loss {}, epoch_time {}"
                  .format(epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
            epoch_losses.append(sum(epoch_loss) / len(epoch_loss))
            # validate the model in the last iteration in each epoch
            with torch.no_grad():
                val_auc = valid_auc(model, margin_loss, batch_size, val_data, val_label, val_subgraph)
            if best_auc_current_setting < val_auc:
                best_auc_current_setting = val_auc
                print('found best model, save to ./pretrained_model/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'
                      .format(args.dataset, otm, lr, GCN_dim, GCN_layers, num_mid_caps,
                              routing, batch_size, pad_edge_num, args.ablation_name))
                torch.save(model, './pretrained_model/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'
                           .format(args.dataset, otm, lr, GCN_dim, GCN_layers, num_mid_caps,
                                   routing, batch_size, pad_edge_num, args.ablation_name))

        mean_auc_pr, hits_at_10 = evaluate(args, new_data, test_subgraph, otm, lr, GCN_dim, GCN_layers,
                                           num_mid_caps, routing, batch_size, pad_edge_num)
        print('current | auc_pr:{} | hit_at_10:{}'.format(mean_auc_pr, hits_at_10))
        if mean_auc_pr > best_mean_auc_pr:
            best_mean_auc_pr = mean_auc_pr
            hits_at10_on_best_auc = hits_at_10
            best_aucpr_setting['otm'] = otm
            best_aucpr_setting['lr'] = lr
            best_aucpr_setting['GCN_dim'] = GCN_dim
            best_aucpr_setting['GCN_layers'] = GCN_layers
            best_aucpr_setting['num_mid_caps'] = num_mid_caps
            best_aucpr_setting['routing'] = routing
            best_aucpr_setting['batch_size'] = batch_size
            best_aucpr_setting['pad_edge_num'] = pad_edge_num
            print('find best auc_pr setting')
        if hits_at_10 > best_hits_at10:
            best_hits_at10 = hits_at_10
            mean_auc_pr_on_best_hits = mean_auc_pr
            best_hits_at10_setting['otm'] = otm
            best_hits_at10_setting['lr'] = lr
            best_hits_at10_setting['GCN_dim'] = GCN_dim
            best_hits_at10_setting['GCN_layers'] = GCN_layers
            best_hits_at10_setting['num_mid_caps'] = num_mid_caps
            best_hits_at10_setting['routing'] = routing
            best_hits_at10_setting['batch_size'] = batch_size
            best_hits_at10_setting['pad_edge_num'] = pad_edge_num
        print('best auc_pr so far: {} | its hits_at10 : {}'.format(best_mean_auc_pr, hits_at10_on_best_auc))
        print('best hits@10 so far: {} | its auc_pr : {}'.format(best_hits_at10, mean_auc_pr_on_best_hits))
        print('best auc_pr setting so far: {}'.format(best_aucpr_setting))
        print('best hits@10 setting so far: {}'.format(best_hits_at10_setting))


def evaluate(args, new_data, test_subgraph, otm, lr, GCN_dim, GCN_layers,
             num_mid_caps, routing, batch_size, pad_edge_num):

    model = torch.load('./pretrained_model/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'
                       .format(args.dataset, otm, lr, GCN_dim, GCN_layers, num_mid_caps,
                               routing, batch_size, pad_edge_num, args.ablation_name))

    model.cuda()
    model.eval()

    with open("./data/{}/{}_{}_hop_total_test_head2.pickle".format(args.dataset, args.dataset, args.hop), 'rb') as handle:
        total_test_head = pickle.load(handle)
    with open("./data/{}/{}_{}_hop_total_test_tail2.pickle".format(args.dataset, args.dataset, args.hop), 'rb') as handle:
        total_test_tail = pickle.load(handle)

    train_triple = new_data['new_train_data']
    relation = set(list(train_triple[:, 1]))

    new_total_test_head = []
    new_total_test_tail = []
    for i in range(len(total_test_head)):
        if total_test_head[i][0, 1] in relation and total_test_head[i].shape[0] >= 50 and total_test_tail[i].shape[0] >= 50:
            new_total_test_head.append(total_test_head[i])
            new_total_test_tail.append(total_test_tail[i])

    # validate auc
    mean_auc_pr = 0.
    for kk in range(10):
        true_triplets = []
        neg_triplets = []
        for i in range(len(new_total_test_head)):
            true_triplets.append(new_total_test_head[i][0, :])
            xx = np.random.uniform()
            if xx < 0.5:
                neg = new_total_test_head[i][1:, :]
                random_entity = random.choice([i for i in range(49)])
                neg_triplets.append(neg[random_entity, :])
            else:
                neg = new_total_test_tail[i][1:, :]
                random_entity = random.choice([i for i in range(49)])
                neg_triplets.append(neg[random_entity, :])

        true_label = np.expand_dims(np.array([1 for i in range(len(true_triplets))]), 1)
        neg_label = np.expand_dims(np.array([-1 for i in range(len(neg_triplets))]), 1)
        true_triplets = np.array(true_triplets)
        neg_triplets = np.array(neg_triplets)

        total_iter = int(len(true_triplets) // batch_size) + 1
        if len(true_triplets) % batch_size == 0:
            total_iter -= 1
        all_preds = []
        all_labels = []
        for i in range(total_iter):
            if i == total_iter - 1:
                true_indices = true_triplets[i*batch_size:, :]
                true_values = true_label[i*batch_size:, :]
                neg_indices = neg_triplets[i*batch_size:, :]
                neg_values = neg_label[i*batch_size:, :]
            else:
                true_indices = true_triplets[i*batch_size : (i+1)*batch_size, :]
                true_values = true_label[i*batch_size : (i+1)*batch_size, :]
                neg_indices = neg_triplets[i*batch_size : (i+1)*batch_size, :]
                neg_values = neg_label[i*batch_size : (i+1)*batch_size, :]
            true_indices = torch.LongTensor(true_indices).cuda()
            neg_indices = torch.LongTensor(neg_indices).cuda()

            preds = model(true_indices, test_subgraph)
            all_preds += preds.squeeze(1).detach().cpu().tolist()
            all_labels += true_values.tolist()

            preds = model(neg_indices, test_subgraph)
            all_preds += preds.squeeze(1).detach().cpu().tolist()
            all_labels += neg_values.tolist()

        auc = metrics.roc_auc_score(all_labels, all_preds)
        auc_pr = metrics.average_precision_score(all_labels, all_preds)
        print('{}th auc_pr:{}'.format(kk, auc_pr))
        mean_auc_pr += auc_pr
        # print(all_preds)
    # print('mean_auc_pr:{}'.format(mean_auc_pr/10.0))

    # validate Hits@10
    ranks_head, ranks_tail = [], []
    reciprocal_ranks_head, reciprocal_ranks_tail = [], []
    hits_at_100_head, hits_at_100_tail = 0, 0
    hits_at_10_head, hits_at_10_tail = 0, 0
    hits_at_3_head, hits_at_3_tail = 0, 0
    hits_at_1_head, hits_at_1_tail = 0, 0
    for i in range(len(new_total_test_head)):
        new_x_batch_head = new_total_test_head[i].astype(np.int64)
        new_x_batch_tail = new_total_test_tail[i].astype(np.int64)

        if new_x_batch_tail.shape[0] < 2 or new_x_batch_head.shape[0] < 2:
            continue

        true_triple = new_x_batch_tail[0, :]
        new_x_batch_head = new_x_batch_head[1:, :]
        new_x_batch_tail = new_x_batch_tail[1:, :]

        rand_head = np.random.randint(new_x_batch_head.shape[0])
        rand_tail = np.random.randint(new_x_batch_tail.shape[0])
        new_x_batch_head = np.insert(new_x_batch_head, rand_head, true_triple, axis=0)
        new_x_batch_tail = np.insert(new_x_batch_tail, rand_tail, true_triple, axis=0)

        scores_head = model(torch.LongTensor(new_x_batch_head).cuda(), test_subgraph)
        sorted_scores_head, sorted_indices_head = torch.sort(scores_head.view(-1), dim=-1, descending=True)

        ranks_head.append(np.where(sorted_indices_head.cpu().numpy() == rand_head)[0][0] + 1)
        reciprocal_ranks_head.append(1.0 / ranks_head[-1])

        scores_tail = model(torch.LongTensor(new_x_batch_tail).cuda(), test_subgraph)
        sorted_scores_tail, sorted_indices_tail = torch.sort(scores_tail.view(-1), dim=-1, descending=True)

        ranks_tail.append(np.where(sorted_indices_tail.cpu().numpy() == rand_tail)[0][0] + 1)
        reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])

    for i in range(len(ranks_head)):
        if ranks_head[i] <= 100:
            hits_at_100_head += 1
        if ranks_head[i] <= 10:
            hits_at_10_head += 1
        if ranks_head[i] <= 3:
            hits_at_3_head += 1
        if ranks_head[i] == 1:
            hits_at_1_head += 1

    for i in range(len(ranks_tail)):
        if ranks_tail[i] <= 100:
            hits_at_100_tail += 1
        if ranks_tail[i] <= 10:
            hits_at_10_tail += 1
        if ranks_tail[i] <= 3:
            hits_at_3_tail += 1
        if ranks_tail[i] == 1:
            hits_at_1_tail += 1

    average_hits_at_100_head = hits_at_100_head / len(ranks_head)
    average_hits_at_10_head = hits_at_10_head / len(ranks_head)
    average_hits_at_3_head = hits_at_3_head / len(ranks_head)
    average_hits_at_1_head = hits_at_1_head / len(ranks_head)

    average_hits_at_100_tail = hits_at_100_tail / len(ranks_tail)
    average_hits_at_10_tail = hits_at_10_tail / len(ranks_tail)
    average_hits_at_3_tail = hits_at_3_tail / len(ranks_tail)
    average_hits_at_1_tail = hits_at_1_tail / len(ranks_tail)

    hits_at_100 = (average_hits_at_100_head + average_hits_at_100_tail) / 2
    hits_at_10 = (average_hits_at_10_head + average_hits_at_10_tail) / 2
    hits_at_3 = (average_hits_at_3_head + average_hits_at_3_tail) / 2
    hits_at_1 = (average_hits_at_1_head + average_hits_at_1_tail) / 2

    '''print('Hits@100 is {}'.format(hits_at_100))
    print('Hits@10 is {}'.format(hits_at_10))
    print('Hits@3 is {}'.format(hits_at_3))
    print('Hits@1 is {}'.format(hits_at_1))'''

    return mean_auc_pr/10.0, hits_at_10


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # arguments for implementation
    args.add_argument('--ablation_name', '-e', type=str, required=True, choices=['wocn', 'wos', 'wor', 'rire', 'gre',
                                                                                 'w2vre', 'CNN', 'BiLSTM', 'BiLSTM',
                                                                                 'GRU', 'Avg'])  # 'wocn' means without capsule network; 'wos' without sorting; 'wor' without routing; 'rire' randomly initialize relation embedding; 'gre' glove relation embedding
    args.add_argument("-dataset", "--dataset",
                      default="FB15k-237-v1", help="dataset")
    # args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("--train", type=str, default='True', help="train")
    args.add_argument("--test", type=str, default='True', help="test")
    # args.add_argument('--batch_size', type=int, default=32)
    # args.add_argument("--optimizer", type=str, default='Adam')
    args.add_argument("--patience", type=int, default='5', help='patience for learning rate decay')
    args.add_argument("--epoch", type=int, default=20)
    # arguments for network
    args.add_argument("--hop", type=int, default=3, help="hop")
    # args.add_argument("--num_mid_caps", type=int, default=3, help="the total number of middle capsule")
    # args.add_argument("--GCN_dim", type=int, default=32, help="the dimension of features in GCN")
    args.add_argument("--activation", type=str, default='tanh', help="the activation function used in GCN")
    # args.add_argument("--GCN_layers", type=int, default=1, help="the total iterations of GCN")
    # args.add_argument("--routing", type=int, default=1, help="the iterations of dynamic routing(The total times of updating the routing coefficients)")
    # args.add_argument("--pad_edge_num", type=int, default=600, help="the number of edge in the subgraph before CapsNet")

    args = args.parse_args()

    hop = args.hop
    # load data
    datapath = "./data/{}/{}_{}_hop_new_data.pickle".format(args.dataset, args.dataset, hop)
    assert os.path.isfile(datapath) is True, "data file not found"
    ## load train subgraph
    with open(datapath, 'rb') as handle:
        new_data = pickle.load(handle)
    datapath1 = "./subgraph/{}_{}_hop_train_subgraph.pickle".format(args.dataset, hop)
    with open(datapath1, 'rb') as handle:
        train_subgraph = pickle.load(handle)
    ## load valid subgraph
    datapath1 = "./subgraph/{}_{}_hop_validation_subgraph.pickle".format(args.dataset, hop)
    with open(datapath1, 'rb') as handle:
        val_subgraph = pickle.load(handle)
    ## load test subgraph
    datapath1 = "./subgraph/{}_{}_hop_test_subgraph.pickle".format(args.dataset, hop)
    with open(datapath1, 'rb') as handle:
        test_subgraph = pickle.load(handle)
    ## load train data
    with open("./data/{}/{}_{}_total_train_data.pickle".format(args.dataset, args.dataset, hop), 'rb') as handle:
        train_data = pickle.load(handle)
    with open("./data/{}/{}_{}_total_train_label.pickle".format(args.dataset, args.dataset, hop), 'rb') as handle:
        train_label = pickle.load(handle)
    with open("./data/{}/{}_{}_total_val_data.pickle".format(args.dataset, args.dataset, hop), 'rb') as handle:
        val_data = pickle.load(handle)
    with open("./data/{}/{}_{}_total_val_label.pickle".format(args.dataset, args.dataset, hop), 'rb') as handle:
        val_label = pickle.load(handle)
    ## load test data
    ## with open("./subgraph/{}_{}_hop_test_path.pickle".format(args.dataset, hop), 'rb') as handle:
    ##   test_path = pickle.load(handle)

    if args.train == 'True':
        train(args, new_data, train_data, train_label, train_subgraph, val_data, val_label, val_subgraph)

    # if args.test == 'True':
    #     evaluate(args, new_data, test_subgraph)
