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
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn import metrics
from torch.utils.data import DataLoader
from matplotlib import cm


from src.model import InTriCaps_refined_interpretation
from src.utils import process_data


def get_relation_embedding():
    path = './data/nell-v2/'
    with open(os.path.join(path, 'id2embed.pkl'), 'rb') as f:
        id2embed = pickle.load(f)
    return id2embed


def tsne_dimension_reduction(tensor):
    '''dimension reduction method by TSNE.
        input: pytorch tensor for reduction; shape [batch, dimensionality]
        output: the numpy tensor after reduction; shape [batch, reduced dimensionality]'''
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(tensor.data.cpu().numpy())
    return low_dim_embs


def plot(numpy_array):
    X_to_cat = []
    Y_to_cat = []
    c_list = []
    label = []
    marker_list = []
    marker_full_list = ['o', '^', 'v', 's', 'p', '*']
    id_relation_ditc = {13: 'state located in geopolitical location', 19: 'country located in geopolitical location',
                        22: 'city located in geopolitical location', 48: 'person leads geopolitical location',
                        0: 'country also known as', 10: 'organization also known as',
                        52: 'team also known as', 69: 'city also known as',
                        12: 'sports game team', 28: 'team plays sports', 50: 'sports uses equipment',
                        85: 'athelete led sports team', 21: 'organization headquartered in state or province',
                        25: 'organization headquartered in in city', 70: 'organization headquartered in country',
                        82: 'person has residence in geopolitical location', 60: 'building located in city',
                        44: 'state located in country', 38: 'stadium located in city', 37: 'city located in state',
                        33: 'city located in country'}
    label_list = []

    group1, group2, group3, group4, group5 = [13, 19, 22, 48], [0, 10, 52, 69], [12, 28, 50, 85], [21, 25, 70], [82, 60, 44, 38, 37, 33]
    j = 0
    for i in group1:
        X_to_cat.append(numpy_array[i, 0])
        Y_to_cat.append(numpy_array[i, 1])
        c_list.append('r')
        marker_list.append(marker_full_list[j])
        label.append(i)
        label_list.append(id_relation_ditc[i])
        j += 1

    j = 0
    for i in group2:
        X_to_cat.append(numpy_array[i, 0])
        Y_to_cat.append(numpy_array[i, 1])
        c_list.append('b')
        marker_list.append(marker_full_list[j])
        label.append(i)
        label_list.append(id_relation_ditc[i])
        j += 1

    j = 0
    for i in group3:
        X_to_cat.append(numpy_array[i, 0])
        Y_to_cat.append(numpy_array[i, 1])
        c_list.append('g')
        marker_list.append(marker_full_list[j])
        label.append(i)
        label_list.append(id_relation_ditc[i])
        j += 1

    j = 0
    for i in group4:
        X_to_cat.append(numpy_array[i, 0])
        Y_to_cat.append(numpy_array[i, 1])
        c_list.append('y')
        marker_list.append(marker_full_list[j])
        label.append(i)
        label_list.append(id_relation_ditc[i])
        j += 1

    j = 0
    for i in group5:
        X_to_cat.append(numpy_array[i, 0])
        Y_to_cat.append(numpy_array[i, 1])
        c_list.append('c')
        marker_list.append(marker_full_list[j])
        label.append(i)
        label_list.append(id_relation_ditc[i])
        j += 1

    fig, ax = plt.subplots()

    for xp, yp, cp, mp, lp in zip(X_to_cat, Y_to_cat, c_list, marker_list, label_list):
        ax.scatter([xp], [yp], c=cp, marker=mp, label=lp)
        # plt.legend(loc='best', framealpha=1)

    # for k in range(len(X_to_cat)):
    #     plt.annotate(label[k], xy = (X_to_cat[k], Y_to_cat[k]), xytext = (X_to_cat[k]+0.1, Y_to_cat[k]+0.1))
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.show()


def vis_re():
    # get relation embedding of BERT
    relation_embed = get_relation_embedding()
    relation_embed_tensor = []
    for key in relation_embed:
        relation_embed_tensor.append(relation_embed[key])
    relation_embed_tensor = torch.cat(relation_embed_tensor)
    # get relation embedding of rire
    model = torch.load('./pretrained_model/rire.pkl')
    relation_embed_rire = model.state_dict()['GetSubgraphFeature.relation_embed']

    low_dim_embs_bert = tsne_dimension_reduction(relation_embed_tensor)  # transform dimensionality
    low_dim_embs_rire = tsne_dimension_reduction(relation_embed_rire)
    plot(low_dim_embs_bert)
    plot(low_dim_embs_rire)


def get_relation_embedding_inter(args):
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
        preds, _, _, _, _, _, _, _ = model(train_indices, val_subgraph)
        all_preds += preds.squeeze(1).detach().cpu().tolist()
        all_labels += train_values.squeeze(1).tolist()
    auc = metrics.roc_auc_score(all_labels, all_preds)
    return auc


def train(args, new_data, train_data, train_label, train_subgraph, val_data, val_label, val_subgraph):
    relation_embed = get_relation_embedding_inter(args)
    # hyperparameters
    otm = args.optimizer
    lr = args.lr
    GCN_dim = args.GCN_dim
    act_func = args.activation
    GCN_layers = args.GCN_layers
    num_mid_caps = args.num_mid_caps
    routing = args.routing
    batch_size = args.batch_size
    pad_edge_num = args.pad_edge_num

    node_dim = 2 * (args.hop + 1)

    model = InTriCaps_refined_interpretation(args, GCN_dim, node_dim, relation_embed,
                                             act_func, GCN_layers, num_mid_caps, routing, pad_edge_num).cuda()

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
    train_iterator = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)

    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5, last_epoch=-1)
    margin_loss = torch.nn.SoftMarginLoss()
    epoch_losses = []   # recording losses of all epochs
    best_loss = 100000
    best_auc = 0
    save_iteration = int(train_data.shape[0]//args.batch_size//3)  # validate models in multiple iterations instead of only the last iteration

    for epoch in range(args.epoch):
        model.train()
        start_time = time.time()
        epoch_loss = []

        for iters, batch in enumerate(train_iterator):
            start_time_per_batch = time.time()
            train_indices, train_values = batch
            train_indices = torch.LongTensor(train_indices).cuda()
            train_values = torch.FloatTensor(train_values).cuda()

            preds, rc_mid, rc_deci, graph, num_edge, node_subgraph, adj_h2e_subgraph, adj_t2e_subgraph = model(train_indices, train_subgraph)

            optimizer.zero_grad()
            loss = margin_loss(preds.view(-1), train_values.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.data.item())
            end_time_per_batch = time.time()

            if (iters+1) % save_iteration == 0:
                with torch.no_grad():
                    val_auc = valid_auc(model, margin_loss, args.batch_size, val_data, val_label, val_subgraph)
                if best_auc < val_auc:
                    best_auc = val_auc
                    print('found best model, save to ./pretrained_model/interpretation_{}.pkl'.format(args.dataset))
                    torch.save(model, './pretrained_model/interpretation_{}.pkl'.format(args.dataset))

        scheduler.step()
        print("epoch {}, average loss {}, epoch_time {}"
              .format(epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))
        # validate the model in the last iteration in each epoch
        with torch.no_grad():
            val_auc = valid_auc(model, margin_loss, args.batch_size, val_data, val_label, val_subgraph)
        if best_auc < val_auc:
            best_auc = val_auc
            print('found best model, save to ./pretrained_model/interpretation_{}.pkl'.format(args.dataset))
            torch.save(model, './pretrained_model/interpretation_{}.pkl'.format(args.dataset))
        # print('the best auc so far:{}'.format(best_auc))


def evaluate(args, new_data, test_subgraph):
    # hyperparameters
    otm = args.optimizer
    lr = args.lr
    GCN_dim = args.GCN_dim
    act_func = args.activation
    GCN_layers = args.GCN_layers
    num_mid_caps = args.num_mid_caps
    routing = args.routing
    batch_size = args.batch_size
    pad_edge_num = args.pad_edge_num

    node_dim = 2 * (args.hop + 1)

    model = torch.load('./pretrained_model/interpretation_{}.pkl'.format(args.dataset))

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

        total_iter = int(len(true_triplets) // args.batch_size) + 1
        if len(true_triplets) % args.batch_size == 0:
            total_iter -= 1
        all_preds = []
        all_labels = []
        for i in range(total_iter):
            if i == total_iter - 1:
                true_indices = true_triplets[i*args.batch_size:, :]
                true_values = true_label[i*args.batch_size:, :]
                neg_indices = neg_triplets[i*args.batch_size:, :]
                neg_values = neg_label[i*args.batch_size:, :]
            else:
                true_indices = true_triplets[i*args.batch_size : (i+1)*args.batch_size, :]
                true_values = true_label[i*args.batch_size : (i+1)*args.batch_size, :]
                neg_indices = neg_triplets[i*args.batch_size : (i+1)*args.batch_size, :]
                neg_values = neg_label[i*args.batch_size : (i+1)*args.batch_size, :]
            true_indices = torch.LongTensor(true_indices).cuda()
            neg_indices = torch.LongTensor(neg_indices).cuda()

            preds, rc_mid, rc_deci, graph, num_edge, node_subgraph, adj_h2e_subgraph, adj_t2e_subgraph = model(true_indices, test_subgraph)
            all_preds += preds.squeeze(1).detach().cpu().tolist()
            all_labels += true_values.tolist()

            preds, rc_mid, rc_deci, graph, num_edge, node_subgraph, adj_h2e_subgraph, adj_t2e_subgraph = model(neg_indices, test_subgraph)
            all_preds += preds.squeeze(1).detach().cpu().tolist()
            all_labels += neg_values.tolist()

        auc = metrics.roc_auc_score(all_labels, all_preds)
        auc_pr = metrics.average_precision_score(all_labels, all_preds)
        print('{}th auc_pr:{}'.format(kk, auc_pr))
        mean_auc_pr += auc_pr
        # print(all_preds)
    print('mean_auc_pr:{}'.format(mean_auc_pr/10.0))

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

        scores_head, _, _, _, _, _, _, _ = model(torch.LongTensor(new_x_batch_head).cuda(), test_subgraph)
        sorted_scores_head, sorted_indices_head = torch.sort(scores_head.view(-1), dim=-1, descending=True)

        ranks_head.append(np.where(sorted_indices_head.cpu().numpy() == rand_head)[0][0] + 1)
        reciprocal_ranks_head.append(1.0 / ranks_head[-1])

        scores_tail, _, _, _, _, _, _, _ = model(torch.LongTensor(new_x_batch_tail).cuda(), test_subgraph)
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

    print('Hits@100 is {}'.format(hits_at_100))
    print('Hits@10 is {}'.format(hits_at_10))
    print('Hits@3 is {}'.format(hits_at_3))
    print('Hits@1 is {}'.format(hits_at_1))


def vis_interpretation(args):
    '''we only visualize example in nell-v2 dataset. The default argument hyperparameters are the best'''
    case_id = args.case_id
    hop = args.hop

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

    if args.train == 'True':
        train(args, new_data, train_data, train_label, train_subgraph, val_data, val_label, val_subgraph)

    if args.test == 'True':
        evaluate(args, new_data, test_subgraph)

    model = torch.load('./pretrained_model/interpretation_{}.pkl'.format(args.dataset)).cuda().eval()

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

    true_triplets = []
    for i in range(len(new_total_test_head)):
        true_triplets.append(new_total_test_head[i][0, :])
    true_label = np.expand_dims(np.array([1 for i in range(len(true_triplets))]), 1)
    true_triplets = np.array(true_triplets)

    true_indices = true_triplets[case_id : (case_id+1), :]
    true_indices = torch.LongTensor(true_indices).cuda()
    preds, rc_mid, rc_deci, graph, num_edge, node_subgraph, adj_h2e_subgraph, adj_t2e_subgraph = model(true_indices, test_subgraph)
    # print(len(new_total_test_head))
    print('case id is {}'.format(case_id))
    # print('prediction is {}'.format(preds))
    # print('there are {} edges in the subgraph'.format(num_edge))
    # print('graph.shape', graph.shape)
    # print('node_subgraph.length', len(node_subgraph))
    # print('adj_h2e_subgraph.shape', adj_h2e_subgraph.shape)
    # print('adj_t2e_subgraph.shape', adj_t2e_subgraph.shape)
    # print(true_indices)
    # print(graph)
    vis_triplets = [(2716, 2614, 12), (2716, 2781, 12), (2781, 2797, 34), (3713, 2797, 34), (3713, 2614, 12), (3713, 2571, 12), (2571, 2614, 12)]
    index_list = []
    for vis_tri in vis_triplets:
        itemindex = np.where((graph == vis_tri).all(axis=1))
        index_list.append(itemindex[0][0])
    rc_mid_all = torch.cat([rc_mid[0], rc_mid[1], rc_mid[2]], 0)
    rc_vis_tri = []
    for index in index_list:
        rc_vis_tri.append(rc_mid_all[index])
    rc_vis_tri = torch.cat(rc_vis_tri, 0)
    rc_vis_tri = np.array(rc_vis_tri.detach().cpu().tolist())
    vis_tri_name = ('1', '2', '3', '4', '5', '6', '7')
    plt.barh(vis_tri_name, rc_vis_tri)
    plt.show()


def vis_hyper(args):
    GCNlayer = [1, 2, 3, 4, 5]
    FB_v1_hits_GCNlayer = [66.52, 67.39, 60.87, 61.30, 59.13]
    FB_v2_hits_GCNlayer = [73.68, 69.87, 73.95, 70.66, 68.95]
    groups = [2, 3, 4, 5, 6]
    FB_v1_hits_groups = [62.61, 67.39, 61.30, 62.17, 60.43]
    FB_v2_hits_groups = [69.08, 73.95, 72.24, 71.05, 69.34]
    routing = [1, 2, 3, 4, 5]
    FB_v1_hits_routing = [64.35, 62.17, 67.39, 60.87, 59.57]
    FB_v2_hits_routing = [73.95, 72.24, 71.97, 71.58, 70.39]

    plt.figure()

    plt.subplot(2, 3, 1)
    plt.plot(GCNlayer, FB_v1_hits_GCNlayer, markerfacecolor='blue', marker='o')
    plt.xlabel('number of layer of GCN', fontsize=15)
    plt.ylabel('Hits@10 on FB15k-237-v1', fontsize=15)
    plt.tick_params(axis='both')
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(groups, FB_v1_hits_groups, markerfacecolor='blue', marker='o')
    plt.xlabel('number of groups', fontsize=15)
    plt.ylabel('Hits@10 on FB15k-237-v1', fontsize=15)
    plt.tick_params(axis='both')
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(routing, FB_v1_hits_routing, markerfacecolor='blue', marker='o')
    plt.xlabel('iterations of routing', fontsize=15)
    plt.ylabel('Hits@10 on FB15k-237-v1', fontsize=15)
    plt.tick_params(axis='both')
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(GCNlayer, FB_v2_hits_GCNlayer, color='orange', markerfacecolor='orange', marker='o')
    plt.xlabel('number of layer of GCN', fontsize=15)
    plt.ylabel('Hits@10 on FB15k-237-v2', fontsize=15)
    plt.tick_params(axis='both')
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(groups, FB_v2_hits_groups, color='orange', markerfacecolor='orange', marker='o')
    plt.xlabel('number of groups', fontsize=15)
    plt.ylabel('Hits@10 on FB15k-237-v2', fontsize=15)
    plt.tick_params(axis='both')
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(routing, FB_v2_hits_routing, color='orange', markerfacecolor='orange', marker='o')
    plt.xlabel('iterations of routing', fontsize=15)
    plt.ylabel('Hits@10 on FB15k-237-v2', fontsize=15)
    plt.tick_params(axis='both')
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--vis_name', required=True, type=str, choices=['relation_embedding', 'interpretation', 'hyperparams'])
    args.add_argument('--case_id', type=int)
    # arguments for implementation
    args.add_argument("-dataset", "--dataset",
                      default="nell-v2", help="dataset")
    args.add_argument("--train", type=str, default='True', help="train")
    args.add_argument("--test", type=str, default='True', help="test")
    # args.add_argument("--pretrained_model", type=str, help="the name of the pretrained model")
    args.add_argument("--epoch", type=int, default=20)
    args.add_argument("--patience", type=int, default='5', help='patience for learning rate decay')
    args.add_argument("--exp_name", type=str, default='pad', help='the name of the experiment')
    # hyperparams
    args.add_argument("--optimizer", type=str, default='RMSprop')
    args.add_argument("-l", "--lr", type=float, default=0.0001)
    args.add_argument("--GCN_dim", type=int, default=16, help="the dimension of features in GCN")
    args.add_argument("--activation", type=str, default='tanh', help="the activation function used in GCN")
    args.add_argument("--GCN_layers", type=str, default=1, help="the total iterations of GCN")
    args.add_argument("--num_mid_caps", type=int, default=3, help="the total number of middle capsule")
    args.add_argument("--routing", type=int, default=1, help="the iterations of dynamic routing(The total times of "
                                                             "updating the routing coefficients)")
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument("--pad_edge_num", type=int, default=200, help="the number of edge in the subgraph before CapsNet")
    args.add_argument("--hop", type=int, default=3, help="hop")
    args = args.parse_args()

    if args.vis_name == 'relation_embedding':
        vis_re()

    if args.vis_name == 'interpretation':
        vis_interpretation(args)

    if args.vis_name == 'hyperparams':
        vis_hyper(args)

