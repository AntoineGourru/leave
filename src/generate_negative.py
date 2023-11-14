import numpy as np
import torch
import random
import collections

# Code taken from M. Buyle - https://github.com/aida-ugent/FIPR
def nd_array_to_tuple(data):
    res = tuple(map(tuple, data))
    return res

# Code taken from M. Buyle - https://github.com/aida-ugent/FIPR
def get_negative_edges(graph):
    edges = np.array(list(graph.edges()))
    neg_edges = generate_negative_edges(edges)
    return neg_edges

# Code taken from M. Buyle - https://github.com/aida-ugent/FIPR
def generate_negative_edges(data):
    lhs_ids = np.unique(data[:, 0])
    rhs_ids = np.unique(data[:, 1])
    nb_lhs_ids = lhs_ids.shape[0]
    nb_rhs_ids = rhs_ids.shape[0]
    lhs_id_to_idx = dict(zip(lhs_ids, np.arange(nb_lhs_ids)))
    rhs_id_to_idx = dict(zip(rhs_ids, np.arange(nb_rhs_ids)))
    nb_negative_samples = data.shape[0]

    # Find linear indexes for the data in a simplified, 0-indexed matrix.
    simplified_data = np.array([(lhs_id_to_idx[edge[0]], rhs_id_to_idx[edge[1]]) for edge in data])
    data_lin_idx = np.ravel_multi_index((simplified_data[:, 0], simplified_data[:, 1]), dims=(nb_lhs_ids, nb_rhs_ids))

    negative_samples = []
    current_nb_negative_samples = 0
    while current_nb_negative_samples < nb_negative_samples:
        # Sample a bunch of edges.
        nb_left_to_sample = nb_negative_samples - current_nb_negative_samples
        lhs_samples = np.random.randint(low=0, high=nb_lhs_ids, size=nb_left_to_sample)
        rhs_samples = np.random.randint(low=0, high=nb_rhs_ids, size=nb_left_to_sample)

        # Check if they are negative by comparing their linear indices.
        candidate_lin_idx = np.ravel_multi_index((lhs_samples, rhs_samples), dims=(nb_lhs_ids, nb_rhs_ids))
        actual_negative_lin_idx = np.setdiff1d(candidate_lin_idx, data_lin_idx)

        # Keep the actually negative samples.
        actual_negative_samples = np.unravel_index(actual_negative_lin_idx, shape=(nb_lhs_ids, nb_rhs_ids))

        # Before storing them, convert the indices to ids.
        sampled_lhs_ids = lhs_ids[actual_negative_samples[0]]
        sampled_rhs_ids = rhs_ids[actual_negative_samples[1]]
        actual_negative_samples = np.vstack((sampled_lhs_ids, sampled_rhs_ids)).T
        negative_samples.append(actual_negative_samples)
        current_nb_negative_samples += actual_negative_samples.shape[0]

    return np.vstack(negative_samples)


def prepare_pairs(data, sens_attr, link_labels):
    pos_pairs = []
    neg_pairs = []

    source_att = []
    target_att = []
    source_id = []
    target_id = []
    for i, j in data:
        source_id.append(i)
        target_id.append(j)
        source_att.append(sens_attr[i])
        target_att.append(sens_attr[j])

    # pair_att = abs(np.array(source_att) - np.array(target_att))
    pair_att = np.array(source_att) != np.array(target_att)
    #pair_att = torch.ne(torch.tensor(source_att), torch.tensor(target_att))
    #for gen in data:
    #    if abs(sens_attr[gen[0]] - sens_attr[gen[1]]) == 0:  # pairs with same sens-attr in negative examples
    #        neg_pairs.append(gen)
    #    else:
    #        pos_pairs.append(gen)

    # neg_pairs = random.choices(neg_pairs, k=len(pos_pairs))  # get equal number of positive and negative samples for training
    # sens_label_pairs = [1 for i in pos_pairs] + [0 for i in neg_pairs]  # labeling according to sens attr
    # data_pairs = pos_pairs + neg_pairs


    # now get link info for the pairs (code to be cleaned)
    # index_list = [(i, j) for i, j in data]
    # index_list is the same thing as data
    # data_pairs_list = [(i, j) for i, j in data_pairs]
    # pairs_link_info = []
    #for pair in data_pairs_list:
        # find pair in index_list and get label from train_labels_all
    #    idx_pair = index_list.index(pair)
    #    pairs_link_info.append(link_labels[idx_pair])

    # n1, n2 = list(zip(*data_pairs))
    # data_list_zipped = list(zip(n1, n2, pairs_link_info, sens_label_pairs))
    # data_stack = torch.vstack([torch.Tensor(data_list_zipped)])

    # x_1, x_2, y, s = torch.split(data_stack, 1, dim=1)

    return torch.Tensor(source_id), torch.Tensor(target_id), torch.Tensor(link_labels), torch.Tensor(pair_att)


def test_pairs(data, sens_attr, link_labels):

    pos_pairs = []
    neg_pairs = []
    for gen in data:
        if abs(sens_attr[gen[0]] - sens_attr[gen[1]]) == 0:  # pairs with same sens-attr in negative examples
            neg_pairs.append(gen)
        else:
            pos_pairs.append(gen)

    # neg_pairs = random.choices(neg_pairs,k=len(pos_pairs)) # get equal number
    # of positive and negative samples for training
    data_pairs = pos_pairs + neg_pairs
    sens_label_pairs = [1 for i in pos_pairs] + [0 for i in neg_pairs]

    # pair_list = [(i, j) for i, j in data]
    index_list = [(i, j) for i, j in data]
    pairs_link_info = []

    for pair in data_pairs:
        idx_pair = index_list.index(pair)
        # data_pairs_list = [(i, j) for i, j in data_pairs]
        pairs_link_info.append(link_labels[idx_pair])

    n1, n2 = list(zip(*data_pairs))
    data_list_zipped = list(zip(n1, n2, pairs_link_info, sens_label_pairs))
    data_stack = torch.vstack([torch.Tensor(data_list_zipped)])

    x_1, x_2, y, s = torch.split(data_stack, 1, dim=1)

    return x_1, x_2, y, s
