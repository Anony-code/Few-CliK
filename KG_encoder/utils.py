import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_add
from torch_geometric.data import Data
import random

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def load_data_1(file_path):

    print("load data from {}".format(file_path))

    with open(os.path.join(file_path, 'ent_ids_1.tsv')) as f:
        entity2id = dict()

        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[int(eid)] = entity

    with open(os.path.join(file_path, 'relations_id.tsv')) as f:
        relation2id = dict()

        for line in f:
            rid, relation = line.strip().split('\t')
            relation2id[int(rid)] = relation

    with open(os.path.join(file_path, 'ref_ent_ids_newid.tsv')) as f:
        cross_links = []

        for line in f:
            le, re = line.strip().split('\t')
            cross_links.append((int(le), int(re)))

        cross_links = random.choices(cross_links, k=int(len(cross_links)*0.1))
        cross_links = np.array(cross_links)

    triples = read_triples_id(os.path.join(file_path, 'triples_1_basic.tsv'))
    triples_meta_train = read_triples_id(os.path.join(file_path, 'triples_1_meta_train.tsv'))
    triples_meta_valid = read_triples_id(os.path.join(file_path, 'triples_1_meta_valid.tsv'))

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_triples_basic: {}'.format(len(triples)))
    print('num_cross_links: {}'.format(len(cross_links)))
    print('num_train_triples: {}'.format(len(triples_meta_train)))
    print('num_valid_triples: {}'.format(len(triples_meta_valid)))

    return entity2id, relation2id, triples, cross_links, triples_meta_train, triples_meta_valid

def load_data_2(file_path):

    print("load data from {}".format(file_path))

    with open(os.path.join(file_path, 'ent_ids_2_newid.tsv')) as f:
        entity2id = dict()

        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[int(eid)] = entity

    with open(os.path.join(file_path, 'relations_id.tsv')) as f:
        relation2id = dict()

        for line in f:
            rid, relation = line.strip().split('\t')
            relation2id[int(rid)] = relation

    triples_basic_train = read_triples_id(os.path.join(file_path, 'triples_2_basic_train.tsv'))
    triples_basic_valid = read_triples_id(os.path.join(file_path, 'triples_2_basic_valid.tsv'))
    triples_meta_train = read_triples_id(os.path.join(file_path, 'triples_2_meta_train.tsv'))
    triples_meta_valid = read_triples_id(os.path.join(file_path, 'triples_2_meta_valid.tsv'))
    triples_meta_test = read_triples_id(os.path.join(file_path, 'triples_2_meta_test.tsv'))

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_basic_triples_train: {}'.format(len(triples_basic_train)))
    print('num_basic_triples_valid: {}'.format(len(triples_basic_valid)))
    print('num_meta_train_triples: {}'.format(len(triples_meta_train)))
    print('num_meta_valid_triples: {}'.format(len(triples_meta_valid)))
    print('num_meta_test_triples: {}'.format(len(triples_meta_test)))

    return entity2id, relation2id, triples_basic_train, triples_basic_valid, triples_meta_train, triples_meta_valid, triples_meta_test

def read_triplets(file_path, entity2id, relation2id):
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return np.array(triplets)

def read_triples_id(file_path):
    triples = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triples.append((int(head), int(relation), int(tail)))

    return np.array(triples)

def sample_edge_uniform(n_triples, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triples)
    return np.random.choice(all_edges, sample_size, replace=False)

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def negative_sampling_alignment(train, k, entity2id_1, entity2id_2):
    t = len(train)
    L = np.ones((t, k)) * (train[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (train[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))

    neg2_left = np.random.choice(len(entity2id_1.keys()), t * k)
    neg_right = np.random.choice(np.arange(len(entity2id_1.keys()), len(entity2id_1.keys())+ len(entity2id_2.keys())), t * k)

    return torch.LongTensor(neg_left), torch.LongTensor(neg_right), torch.LongTensor(neg2_left), torch.LongTensor(neg2_right)


def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm

def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_entity, num_rels, negative_rate):
    """
        Get training graph and signals
        First perform edge neighborhood sampling on graph, then perform negative
        sampling to generate negative samples
    """

    edges = sample_edge_uniform(len(triplets), sample_size)

    # Select sampled edges
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)

    src = torch.tensor(src[graph_split_ids], dtype = torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype = torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype = torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    return data

def build_test_graph(num_nodes, num_rels, triplets):
    src, rel, dst = triplets.transpose()

    src = torch.from_numpy(src)
    rel = torch.from_numpy(rel)
    dst = torch.from_numpy(dst)

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(np.arange(num_nodes))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, num_nodes, num_rels)

    return data

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

# return MRR (filtered), and Hits @ (1, 3, 10)
def calc_mrr(embedding, w, test_triplets, all_triplets, hits=[]):
    with torch.no_grad():
        
        num_entity = len(embedding)

        ranks_s = []
        ranks_o = []

        head_relation_triplets = all_triplets[:, :2]
        tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)

        for test_triplet in tqdm(test_triplets):

            # Perturb object
            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            subject_relation = test_triplet[:2]
            delete_index = torch.sum(head_relation_triplets == subject_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, object_.view(-1)))
            
            emb_ar = embedding[subject] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)
            
            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)
            
            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_s.append(sort_and_rank(score, target))

            # Perturb subject
            object_ = test_triplet[2]
            relation = test_triplet[1]
            subject = test_triplet[0]

            object_relation = torch.tensor([object_, relation])
            delete_index = torch.sum(tail_relation_triplets == object_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 0].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, subject.view(-1)))

            emb_ar = embedding[object_] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_o.append(sort_and_rank(score, target))

        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))
            
    return mrr.item()

def meta_processing(input):
    input = input.tolist()
    rel2trip = {}
    for item in input:
        if item[1] not in rel2trip.keys():
            rel2trip[item[1]] = [item]
        else:
            rel2trip[item[1]].append(item)

    return rel2trip


def cal_meta_mrr(test_triplets, all_triplets, rel2cand_meta_valid_list, num_entity, emb_layer, rel_emb, use_cuda):

    ranks_s = []
    ranks_o = []

    head_relation_triplets = all_triplets[:, :2]
    tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)

    for test_triplet in test_triplets:

        # Perturb object
        subject = test_triplet[0]
        relation = test_triplet[1]
        object_ = test_triplet[2]

        # subject_relation = test_triplet[:2]
        # delete_index = torch.sum(head_relation_triplets == subject_relation, dim=1)
        # delete_index = torch.nonzero(delete_index == 2).squeeze()
        #
        # delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
        # perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
        # perturb_entity_index_ = np.random.choice(perturb_entity_index, size=1000)
        perturb_entity_index_ = rel2cand_meta_valid_list
        perturb_entity_index = torch.from_numpy(perturb_entity_index_)
        perturb_entity_index = torch.cat((perturb_entity_index, object_.view(-1)))

        emb_ar = emb_layer(subject.cuda()) * rel_emb
        emb_ar = emb_ar.view(-1, 1, 1)

        emb_c = emb_layer(perturb_entity_index.cuda())
        emb_c = emb_c.transpose(0, 1).unsqueeze(1)

        out_prod = torch.bmm(emb_ar, emb_c)
        score = torch.sum(out_prod, dim=0)

        target = torch.tensor(len(perturb_entity_index) - 1)
        ranks_s.append(sort_and_rank(score, target.cuda()))

        # Perturb subject
        object_ = test_triplet[2]
        relation = test_triplet[1]
        subject = test_triplet[0]

        object_relation = torch.tensor([object_, relation])
        delete_index = torch.sum(tail_relation_triplets == object_relation, dim=1)
        delete_index = torch.nonzero(delete_index == 2).squeeze()

        delete_entity_index = all_triplets[delete_index, 0].view(-1).numpy()
        perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
        perturb_entity_index_ = np.random.choice(perturb_entity_index, size=1000)
        perturb_entity_index = torch.from_numpy(perturb_entity_index_)
        perturb_entity_index = torch.cat((perturb_entity_index, subject.view(-1)))

        emb_ar = emb_layer(object_.cuda()) * rel_emb
        emb_ar = emb_ar.view(-1, 1, 1)

        emb_c = emb_layer(perturb_entity_index.cuda())
        emb_c = emb_c.transpose(0, 1).unsqueeze(1)

        out_prod = torch.bmm(emb_ar, emb_c)
        score = torch.sum(out_prod, dim=0)
        score = torch.sigmoid(score)

        target = torch.tensor(len(perturb_entity_index) - 1)
        ranks_o.append(sort_and_rank(score, target.cuda()))

        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)
        ranks = torch.cat((ranks_s, ranks_o))

        return ranks, ranks_s, ranks_o

def cal_meta_mrr_oneway(test_triplets, all_triplets, rel2cand_meta_valid_list, num_entity, emb_layer, rel_emb, use_cuda):
    ranks_s = []

    head_relation_triplets = all_triplets[:, :2]
    tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)

    for test_triplet in test_triplets:

        # Perturb object
        subject = test_triplet[0]
        relation = test_triplet[1]
        object_ = test_triplet[2]

        # subject_relation = test_triplet[:2]
        # delete_index = torch.sum(head_relation_triplets == subject_relation, dim=1)
        # delete_index = torch.nonzero(delete_index == 2).squeeze()
        #
        # delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
        # perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
        # perturb_entity_index_ = np.random.choice(perturb_entity_index, size=1000)
        perturb_entity_index_ = np.asarray(rel2cand_meta_valid_list)
        perturb_entity_index = torch.from_numpy(perturb_entity_index_)
        perturb_entity_index = torch.cat((perturb_entity_index, object_.view(-1)))

        emb_ar = emb_layer(subject.cuda()) * rel_emb
        emb_ar = emb_ar.view(-1, 1, 1)

        emb_c = emb_layer(perturb_entity_index.cuda())
        emb_c = emb_c.transpose(0, 1).unsqueeze(1)

        out_prod = torch.bmm(emb_ar, emb_c)
        score = torch.sum(out_prod, dim=0)

        target = torch.tensor(len(perturb_entity_index) - 1)
        ranks_s.append(sort_and_rank(score, target.cuda()))

        ranks_s = torch.cat(ranks_s)

        return ranks_s