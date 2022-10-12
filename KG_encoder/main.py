import argparse
import random
import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import torch.nn.functional as F


from utils import load_data_1, load_data_2, generate_sampled_graph_and_labels, build_test_graph, calc_mrr, \
    negative_sampling_alignment, meta_processing, cal_meta_mrr, cal_meta_mrr_oneway
from models import RGCN, MetaGNN, Embedding, RelationMetaLearner

logging.basicConfig(level=logging.INFO, filename='logging-en-es.txt', filemode='w')

def train(train_triplets, model, use_cuda, batch_size, split_size, negative_sample, reg_ratio, num_entities, num_relations):

    train_data = generate_sampled_graph_and_labels(train_triplets, batch_size, split_size, num_entities, num_relations,
                                                   negative_sample)

    if use_cuda:
        device = torch.device('cuda')
        train_data.to(device)

    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
    loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + \
           reg_ratio * model.reg_loss(entity_embedding)
    return loss

def train_alignment(cross_links, entity2id_1, entity2id_2, model, gamma):
    k = 5
    neg_left, neg_right, neg2_left, neg2_right = negative_sampling_alignment(cross_links, k, entity2id_1, entity2id_2)
    cross_links = torch.from_numpy(cross_links)
    t = len(cross_links)

    left_x = model.retrival_embedding(cross_links[:,0].cuda())
    right_x = model.retrival_embedding(cross_links[:,1].cuda())
    A = torch.sum(torch.abs(left_x - right_x), 1)
    neg_l_x = model.retrival_embedding(neg_left.cuda())
    neg_r_x = model.retrival_embedding(neg_right.cuda())
    B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
    C = - torch.reshape(B, (t, k))
    D = A + gamma
    L1 = F.relu(torch.add(C, torch.reshape(D, (t, 1))))
    neg_l_x = model.retrival_embedding(neg2_left.cuda())
    neg_r_x = model.retrival_embedding(neg2_right.cuda())
    B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
    C = - torch.reshape(B, (t, k))
    L2 = F.relu(torch.add(C, torch.reshape(D, (t, 1))))

    return 0.01*(torch.sum(L1) + torch.sum(L2)) / (2.0 * k * t)

def valid(valid_triplets, model, test_graph, all_triplets):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr = calc_mrr(entity_embedding, model.relation_embedding, valid_triplets, all_triplets, hits=[1, 3, 10])

    return mrr

def test(test_triplets, model, test_graph, all_triplets):

    entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type, test_graph.edge_norm)
    mrr = calc_mrr(entity_embedding, model.relation_embedding, test_triplets, all_triplets, hits=[1, 3, 10])

    return mrr

def main(args):

    model_pretrain = True

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    best_mrr = 0

    entity2id_1, relation2id, basic_triplets_1, cross_links, triples_1_meta_train, triples_1_meta_valid = load_data_1('../dataset/data/en_es')
    entity2id_2, relation2id, basic_triplets_2_train, basic_triplets_2_valid, \
    triplets_2_meta_train, triplets_2_meta_valid, triplets_2_meta_test = load_data_2('../dataset/data/en_es')

    all_triplets = torch.LongTensor(np.concatenate((basic_triplets_1, basic_triplets_2_train, basic_triplets_2_valid)))

    all_train_triplets = np.concatenate((basic_triplets_1, basic_triplets_2_train))

    print('num_total_train_triples: {}'.format(len(all_train_triplets)))

    entity2id = {**entity2id_1, **entity2id_2}
    print('num_total_entity: {}'.format(len(entity2id.keys())))

    test_graph = build_test_graph(len(entity2id), len(relation2id), all_train_triplets)
    valid_triplets = torch.LongTensor(basic_triplets_2_valid)
    # test_triplets = torch.LongTensor(triplets_test)

    model = RGCN(len(entity2id), len(relation2id), num_bases=args.n_bases, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(model)

    if use_cuda:
        model.cuda()

    if model_pretrain == True:

        for epoch in trange(1, (args.n_epochs + 1), desc='Epochs', position=0):

            model.train()
            optimizer.zero_grad()

            loss = train(all_train_triplets, model, use_cuda, batch_size=args.graph_batch_size, split_size=args.graph_split_size,
                negative_sample=args.negative_sample, reg_ratio = args.regularization,
                         num_entities=len(entity2id), num_relations=len(relation2id))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()

            ######
            if epoch % 100 == 0:

                tqdm.write("Train Loss {} at epoch {}".format(loss, epoch))

                for _ in trange(0, 5, desc='Epochs', position=0):
                    loss_alignment = train_alignment(cross_links, entity2id_1, entity2id_2, model, gamma=3.0)
                    loss_alignment.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                    optimizer.step()
                    print("the loss of alignment is {}".format(loss_alignment))
            ######

            if epoch % args.evaluate_every == 0:

                if use_cuda:
                    model.cpu()

                model.eval()
                valid_mrr = valid(valid_triplets, model, test_graph, all_triplets)

                if valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                                'best_mrr_model.pth')

                if use_cuda:
                    model.cuda()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RGCN')
    
    parser.add_argument("--graph-batch-size", type=int, default=30000)
    parser.add_argument("--graph-split-size", type=float, default=0.5)
    parser.add_argument("--negative-sample", type=int, default=1)

    parser.add_argument("--n-epochs", type=int, default=10000)
    parser.add_argument("--evaluate-every", type=int, default=100)
    
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-bases", type=int, default=4)
    parser.add_argument("--score-function", type=str, default='DistMult')
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1)
    
    parser.add_argument("--regularization", type=float, default=1e-2)
    parser.add_argument("--grad-norm", type=float, default=1.0)

    args = parser.parse_args()
    print(args)

    main(args)