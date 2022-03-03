import argparse
import numpy as np
import torch
from utils import load_data_1, load_data_2
from models import RGCN

def main(args):

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)


    entity2id_1, relation2id, basic_triplets_1, cross_links, triples_1_meta_train, triples_1_meta_valid = load_data_1('../dataset/data/en_es')
    entity2id_2, relation2id, basic_triplets_2_train, basic_triplets_2_valid, \
    triplets_2_meta_train, triplets_2_meta_valid, triplets_2_meta_test = load_data_2('../dataset/data/en_es')

    all_train_triplets = np.concatenate((basic_triplets_1, basic_triplets_2_train))

    print('num_total_train_triples: {}'.format(len(all_train_triplets)))

    entity2id = {**entity2id_1, **entity2id_2}
    print('num_total_entity: {}'.format(len(entity2id.keys())))


    model = RGCN(len(entity2id), len(relation2id), num_bases=args.n_bases, dropout=args.dropout)

    if use_cuda:
        model.cuda()

    checkpoint = torch.load('best_mrr_model.pth')
    model.load_state_dict(checkpoint['state_dict'])

    emb = model.entity_embedding.weight.clone().detach().cpu()

    with open('emb.npy', 'wb') as f:
        np.save(f, emb)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')

    parser.add_argument("--graph-batch-size", type=int, default=30000)
    parser.add_argument("--graph-split-size", type=float, default=0.5)
    parser.add_argument("--negative-sample", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=10000)
    parser.add_argument("--evaluate-every", type=int, default=100)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-bases", type=int, default=4)

    parser.add_argument("--regularization", type=float, default=1e-2)
    parser.add_argument("--grad-norm", type=float, default=1.0)

    args = parser.parse_args()
    print(args)

    main(args)