import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
import numpy as np
from utils import uniform

class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases)

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type, edge_norm):
        x = self.entity_embedding(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = self.conv2(x, edge_index, edge_type, edge_norm)
        
        return x

    def retrival_embedding(self, entity):
        return self.entity_embedding(entity)

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))

class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)


    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)


    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)


class MetaGNN(nn.Module):

    def __init__(self, entity_embedding_dim, relation_embedding_dim, num_entities, num_relations, args, entity_embedding, relation_embedding):

        super(MetaGNN, self).__init__()

        self.args = args

        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.entity_embedding = nn.Embedding(num_entities, entity_embedding_dim)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, relation_embedding_dim))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.entity_embedding.weight.data.copy_(entity_embedding.clone().detach())
        self.relation_embedding.data.copy_(relation_embedding.clone().detach())

        self.dropout = nn.Dropout(args.dropout)

        self.gnn = GENConv(self.entity_embedding_dim + self.relation_embedding_dim, self.entity_embedding_dim,
                           self.num_relations * 2, num_bases=self.args.n_bases, root_weight=False, bias=False)
        self.score_function = self.args.score_function

    def forward(self, triplets, use_cuda):

        src, rel, dst = triplets.transpose()
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))

        rel_index = np.concatenate((rel, rel))

        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + self.num_relations))

        # Torch
        node_id = torch.LongTensor(uniq_v)
        edge_index = torch.stack((
            torch.LongTensor(src),
            torch.LongTensor(dst)
        ))
        edge_type = torch.LongTensor(rel)

        if use_cuda:
            node_id = node_id.cuda()
            edge_index = edge_index.cuda()
            edge_type = edge_type.cuda()

        x = self.entity_embedding(node_id)
        rel_emb = self.relation_embedding[rel_index]

        emb = self.gnn(x, edge_index, edge_type, rel_emb, edge_norm=None)

        return emb

    def score_loss(self, triplets, target, use_cuda):

        head_embeddings = self.entity_embedding(triplets[:, 0])
        relation_embeddings = self.relation_embedding[triplets[:, 1]]
        tail_embeddings = self.entity_embedding(triplets[:, 2])

        len_positive_triplets = int(len(target) / (self.args.negative_sample_meta + 1))

        if self.score_function == 'DistMult':

            score = head_embeddings * relation_embeddings * tail_embeddings
            score = torch.sum(score, dim = 1)

            positive_score = score[:len_positive_triplets]
            negative_score = score[len_positive_triplets:]

        else:

            raise ValueError("Score Function Name <{}> is Wrong".format(self.score_function))

        y = torch.ones(len_positive_triplets * self.args.negative_sample_meta)

        if use_cuda:
            y = y.cuda()

        positive_score = positive_score.repeat(self.args.negative_sample_meta)

        loss = F.margin_ranking_loss(positive_score, negative_score, y, margin=self.args.margin)

        return loss


class GENConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(GENConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(int(in_channels / 2), out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, rel_emb, edge_norm=None, size=None):

        self.rel_emb = rel_emb

        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_index_i, edge_type, edge_norm):

        x_j = torch.cat((
            x_j,
            self.rel_emb
        ), dim=1)

        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):

        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias

        if (self.root is None) and (self.bias is None):
            return aggr_out

        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)


