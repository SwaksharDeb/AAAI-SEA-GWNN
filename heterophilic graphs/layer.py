from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from scipy import sparse 
import numpy as np
from torch.nn.functional import normalize
from torch_geometric.utils import remove_self_loops


class noflayer(nn.Module):
    def __init__(self, nnode, in_features, out_features, hop, adj, max_degree, residual=False, variant=False):
        super(noflayer, self).__init__()
        self.max_degree = 170
        self.variant = variant
        self.hop = hop
        self.nnode = nnode
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        #self.adj = adj.to_dense()
        self.adj = adj
        edge_index = self.adj.coalesce().indices()
        edge_weight = self.adj.coalesce().values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        self.adj = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([self.adj.shape[0], self.adj.shape[1]]))
        
        #self.adj = torch.sparse.mm(self.adj, self.adj)
        #self.rowsum_adj = torch.sum(self.adj,1)
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.a = nn.Parameter(torch.empty(size=(2*self.in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(2*self.in_features, 1)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.act_fn = nn.ReLU()
        self.f = Parameter(torch.ones(self.nnode))
        self.weight_matrix_att = Parameter(torch.FloatTensor(self.in_features, self.in_features))
        self.temp = Parameter(torch.Tensor(4))
        self.weight_matrix_att_prime_1 = torch.nn.Parameter(torch.Tensor(self.in_features, self.in_features))
        self.weight_matrix_att_prime_2 = torch.nn.Parameter(torch.Tensor(self.in_features, 1))
        self.weight_matrix_att_prime_3 = torch.nn.Parameter(torch.Tensor(self.in_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix_att)
        torch.nn.init.xavier_uniform_(self.weight)
        #torch.nn.init.xavier_uniform_(self.weight_matrix2)
        self.temp.data.fill_(0.0)
        torch.nn.init.xavier_uniform_(self.weight_matrix_att_prime_1)
        torch.nn.init.xavier_uniform_(self.weight_matrix_att_prime_2)
        torch.nn.init.xavier_uniform_(self.weight_matrix_att_prime_3)
        

    def attention(self, feature):
        feature = torch.mm(feature, self.weight_matrix_att) 
        feat_1 = torch.matmul(feature, self.a[:self.in_features, :].clone())
        feat_2 = torch.matmul(feature, self.a[self.in_features:, :].clone())
        e = feat_1 + feat_2.T
        e = self.leakyrelu(e)
        nonzero_indices = self.adj.coalesce().indices().t()
        values = e[nonzero_indices[:, 0], nonzero_indices[:, 1]]
        att = torch.sparse.FloatTensor(nonzero_indices.t(), values, torch.Size([self.adj.shape[0], self.adj.shape[1]]))
        att = torch.sparse.softmax(att,dim=1)
        U = att.clone()
        
        P = 0.5*att.clone()
        return U,P, att
    
    def forward_lifting_bases(self, feature, P, U, adj):
        coe = torch.sigmoid(self.temp)
        adj_value = adj.coalesce().values()
        P_value = P.coalesce().values()
        values = torch.mul(adj_value,P_value)
        nonzero_indices = adj.coalesce().indices().t()
        Adj_handmard_P = torch.sparse.FloatTensor(nonzero_indices.t(), values, torch.Size([adj.shape[0], adj.shape[1]]))
        rowsum = torch.sparse.sum(Adj_handmard_P,1)
        update = feature
        for step in range(self.hop):
            update = torch.sparse.mm(U,update)
            feat_even_bar = coe[0]*feature + update
            feat_odd_bar = update - torch.einsum('ij, i -> ij',  feat_even_bar, rowsum.to_dense())
            if step == 0:
                feat_prime = feat_odd_bar
            else:
                feat_fuse = feat_odd_bar
                feat_prime = coe[2]*feat_prime + (1-coe[2])*feat_fuse
        return feat_prime
    
    
    def forward(self, input, h0):
        hi = input
        U,P,att = self.attention(hi)
        output = self.forward_lifting_bases(hi, P, U, self.adj)  
        return output
