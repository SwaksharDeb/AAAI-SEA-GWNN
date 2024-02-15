from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from scipy import sparse 
import numpy as np
from torch.nn.functional import normalize

class noflayer(nn.Module):
    def __init__(self, nnode, in_features, out_features, hop, alpha, adj, max_degree, residual=False, variant=False):
        super(noflayer, self).__init__()
        self.variant = variant
        self.nnode = nnode
        self.alpha_ = alpha
        self.hop = hop
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.adj_1 = adj
        self.adj_2 = torch.sparse.mm(self.adj_1, self.adj_1)
        self.adj_3 = torch.sparse.mm(self.adj_2, self.adj_1)
        self.a = nn.Parameter(torch.empty(size=(2*self.in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.alpha = 0.2
        self.alp = 0.9
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.f = Parameter(torch.ones(self.nnode))
        self.weight_matrix_att = Parameter(torch.FloatTensor(self.in_features, self.in_features))
        self.temp = Parameter(torch.Tensor(self.hop+1))
        #self.cheb = Parameter(torch.Tensor(self.hop))
        Temp = self.alp*np.arange(self.hop+1)
        Temp = Temp/np.sum(np.abs(Temp))
        self.cheb = Parameter(torch.tensor(Temp))
        self.weight_matrix_att_prime_1 = torch.nn.Parameter(torch.Tensor(self.in_features, self.in_features))
        self.weight_matrix_att_prime_2 = torch.nn.Parameter(torch.Tensor(self.in_features, 1))
        self.weight_matrix_att_prime_3 = torch.nn.Parameter(torch.Tensor(self.in_features, 1))

        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix_att)
        torch.nn.init.xavier_uniform_(self.weight)
        #torch.nn.init.xavier_uniform_(self.weight_matrix2)
        self.temp.data.fill_(0.0)
        for k in range(self.hop+1):
            self.cheb.data[k] = self.alp*(1-self.alp)**k
        self.cheb.data[-1] = (1-self.alp)**2
        torch.nn.init.xavier_uniform_(self.weight_matrix_att_prime_1)
        torch.nn.init.xavier_uniform_(self.weight_matrix_att_prime_2)
        torch.nn.init.xavier_uniform_(self.weight_matrix_att_prime_3)

    def attention(self, feature, adj):
        feat_1 = torch.matmul(feature, self.a[:self.in_features, :].clone())
        feat_2 = torch.matmul(feature, self.a[self.in_features:, :].clone())
        
        e = feat_1 + feat_2.T
        e = self.leakyrelu(e)
        
        nonzero_indices = adj.coalesce().indices().t()
        values = e[nonzero_indices[:, 0], nonzero_indices[:, 1]]
        att = torch.sparse.FloatTensor(nonzero_indices.t(), values, torch.Size([adj.shape[0], adj.shape[1]]))
        att = torch.sparse.softmax(att,dim=1)
        
        U = att.clone()
        P = 0.5*U.clone()
        return U, P
    
    def forward_lifting_bases(self, feature, h0, P, U, adj):
        coe = torch.sigmoid(self.temp)
        cheb_coe = torch.sigmoid(self.cheb)
        adj_value = adj.coalesce().values()
        P_value = P.coalesce().values()
        values = torch.mul(adj_value,P_value)
        nonzero_indices = adj.coalesce().indices().t()
        Adj_handmard_P = torch.sparse.FloatTensor(nonzero_indices.t(), values, torch.Size([adj.shape[0], adj.shape[1]]))
        rowsum = torch.sparse.sum(Adj_handmard_P,1).to_dense()
        rowsum_present = rowsum
        update = feature
        for step in range(self.hop):
            update = torch.sparse.mm(U,update)
            if self.alpha_== None:
                feat_even_bar = coe[0]*feature + update
            else:
                feat_even_bar = update
            if step >= 1:
                rowsum = (cheb_coe[step-1])*rowsum
            feat_odd_bar = update - torch.einsum('ij, i -> ij',  feat_even_bar, rowsum)
            if step == 0:
                if self.alpha_ == None:
                    feat_fuse = coe[1]*feat_even_bar + (1-coe[1])*(feat_odd_bar)
                    feat_prime = coe[2]*feature + (1-coe[2])*feat_fuse
                else:
                    feat_fuse = self.alpha_*feat_even_bar + (1-self.alpha_)*(feat_odd_bar)
                    feat_prime = self.alpha_*feature + (1-self.alpha_)*feat_fuse  
            else:
                if  self.alpha_ == None:
                    feat_fuse = coe[1]*feat_even_bar + (1-coe[1])*(feat_odd_bar)
                    feat_prime = coe[2]*feat_prime + (1-coe[2])*feat_fuse
                else:
                    feat_fuse = self.alpha_*feat_even_bar + (1-self.alpha_)*(feat_odd_bar)
                    feat_prime = self.alpha_*feat_prime + (1-self.alpha_)*feat_fuse
        return feat_prime
    
    def lifting_block(self, hi, h0, adj):
        U,P = self.attention(hi, adj)
        feat_prime = self.forward_lifting_bases(hi, h0, P, U, adj)
        return feat_prime
    
    def forward(self, input, h0):
        #hi = input + torch.matmul(input, self.weight)
        hi = input
        output = self.lifting_block(hi, h0, self.adj_1)
        return output
