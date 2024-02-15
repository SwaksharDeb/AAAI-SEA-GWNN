import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layer import *
import torch
from sklearn.manifold import TSNE

class nof(nn.Module):
    def __init__(self, mydev, myf, max_degree, adj, gamma, nnode, nfeat, nlayers, hop, alpha, nhidden, nclass, dropout, lamda, variant):
        super(nof, self).__init__()
        self.myf=myf
        self.max_degree = max_degree
        self.adj=adj
        self.gamma=gamma
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(noflayer(nnode, nhidden, nhidden, hop, alpha, self.adj, self.max_degree, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.leakyrelu = nn.ELU()

    def forward(self, x):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        
        for i, con in enumerate(self.convs):
             layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
             layer_inner = self.act_fn(self.convs[0](layer_inner, _layers[0]))
             _layers.append(layer_inner)
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        mylast=layer_inner

        return F.log_softmax(layer_inner, dim=1)

if __name__ == '__main__':
    pass