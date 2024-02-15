from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils import *
from model import *
import uuid
import nni
from sklearn.preprocessing import normalize
import statistics
from config import configure

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--epochs', type=int, default=1000, help= 'Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--layer', type=int, default=1, help='Number of layers.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--variant', action='store_true', default=False, help=
    'GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help=
    'evaluation on test set.')
args = parser.parse_args()
print(args)

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)

def build_and_train(hype_space,datastr, splitstr,i):
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr,splitstr)

    adj = adj.to(device)
    features = features.to(device)
    max_degree = 55
    modelname = 'nof'
    np.random.seed(hype_space['seed'])
    torch.manual_seed(hype_space['seed'])
    modeltype = {'nof':nof}
    model = modeltype[modelname](
        mydev=device,
        myf=hype_space['myf'],
        max_degree=max_degree,
        adj=adj,
        gamma=hype_space['gamma'],
        nnode=features.shape[0],
        nfeat=features.shape[1],
        nlayers=hype_space['layer'],
        hop=hype_space['hop'],
        alpha=hype_space['alpha'],
        nhidden=hype_space['hidden'],
        nclass=int(labels.max()) + 1,
        dropout=hype_space['dropout'],
        lamda=hype_space['lambda'],
        variant=args.variant).to(device)
        
    optimizer = optim.Adam([{'params': model.params1, 'weight_decay': hype_space['wd1']},
                            {'params': model.params2, 'weight_decay': hype_space['wd2']},],
                           lr=0.1*hype_space['lr_rate_mul'])

    def train():
        model.train()
        optimizer.zero_grad()
        output=model(features)
        acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
        loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
        loss_train.backward()
        optimizer.step()
        return loss_train.item(), acc_train.item()

    def validate():
        model.eval()
        with torch.no_grad():
            output=model(features)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
            acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
            return loss_val.item(), acc_val.item()

    def test():
        model.eval()
        with torch.no_grad():
            output=model(features)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
            acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
            return loss_test.item(), acc_test.item()

    bad_counter = 0
    best = 999999999
    best_test_acc = 0
    best_val_acc = 0
    lrnow = optimizer.state_dict()['param_groups'][0]['lr']
    for epoch in range(args.epochs):
        t0 = time.time()
        loss_tra, acc_tra = train()
        loss_val,acc_val = validate()
        loss_tes,acc_tes = test()
        pred_label = F.log_softmax(model(features),dim=1)
        preds = pred_label.max(1)[1].type_as(labels)
        np.save('coralabel.npy', preds.detach().cpu())
        
        if epoch==0 or loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            best_test_acc = acc_tes
            best_test_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        
        print('Epoch:%4d|train loss:%.3f acc:%.2f|val loss:%.3f acc:%.2f|test loss:%.3f acc:%.2f|best acc:%.4f epoch:%d'
                %(epoch + 1,loss_tra,acc_tra * 100,loss_val,acc_val * 100,loss_tes,acc_tes*100,best_test_acc*100,best_test_epoch))
        print('{} seconds'.format(time.time() - t0))
        if bad_counter == args.patience:
            print('early stopping at epoch %d'%epoch)
            break
    nni.report_final_result(best_test_acc)
    return best_test_acc

t_total = time.time()
acc_list = []
for i in range(10):
    datastr = args.data
    splitstr ='../splits/'+args.data+'_split_0.6_0.2_'+str(i)+'.npz'
    params = configure(datastr)
    acc = build_and_train(params, datastr, splitstr,str(i))
    acc_list.append(acc)
    print(i,": {:.2f}".format(acc_list[-1]))
print("Test acc.:{:.4f}".format(np.mean(acc_list)))

print("Test std.:{:.4f}".format(np.std(acc_list)))
    
