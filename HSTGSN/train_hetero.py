import argparse
from functools import partial
import time
import pickle

import dgl
import dgl.function as fn

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, dataset, Subset
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split

from dataloading_hetero import TrafficAssignmentDataset
from modelzoo_hetero import GraphRNN_Transformer, TransformerModel_Homo, TransformerModel_Hetero, RegressionBranch, RGNN
from utils_log import createLogger
log = createLogger(__name__)
log.info('in train.py')

parser = argparse.ArgumentParser()
# parser.add_argument("--map_name", help="map_name", type=str)
# parser.add_argument("--gpu", help="gpu", type=int, default = -1)

parser.add_argument("--train_dir", help="train_dir", type=str)
parser.add_argument("--save_dir", help="save_dir", type=str, default = 'toy')

# parser.add_argument("--train_data_dir_list", help="train_data_dir_list", type=str, nargs="+")
# parser.add_argument("--train_num_sample_list", help="train_num_sample_list", type=int, nargs="+")
parser.add_argument("--train_ratio", help="train_ratio", type=float, default = 0.8)
parser.add_argument("--test_ratio", help="test_ratio", type=float, default = 0.2)

parser.add_argument("--model_idx", help="model_idx", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 32)
parser.add_argument("--n_sample", help="n_sample", type=int)
parser.add_argument("--epoch", help="epoch", type=int, default = 100)
parser.add_argument("--lr", help="lr", type=float, default = 0.001)
parser.add_argument("--train_loss", help="train_loss function", type=int, default = 1)
parser.add_argument("--need_train", help="need_train", type=int, default = 1)
parser.add_argument("--need_test", help="need_test", type=int, default = 1)
parser.add_argument("--need_residue", help="need_residue", type=int)
parser.add_argument("--residue_norm", help="residue_norm", type=float, default = 1e-5)
parser.add_argument("--pred_type", help="prediction type", type=str, default = 'ratio')

parser.add_argument("--rnn_name", help="rnn_name", type=str)
parser.add_argument("--model_name", help="model_name", type=str)

args = parser.parse_args()
need_residue = args.need_residue
residue_norm = args.residue_norm
n_sample = args.n_sample

device = 'cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu'

def collate(samples):
    n = len(samples)
    res_density = [samples[i].edges['connect'].data['res_density'] for i in range(n)]
    res_ratio = [samples[i].edges['connect'].data['res_ratio'] for i in range(n)]
    res_CA = [samples[i].edges['connect'].data['res_CA'] for i in range(n)]
    res_CD = [samples[i].edges['connect'].data['res_CD'] for i in range(n)]

    batched_graph = dgl.batch(samples)
    return batched_graph, torch.vstack((*res_density,)), torch.vstack((*res_ratio,)), torch.vstack((*res_CA,)), torch.vstack((*res_CD,))

def train(train_dataloader, test_dataloader, model, args):
    time_total = 0
    min_test_loss = 1e9
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[300], gamma=0.2)
    
    train_loss_list, test_loss_list = [], []
    for e in range(args.epoch):
        t0 = time.time()
        model.train()
        train_loss_epoch, train_residue = [], []
        for _, (g, res_density, res_ratio, res_CA, res_CD) in enumerate(train_dataloader):
            g = g.to(device)
            res_density = res_density.to(device)
            res_ratio = res_ratio.to(device)
            res_CA = res_CA.to(device)
            res_CD = res_CD.to(device)
            
            # Forward
            pred_flow, pred_cnt = model(g, args.batch_size, device)
            if args.pred_type == 'ratio':

                weight = (res_ratio + 0.1).detach()
                weight = weight / weight.max()
                log.debug(res_CA.shape, res_CD.shape, l=10)
                log.debug(pred_cnt.shape, pred_cnt.shape, l=10)
                # asdf
                train_loss = torch.mean(torch.abs(weight*(pred_flow.squeeze() - res_ratio))) #+ \
                                # 0.01*torch.mean(torch.abs(weight*(pred_cnt[:,:,0].squeeze()*1e3 - res_CA))) + \
                                # 0.01*torch.mean(torch.abs(weight*(pred_cnt[:,:,1].squeeze()*1e3 - res_CD)))
                train_loss_epoch.append(torch.mean(torch.abs(pred_flow.squeeze() - res_ratio)).detach().cpu().numpy())
                
            # Backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        # save training history
        train_loss_list.append(np.mean(np.hstack(train_loss_epoch)))
        time_total += time.time() - t0
        
        # test every 5 epoch
        if e % 5 == 0:
            test_loss_epoch = []
            model.eval()
            for _, (g, res_density, res_ratio, res_CA, res_CD) in enumerate(test_dataloader):
                g = g.to(device)
                res_density = res_density.to(device)
                res_ratio = res_ratio.to(device)
                res_CA = res_CA.to(device)
                res_CD = res_CD.to(device)
                
                # Forward
                pred_flow, pred_cnt = model(g, args.batch_size, device)
                
                # normal train_loss
                if args.pred_type == 'ratio':
                        
                    weight = (res_ratio + 0.1).detach()
                    weight = weight / weight.max()
                    test_loss_epoch.append(torch.mean(torch.abs(pred_flow.squeeze() - res_ratio)).detach().cpu().numpy())
                
            if np.mean(np.array(test_loss_epoch)) < min_test_loss:
                min_test_loss = np.mean(np.array(test_loss_epoch))
                # save model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    }, '../saved_model/{}_{}_{}_{}_{}.pt'.format(args.save_dir, \
                        args.train_ratio, args.test_ratio, args.rnn_name, args.model_name))
            
            if need_residue:
                # log.info('In epoch {}, train_loss_epoch: {:.3f}, test_loss_epoch: {:.3f}, residue: {:.3f}'.format(e, train_loss, np.mean(np.array(test_loss_epoch)), residue), l=1)
                log.info('In epoch {}, train_loss_epoch: {:.3f}, test_loss_epoch: {:.3f}, residue: {:.3f}'.format(e, np.mean(np.array(train_loss_epoch)), np.mean(np.array(test_loss_epoch)), residue), l=1)
            else:
                log.info('In epoch {}, train_loss_epoch: {:.3f}, test_loss_epoch: {:.3f}'.format(e, np.mean(np.array(train_loss_epoch)), np.mean(np.array(test_loss_epoch))), l=1)
                
        # save testing history
        test_loss_list.append(np.mean(np.hstack(test_loss_epoch)))

    result = dict()
    result['train_loss_list'] = np.vstack(train_loss_list)
    result['test_loss_list'] = np.vstack(test_loss_list)
    result['args'] = args
    result['time'] = time_total
    result['min_test_loss'] = min_test_loss
    with open('../saved_result/{}_{}_{}_{}_{}.pickle'.format(args.save_dir, \
        args.train_ratio, args.test_ratio, args.rnn_name, args.model_name), 'wb') as handle:
        pickle.dump(result, handle)

def test(test_dataloader, model, args):
    # save result in pickle
    checkpoint = torch.load('../saved_model/{}_{}_{}_{}_{}.pt'.format(args.save_dir, \
        args.train_ratio, args.test_ratio, args.rnn_name, args.model_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    all_res_flow, all_res_ratio, all_pred_flow = [], [], []
    for _, (g, res_density, res_ratio, res_CA, res_CD) in enumerate(test_dataloader):
        g = g.to(device)
        res_density = res_density.to(device)
        res_ratio = res_ratio.to(device)
        res_CA = res_CA.to(device)
        res_CD = res_CD.to(device)
        
        # Forward
        pred_flow, _ = model(g, args.batch_size, device)
        
        all_res_flow.append(res_density.detach().cpu().numpy())
        all_res_ratio.append(res_ratio.detach().cpu().numpy())
        all_pred_flow.append(pred_flow.detach().cpu().numpy())
    
    with open('../saved_result/{}_{}_{}_{}_{}.pickle'.format(args.save_dir, \
        args.train_ratio, args.test_ratio, args.rnn_name, args.model_name), 'rb') as handle:
        result = pickle.load(handle)
    
    print(np.vstack(all_res_flow).shape)
    result['all_gt_flow'] = np.vstack(all_res_flow)
    result['all_gt_ratio'] = np.vstack(all_res_ratio)
    result['all_pred'] = np.vstack(all_pred_flow)
    
    with open('../saved_result/{}_{}_{}_{}_{}.pickle'.format(args.save_dir, \
        args.train_ratio, args.test_ratio, args.rnn_name, args.model_name), 'wb') as handle:
        pickle.dump(result, handle)

if 'toy' in args.train_dir: in_feats = 4
if 'siouxfalls' in args.train_dir: in_feats = 24
if 'chicago' in args.train_dir: in_feats = 933
if 'anaheim' in args.train_dir: in_feats = 416


### change n_T when change the dataset

# training setting
n_T = 25

# in pre_emb_gnn
# n_head used in pre_emb_gnn and RGNN
# rnn_gnn_name = 'GRU' # 'LSTM' 'GRU' 'RGNN'
n_head = 4

emb_feats = 32 # 128
# pre_emb_gnn = TransformerModel_Homo(in_feats, emb_feats, n_head, n_T)
if args.model_name == 'hetero_2rnn': # add RNN after both node embedding and edge embedding
    pre_emb_gnn = TransformerModel_Hetero(in_feats, emb_feats, n_head, n_T)
# pre_emb_gnn = RegressionBranch(in_feats, 64, emb_feats*n_head, n_layer=3)
# in rnn_gnn_1

out_feats, n_rnn_layers, hidden_in_cell = 64, 2, False

if args.rnn_name == 'GRU':
    rnn_gnn_1 = nn.GRU(emb_feats*n_head, out_feats, n_rnn_layers, batch_first=True)
    rnn_gnn_2 = nn.GRU(out_feats*2, out_feats, n_rnn_layers, batch_first=True)
if args.rnn_name == 'LSTM':   
    rnn_gnn_1 = nn.LSTM(emb_feats*n_head, out_feats, n_rnn_layers, batch_first=True)
    rnn_gnn_2 = nn.LSTM(out_feats*2, out_feats, n_rnn_layers, batch_first=True)
if args.rnn_name == 'RGNN':   
    rnn_gnn_1 = RGNN(emb_feats*n_head, out_feats, n_rnn_layers, n_head, n_T, hidden_in_cell)
    rnn_gnn_2 = RGNN(emb_feats*n_head, out_feats, n_rnn_layers, n_head, n_T, hidden_in_cell)

if isinstance(rnn_gnn_1, nn.GRU): 
    out_fnn_ratio = RegressionBranch(out_feats*2+6, 64, 1, n_layer=2)
    out_fnn_cnt = RegressionBranch(out_feats*2+6, 64, 2, n_layer=2)
if isinstance(rnn_gnn_1, nn.LSTM): 
    out_fnn_ratio = RegressionBranch(out_feats*2+6, 64, 1, n_layer=2)
    out_fnn_cnt = RegressionBranch(out_feats*2+6, 64, 2, n_layer=2)
if isinstance(rnn_gnn_1, RGNN): 
    out_fnn_ratio = RegressionBranch(out_feats*n_head*2+6, 64, 1, n_layer=2)
    out_fnn_cnt = RegressionBranch(out_feats*2+6, 64, 2, n_layer=2)

model = GraphRNN_Transformer(pre_emb_gnn=pre_emb_gnn, rnn_gnn=(rnn_gnn_1, rnn_gnn_2), \
    out_fnn_ratio=out_fnn_ratio, out_fnn_cnt=out_fnn_cnt, n_T=n_T, device=device).to(device)

dataset = TrafficAssignmentDataset(args.train_dir, n_sample, n_T)
# train_idx, _ = train_test_split(list(range(n_sample)), train_size=args.train_ratio, shuffle=True, random_state=42)
# _, test_idx = train_test_split(list(range(n_sample)), train_size=0.8, shuffle=True, random_state=42)
# train_idx, _ = train_test_split(list(range(n_sample)), train_size=args.train_ratio, shuffle=False)
# _, test_idx = train_test_split(list(range(n_sample)), train_size=args.train_ratio, shuffle=False)
train_idx, test_idx = train_test_split(list(range(n_sample)), train_size=args.train_ratio, shuffle=True, random_state=43)
train_batch = Subset(dataset, train_idx)
test_batch = Subset(dataset, test_idx)

train_dataloader = DataLoader(train_batch, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
test_dataloader = DataLoader(test_batch, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

if args.need_train:
    train(train_dataloader, test_dataloader, model, args)
if args.need_test:
    test(test_dataloader, model, args)



