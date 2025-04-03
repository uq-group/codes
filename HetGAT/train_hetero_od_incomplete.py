import argparse

import os
import pickle5 as pickle
from itertools import product
import matplotlib.pyplot as plt
import time
import dgl
from dgl.data import DGLDataset
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, dataset
from torch.utils.data import ConcatDataset
from torch.autograd.functional import jacobian
from torch.optim.lr_scheduler import MultiStepLR
from modelzoo_hetero_od import *

parser = argparse.ArgumentParser()
parser.add_argument("--map_name", help="map_name", type=str)
parser.add_argument("--gpu", help="gpu", type=int, default = -1)

parser.add_argument("--train_data_dir_list", help="train_data_dir_list", type=str, nargs="+")
parser.add_argument("--train_num_sample_list", help="train_num_sample_list", type=int, nargs="+")
parser.add_argument("--train_ratio", help="train_ratio", type=float, default = 0.8)
parser.add_argument("--test_ratio", help="test_ratio", type=float, default = 0.2)

parser.add_argument("--model_idx", help="model_idx", type=int)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 64)
parser.add_argument("--epoch", help="epoch", type=int, default = 200)
parser.add_argument("--lr", help="lr", type=float, default = 0.001)
parser.add_argument("--miss", help="missing data", type=float, default = 0)
parser.add_argument("--conservation_loss", help="conservation_loss", type=int)
parser.add_argument("--loss", help="loss function", type=int, default = 1)

args = parser.parse_args()
train_num_sample_list = args.train_num_sample_list
train_data_dir_list = args.train_data_dir_list
map_name = args.map_name

# dgl.seed(234)
# torch.manual_seed(234)
# np.random.seed(234)
torch.set_printoptions(linewidth=200)

if args.gpu >= 0:
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'
    
def collate(samples):
    n = len(samples)
    res_ratio = [samples[i].edges['connect'].data['res_ratio'] for i in range(n)]
    res_flow = [samples[i].edges['connect'].data['res_flow'] for i in range(n)]
    batched_graph = dgl.batch(samples)
    return batched_graph, torch.vstack((*res_ratio,)), torch.vstack((*res_flow,))

class TrafficAssignmentDataset(DGLDataset):
    def __init__(self, num_sample, data_dir, n_node, map_name):
        self.num_sample = num_sample
        self.data_dir = data_dir
        self.n_node = n_node
        self.map_name = map_name
        self.cap_ratio = {'Sioux':1000, 'EMA':1000, 'Anaheim':800, 'ANAHEIM':800}
        super(TrafficAssignmentDataset, self).__init__(name='bridge', hash_key={num_sample, data_dir})
     
    def process(self):
        self.graphs = []
        
        coord_all = np.loadtxt('./{}/coord.csv'.format(map_name), delimiter=' ')
        coord = coord_all[:, 3:]
        
        # load data from imp folder
        for graph_id in range(self.num_sample):
            # load each graph
            print(graph_id)
            with open(self.data_dir+'/data_{}.pickle'.format(graph_id), 'rb') as handle:
                graph_data = pickle.load(handle)
                
            ratio, flow_list = [], []
            flow, capacity = graph_data['flow'], graph_data['capacity']
            for k1, k2 in zip(graph_data['ca_list'][:,0].squeeze(), graph_data['ca_list'][:,1].squeeze()):
                ratio.append([k1, k2, flow[(int(k1)), int(k2)]/capacity[(int(k1)), int(k2)] ])
                flow_list.append([k1, k2, flow[(int(k1)), int(k2)]])
            ratio = np.vstack(ratio)
            flow_list = np.vstack(flow_list)
            
            assert(np.array_equal(ratio[:, 0].squeeze(), graph_data['ca_list'][:, 0].squeeze()))
            assert(np.array_equal(ratio[:, 1].squeeze(), graph_data['ca_list'][:, 1].squeeze()))
            
            # considier OD as node feature
            od_node_feat = torch.tensor(graph_data['demand_matrix'], dtype=torch.float32)
            coord_feat = torch.tensor(coord, dtype=torch.float32)
            connect_edge_feat = torch.tensor(graph_data['ca_list'][:, 2:], dtype=torch.float32)
            connect_edge_res_ratio = torch.tensor(ratio[:, 2:], dtype=torch.float32)
            connect_edge_res_flow = torch.tensor(flow_list[:, 2:], dtype=torch.float32)
            
            # mask the od matrix
            if args.miss > 1e-3:
                mask = torch.rand(*od_node_feat.shape)
                od_node_feat_incomplete = od_node_feat*mask.ge(args.miss)
            else:
                od_node_feat_incomplete = od_node_feat
            
            # data is directed, make it undirected
            src_ncn = torch.tensor(graph_data['ca_list'][:,0].squeeze(), dtype=torch.int32)
            dst_ncn = torch.tensor(graph_data['ca_list'][:,1].squeeze(), dtype=torch.int32)

            ## now nodn only connect edge that od > 0 --> considering add edge feature in the future
            src_nodn = torch.tensor(graph_data['od_list'][:,0].squeeze(), dtype=torch.int32)
            dst_nodn = torch.tensor(graph_data['od_list'][:,1].squeeze(), dtype=torch.int32)
            
            hetergraph_data = {
                ('node', 'connect', 'node'): (src_ncn, dst_ncn),
                ('node', 'od', 'node'): (src_nodn, dst_nodn)
            }
            g = dgl.heterograph(hetergraph_data)
            
            # -- dont delete --> may be a different way --> considier OD as edge feature
            # connect_edge_feat = torch.tensor(pickle_data['ca_list'][:, 2:], dtype=torch.float32)
            # od_edge_feat = torch.tensor(pickle_data['od_list'][:, 2:], dtype=torch.float32)
            # connect_edge_res = torch.tensor(ratio[:, 2:], dtype=torch.float32)
            
            g.nodes['node'].data['feat'] = od_node_feat_incomplete / 1e3 # od vector
            g.nodes['node'].data['feat_T'] = od_node_feat_incomplete.T / 1e3
            g.ndata['feat_whole'] = od_node_feat / 1e3
            g.ndata['feat_whole_T'] = od_node_feat.T / 1e3
            g.nodes['node'].data['coord'] = coord_feat
            g.edges['connect'].data['feat'] = connect_edge_feat # capacity
            g.edges['connect'].data['capacity'] = connect_edge_feat[:, 0:1]*self.cap_ratio[map_name] # capacity
            g.edges['connect'].data['res_ratio'] = connect_edge_res_ratio # result_ratio
            g.edges['connect'].data['res_flow'] = connect_edge_res_flow # result_flow

            self.graphs.append(g)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)

def train(train_dataloader, test_dataloader, model, args):
    time_total = 0
    min_test_loss = 1e9
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[300], gamma=0.2)
    
    train_loss_list, train_residue_list, test_loss_list, test_residue_list = [], [], [], []
    for e in range(args.epoch):
        t0 = time.time()
        model.train()
        train_loss, train_residue = [], []
        for _, (g, edge_ratio, edge_flow) in enumerate(train_dataloader):
            g = g.to(device)
            edge_ratio = edge_ratio.to(device)
            edge_flow = edge_flow.to(device)
            
            node_feat = g.nodes['node'].data['feat']
            edge_feat = g.edges['connect'].data['feat']

            # ratio
            pred_ratio = model(g, node_feat, edge_feat)
            # flow
            pred_flow = pred_ratio * g.edges['connect'].data['capacity']
            
            # normal loss
            if args.loss == 1:
                loss = torch.mean(torch.abs(pred_ratio - edge_ratio)) + torch.mean(torch.abs(pred_flow - edge_flow))*0.005
            if args.loss == 2:
                loss = torch.mean((edge_ratio + 1.0)*torch.abs(pred_ratio - edge_ratio)) + torch.mean(torch.abs(pred_flow - edge_flow))*0.005
                
            train_loss.append(torch.mean(torch.abs(pred_ratio - edge_ratio)).detach().cpu().numpy())
            
            if args.conservation_loss:
                # conservation loss
                g.edges['connect'].data['flow'] = pred_ratio * g.edges['connect'].data['feat'][:, 0:1]
                rg = dgl.reverse(g, copy_ndata=False, copy_edata=True)
                
                g['connect'].update_all(fn.copy_e('flow', 'm'), fn.sum('m', 'in_flow'), etype='connect')
                rg['connect'].update_all(fn.copy_e('flow', 'm'), fn.sum('m', 'out_flow'), etype='connect')
                
                # out_flow - inflow = out_demand (sum(X[0, :])) - in_demand(sum(X[:, 0]))
                residue = rg.nodes['node'].data['out_flow'] - g.nodes['node'].data['in_flow'] - \
                    (torch.sum(g.nodes['node'].data['feat_whole'], dim=1).view(-1, 1) - \
                    torch.sum(g.nodes['node'].data['feat_whole_T'], dim=1).view(-1, 1))

                if args.conservation_loss:  loss += 0.05*torch.mean(torch.abs(residue))
                train_residue.append(torch.mean(torch.abs(residue)).detach().cpu().numpy())
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save training history
        train_loss_list.append(np.mean(np.hstack(train_loss)))
        if args.conservation_loss:  train_residue_list.append(np.mean(np.hstack(train_residue)))
        
        time_total += time.time() - t0
        
        if e % 5 == 0:
            test_loss, test_residue = [], []
            model.eval()
            for _, (g, edge_ratio, edge_flow) in enumerate(test_dataloader):
                g = g.to(device)
                edge_ratio = edge_ratio.to(device)
                edge_flow = edge_flow.to(device)
                
                node_feat = g.nodes['node'].data['feat']
                edge_feat = g.edges['connect'].data['feat']

                # Forward
                pred_ratio = model(g, node_feat, edge_feat)
                
                # normal loss
                l = torch.mean(torch.abs(pred_ratio - edge_ratio))
                test_loss.append(torch.mean(torch.abs(pred_ratio - edge_ratio)).detach().cpu().numpy())
                
                # conservation loss
                g.edges['connect'].data['flow'] = pred_ratio * g.edges['connect'].data['feat'][:, 0:1]
                rg = dgl.reverse(g, copy_ndata=False, copy_edata=True)
                
                g['connect'].update_all(fn.copy_e('flow', 'm'), fn.sum('m', 'in_flow'), etype='connect')
                rg['connect'].update_all(fn.copy_e('flow', 'm'), fn.sum('m', 'out_flow'), etype='connect')
                
                # out_flow - inflow = out_demand (sum(X[0, :])) - in_demand(sum(X[:, 0]))
                res_test = rg.nodes['node'].data['out_flow'] - g.nodes['node'].data['in_flow'] - \
                    (torch.sum(g.nodes['node'].data['feat_whole'], dim=1).view(-1, 1) - \
                    torch.sum(g.nodes['node'].data['feat_whole_T'], dim=1).view(-1, 1))
                test_residue.append(torch.mean(torch.abs(res_test)).detach().cpu().numpy())

                # Compute accuracy on training/validation/test
                test_loss.append(l.cpu().detach().numpy().item())
            
            if np.mean(np.array(test_loss)) < min_test_loss:
                min_test_loss = np.mean(np.array(test_loss))
                torch.save(model.state_dict(), 'model/{}_{}_{}_{}_{}_{}_{}.pth'.format(' '.join(args.train_data_dir_list), \
                    args.model_idx, args.train_ratio, args.test_ratio, args.conservation_loss, args.loss, args.miss))
            print('In epoch {}, train_loss: {:.3f}, test_loss: {:.3f}, residue_test: {:.3f}'.format(e, \
                loss, np.mean(np.array(test_loss)), torch.mean(torch.abs(res_test))))
        
        # save testing history
        test_loss_list.append(np.mean(np.hstack(test_loss)))
        if args.conservation_loss:  test_residue_list.append(np.mean(np.hstack(test_residue)))
        
    # save result in pickle
    # gt, pred
    all_gt, all_pred, all_conservation = [], [], []
    model.eval()
    for _, (g, edge_ratio, edge_flow) in enumerate(test_dataloader):
        g = g.to(device)
        edge_ratio = edge_ratio.to(device)
        edge_flow = edge_flow.to(device)
        
        node_feat = g.nodes['node'].data['feat']
        edge_feat = g.edges['connect'].data['feat']

        # Forward
        pred_ratio = model(g, node_feat, edge_feat)
        
        # conservation loss
        g.edges['connect'].data['flow'] = pred_ratio * g.edges['connect'].data['feat'][:, 0:1]
        rg = dgl.reverse(g, copy_ndata=False, copy_edata=True)
        
        g['connect'].update_all(fn.copy_e('flow', 'm'), fn.sum('m', 'in_flow'), etype='connect')
        rg['connect'].update_all(fn.copy_e('flow', 'm'), fn.sum('m', 'out_flow'), etype='connect')
        
        res_test = rg.nodes['node'].data['out_flow'] - g.nodes['node'].data['in_flow'] - \
            (torch.sum(g.nodes['node'].data['feat_whole'], dim=1).view(-1, 1) - \
            torch.sum(g.nodes['node'].data['feat_whole_T'], dim=1).view(-1, 1))
        
        all_gt.append(edge_ratio.detach().cpu().numpy())
        all_pred.append(pred_ratio.detach().cpu().numpy())
        all_conservation.append(res_test.detach().cpu().numpy())
    
    result = dict()
    all_gt = np.vstack(all_gt)
    all_pred = np.vstack(all_pred)
    result['train_loss_list'] = np.vstack(train_loss_list)
    if args.conservation_loss:  result['train_residue_list'] = np.vstack(train_residue_list)
    result['test_loss_list'] = np.vstack(test_loss_list)
    if args.conservation_loss:  result['test_residue_list'] = np.vstack(test_residue_list)
    result['all_conservation'] = np.vstack(all_conservation)
    result['all_res'] = np.column_stack((all_gt, all_pred)) # gt, pred
    result['args'] = args
    result['time'] = time_total
    result['min_test_loss'] = min_test_loss
    with open('./saved_result/{}_{}_{}_{}_{}_{}_{}.pickle'.format(' '.join(args.train_data_dir_list), \
        args.model_idx, args.train_ratio, args.test_ratio, args.conservation_loss, args.loss, args.miss), 'wb') as handle:
        pickle.dump(result, handle)
  
if 'Sioux' in map_name:
    n_node = 24
    map_name = 'Sioux'
if 'EMA' in map_name:
    n_node = 74
    map_name = 'EMA'
if 'Anaheim' in map_name:
    n_node = 416
    map_name = 'Anaheim'
if 'ANAHEIM' in map_name:
    n_node = 416
    map_name = 'ANAHEIM'
    
if args.model_idx == 11: model = TransformerModel_Hetero2(in_feats=n_node, h_feats=32, num_head=4).to(device)
if args.model_idx == 12: model = TransformerModel_Hetero3(in_feats=n_node, h_feats=32, num_head=4).to(device)
if args.model_idx == 13: model = TransformerModel_Hetero4(in_feats=n_node, h_feats=32, num_head=4).to(device)

all_train_batch, all_test_batch = [], []
for train_num_sample, train_data_dir in zip(train_num_sample_list, train_data_dir_list):
    whole_dataset = TrafficAssignmentDataset(train_num_sample, train_data_dir, n_node, map_name)
    _, test_idx = train_test_split(list(range(train_num_sample)), test_size=args.test_ratio, shuffle=False)
    train_idx, _ = train_test_split(list(range(train_num_sample)), train_size=args.train_ratio, shuffle=False)
    train_batch, test_batch = Subset(whole_dataset, train_idx), Subset(whole_dataset, test_idx)
    all_train_batch.append(train_batch)
    all_test_batch.append(test_batch)

train_batch = ConcatDataset(all_train_batch)
test_batch = ConcatDataset(all_test_batch)

train_dataloader = DataLoader(train_batch, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=8)
test_dataloader = DataLoader(test_batch, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=8)

train(train_dataloader, test_dataloader, model, args)
