import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian

import dgl
from dgl.data import DGLDataset
import dgl.function as fn
from dgl.nn import GATConv, GraphConv
import dgl.ops as ops

import os
import numpy as np
import pickle

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        # feat_dict: learnable node feature
        # self.weight: weight matrix for node feature
        print(self.weight)
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_{}'.format(etype)] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_{}'.format(etype), 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
    
class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        # self.embed: learnable node feature
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G):
        h_dict = self.layer1(G, self.embed)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get paper logits
        return h_dict['paper']

class RegressionBranch(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat, n_layer=1):
        super(RegressionBranch, self).__init__()
        self.n_layer = n_layer
        if self.n_layer == 1:
            self.linear1 = nn.Linear(in_feat, out_feat)
        else:
            self.linear1 = nn.Linear(in_feat, h_feat)
            self.linears = nn.ModuleList([nn.Linear(h_feat, h_feat) for i in range(self.n_layer-2)])
            self.linear2 = nn.Linear(h_feat, out_feat)
        
    def forward(self, h):
        if self.n_layer == 1:
            h = self.linear1(h)
        else:
            h = self.linear1(h)
            h = torch.relu(h)
            for layer in self.linears:
                h = layer(h)
                h = torch.relu(h)
            h = self.linear2(h)
        return h

class ResidualBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_feat, out_feat)
        self.linear2 = nn.Linear(out_feat, out_feat)
    def forward(self, x):
        residual = x
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        out = out + residual
        return out

class Pre_RegressionBranch(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat, n_layer=2):
        super(Pre_RegressionBranch, self).__init__()
        self.n_layer = n_layer
        if self.n_layer == 1:
            self.block1 = ResidualBlock(in_feat, h_feat)
        else:
            self.block1 = ResidualBlock(in_feat, h_feat)
            self.blocks = nn.ModuleList([ResidualBlock(h_feat, h_feat) for i in range(self.n_layer-2)])
            self.block2 = ResidualBlock(h_feat, h_feat)
        self.linear = nn.Linear(h_feat, out_feat)
        
    def forward(self, h):
        if self.n_layer == 1:
            h = self.block1(h)
            h = torch.relu(h)
        else:
            h = self.block1(h)
            h = torch.relu(h)
            for layer in self.blocks:
                h = layer(h)
                h = torch.relu(h)
            h = self.block2(h)
            h = torch.relu(h)
        h = self.linear(h)
        return h

class RegressionBranch_EdgeConnect(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat, n_layer=1):
        super(RegressionBranch_EdgeConnect, self).__init__()
        self.n_layer = n_layer
        self.act = nn.LeakyReLU()
        if self.n_layer == 1:
            self.linear1 = nn.Linear(in_feat+2, out_feat)
        else:
            self.linear1 = nn.Linear(in_feat+2, h_feat)
            self.linears = nn.ModuleList([nn.Linear(h_feat, h_feat) for i in range(self.n_layer-2)])
            self.linear2 = nn.Linear(h_feat, out_feat)
        
    def forward(self, h, edge_feat):
        if self.n_layer == 1:
            h = torch.cat([h, edge_feat], -1)
            h = self.linear1(h)
        else:
            h = torch.cat([h, edge_feat], -1)
            h = self.linear1(h)
            h = self.act(h) #torch.relu(h)
            for layer in self.linears:
                h = layer(h)
                h = self.act(h) #torch.relu(h)
            h = torch.cat([h, edge_feat], -1)
            h = self.linear2(h)
            # h = torch.relu(h)
        return h

# add capacity edge feature into heterograph
class TransformerModel_Hetero2(nn.Module):
    def __init__(self, in_feats, h_feats, num_head):
        super(TransformerModel_Hetero2, self).__init__()
        self.conv1 = MultiHeadAttention_Hetero2(in_feats = in_feats, \
            o_feats = h_feats, num_head = num_head)
        # self.convmid1 = MultiHeadAttention_Hetero2(in_feats = h_feats*num_head, \
        #     o_feats = h_feats, num_head = num_head)
        # self.conv2 = MultiHeadAttention_Hetero2(in_feats = h_feats*num_head, \
            # o_feats = h_feats, num_head = num_head)
        self.connconv1 = MultiHeadAttention_Conn(in_feats = h_feats*num_head, \
            o_feats = h_feats, num_head = num_head)
        self.connconv2 = MultiHeadAttention_Conn(in_feats = h_feats*num_head, \
            o_feats = h_feats, num_head = num_head)
        self.reg = RegressionBranch(h_feats*num_head*2+2, 256, 1, n_layer=5) # 128
        # self.reg = RegressionBranch_EdgeConnect(h_feats*num_head*2, 128, 1, n_layer=5)
        self.act = nn.LeakyReLU()
        
    def forward(self, g, node_feat, edge_feat):
        h = self.conv1(g, node_feat)
        h = self.act(h) #torch.sigmoid(h)
        # print("h.shape", h.shape)
        # print('--inish conv1---')
        
        # ##### remove this layer 
        # h = self.convmid1(g, h)
        # h = self.act(h) #torch.sigmoid(h)
        
        # h = self.conv2(g, h)
        # h = torch.sigmoid(h)
        
        h = self.connconv1(g, h)
        h = self.act(h) #torch.sigmoid(h)
        
        h = self.connconv2(g, h)
        h = self.act(h) #torch.sigmoid(h)
        
        if self.reg.__class__.__name__ == 'RegressionBranch':
            g.nodes['node'].data['z'] = h
            g.apply_edges(lambda edges: {'hcat': torch.cat([edges.src['z'], edges.dst['z'], edge_feat], -1)}, etype='connect')
            h = self.reg(g.edges['connect'].data['hcat'])
        
        if self.reg.__class__.__name__ == 'RegressionBranch_EdgeConnect':
            g.nodes['node'].data['z'] = h
            g.apply_edges(lambda edges: {'hcat': torch.cat([edges.src['z'], edges.dst['z']], -1)}, etype='connect')
            h = self.reg(g.edges['connect'].data['hcat'], edge_feat)
        
        return h

# add capacity edge feature into heterograph
class TransformerModel_Hetero3(nn.Module):
    def __init__(self, in_feats, h_feats, num_head):
        super(TransformerModel_Hetero3, self).__init__()
        # self.pre_reg = Pre_RegressionBranch(in_feats, 64, 1, n_layer=5)
        self.pre_reg = RegressionBranch(in_feats, 64, 64, n_layer=2)
        self.conv1 = MultiHeadAttention_Hetero2(in_feats = 64, \
            o_feats = h_feats, num_head = num_head)
        self.connconv1 = MultiHeadAttention_Conn(in_feats = h_feats*num_head, \
            o_feats = h_feats, num_head = num_head)
        self.connconv2 = MultiHeadAttention_Conn(in_feats = h_feats*num_head, \
            o_feats = h_feats, num_head = num_head)
        self.reg = RegressionBranch(h_feats*num_head*2+2, 256, 1, n_layer=5) # 128
        # self.reg = RegressionBranch_EdgeConnect(h_feats*num_head*2, 128, 1, n_layer=5)
        self.act = nn.LeakyReLU()
        
    def forward(self, g, node_feat, edge_feat):
        # preprocessing
        node_feat = self.pre_reg(node_feat)
        
        h = self.conv1(g, node_feat)
        h = self.act(h) #torch.sigmoid(h)
        # print("h.shape", h.shape)
        # print('--inish conv1---')
        
        h = self.connconv1(g, h)
        h = self.act(h) #torch.sigmoid(h)
        
        h = self.connconv2(g, h)
        h = self.act(h) #torch.sigmoid(h)
        
        if self.reg.__class__.__name__ == 'RegressionBranch':
            g.nodes['node'].data['z'] = h
            g.apply_edges(lambda edges: {'hcat': torch.cat([edges.src['z'], edges.dst['z'], edge_feat], -1)}, etype='connect')
            h = self.reg(g.edges['connect'].data['hcat'])
        
        if self.reg.__class__.__name__ == 'RegressionBranch_EdgeConnect':
            g.nodes['node'].data['z'] = h
            g.apply_edges(lambda edges: {'hcat': torch.cat([edges.src['z'], edges.dst['z']], -1)}, etype='connect')
            h = self.reg(g.edges['connect'].data['hcat'], edge_feat)
        
        return h

class MultiHeadAttention_Hetero2(nn.Module):
    "Multi-Head Attention"
    def __init__(self, in_feats, o_feats, num_head):
        "h: number of heads; dim_model: hidden dimension"
        super(MultiHeadAttention_Hetero2, self).__init__()
        self.in_feats = in_feats
        self.o_feats = o_feats
        self.in_head = in_feats * num_head
        self.o_head = o_feats * num_head
        self.num_head = num_head
        # W_q, W_k, W_v, W_o
        self.linears = nn.ModuleList([nn.Linear(self.in_feats, self.o_head), \
                                      nn.Linear(self.in_feats, self.o_head), \
                                      nn.Linear(self.in_feats, self.o_head), \
                                      nn.Linear(self.o_head, self.o_head)])
        self.FFN = nn.Linear(self.in_feats, self.o_head)
        self.layer_norm = nn.LayerNorm([self.o_head])
        #print(self.in_feats, self.o_head)
        #print("--finish initializaiton---")
        
    def get(self, g, x, fields='qkv'):
        "Return a dict of queries / keys / values."
        batch_size = x.shape[0]
        # print(x.shape)
        # #print(self.in_feats, self.o_head)
        # print("--in get---")
        ret = {}
        if 'q' in fields:
            g.nodes['node'].data['q'] = self.linears[0](x).view(batch_size, self.num_head, self.o_feats)
        if 'k' in fields:
            g.nodes['node'].data['k'] = self.linears[1](x).view(batch_size, self.num_head, self.o_feats)
        if 'v' in fields:
            g.nodes['node'].data['v'] = self.linears[2](x).view(batch_size, self.num_head, self.o_feats)

    def propagate_attention(self, g):
        # Compute attention score
        eids = g.edges(form='eid', etype='od')

        g.apply_edges(lambda edges: {'score': (edges.src['k'] * edges.dst['q']).sum(-1, keepdim=True)}, etype='od')   
        g.apply_edges(lambda edges: {'score': torch.exp((edges.data['score'] / np.sqrt(self.in_feats)).clamp(-5, 5))}, etype='od')
        
        # Update node state
        g.send_and_recv(g['od'].edges(), fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'), etype='od')
        g.send_and_recv(g['od'].edges(), fn.copy_edge('score', 'score'), fn.sum('score', 'z'), etype='od')

    def get_o(self, g, x):
        "get output of the multi-head attention"
        batch_size = g.ndata['feat'].shape[0]
        wv, z = g.nodes['node'].data['wv'], g.nodes['node'].data['z']

        g.nodes['node'].data['wv'] = wv / (z + 1)
        o = self.linears[3](g.nodes['node'].data['wv'].view(batch_size, -1))
        
        h = self.FFN(x) + o
        h = h + self.layer_norm(h)
        
        return h
    
    def forward(self, g, feats):
        self.get(g, feats)
        self.propagate_attention(g)
        h = self.get_o(g, feats)
        
        return h

class MultiHeadAttention_Conn(nn.Module):
    "Multi-Head Attention"
    def __init__(self, in_feats, o_feats, num_head):
        "h: number of heads; dim_model: hidden dimension"
        super(MultiHeadAttention_Conn, self).__init__()
        self.in_feats = in_feats
        self.o_feats = o_feats
        self.in_head = in_feats * num_head
        self.o_head = o_feats * num_head
        self.num_head = num_head
        # W_q, W_k, W_v, W_o
        self.linears = nn.ModuleList([nn.Linear(self.in_feats, self.o_head), \
                                      nn.Linear(self.in_feats, self.o_head), \
                                      nn.Linear(self.in_feats, self.o_head), \
                                      nn.Linear(self.o_head, self.o_head)])
        self.FFN = nn.Linear(self.in_feats, self.o_head)
        self.layer_norm = nn.LayerNorm([self.o_head])
        #print(self.in_feats, self.o_head)
        #print("--finish initializaiton---")
        
    def get(self, g, x, fields='qkv'):
        "Return a dict of queries / keys / values."
        batch_size = x.shape[0]
        # print(x.shape)
        # #print(self.in_feats, self.o_head)
        # print("--in get---")
        ret = {}
        if 'q' in fields:
            g.nodes['node'].data['q'] = self.linears[0](x).view(batch_size, self.num_head, self.o_feats)
        if 'k' in fields:
            g.nodes['node'].data['k'] = self.linears[1](x).view(batch_size, self.num_head, self.o_feats)
        if 'v' in fields:
            g.nodes['node'].data['v'] = self.linears[2](x).view(batch_size, self.num_head, self.o_feats)

    def propagate_attention(self, g):
        # Compute attention score
        eids = g.edges(form='eid', etype='connect')

        g.apply_edges(lambda edges: {'score': (edges.src['k'] * edges.dst['q']).sum(-1, keepdim=True)}, etype='connect')
        g.apply_edges(lambda edges: {'score': torch.einsum('bij,bj->bij', edges.data['score'], edges.data['feat'][:, 0:1])}, etype='connect')
        g.apply_edges(lambda edges: {'score': torch.exp((edges.data['score'] / np.sqrt(self.in_feats)).clamp(-5, 5))}, etype='connect')

        # Update node state
        g.send_and_recv(g['connect'].edges(), fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'), etype='connect')
        g.send_and_recv(g['connect'].edges(), fn.copy_edge('score', 'score'), fn.sum('score', 'z'), etype='connect')

    def get_o(self, g, x):
        "get output of the multi-head attention"
        batch_size = g.ndata['feat'].shape[0]
        wv, z = g.nodes['node'].data['wv'], g.nodes['node'].data['z']
        g.nodes['node'].data['wv'] = wv / z
        o = self.linears[3](g.nodes['node'].data['wv'].view(batch_size, -1))

        h = self.FFN(x) + o
        h = h + self.layer_norm(h)

        return h
    
    def forward(self, g, feats):
        self.get(g, feats)
        self.propagate_attention(g)
        h = self.get_o(g, feats)
        
        return h

# add capacity edge feature into heterograph
class TransformerModel_Hetero4(nn.Module):
    def __init__(self, in_feats, h_feats, num_head):
        super(TransformerModel_Hetero4, self).__init__()
        # self.pre_reg = Pre_RegressionBranch(in_feats, 64, 1, n_layer=5)
        self.pre_reg = RegressionBranch(in_feats, 64, 64, n_layer=2)
        self.conv1 = MultiHeadAttention_Hetero2(in_feats = 64+2, \
            o_feats = h_feats, num_head = num_head)
        self.connconv1 = MultiHeadAttention_Conn(in_feats = h_feats*num_head, \
            o_feats = h_feats, num_head = num_head)
        self.connconv2 = MultiHeadAttention_Conn(in_feats = h_feats*num_head, \
            o_feats = h_feats, num_head = num_head)
        self.reg = RegressionBranch(h_feats*num_head*2+2+4, 128, 1, n_layer=5) # 128
        # self.reg = RegressionBranch_EdgeConnect(h_feats*num_head*2, 128, 1, n_layer=5)
        self.act = nn.LeakyReLU()
        
    def forward(self, g, node_feat, edge_feat):
        # preprocessing
        node_feat = self.pre_reg(node_feat)
        node_feat = torch.cat([node_feat, g.nodes['node'].data['coord']], -1)
        
        h = self.conv1(g, node_feat)
        h = self.act(h) #torch.sigmoid(h)
        # print("h.shape", h.shape)
        # print('--inish conv1---')
        
        h = self.connconv1(g, h)
        h = self.act(h) #torch.sigmoid(h)
        
        h = self.connconv2(g, h)
        h = self.act(h) #torch.sigmoid(h)
        
        if self.reg.__class__.__name__ == 'RegressionBranch':
            g.nodes['node'].data['z'] = h
            g.apply_edges(lambda edges: {'hcat': torch.cat([edges.src['z'], edges.dst['z'], edges.src['coord'], edges.dst['coord'], edge_feat], -1)}, etype='connect')
            h = self.reg(g.edges['connect'].data['hcat'])
        
        if self.reg.__class__.__name__ == 'RegressionBranch_EdgeConnect':
            g.nodes['node'].data['z'] = h
            g.apply_edges(lambda edges: {'hcat': torch.cat([edges.src['z'], edges.dst['z'], edges.src['coord'], edges.dst['coord']], -1)}, etype='connect')
            h = self.reg(g.edges['connect'].data['hcat'], edge_feat)
        
        return h

