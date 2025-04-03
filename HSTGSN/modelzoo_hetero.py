import dgl
from dgl.data import DGLDataset
import dgl.function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
from dgl.nn import GATConv, GraphConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian

import os
import numpy as np
import pickle
import scipy.sparse as sparse


from utils_log import createLogger
log = createLogger(__name__)
log.debug('in modelzoo.py')


# dgl.seed(234)
# torch.manual_seed(234)
# np.random.seed(234)
torch.set_printoptions(linewidth=200)

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

class GATmodel(nn.Module):
    def __init__(self, in_feats, h_feats, o_feats, n_head = 4, n_head_o=4):
        super(GATmodel, self).__init__()
        self.h_feats = h_feats
        self.o_feats = o_feats
        self.n_head = n_head
        self.n_head_o = n_head_o
        self.pre_reg = RegressionBranch(in_feats, 64, 64, n_layer=2)
        self.conv1 = GATConv(64, h_feats, num_heads=n_head)
        self.conv2 = GATConv(h_feats*n_head, o_feats, num_heads=n_head_o)
        
    def forward(self, g):
        # in the forward, the feat dim should at the last axis
        g.ndata['feat'] = torch.permute(g.ndata['feat'], (0, 2, 1))
        node_feat = self.pre_reg(g.ndata['feat'])
        
        n_node, n_T, n_feat = g.ndata['feat'].shape
        log.debug("g.ndata['feat']", g.ndata['feat'].shape)

        h = self.conv1(g, node_feat)
        h = h.view(n_node, -1, self.h_feats*self.n_head)
        h = torch.sigmoid(h)
        
        h = self.conv2(g, h)
        h = h.view(n_node, -1, self.o_feats*self.n_head_o)
        h = torch.sigmoid(h)
        
        return h

class MultiHeadAttention_Homo(nn.Module):
    "Multi-Head Attention" # no edge feature into attention score
    def __init__(self, in_feats, o_feats, num_head, n_T):
        "h: number of heads; dim_model: hidden dimension"
        super(MultiHeadAttention_Homo, self).__init__()
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
        self.n_T = n_T
        
    def get(self, g, x, fields='qkv'):
        "Return a dict of queries / keys / values."
        batch_size = x.shape[0]
        # log.debug(x.shape)
        # log.debug(self.in_feats, self.o_head)
        # log.debug("--in get---")
        
        ret = {}
        if 'q' in fields:
            log.debug(x.shape)
            g.ndata['q'] = self.linears[0](x).view(batch_size, self.n_T, self.num_head, self.o_feats)
        if 'k' in fields:
            g.ndata['k'] = self.linears[1](x).view(batch_size, self.n_T, self.num_head, self.o_feats)
        if 'v' in fields:
            g.ndata['v'] = self.linears[2](x).view(batch_size, self.n_T, self.num_head, self.o_feats)
    
    def src_dot_dst(self, src_field, dst_field, out_field):
        def func(edges):
            return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

        return func

    def scaled_exp(self, field, scale_constant):
        def func(edges):
            # clamp for softmax numerical stability
            return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

        return func

    def propagate_attention(self, g):
        # Compute attention score
        eids = g.edges(form='eid')
        g.apply_edges(self.src_dot_dst('k', 'q', 'score'), eids)
        g.apply_edges(self.scaled_exp('score', np.sqrt(self.in_feats)))
        
        # Update node state

        # log.info(g.ndata['v'].shape)
        # log.info(g.edata['score'].shape)
        # asdf
        
        g.send_and_recv(eids, fn.u_mul_e('v', 'score', 'v'), fn.sum('v', 'wv'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))
        
    def get_o(self, g, x):
        "get output of the multi-head attention"
        batch_size = g.ndata['feat'].shape[0]
        wv, z = g.ndata['wv'], g.ndata['z']
        
        log.info(g.ndata['wv'].shape)
        log.info(g.ndata['z'].shape)
        
        g.ndata['wv'] = wv / z
        o = self.linears[3](g.ndata['wv'].view(batch_size, self.n_T, -1))
        
        h = self.FFN(x) + o
        h = h + self.layer_norm(h)
        return h
    
    def forward(self, g, feats):
        self.get(g, feats)
        self.propagate_attention(g)
        h = self.get_o(g, feats)
        
        return h
        
class TransformerModel_Homo(nn.Module):
    def __init__(self, in_feats, h_feats, num_head, n_T):
        super(TransformerModel_Homo, self).__init__()
        self.conv1 = MultiHeadAttention_Homo(in_feats = in_feats, \
            o_feats = h_feats, num_head = num_head, n_T=n_T)
        self.convmid1 = MultiHeadAttention_Homo(in_feats = h_feats*num_head, \
            o_feats = h_feats, num_head = num_head, n_T=n_T)
        self.conv2 = MultiHeadAttention_Homo(in_feats = h_feats*num_head, \
            o_feats = h_feats, num_head = num_head, n_T=n_T)
        
    def forward(self, g):
        g.ndata['feat'] = torch.permute(g.ndata['feat'], (0, 2, 1))
        log.info("before conv feat.shape", g.ndata['feat'].shape)
        
        h = self.conv1(g, g.ndata['feat'])
        h = torch.sigmoid(h)
        
        log.info("conv1 h.shape", h.shape)
        h = self.convmid1(g, h)
        h = torch.sigmoid(h)
        
        h = self.conv2(g, h)
        h = torch.sigmoid(h)
        log.info("conv2 h.shape", h.shape)
        return h

class MultiHeadAttention_OD(nn.Module):
    "Multi-Head Attention"
    def __init__(self, in_feats, o_feats, num_head, n_T):
        "h: number of heads; dim_model: hidden dimension"
        super(MultiHeadAttention_OD, self).__init__()
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
        self.n_T = n_T
        #print(self.in_feats, self.o_head)
        #print("--finish initializaiton---")
        
    def get(self, g, x, fields='qkv'):
        batch_size = x.shape[0]
        "Return a dict of queries / keys / values."
        # #print(self.in_feats, self.o_head)
        # print("--in get---")
        ret = {}
        if 'q' in fields:
            g.nodes['node'].data['q'] = self.linears[0](x).view(batch_size, self.n_T, self.num_head, self.o_feats)
        if 'k' in fields:
            g.nodes['node'].data['k'] = self.linears[1](x).view(batch_size, self.n_T, self.num_head, self.o_feats)
        if 'v' in fields:
            g.nodes['node'].data['v'] = self.linears[2](x).view(batch_size, self.n_T, self.num_head, self.o_feats)

    def propagate_attention(self, g):
        # Compute attention score
        eids = g.edges(form='eid', etype='od')

        g.apply_edges(lambda edges: {'score': (edges.src['k'] * edges.dst['q']).sum(-1, keepdim=True)}, etype='od')
        # log.debug(g.edges['od'].data['score'].shape)
        g.apply_edges(lambda edges: {'score': torch.exp((edges.data['score'] / np.sqrt(self.in_feats)).clamp(-5, 5))}, etype='od')
        # log.debug(g.edges['od'].data['score'].shape)
        # sadf
        # # Update node state
        # g.send_and_recv(g['od'].edges(), fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'), etype='od')
        # g.send_and_recv(g['od'].edges(), fn.copy_edge('score', 'score'), fn.sum('score', 'z'), etype='od')
        
        # Update node state
        g.send_and_recv(g['od'].edges(), fn.u_mul_e('v', 'score', 'v'), fn.sum('v', 'wv'), etype='od')
        g.send_and_recv(g['od'].edges(), fn.copy_e('score', 'score'), fn.sum('score', 'z'), etype='od')

    def get_o(self, g, x):
        "get output of the multi-head attention"
        batch_size = g.ndata['feat'].shape[0]
        wv, z = g.nodes['node'].data['wv'], g.nodes['node'].data['z']

        g.nodes['node'].data['wv'] = wv / (z + 1)
        
        # log.debug(wv.shape, z.shape)
        # asdf
        
        o = self.linears[3](g.nodes['node'].data['wv'].view(batch_size, self.n_T, -1))
        
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
    def __init__(self, in_feats, o_feats, num_head, n_T):
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
        self.n_T = n_T
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
            g.nodes['node'].data['q'] = self.linears[0](x).view(batch_size, self.n_T, self.num_head, self.o_feats)
        if 'k' in fields:
            g.nodes['node'].data['k'] = self.linears[1](x).view(batch_size, self.n_T, self.num_head, self.o_feats)
        if 'v' in fields:
            g.nodes['node'].data['v'] = self.linears[2](x).view(batch_size, self.n_T, self.num_head, self.o_feats)

    def propagate_attention(self, g):
        # Compute attention score
        eids = g.edges(form='eid', etype='connect')
        log.debug(g.nodes['node'].data['k'].shape)
        g.apply_edges(lambda edges: {'score': (edges.src['k'] * edges.dst['q']).sum(-1, keepdim=True)}, etype='connect')
        # log.debug(g.edges['connect'].data['score'].shape)
        g.apply_edges(lambda edges: {'score': torch.einsum('bijk,bk->bijk', edges.data['score'], edges.data['length'].view(-1, 1))}, etype='connect')
        g.apply_edges(lambda edges: {'score': torch.exp((edges.data['score'] / np.sqrt(self.in_feats)).clamp(-5, 5))}, etype='connect')
        log.debug(g.edges['connect'].data['score'].shape)
        # sadf
        # Update node state
        g.send_and_recv(g['connect'].edges(), fn.u_mul_e('v', 'score', 'v'), fn.sum('v', 'wv'), etype='connect')
        g.send_and_recv(g['connect'].edges(), fn.copy_e('score', 'score'), fn.sum('score', 'z'), etype='connect')

    def get_o(self, g, x):
        "get output of the multi-head attention"
        batch_size = g.ndata['feat'].shape[0]
        wv, z = g.nodes['node'].data['wv'], g.nodes['node'].data['z']
        g.nodes['node'].data['wv'] = wv / z
        o = self.linears[3](g.nodes['node'].data['wv'].view(batch_size, self.n_T, -1))

        h = self.FFN(x) + o
        h = h + self.layer_norm(h)

        return h
    
    def forward(self, g, feats):
        self.get(g, feats)
        self.propagate_attention(g)
        h = self.get_o(g, feats)
        
        return h

class TransformerModel_Hetero(nn.Module):
    def __init__(self, in_feats, h_feats, num_head, n_T):
        super(TransformerModel_Hetero, self).__init__()
        self.pre_reg = RegressionBranch(in_feats+2, 64, 64, n_layer=2)
        # self.pre_reg = RegressionBranch(in_feats, 64, 64, n_layer=2)
        self.conv1 = MultiHeadAttention_OD(in_feats = 64, \
            o_feats = h_feats, num_head = num_head, n_T=n_T) #in_feats = 64+2
        self.connconv1 = MultiHeadAttention_Conn(in_feats = h_feats*num_head, \
            o_feats = h_feats, num_head = num_head, n_T=n_T)
        self.connconv2 = MultiHeadAttention_Conn(in_feats = h_feats*num_head, \
            o_feats = h_feats, num_head = num_head, n_T=n_T)
        self.reg = RegressionBranch(h_feats*num_head*2, 128, 1, n_layer=5) # 128
        self.act = nn.LeakyReLU()
        self.n_T = n_T
        
    def forward(self, g):
        # preprocessing
        g.nodes['node'].data['feat'] = torch.permute(g.nodes['node'].data['feat'], (0, 2, 1))
        
        x_coord = torch.unsqueeze(g.nodes['node'].data['x_coord'].repeat(1, self.n_T), 2)
        y_coord = torch.unsqueeze(g.nodes['node'].data['y_coord'].repeat(1, self.n_T), 2)
        
        g.nodes['node'].data['feat'] = torch.cat([g.nodes['node'].data['feat'], x_coord, y_coord], -1)
        log.debug(g.nodes['node'].data['feat'].shape)
        log.debug(g.nodes['node'].data['feat'][:, 2, :])
        node_feat = self.pre_reg(g.nodes['node'].data['feat'])
        # asdf
        
        # node_feat = torch.cat([node_feat, g.nodes['node'].data['x_coord']], -1)
        log.debug(node_feat.shape)

        h = self.conv1(g, node_feat)
        h = self.act(h) #torch.sigmoid(h)
        log.debug("h.shape", h.shape)
        # print('--inish conv1---')
        
        h = self.connconv1(g, h)
        log.debug("h.shape", h.shape)
        h = self.act(h) #torch.sigmoid(h)
        
        h = self.connconv2(g, h)
        h = self.act(h) #torch.sigmoid(h)
        log.debug("h.shape", h.shape)
        # asdf
        # if self.reg.__class__.__name__ == 'RegressionBranch':
        #     g.nodes['node'].data['z'] = h
        #     g.apply_edges(lambda edges: {'hcat': torch.cat([edges.src['z'], edges.dst['z']], -1)}, etype='connect')

        #     # g.apply_edges(lambda edges: {'hcat': torch.cat([edges.src['z'], edges.dst['z'], edges.src['x_coord'], edges.src['y_coord'], \
        #     #     edges.dst['x_coord'], edges.dst['y_coord'], g.edges['connect'].data['length'].view(-1, 1), g.edges['connect'].data['capacity'].view(-1, 1)], -1)}, etype='connect')
        #     h = self.reg(g.edges['connect'].data['hcat'])
        
        # if self.reg.__class__.__name__ == 'RegressionBranch_EdgeConnect':
        #     g.nodes['node'].data['z'] = h
        #     g.apply_edges(lambda edges: {'hcat': torch.cat([edges.src['z'], edges.dst['z'], edges.src['coord'], edges.dst['coord']], -1)}, etype='connect')
        #     h = self.reg(g.edges['connect'].data['hcat'], edge_feat)
        
        return h

class GraphGRUCell(nn.Module):
    def __init__(self, in_feats, out_feats, net, n_head, hidden_in_cell=False):
        super(GraphGRUCell, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.n_head = n_head
        self.use_hidden = hidden_in_cell
        if net == GATConv and not self.use_hidden:
            log.info('in GraphGRUCell, GATConv')
            self.r_net = net(in_feats, out_feats, num_heads=n_head)
            self.u_net = net(in_feats, out_feats, num_heads=n_head)
            self.c_net = net(in_feats, out_feats, num_heads=n_head)
            self.r_bias = nn.Parameter(torch.rand(n_head, out_feats))
            self.u_bias = nn.Parameter(torch.rand(n_head, out_feats))
            self.c_bias = nn.Parameter(torch.rand(n_head, out_feats))
        if net == GATConv and self.use_hidden:
            log.info('in GraphGRUCell, GATConv, True')
            self.r_net = net(in_feats + out_feats*n_head, out_feats, num_heads=n_head)
            self.u_net = net(in_feats + out_feats*n_head, out_feats, num_heads=n_head)
            self.c_net = net(in_feats + out_feats*n_head, out_feats, num_heads=n_head)
            self.r_bias = nn.Parameter(torch.rand(out_feats))
            self.u_bias = nn.Parameter(torch.rand(out_feats))
            self.c_bias = nn.Parameter(torch.rand(out_feats))
        
    def forward(self, g, x, h):
        if self.use_hidden:
            log.info("x.shape, h.shape, ", x.shape, h.shape, self.r_bias.shape, torch.cat([x, h.view(g.num_nodes(), -1)], dim=1).shape)
            log.info("self.r_net(g, torch.cat([x, h.view(g.num_nodes(), -1)], dim=1)), ", self.r_net(g, torch.cat([x, h.view(g.num_nodes(), -1)], dim=1)).shape)
            # use hidden state in the forward propagation
            r = torch.sigmoid(self.r_net(g, torch.cat([x, h.view(g.num_nodes(), -1)], dim=1)).squeeze() + self.r_bias)
            u = torch.sigmoid(self.u_net(g, torch.cat([x, h.view(g.num_nodes(), -1)], dim=1)).squeeze() + self.u_bias)
            log.info("r, u, h", r.shape, u.shape, h.shape, self.r_bias.shape, self.u_bias.shape)
            h_ = r * h
            c = torch.sigmoid(self.c_net(g, torch.cat([x, h_.view(g.num_nodes(), -1)], dim=1)).squeeze() + self.c_bias)
            new_h = u * h + (1 - u) * c
        else:
            # squeeze() is needed in the case n_head=1
            log.info("x.shape, h.shape, r_bias.shape, ", x.shape, h.shape, self.r_bias.shape)
            log.info("self.r_net(g, x).shape, ", self.r_net(g, x).squeeze().shape)

            r = torch.sigmoid(self.r_net(g, x).squeeze() + self.r_bias)
            u = torch.sigmoid(self.u_net(g, x).squeeze() + self.u_bias)
            log.info("r, u, h", r.shape, u.shape, h.shape)
            h_ = r * h
            c = torch.sigmoid(self.c_net(g, x).squeeze() + self.c_bias)
            new_h = u * h + (1 - u) * c
        log.info("x.shape, h.shape, new_h.shape", x.shape, h.shape, new_h.shape)
        
        return new_h

class RGNN(nn.Module):
    def __init__(self, in_feats, out_feats, n_rnn_layers, n_head, n_T, hidden_in_cell):
        super(RGNN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        
        self.n_head = n_head
        self.n_rnn_layers = n_rnn_layers
        self.n_T = n_T
        
        self.net = GATConv
        self.hidden_in_cell = hidden_in_cell
        
        self.layers = nn.ModuleList()
        self.layers.append(GraphGRUCell(self.in_feats, self.out_feats, self.net, self.n_head, self.hidden_in_cell))
        log.info(self.in_feats, self.out_feats)
        for _ in range(self.n_rnn_layers - 1):
            self.layers.append(GraphGRUCell(self.out_feats*self.n_head, self.out_feats, self.net, self.n_head, self.hidden_in_cell))
            log.info(self.out_feats, self.out_feats)
    
    def forward(self, g, x, hidden_states):
        hiddens = [] # hidden_states should be a list which for different layer
        for i, layer in enumerate(self.layers):
            log.info("x, hidden_states[{}], ".format(i), x.shape, hidden_states[i].shape)
            # log.info("layer(g, x, hidden_states[i]).shape, ", layer(g, x, hidden_states[i]).shape)
            x = layer(g, x, hidden_states[i]).reshape(x.shape[0], -1)
            log.info("x after reshape", x.shape)
            hiddens.append(x.reshape(x.shape[0], self.n_head, -1).squeeze())
        log.info("hiddens[0], hiddens[1], ", hiddens[0].shape, hiddens[1].shape)
        
        return x, hiddens

class GraphRNN_Transformer(nn.Module):
    def __init__( self, pre_emb_gnn, rnn_gnn, out_fnn_ratio, out_fnn_cnt, n_T, device):
        super(GraphRNN_Transformer, self).__init__()
        self.device = device
        self.pre_emb_gnn = pre_emb_gnn
        self.rnn_gnn_1 = rnn_gnn[0] #nn.GRU(128, self.out_feats, self.n_rnn_layers)
        self.rnn_gnn_2 = rnn_gnn[1] #nn.GRU(128, self.out_feats, self.n_rnn_layers)
        self.out_fnn_ratio = out_fnn_ratio
        self.out_fnn_cnt = out_fnn_cnt
        
        self.n_T = n_T
        
    # Threshold For Teacher Forcing
    def compute_thresh(self, batch_cnt):
        # return 1. / ( 1. + np.exp(batch_cnt / self.decay_steps) )
        return self.decay_steps / ( self.decay_steps + np.exp(batch_cnt / self.decay_steps) )

    def encode(self, g): #, inputs, device
        # if create rnn_hs is here: reset hidden state every batch
        # if dont reset --> move to __init__
        # if reset every epoch: need to use a flag during training
        if isinstance(self.rnn_gnn_1, nn.GRU):
            self.rnn_hs = torch.randn(self.rnn_gnn_1.num_layers, g.num_nodes(), self.rnn_gnn_1.hidden_size).to(self.device)
            log.info('we are here, nn.GRU', self.rnn_hs.shape, g.ndata['emb_feat'].shape) # n_rnn_layers x n_T x out_feats
            outputs, hn = self.rnn_gnn_1(g.ndata['emb_feat'], self.rnn_hs)
        
        if isinstance(self.rnn_gnn_1, nn.LSTM):
            self.rnn_hs = torch.randn(self.rnn_gnn_1.num_layers, g.num_nodes(), self.rnn_gnn_1.hidden_size).to(self.device)
            self.rnn_cs = torch.randn(self.rnn_gnn_1.num_layers, g.num_nodes(), self.rnn_gnn_1.hidden_size).to(self.device)
            log.info('we are here, nn.GRU', self.rnn_hs.shape) # n_rnn_layers x n_T x out_feats
            
            outputs, (hn, cn) = self.rnn_gnn_1(g.ndata['emb_feat'], (self.rnn_hs, self.rnn_cs))
            log.debug(g.ndata['emb_feat'].shape, outputs.shape)
        if isinstance(self.rnn_gnn_1, RGNN):
            outputs = []
            if self.rnn_gnn_1.n_head == 1:
                self.rnn_hs = [torch.randn(g.num_nodes(), self.rnn_gnn_1.out_feats).to(self.device) for _ in range(self.rnn_gnn_1.n_rnn_layers)]
            else:
                self.rnn_hs = [torch.randn(g.num_nodes(), self.rnn_gnn_1.n_head, self.rnn_gnn_1.out_feats).to(self.device) for _ in range(self.rnn_gnn_1.n_rnn_layers)]
            log.info('we are here, RGNN', self.rnn_hs[0].shape, g.ndata['emb_feat'].shape)
            
            # output: save from every time step --> used for prediction
            # hn: only save the last time step --> used in GNNRNN
            log.info('we are here, RGNN_new {}'.format(0), self.rnn_hs[0].shape, g.ndata['emb_feat'].shape)
            log.info("len(self.rnn_hs), self.rnn_hs[0].shape, ", len(self.rnn_hs), self.rnn_hs[0].shape)
            for i in range(self.rnn_gnn_1.n_T):
                if i == 0:
                    output, hn = self.rnn_gnn_1(g, g.ndata['emb_feat'][:, i, :], self.rnn_hs)
                else:
                    output, hn = self.rnn_gnn_1(g, g.ndata['emb_feat'][:, i, :], hn)
                log.info('we are here, RGNN_new {}'.format(i), hn[0].shape, g.ndata['emb_feat'].shape)
                log.info("len(self.rnn_hs), self.rnn_hs[0].shape, ", len(hn), hn[0].shape)
                outputs.append(output)
            outputs = torch.stack(outputs, axis=1)
        
        log.info("outputs, hn", outputs.shape, hn[0].shape, hn[1].shape)
        return outputs, hn
        
    def decode(self, g):            
        g.apply_edges(lambda edges: {'rnn_e_emb': torch.cat([edges.src['rnn_n_emb'], edges.dst['rnn_n_emb']], -1)}, etype='connect')
        log.debug("g.edata['rnn_e_emb'], g.edata['res_ratio']", g.edges['connect'].data['rnn_e_emb'].shape, g.edges['connect'].data['res_ratio'].shape)
        
        if isinstance(self.rnn_gnn_2, nn.LSTM):
            self.rnn_hs = torch.randn(self.rnn_gnn_2.num_layers, g.num_edges('connect'), self.rnn_gnn_2.hidden_size).to(self.device)
            self.rnn_cs = torch.randn(self.rnn_gnn_2.num_layers, g.num_edges('connect'), self.rnn_gnn_2.hidden_size).to(self.device)
            log.info('we are here, nn.GRU', self.rnn_hs.shape) # n_rnn_layers x n_T x out_feats
            log.debug(g.edges['connect'].data['rnn_e_emb'].shape, g.num_edges('connect'))
            outputs, (hn, cn) = self.rnn_gnn_2(g.edges['connect'].data['rnn_e_emb'], (self.rnn_hs, self.rnn_cs))
            log.debug(g.edges['connect'].data['rnn_e_emb'].shape, outputs.shape)
            
        g.apply_edges(lambda edges: {'aux_emd': torch.cat([edges.src['x_coord'], edges.src['y_coord'], edges.dst['x_coord'], \
            edges.dst['y_coord'], ], -1)}, etype='connect')
        aux_emd = torch.unsqueeze(torch.cat([g.edges['connect'].data['aux_emd'], g.edges['connect'].data['capacity'], g.edges['connect'].data['length'].view(-1, 1)], -1), 1).repeat(1, self.n_T, 1)
        
        g.edges['connect'].data['rnn_e_emb'] = torch.cat([g.edges['connect'].data['rnn_e_emb'], aux_emd], -1)
        
        log.info(g.edges['connect'].data['rnn_e_emb'].shape, l=10)
        output_ratio = self.out_fnn_ratio(g.edges['connect'].data['rnn_e_emb'])
        output_cnt = self.out_fnn_cnt(g.edges['connect'].data['rnn_e_emb'])
        log.debug("output_ratio", output_ratio.shape)
        
        return output_ratio, output_cnt
    
    def forward(self, g, batch_cnt, device): # , inputs, teacher_states, batch_cnt, device
        log.debug("g.ndata['feat'].shape, g.edata['res_ratio'].shape, g.edata['free_time'], ", \
            g.nodes['node'].data['feat'].shape, g.edges['connect'].data['res_ratio'].shape, \
            g.edges['connect'].data['free_time'].shape)
        
        # pre_emb_gnn: use node_feat + pred_flow from previous time step
        if isinstance(self.pre_emb_gnn, TransformerModel_Homo):
            emd_feat = self.pre_emb_gnn(g)
        
        if isinstance(self.pre_emb_gnn, TransformerModel_Hetero):
            emd_feat = self.pre_emb_gnn(g)
        
        if isinstance(self.pre_emb_gnn, RegressionBranch):
            g.ndata['feat'] = torch.permute(g.ndata['feat'], (0, 2, 1))
            emd_feat = self.pre_emb_gnn(g.ndata['feat'])
            
        g.ndata['emb_feat'] = emd_feat
        log.debug("emd_feat, ", emd_feat.shape)
        
        # in encoder: from od_feat --> output and hidden state
        rnn_n_emb, hidden = self.encode(g) # , inputs, device
        log.info("hidden[0], hidden[1], ", hidden[0].shape, hidden[1].shape, rnn_n_emb.shape)
        
        g.ndata['rnn_n_emb'] = rnn_n_emb
        # dont need the teacher state
        # in decoder: make node rnn emb(rnn_n_emb) into edge feat and then make prediction
        outputs = self.decode(g)
        return outputs

