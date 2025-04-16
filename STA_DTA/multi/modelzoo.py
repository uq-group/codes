import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GATConv, GraphConv
import numpy as np
from functools import partial


from utils import get_parse_args
args = get_parse_args()

from utils_log import createLogger
log = createLogger(__name__, args.log_save_path, args.log_level, args.log_file_name)
log.info('in modelzoo.py')

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

class BaseTransformerLayer(nn.Module):
    def __init__(self, in_feats, o_feats, num_head, v_id_list):
        super(BaseTransformerLayer, self).__init__()
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
        
        self.v_id_list = v_id_list
        
        self.aggr_linears = nn.ModuleList([nn.Linear(self.in_feats*len(self.v_id_list), self.o_head), \
                                        nn.Linear(self.in_feats*len(self.v_id_list), self.o_head), \
                                        nn.Linear(self.in_feats*len(self.v_id_list), self.o_head), \
                                        nn.Linear(self.o_head, self.o_head)])
        self.aggr_FFN = nn.Linear(self.in_feats*len(self.v_id_list), self.o_head)
        self.aggr_layer_norm = nn.LayerNorm([self.o_head])
        
    def get(self, g, x_dict, fields='qkv'):
        "Return a dict of queries / keys / values."
        for v_id in self.v_id_list + [99]:
            linear_layer = self.linears if v_id != 99 else self.aggr_linears
            batch_size = x_dict[v_id].shape[0]
            if 'q' in fields:
                g.nodes['node'].data[f'q_{v_id}'] = \
                    linear_layer[0](x_dict[v_id]).view(batch_size, self.num_head, self.o_feats)
            if 'k' in fields:
                g.nodes['node'].data[f'k_{v_id}'] = \
                    linear_layer[1](x_dict[v_id]).view(batch_size, self.num_head, self.o_feats)
            if 'v' in fields:
                g.nodes['node'].data[f'v_{v_id}'] = \
                    linear_layer[2](x_dict[v_id]).view(batch_size, self.num_head, self.o_feats)
        
    def propagate_attention(self, g, conn_type):
        for v_id in self.v_id_list:
            e_type = f'{conn_type}_{v_id}'
            ## Compute attention score
            ## first within class (0,1,2), then between class (99)
            g.apply_edges(lambda edges: {f'score_{v_id}': (edges.src[f'k_{v_id}'] * \
                edges.dst[f'q_{v_id}']).sum(-1, keepdim=True)}, etype=e_type)
            g.apply_edges(lambda edges: {f'score_99_{v_id}': (edges.src['k_99'] * \
                edges.dst['q_99']).sum(-1, keepdim=True)}, etype=e_type)
            
            if 'connect' in e_type:
                g.apply_edges(lambda edges: {f'score_{v_id}': torch.einsum('bij,bj->bij', 
                                                                           edges.data[f'score_{v_id}'], 
                                                                           edges.data['feat'][:, 0:1])}, etype=e_type)
                g.apply_edges(lambda edges: {f'score_99_{v_id}': torch.einsum('bij,bj->bij', 
                                                                           edges.data[f'score_99_{v_id}'], 
                                                                           edges.data['feat'][:, 0:1])}, etype=e_type)
                
            g.apply_edges(lambda edges: {f'score_{v_id}': torch.exp((edges.data[f'score_{v_id}'] / \
                np.sqrt(self.in_feats)).clamp(-5, 5))}, etype=e_type)
            g.apply_edges(lambda edges: {f'score_99_{v_id}': torch.exp((edges.data[f'score_99_{v_id}'] / \
                np.sqrt(self.in_feats)).clamp(-5, 5))}, etype=e_type)
            
            # Update node state
            g.send_and_recv(g[e_type].edges(), 
                            fn.u_mul_e(f'v_{v_id}', f'score_{v_id}', f'v_{v_id}'),
                            fn.sum(f'v_{v_id}', f'wv_{v_id}'), etype=e_type)
            g.send_and_recv(g[e_type].edges(), 
                            fn.copy_e(f'score_{v_id}', f'score_{v_id}'), 
                            fn.sum(f'score_{v_id}', f'z_{v_id}'), etype=e_type)
            
            g.send_and_recv(g[e_type].edges(), 
                            fn.u_mul_e('v_99', f'score_99_{v_id}', 'v_99'),
                            fn.sum('v_99', f'wv_99_{v_id}'), etype=e_type)
            g.send_and_recv(g[e_type].edges(), 
                            fn.copy_e(f'score_99_{v_id}', f'score_99_{v_id}'), 
                            fn.sum(f'score_99_{v_id}', f'z_99_{v_id}'), etype=e_type)
                   
    def get_o(self, g, x_dict):
        "get output of the multi-head attention"
        o_dict, h_dict, aggr_o_dict, aggr_h_dict = {}, {}, {}, {}
        for v_id in self.v_id_list:
            batch_size = g.ndata[f'feat_{v_id}'].shape[0]
            g.nodes['node'].data[f'wv_{v_id}'] = g.nodes['node'].data[f'wv_{v_id}'] \
                / (g.nodes['node'].data[f'z_{v_id}'] + 1)
            o_dict[v_id] = self.linears[3](g.nodes['node'].data[f'wv_{v_id}'].view(batch_size, -1))
            
            h_dict[v_id] = self.FFN(x_dict[v_id]) + o_dict[v_id]
            h_dict[v_id] = h_dict[v_id] + self.layer_norm(h_dict[v_id])
        
        for v_id in self.v_id_list:
            g.nodes['node'].data[f'wv_99_{v_id}'] = g.nodes['node'].data[f'wv_99_{v_id}'] \
                / (g.nodes['node'].data[f'z_99_{v_id}'] + 1)

            # print(g.nodes['node'].data[f'wv_99_{v_id}'].shape)
            # print(g.nodes['node'].data[f'wv_99_{v_id}'].view(batch_size, -1).shape)
            aggr_o_dict[v_id] = self.aggr_linears[3](g.nodes['node'].data[f'wv_99_{v_id}'].view(batch_size, -1))

            aggr_h_dict[v_id] = self.aggr_FFN(x_dict[99]) + aggr_o_dict[v_id]
            aggr_h_dict[v_id] = aggr_h_dict[v_id] + self.aggr_layer_norm(aggr_h_dict[v_id])
            
        return aggr_h_dict
    
class TransformerLayer_OD(BaseTransformerLayer):
    "Multi-Head Attention"
    def __init__(self, in_feats, o_feats, num_head, v_id_list):
        "h: number of heads; dim_model: hidden dimension"
        super(TransformerLayer_OD, self).__init__(in_feats, o_feats, num_head, v_id_list)

    def forward(self, g, x_dict):
        # x_dict: {0: reg_feat, 1: reg_feat, 2: reg_feat}
        x_dict[99] = torch.cat([x_dict[v_id] for v_id in self.v_id_list], dim=1)
        with g.local_scope():
            self.get(g, x_dict)
            self.propagate_attention(g, conn_type='od')
            h_dict = self.get_o(g, x_dict)

        return h_dict

class TransformerLayer_Conn(BaseTransformerLayer):
    "Multi-Head Attention"
    def __init__(self, in_feats, o_feats, num_head, v_id_list):
        "h: number of heads; dim_model: hidden dimension"
        super(TransformerLayer_Conn, self).__init__(in_feats, o_feats, num_head, v_id_list)

    def forward(self, g, x_dict):
        # x_dict: {0: reg_feat, 1: reg_feat, 2: reg_feat}
        x_dict[-1] = torch.cat([x_dict[v_id] for v_id in self.v_id_list], dim=1)
        with g.local_scope():
            self.get(g, x_dict)
            self.propagate_attention(g, conn_type='connect')
            h_dict = self.get_o(g, x_dict)
        return h_dict

class BaseHetTransformer(nn.Module):
    def __init__(self, in_feats, pre_reg_feats, h_feats, num_head, base_v_id, v_id_list):
        super(BaseHetTransformer, self).__init__()
        self.base_v_id = base_v_id
        self.v_id_list = v_id_list
        self.pre_reg = RegressionBranch(in_feats, pre_reg_feats, pre_reg_feats, n_layer=2)
        
        self.connconv1 = TransformerLayer_Conn(in_feats = h_feats*num_head, \
            o_feats = h_feats, num_head = num_head, v_id_list=v_id_list)
        self.connconv2 = TransformerLayer_Conn(in_feats = h_feats*num_head, \
            o_feats = h_feats, num_head = num_head, v_id_list=v_id_list)
        self.reg_dict = nn.ModuleDict({
            f'connect_{v_id}': RegressionBranch(h_feats*num_head*2+2+4, 128, 1, n_layer=5) for v_id in self.v_id_list # 128
        })
        self.act = nn.LeakyReLU()
    
    def forward_pre_reg(self, g):
        # preprocessing
        node_feat_dict = {}
        for v_id in self.v_id_list:
            node_feat = g.nodes['node'].data[f'feat_{v_id}']
            node_feat = self.pre_reg(node_feat)
            node_feat_dict[v_id] = torch.cat([node_feat, g.nodes['node'].data['coord']], -1)
        return node_feat_dict
    
    def forward_conn(self, g, h_dict):
        h_dict = self.connconv1(g, h_dict)
        for v_id in self.v_id_list:
            h_dict[v_id] = self.act(h_dict[v_id]) #torch.sigmoid(h)
        
        h_dict = self.connconv2(g, h_dict)
        for v_id in self.v_id_list:
            h_dict[v_id] = self.act(h_dict[v_id]) #torch.sigmoid(h)
            
        return h_dict

    def forward_predict(self, g, h_dict):
        ratio_dict, flow_dict = {}, {}
        with g.local_scope():
            for v_id in self.v_id_list:
                edge_feat = g.edges[f'connect_{v_id}'].data['feat']
                g.nodes['node'].data[f'z_{v_id}'] = h_dict[v_id]
                g.apply_edges(lambda edges: {f'hcat_{v_id}': torch.cat([edges.src[f'z_{v_id}'], edges.dst[f'z_{v_id}'], 
                                                                edges.src['coord'], edges.dst['coord'], 
                                                                edge_feat], -1)}, etype=f'connect_{v_id}')
                ratio_dict[v_id] = self.reg_dict[f'connect_{v_id}'](g.edges[f'connect_{v_id}'].data[f'hcat_{v_id}'])
                flow_dict[v_id] = ratio_dict[v_id] * g.edges[f'connect_{v_id}'].data['capacity']

        return ratio_dict, flow_dict
      
class HetTransformer(BaseHetTransformer):
    def __init__(self, in_feats, pre_reg_feats, h_feats, 
                 num_head, base_v_id, v_id_list):
        super(HetTransformer, self).__init__(in_feats, pre_reg_feats, h_feats, 
                                             num_head, base_v_id, v_id_list)
        self.conv1 = TransformerLayer_OD(in_feats = pre_reg_feats+2, \
            o_feats = h_feats, num_head = num_head, v_id_list=v_id_list)

        self.cross_conv = RegressionBranch(h_feats*num_head*len(v_id_list), 64, 64, n_layer=3)

    def forward_od(self, g, h_dict):
        h_dict = self.conv1(g, h_dict)
        for v_id in self.v_id_list:
            h_dict[v_id] = self.act(h_dict[v_id]) #torch.sigmoid(h)
        return h_dict
          
    def forward(self, g):
        node_feat_dict = self.forward_pre_reg(g)
        h_dict = self.forward_od(g, node_feat_dict)
        
        for v_id in self.v_id_list:
            log.debug(f"h_dict[{v_id}].shape: ", h_dict[v_id].shape)
        h_dict = self.forward_conn(g, h_dict)
        ratio_dict, flow_dict = self.forward_predict(g, h_dict)
        return ratio_dict, flow_dict
