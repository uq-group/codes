import dgl
from dgl.nn import GraphConv, GATConv
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils_log import createLogger
log = createLogger(__name__)
log.info('in modelzoo.py')


class SageCONV(nn.Module):
    def __init__(self, mode):
        super(SageCONV, self).__init__()
        self.mode = mode

    def forward(self, g, hn):
        # g.ndata['hn'], hn stores the input node features
        # g.edata['he'], he stores the input edge features
        with g.local_scope():
            g.ndata['hn'] = hn
            # update_all is a message passing API.
            # g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
            if self.mode == 'mean':
                g.update_all(fn.copy_u('hn', 'm'), fn.mean('m', 'hn_aggr'))
            if self.mode == 'sum':
                g.update_all(fn.copy_u('hn', 'm'), fn.sum('m', 'hn_aggr'))
            if self.mode == 'max':
                g.update_all(fn.copy_u('hn', 'm'), fn.max('m', 'hn_aggr'))

            hn_aggr = g.ndata['hn_aggr']
            return hn_aggr

class PPOActorCritic_GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(PPOActorCritic_GNN, self).__init__()
                
        # actor layer
        self.actor_conv1 = nn.Linear(5, hidden_channels)
        self.actor_conv2 = nn.Linear(hidden_channels, hidden_channels)
        self.actor_conv3 = nn.Linear(hidden_channels, hidden_channels)
        self.actor_conv4 = nn.Linear(hidden_channels, 1)
        self.actor_softmax = nn.Softmax(dim=-1)
        
        # critic layer
        self.critic_conv1 = nn.Linear(3, hidden_channels)
        self.critic_conv2 = nn.Linear(hidden_channels, hidden_channels)
        self.critic_conv3 = nn.Linear(hidden_channels, hidden_channels)
        self.critic_conv4 = nn.Linear(hidden_channels, 1)
        
        # other layer
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.normalize = torch.nn.functional.normalize
    
    def actor_forward(self, graph, curr_node=None, curr_node_list=None):
        if type(graph) == list:
            all_policy = []
            for g, curr_n in zip(graph, curr_node_list):
                with g.local_scope():
                    policy_complement = self.actor_forward_single(g, curr_n)
                all_policy.append(policy_complement)
            all_policy = torch.stack(all_policy)
            return all_policy
        else:
            with graph.local_scope():
                # there is the minor problem with the cosine similairy at the last step
                policy_complement = self.actor_forward_single(graph, curr_node)

                return policy_complement

    def actor_forward_single(self, graph, curr_node):
        graph.ndata['cosSim'] = self.cos(graph.ndata['curr2neigh'], graph.ndata['curr2target']).view(-1, 1)
        
        # feat does not follow the sorted order
        # TODO:
        # (done) normalize the feature, but not in 0-1 scale
        feat = torch.cat((graph.ndata['hop'],
                          graph.ndata['cosSim'],
                          graph.ndata['remaining_distance'],
                          graph.ndata['expected_travel_time']), dim=1)
        feat = self.normalize(feat, p=float('inf'), dim=0)
        feat = torch.cat((graph.ndata['current_time'] / 1440.0,
                          feat), dim=1)
          
        x = self.actor_conv1(feat)
        x = F.relu(x)
        x = self.actor_conv2(x)
        x = F.relu(x)
        x = self.actor_conv3(x)
        x = F.relu(x)
        policy = self.actor_conv4(x).squeeze(-1)
        
        # post process for policy output
        # 1. remove current node action
        # 2. expand the dim to MAX_DIM
        # curr_idx = torch.where(graph.ndata['nid'] == graph.ndata['nid'][graph.ndata['current_node']])
        curr_idx = torch.where(graph.ndata['nid'] == curr_node)

        policy[curr_idx] = float('-inf')
        policy_mask = self.actor_softmax(policy)
        
        if policy_mask.shape[0] < 9: # 9 = maximum # out edges + 1
            policy_mask = torch.cat((policy_mask, torch.zeros(9 - policy_mask.shape[0])))
        else:
            pass
        return policy_mask

    def critic_forward(self, graph):
        with graph.local_scope():
            # there is the minor problem with the cosine similairy at the last step
            # graph.ndata['cosSim'] = self.cos(graph.ndata['curr2neigh'], graph.ndata['curr2target']).view(-1, 1)
            
            # feat does not follow the sorted order
            feat = torch.cat((graph.ndata['hop'], 
                              graph.ndata['remaining_distance']), dim=1)
            # dont normalize the critic feature, because it need to evaluate the value of current state
            feat = self.normalize(feat, p=float('inf'), dim=0)
            feat = torch.cat((graph.ndata['current_time'] / 1440.0,
                                feat), dim=1)
            
            graph.ndata['feat'] = feat
            average_feature = dgl.mean_nodes(graph, 'feat')
            
            x = self.critic_conv1(average_feature)
            x = F.relu(x)
            x = self.critic_conv2(x)
            x = F.relu(x)
            x = self.critic_conv3(x)
            x = F.relu(x)
            # log.info(x.shape)
            value = self.critic_conv4(x).squeeze()
            return value

    def forward(self, state, curr_node=None, curr_node_list=None): #, actor_feat, critic_feat
        policy_mask = self.actor_forward(state, curr_node, curr_node_list)        
        value_mask = self.critic_forward(state)
        
        return policy_mask, value_mask

class PPOActorCritic_GNN_new(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(PPOActorCritic_GNN_new, self).__init__()
                
        # actor layer
        # self.actor_conv1 = GraphConv(5, hidden_channels, norm='both', weight=True, bias=True)
        # self.actor_conv1 = GATConv(5, hidden_channels, num_heads=1)
        self.actor_conv1 = SageCONV('mean')
        # self.actor_conv1 = nn.Linear(5, hidden_channels)
        self.actor_conv2 = nn.Linear(5, hidden_channels)
        self.actor_conv3 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.actor_conv4 = nn.Linear(hidden_channels // 2, 1)
        self.actor_softmax = nn.Softmax(dim=-1)
        
        # critic layer
        # self.critic_conv1 = GraphConv(3, hidden_channels, norm='both', weight=True, bias=True)
        # self.critic_conv1 = GATConv(3, hidden_channels, num_heads=1)
        self.critic_conv1 = SageCONV('mean')
        # self.critic_conv1 = nn.Linear(3, hidden_channels)
        self.critic_conv2 = nn.Linear(3*2, hidden_channels)
        self.critic_conv3 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.critic_conv4 = nn.Linear(hidden_channels // 2, 1)
        
        # other layer
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.normalize = torch.nn.functional.normalize
    
    def actor_forward(self, graph, curr_node=None, curr_node_list=None):
        if type(graph) == list:
            all_policy = []
            for g, curr_n in zip(graph, curr_node_list):
                with g.local_scope():
                    policy_complement = self.actor_forward_single(g, curr_n)
                all_policy.append(policy_complement)
            all_policy = torch.stack(all_policy)
            return all_policy
        else:
            with graph.local_scope():
                # there is the minor problem with the cosine similairy at the last step
                policy_complement = self.actor_forward_single(graph, curr_node)

                return policy_complement

    def actor_forward_single(self, graph, curr_node):
        graph.ndata['cosSim'] = self.cos(graph.ndata['curr2neigh'], graph.ndata['curr2target']).view(-1, 1)
        
        # feat does not follow the sorted order
        # TODO:
        # (done) normalize the feature, but not in 0-1 scale
        feat = torch.cat((graph.ndata['hop'],
                          graph.ndata['cosSim'],
                          graph.ndata['remaining_distance'],
                          graph.ndata['expected_travel_time']), dim=1)
        feat = self.normalize(feat, p=float('inf'), dim=0)
        feat = torch.cat((graph.ndata['current_time'] / 1440.0,
                          feat), dim=1)
                
        x = self.actor_conv1(graph, feat).squeeze()        
        # concat_x = torch.cat((feat, x), dim=1)
        concat_x = (feat + x) / 2
        
        x = self.actor_conv2(concat_x)
        x = F.sigmoid(x)
        x = self.actor_conv3(x)
        x = F.sigmoid(x)
        policy = self.actor_conv4(x).squeeze(-1)
        
        # post process for policy output
        # 1. remove current node action
        # 2. expand the dim to MAX_DIM
        # curr_idx = torch.where(graph.ndata['nid'] == graph.ndata['nid'][graph.ndata['current_node']])
        curr_idx = torch.where(graph.ndata['nid'] == curr_node)

        policy[curr_idx] = float('-inf')
        policy_mask = self.actor_softmax(policy)
        
        if policy_mask.shape[0] < 9: # 9 = maximum # out edges + 1
            policy_mask = torch.cat((policy_mask, torch.zeros(9 - policy_mask.shape[0])))
        else:
            pass
        return policy_mask

    def critic_forward(self, graph):
        with graph.local_scope():
            # there is the minor problem with the cosine similairy at the last step
            # graph.ndata['cosSim'] = self.cos(graph.ndata['curr2neigh'], graph.ndata['curr2target']).view(-1, 1)
            
            # feat does not follow the sorted order
            feat = torch.cat((graph.ndata['hop'], 
                              graph.ndata['remaining_distance']), dim=1)
            # dont normalize the critic feature, because it need to evaluate the value of current state
            feat = self.normalize(feat, p=float('inf'), dim=0)
            feat = torch.cat((graph.ndata['current_time'] / 1440.0,
                                feat), dim=1)
            
            graph.ndata['feat'] = feat
            
            x = self.critic_conv1(graph, feat).squeeze()
            graph.ndata['feat1'] = x
            average_feat0 = dgl.mean_nodes(graph, 'feat')
            average_feat1 = dgl.mean_nodes(graph, 'feat1')
            concat_x = torch.cat((average_feat0, average_feat1), dim=1)
            
            x = self.critic_conv2(concat_x)
            x = F.sigmoid(x)
            x = self.critic_conv3(x)
            x = F.sigmoid(x)
            value = self.critic_conv4(x).squeeze()
            return value

    def forward(self, state, curr_node=None, curr_node_list=None): #, actor_feat, critic_feat
        policy_mask = self.actor_forward(state, curr_node, curr_node_list)        
        value_mask = self.critic_forward(state)
        
        return policy_mask, value_mask

class PPOActorCritic_MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(PPOActorCritic_MLP, self).__init__()
                
        # actor layer
        self.actor_conv1 = nn.Linear(7, hidden_channels)
        self.actor_conv2 = nn.Linear(hidden_channels, hidden_channels)
        self.actor_conv3 = nn.Linear(hidden_channels, hidden_channels)
        self.actor_conv4 = nn.Linear(hidden_channels, 1)
        self.actor_softmax = nn.Softmax(dim=-1)
        
        # critic layer
        self.critic_conv1 = nn.Linear(4, hidden_channels)
        self.critic_conv2 = nn.Linear(hidden_channels, hidden_channels)
        self.critic_conv3 = nn.Linear(hidden_channels, hidden_channels)
        self.critic_conv4 = nn.Linear(hidden_channels, 1)
        
        # other layer
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.normalize = torch.nn.functional.normalize
    
    def actor_forward(self, graph, curr_node=None, curr_node_list=None):
        if type(graph) == list:
            all_policy = []
            for g, curr_n in zip(graph, curr_node_list):
                with g.local_scope():
                    policy_complement = self.actor_forward_single(g, curr_n)
                all_policy.append(policy_complement)
            all_policy = torch.stack(all_policy)
            return all_policy
        else:
            with graph.local_scope():
                # there is the minor problem with the cosine similairy at the last step
                policy_complement = self.actor_forward_single(graph, curr_node)

                return policy_complement

    def actor_forward_single(self, graph, curr_node):
        graph.ndata['cosSim'] = self.cos(graph.ndata['curr2neigh'], graph.ndata['curr2target']).view(-1, 1)
        
        # feat does not follow the sorted order
        feat = torch.cat((graph.ndata['hop'],
                          graph.ndata['cosSim'],
                          graph.ndata['curr2target'],
                          graph.ndata['curr2neigh']), dim=1)
        feat = self.normalize(feat, p=float('inf'), dim=0)
        feat = torch.cat((graph.ndata['current_time'] / 1440.0,
                          feat), dim=1)
        
        x = self.actor_conv1(feat)
        x = F.relu(x)
        x = self.actor_conv2(x)
        x = F.relu(x)
        x = self.actor_conv3(x)
        x = F.relu(x)
        policy = self.actor_conv4(x).squeeze(-1)
                
        # post process for policy output
        # 1. remove current node action
        # 2. expand the dim to MAX_DIM
        # curr_idx = torch.where(graph.ndata['nid'] == graph.ndata['nid'][graph.ndata['current_node']])
        curr_idx = torch.where(graph.ndata['nid'] == curr_node)

        policy[curr_idx] = float('-inf')
        policy_mask = self.actor_softmax(policy)
        
        if policy_mask.shape[0] < 9: # 9 = maximum # out edges + 1
            policy_mask = torch.cat((policy_mask, torch.zeros(9 - policy_mask.shape[0])))
        else:
            pass
        return policy_mask

    def critic_forward(self, graph):
        with graph.local_scope():
            # there is the minor problem with the cosine similairy at the last step
            # graph.ndata['cosSim'] = self.cos(graph.ndata['curr2neigh'], graph.ndata['curr2target']).view(-1, 1)
            
            # feat does not follow the sorted order
            feat = torch.cat((graph.ndata['hop'], graph.ndata['curr2target']), dim=1)
            # dont normalize the critic feature, because it need to evaluate the value of current state
            feat = self.normalize(feat, p=float('inf'), dim=0)
            feat = torch.cat((graph.ndata['current_time'] / 1440.0,
                                feat), dim=1)
            
            graph.ndata['feat'] = feat
            average_feature = dgl.mean_nodes(graph, 'feat')
            
            x = self.critic_conv1(average_feature)
            x = F.relu(x)
            x = self.critic_conv2(x)
            x = F.relu(x)
            x = self.critic_conv3(x)
            x = F.relu(x)
            # log.info(x.shape)
            value = self.critic_conv4(x).squeeze()
            return value

    def forward(self, state, curr_node=None, curr_node_list=None): #, actor_feat, critic_feat
        policy_mask = self.actor_forward(state, curr_node, curr_node_list)        
        value_mask = self.critic_forward(state)
        
        return policy_mask, value_mask

class DeepQ_MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(DeepQ_MLP, self).__init__()
                
        self.conv1 = nn.Linear(6, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, hidden_channels)
        self.conv3 = nn.Linear(hidden_channels, hidden_channels)
        self.conv4 = nn.Linear(hidden_channels, 1)
        
        # other layer
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.normalize = torch.nn.functional.normalize

    def forward(self, state, curr_node=None): #, actor_feat, critic_feat
        # need to change the feature,
        # cosine similarity
        state.ndata['cosSim'] = self.cos(state.ndata['curr2neigh'], \
            state.ndata['curr2target']).view(-1, 1)
        
        # feat does not follow the sorted order
        feat = torch.cat((state.ndata['hop'],
                          state.ndata['cosSim'],
                          state.ndata['curr2target'],
                          state.ndata['curr2neigh']), dim=1)
        
        feat = self.normalize(feat, p=float('inf'), dim=0)
        
        x = self.conv1(feat)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        
        return x.squeeze(-1) # [adjacent_nodes]
    
    def get_q_values(self, state, curr_node):
        q_values_all = self.forward(state, curr_node)
        return q_values_all[:-1]

class DeepQ_GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(DeepQ_GNN, self).__init__()
                
        self.conv1 = nn.Linear(4, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, hidden_channels)
        self.conv3 = nn.Linear(hidden_channels, hidden_channels)
        self.conv4 = nn.Linear(hidden_channels, 1)
        
        # other layer
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.normalize = torch.nn.functional.normalize

    def forward(self, state, curr_node=None): #, actor_feat, critic_feat
        # need to change the feature,
        # cosine similarity
        state.ndata['cosSim'] = self.cos(state.ndata['curr2neigh'], \
            state.ndata['curr2target']).view(-1, 1)
        
        # feat does not follow the sorted order
        feat = torch.cat((state.ndata['hop'],
                          state.ndata['cosSim'],
                          state.ndata['remaining_distance'],
                          state.ndata['expected_travel_time']), dim=1)
        
        feat = self.normalize(feat, p=float('inf'), dim=0)
        
        x = self.conv1(feat)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        
        return x.squeeze(-1) # [adjacent_nodes]
    
    def get_q_values(self, state, curr_node):
        q_values_all = self.forward(state, curr_node)
        return q_values_all[:-1]

