import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import os
import pickle

import dgl
import torch
from torch.optim import Adam

from utils_data import MultiRoutingDataset
from utils_log import createLogger
from utils import temp_seed, find_cloest_idx, extract_test_pairs

log = createLogger(__name__)
log.info('in utils_env.py')

def test_env_travel_time(num_stop_action_list, agent, env_idx, env):
    travel_time_per_test = 0
    state, curr_node = env.reset(src = 3, dst = 100, current_t=8*60., train_env=False, is_congest=True) # state is the subgraph of the dgl graph.
    for _ in range(num_stop_action_list[env_idx]):  # Limit each episode to a maximum of 100 steps
                    # input:    state: subgraph
                    # output:   actor feat, critic feat, action, log_prob, reward, new_state
        action = agent.select_action(state, curr_node)
        (next_state, curr_node), (_, travel_time), done = env.step(action)
        travel_time_per_test += travel_time
                    
        if done:    break
        state = next_state
    return travel_time_per_test

def test_gnn_env_travel_time(num_stop_action_list, agent, env_idx, env, tdsp_result, static_result):
    travel_time_per_test_list, done_per_test_list = [], []
    diff_tdsp, diff_static = [], []
    improve = []
    log.info(env.test_node_pair, l=1)
    for src, dst in env.test_node_pair:
        temp_list = []
        min_travel_time_per_test, done_per_test = 1000, False
        for _ in range(5):
            travel_time_per_test, done = select_action_trial(num_stop_action_list, agent, env_idx, env, src, dst)
            if done and (min_travel_time_per_test > travel_time_per_test):
                min_travel_time_per_test, done_per_test = travel_time_per_test, done
            temp_list.append(round(travel_time_per_test, 2))
            
        log.info(src, dst, temp_list, round(min_travel_time_per_test, 2), done, round(tdsp_result[(src, dst)]['mean_travel_time'], 2), \
            round(static_result[(src, dst)]['mean_travel_time'], 2), l=1)
        
        diff_tdsp.append(tdsp_result[(src, dst)]['mean_travel_time'] - min_travel_time_per_test)
        diff_static.append(static_result[(src, dst)]['mean_travel_time'] - min_travel_time_per_test)
        improve.append((tdsp_result[(src, dst)]['mean_travel_time'] - min_travel_time_per_test)/tdsp_result[(src, dst)]['mean_travel_time'])
        travel_time_per_test_list.append(min_travel_time_per_test)
        done_per_test_list.append(done_per_test)
    
    improve = np.array(improve)
    improve = improve[improve > -1.0]
    return np.array(travel_time_per_test_list), done_per_test_list, np.array(diff_tdsp), np.array(diff_static), improve

def select_action_trial(num_stop_action_list, agent, env_idx, env, src, dst):
    state, curr_node = env.reset(src = src, dst = dst, current_t=8*60., train_env=False, is_congest=True) # state is the subgraph of the dgl graph.
    travel_time_per_test = 0
    for _ in range(num_stop_action_list[env_idx]):  # Limit each episode to a maximum of 100 steps
        # input:    state: subgraph
        # output:   actor feat, critic feat, action, log_prob, reward, new_state
        action, _, _, _ = agent.select_action(state, curr_node)
        (next_state, curr_node), (_, travel_time), done = env.step(action)
        travel_time_per_test += travel_time
                            
        if done:    break
        state = next_state
    return travel_time_per_test, done

def remove_temp_model():
    try:
        os.remove("model_multi/temp.pt")
    except:
        pass

def cal_reward_to_go(ep_rewards):
    ep_reward_rtg, discounted_reward = [], 0

    for reward in reversed(ep_rewards):
        discounted_reward = reward + 0.95 * discounted_reward
        ep_reward_rtg.insert(0, discounted_reward)
    ep_reward_rtg = torch.tensor(ep_reward_rtg)
    
    # # TODO:
    # (done) normalize the cumulative rewards
    ep_reward_rtg = (ep_reward_rtg - torch.mean(ep_reward_rtg)) / (torch.std(ep_reward_rtg) + 1e-10)
    
    return ep_reward_rtg

class GNN_ENV_MULTI:
    def __init__(self, args, env_idx=None, test_env=False):
        self.args = args
        self.env_idx = env_idx
        self.G_dgl_list, self.G_nx_list, self.node_pair_list = self.load_datasets() #, self.tdsp_list

        
        self.start_node = None
        self.target_node = None
        self.current_node = None
        self.is_congest = None     # initialize with a congested state
        
        self.final_reward = self.load_final_reward() #0 #None # shortest path from src to dst
        
        # initialize the graph
        self.train_count = 0
        self.chosen_graph_id = None #random.randint(0, len(self.G_dgl_list)-1) # reset the graph_id every reset
        
        self.G_dgl = None #self.G_dgl_list[self.chosen_graph_id]
        self.G_nx = None #self.G_nx_list[self.chosen_graph_id]
        self.test_env = test_env
        if self.test_env:
            self.test_node_pair = self.get_test_node_pair() #self.node_pair_list[self.chosen_graph_id]
        self.time_tick = np.arange(0, 24*60., args.time_interval_list[self.env_idx]).round(2)        
        
        self.softmax = torch.nn.Softmax(dim=0)
        self.normalize = torch.nn.functional.normalize
        
        # debug purpose
        self.current_step = 0
        self.current_reward = []

    def load_datasets(self):
        # create a dataloader and create separated DGL graphs for each nx graph in the dir
        dataset = MultiRoutingDataset(self.args, self.env_idx)
        graph_num = len(dataset)

        graph_dgl_list, graph_nx_list, node_pair_list = [], [], [] #, tdsp_list, []
        for graph_id in range(graph_num):
            graph_dgl, graph_nx, curriculum = dataset[graph_id]
            node_pairs = curriculum['curriculum_list']
            
            print(curriculum['nodes'])
            graph_dgl_list.append(graph_dgl)
            graph_nx_list.append(graph_nx)
            # tdsp_list.append(tdsp)
            node_pair_list.append(node_pairs)

        return graph_dgl_list, graph_nx_list, node_pair_list #, tdsp_list
    
    def load_final_reward(self):
        print("in load_final_reward", self.args.map_name[self.env_idx], self.args.graph_size_list[self.env_idx])
        dir_path = "./{}/graph/{}".format(self.args.map_name[self.env_idx], self.args.graph_size_list[self.env_idx])
        with open('{}/final_reward.pickle'.format(dir_path), 'rb') as handle:
            res = pickle.load(handle)
        return res
    
    def reset(self, src = None, dst = None, current_t = None, train_env=True, is_congest=None):
        # change to a new graph every 20 iteration
        if train_env:
            # in training, update the training counter            
            if self.train_count % 20 == 0:
                self.chosen_graph_id = random.randint(0, len(self.G_dgl_list)-1)
            self.train_count += 1
        else:
            # in test, only choose the last graph
            self.chosen_graph_id = -1
        
        self.G_dgl = self.G_dgl_list[self.chosen_graph_id]
        self.G_nx = self.G_nx_list[self.chosen_graph_id]
        self.node_pair = self.node_pair_list[self.chosen_graph_id]
        
        self.static_travel_time = nx.get_edge_attributes(self.G_nx, 'static_travel_time')
        self.travel_time_mean = {True: nx.get_edge_attributes(self.G_nx, 'congest_mean'),
                                 False: nx.get_edge_attributes(self.G_nx, 'uncongest_mean')}
        self.travel_time_lb = {True: nx.get_edge_attributes(self.G_nx, 'congest_lb'),
                               False: nx.get_edge_attributes(self.G_nx, 'uncongest_lb')}
        self.travel_time_ub = {True: nx.get_edge_attributes(self.G_nx, 'congest_ub'),
                               False: nx.get_edge_attributes(self.G_nx, 'uncongest_ub')}
        
        self.prob_c2c = nx.get_edge_attributes(self.G_nx, 'prob_c2c')
        self.prob_u2u = nx.get_edge_attributes(self.G_nx, 'prob_u2u')
        
        # reset the environment: source, destination, starting time.
        if src is None and dst is None:
            self.start_node, self.target_node = random.choice(self.node_pair)
            while self.start_node == self.target_node:
                self.start_node, self.target_node = random.choice(self.node_pair)
            self.current_t = random.sample(np.arange(7*60., 9*60., self.args.time_interval_list[self.env_idx]).round(2).tolist(), k=1)[0]
            self.is_congest = random.choice([True, False])
        else:
            self.start_node, self.target_node = src, dst
            self.current_t = current_t
            self.is_congest = is_congest
            
        self.current_node = self.start_node
        # self.final_reward = 0
        
        self._set_ndata()
        self._set_edata()

        self.current_step = 0
        self.current_reward = []
            
        return self._get_state()

    def get_test_node_pair(self, k = 40):
        G_nx = self.G_nx_list[-1]
        node_pairs = extract_test_pairs(G_nx, n_pair=k)
        return node_pairs
        
    def step(self, action):
        next_node = action
        past_node = self.current_node
        # You might want to check here if next_node is a valid action
        # (i.e., if it is connected to current_node)
        self.current_node = next_node
        
        # TODO: 
        # (done) calculate the time index recurrently --> just in case the time going to next 24h.
        # time_idx = np.argmax(self.time_tick > (self.current_t % 1440.0)) # find the suitable time slot
        time_idx = find_cloest_idx(self.time_tick, (self.current_t % 1440.0))
        
        # update the congestion state
        if (self.is_congest == True) and np.random.uniform() > self.prob_c2c[(past_node, next_node)][time_idx]:
            self.is_congest = False
        elif (self.is_congest == False) and np.random.uniform() > self.prob_u2u[(past_node, next_node)][time_idx]:
            self.is_congest = True
        else:
            pass
        
        lb = self.travel_time_lb[self.is_congest][(past_node, next_node)][time_idx]
        ub = self.travel_time_ub[self.is_congest][(past_node, next_node)][time_idx]
        travel_time = np.random.uniform(lb, ub)
        
        if self.current_node == self.target_node:
            if self.chosen_graph_id == -1:
                try:
                    reward = (self.final_reward[0][self.start_node][self.target_node] - travel_time, travel_time)  # Large reward when reaching the target
                except:
                    reward = (- travel_time, travel_time)
            else:
                reward = (self.final_reward[self.chosen_graph_id][self.start_node][self.target_node] - travel_time, travel_time)  # Large reward when reaching the target
            done = True
        else:
            reward = (-travel_time, travel_time)  # Small penalty for each step            
            done = False
        
        self.current_t += travel_time # update the current time, 
        
        self.G_dgl.ndata['current_node'][past_node] = 0
        self.G_dgl.ndata['current_node'][self.current_node] = 1
        
        self.current_step += 1
        self.current_reward.append(reward[1])
        
        return self._get_state(), reward, done

    def _set_ndata(self):
        self.G_dgl.ndata['hop'] = dgl.shortest_dist(self.G_dgl, root=self.target_node).view(-1, 1)
        self.G_dgl.ndata['nid'] = self.G_dgl.nodes()
        
        self.G_dgl.ndata['curr2neigh'] = self.G_dgl.ndata['pos'] - self.G_dgl.ndata['pos'][self.current_node, :]
        
        # curr2target --> actually is neighbor to target
        self.G_dgl.ndata['curr2target'] = self.G_dgl.ndata['pos'][self.target_node, :] - self.G_dgl.ndata['pos'] 
        
        self.G_dgl.ndata['current_node'] = torch.zeros_like(self.G_dgl.ndata['nid'], dtype=torch.bool)
        self.G_dgl.ndata['current_node'][self.current_node] = 1
        
        # TODO:
        # -- try new feature --> include attention or RNN sequence in the routing, add time in the feature
        # -- (done) add time in the feature
        
        self.G_dgl.ndata['current_time'] = torch.ones_like(self.G_dgl.ndata['hop'], dtype=torch.float64) * self.current_t

        neighbors = self.G_dgl.successors(self.current_node).cpu().numpy().tolist()
        path_count = [0 for _ in neighbors]
        
        trace_list, time_each_list = [], []
        
        while not all(path_count):
            current_hop = self.G_dgl.ndata['hop'][self.current_node].item()
            if current_hop > 50:
                walk_length = 3
            elif current_hop > 20:
                walk_length = 3
            else:
                walk_length = 3
                
            trace, path = dgl.sampling.node2vec_random_walk(self.G_dgl, [self.current_node]*10, p=10, q=0.1, walk_length=walk_length, return_eids=True)

            time_each = torch.zeros_like(path, dtype=torch.float64)
            time_curr = torch.ones_like(path[:, 0], dtype=torch.float64) * self.current_t

            self.G_dgl.ndata['expected_travel_time'] = torch.zeros_like(self.G_dgl.ndata['hop'], dtype=torch.float64)
            self.G_dgl.ndata['remaining_distance'] = torch.zeros_like(self.G_dgl.ndata['hop'], dtype=torch.float64)
                        
            for idx in range(time_each.shape[1]):
                edge_visited = path[:, idx]
                # TODO:
                # (done) debug: out of bound error
                # (done) replace 0.1 with a finer_time_interval
                # (done) calculate the time index recurrently --> just in case the time going to next 24h.

                time_idx = ((time_curr % 1440.0)/self.args.time_interval_list[self.env_idx]).type(torch.int64)
            
                mean_c = self.G_dgl.edata['congest_mean'][edge_visited, time_idx]
                std_c = (self.G_dgl.edata['congest_ub'][edge_visited, time_idx] - self.G_dgl.edata['congest_lb'][edge_visited, time_idx]) / 6
                mean_u = self.G_dgl.edata['uncongest_mean'][edge_visited, time_idx]
                std_u = (self.G_dgl.edata['uncongest_ub'][edge_visited, time_idx] - self.G_dgl.edata['uncongest_lb'][edge_visited, time_idx]) / 6
            
                # time_each[:, idx] = torch.normal(mean, std) # travel_time
                try:
                    
                    travel_time_1 = torch.distributions.Uniform(self.G_dgl.edata['congest_lb'][edge_visited, time_idx], \
                                                                    self.G_dgl.edata['congest_ub'][edge_visited, time_idx]).sample()
                    travel_time_2 = torch.distributions.Uniform(self.G_dgl.edata['uncongest_lb'][edge_visited, time_idx], \
                                                                    self.G_dgl.edata['uncongest_ub'][edge_visited, time_idx]).sample()
                    
                    # print(self.G_dgl.edata['prob_c2c'].shape)
                    # print(self.G_dgl.edata['prob_c2c'][edge_visited, time_idx])
                    # print(edge_visited)
                    
                    if self.is_congest == True:
                        time_each[:, idx] = self.G_dgl.edata['prob_c2c'][edge_visited, time_idx] * travel_time_1 + \
                                            (1-self.G_dgl.edata['prob_c2c'][edge_visited, time_idx]) * travel_time_2
                    else:
                        time_each[:, idx] = self.G_dgl.edata['prob_u2u'][edge_visited, time_idx] * travel_time_2 + \
                                            (1-self.G_dgl.edata['prob_u2u'][edge_visited, time_idx]) * travel_time_1
                except:
                    travel_time_1 = torch.normal(mean_c, std_c) # travel_time
                    travel_time_2 = torch.normal(mean_u, std_u) # travel_time
                    
                    if self.is_congest == True:
                        time_each[:, idx] = self.G_dgl.edata['prob_c2c'][edge_visited, time_idx] * travel_time_1 + \
                                            (1-self.G_dgl.edata['prob_c2c'][edge_visited, time_idx]) * travel_time_2
                    else:
                        time_each[:, idx] = self.G_dgl.edata['prob_u2u'][edge_visited, time_idx] * travel_time_2 + \
                                            (1-self.G_dgl.edata['prob_u2u'][edge_visited, time_idx]) * travel_time_1
                    
                # print(time_each[:, idx])
                # print('-------------------------------')
                time_curr += time_each[:, idx]

            # collect all result
            trace_list.append(trace)
            time_each_list.append(time_each)
            
            # update path count
            for neigh_idx, neigh in enumerate(neighbors):
                path_count[neigh_idx] += sum(trace[:,1]==neigh).item()
        
        # collect trace, time_each
        trace_list = torch.vstack(trace_list)
        time_each_list = torch.vstack(time_each_list)
        time_overall = torch.sum(time_each_list, dim=1, keepdim=True)
                
        for neigh in neighbors:
            trace_id = trace_list[:,1]==neigh
            trace_each_node = trace_list[trace_id, 1:]
            log.info("trace_each_node", trace_each_node)
            
            # find the remaining distance of each trace
            remaining_distance = torch.sqrt(torch.sum(torch.square(self.G_dgl.ndata['pos'][trace_each_node[:, -1]] - \
                self.G_dgl.ndata['pos'][self.target_node]), dim=1))
            
            # calculate the softmax
            travel_time = time_overall[trace_id]
            weight = self.softmax(-self.normalize(remaining_distance, dim=0, p=float('inf')))
            expected_travel_time = torch.sum(travel_time*weight)
            
            # update the ndata['expected_travel_time']
            # TODO:
            # normalize the expected travel time by number of hop
            self.G_dgl.ndata['expected_travel_time'][neigh] = expected_travel_time
            self.G_dgl.ndata['remaining_distance'][neigh] = torch.min(remaining_distance)
    
    def _set_edata(self):
        pass
    
    def _get_state(self):
        # TODO
        # (done) 1. update all the graph data after each step.
        self._set_ndata()
        self._set_edata()
        
        # the state will be a subgraph of the whole graph. And the graph will be passed in the actor and critic.
        neighbors = self.G_dgl.successors(self.current_node).cpu().numpy().tolist()
        subgraph = dgl.node_subgraph(self.G_dgl, neighbors + [self.current_node] ) # [0,1,2] + [3] the order will determine the order in the subgraph
        
        log.info(self)
        log.info(subgraph.ndata['nid'])
        log.info("expected_travel_time", subgraph.ndata['expected_travel_time'])
        log.info("remaining distance", subgraph.ndata['remaining_distance'])

        log.info(self.current_node, self.target_node)

        return (subgraph, self.current_node)

    def __str__(self):
        return "start node {}, current node {}, target_node {}, current_t {}".format(self.start_node, \
            self.current_node, self.target_node, round(self.current_t, 2))

class MLP_ENV_MULTI:
    def __init__(self, args, env_idx=None, test_env=False):
        self.args = args
        self.env_idx = env_idx
        self.G_dgl_list, self.G_nx_list, self.node_pair_list = self.load_datasets() #, self.tdsp_list

        
        self.start_node = None
        self.target_node = None
        self.current_node = None
        self.is_congest = None     # initialize with a congested state
        
        self.final_reward = self.load_final_reward() # = 0 #None # shortest path from src to dst
        
        # initialize the graph
        self.train_count = 0
        self.chosen_graph_id = None #random.randint(0, len(self.G_dgl_list)-1) # reset the graph_id every reset
        
        self.G_dgl = None #self.G_dgl_list[self.chosen_graph_id]
        self.G_nx = None #self.G_nx_list[self.chosen_graph_id]
        self.test_env = test_env
        if self.test_env:
            self.test_node_pair = self.get_test_node_pair() #self.node_pair_list[self.chosen_graph_id]
        # self.node_pair = None #self.node_pair_list[self.chosen_graph_id]
        self.time_tick = np.arange(0, 24*60., args.time_interval_list[self.env_idx]).round(2)
            
        self.softmax = torch.nn.Softmax(dim=0)
        self.normalize = torch.nn.functional.normalize
        
        # debug purpose
        self.current_step = 0
        self.current_reward = []

    def load_datasets(self):
        # create a dataloader and create separated DGL graphs for each nx graph in the dir
        dataset = MultiRoutingDataset(self.args, self.env_idx)
        graph_num = len(dataset)

        graph_dgl_list, graph_nx_list, node_pair_list = [], [], [] #, tdsp_list, []
        for graph_id in range(graph_num):
            graph_dgl, graph_nx, curriculum = dataset[graph_id]
            node_pairs = curriculum['curriculum_list']
            
            print(curriculum['nodes'])
            graph_dgl_list.append(graph_dgl)
            graph_nx_list.append(graph_nx)
            # tdsp_list.append(tdsp)
            node_pair_list.append(node_pairs)

        return graph_dgl_list, graph_nx_list, node_pair_list #, tdsp_list
    
    def load_final_reward(self):
        print("in load_final_reward", self.args.map_name[self.env_idx], self.args.graph_size_list[self.env_idx])
        dir_path = "./{}/graph/{}".format(self.args.map_name[self.env_idx], self.args.graph_size_list[self.env_idx])
        with open('{}/final_reward.pickle'.format(dir_path), 'rb') as handle:
            res = pickle.load(handle)
        return res
    
    def reset(self, src = None, dst = None, current_t = None, train_env=True, is_congest=None):
        # change to a new graph every 20 iteration
        if train_env:
            # in training, update the training counter            
            if self.train_count % 20 == 0:
                self.chosen_graph_id = random.randint(0, len(self.G_dgl_list)-1)
            self.train_count += 1
        else:
            # in test, only choose the last graph
            self.chosen_graph_id = -1
        
        self.G_dgl = self.G_dgl_list[self.chosen_graph_id]
        self.G_nx = self.G_nx_list[self.chosen_graph_id]
        self.node_pair = self.node_pair_list[self.chosen_graph_id]
        
        self.static_travel_time = nx.get_edge_attributes(self.G_nx, 'static_travel_time')
        self.travel_time_mean = {True: nx.get_edge_attributes(self.G_nx, 'congest_mean'),
                                 False: nx.get_edge_attributes(self.G_nx, 'uncongest_mean')}
        self.travel_time_lb = {True: nx.get_edge_attributes(self.G_nx, 'congest_lb'),
                               False: nx.get_edge_attributes(self.G_nx, 'uncongest_lb')}
        self.travel_time_ub = {True: nx.get_edge_attributes(self.G_nx, 'congest_ub'),
                               False: nx.get_edge_attributes(self.G_nx, 'uncongest_ub')}
        
        self.prob_c2c = nx.get_edge_attributes(self.G_nx, 'prob_c2c')
        self.prob_u2u = nx.get_edge_attributes(self.G_nx, 'prob_u2u')
        
        # reset the environment: source, destination, starting time.
        if src is None and dst is None:
            self.start_node, self.target_node = random.choice(self.node_pair)
            while self.start_node == self.target_node:
                self.start_node, self.target_node = random.choice(self.node_pair)
            self.current_t = random.sample(np.arange(7*60., 9*60., self.args.time_interval_list[self.env_idx]).round(2).tolist(), k=1)[0]
            self.is_congest = random.choice([True, False])
        else:
            self.start_node, self.target_node = src, dst
            self.current_t = current_t
            self.is_congest = is_congest
            
        self.current_node = self.start_node
        # self.final_reward = 0
        
        self._set_ndata()
        self._set_edata()

        self.current_step = 0
        self.current_reward = []
            
        return self._get_state()

    def get_test_node_pair(self, k = 40):
        G_nx = self.G_nx_list[-1]
        node_pairs = extract_test_pairs(G_nx, n_pair=k)
        return node_pairs
        
    def step(self, action):
        next_node = action
        past_node = self.current_node
        # You might want to check here if next_node is a valid action
        # (i.e., if it is connected to current_node)
        self.current_node = next_node
        
        # TODO: 
        # (done) calculate the time index recurrently --> just in case the time going to next 24h.
        # time_idx = np.argmax(self.time_tick > (self.current_t % 1440.0)) # find the suitable time slot
        time_idx = find_cloest_idx(self.time_tick, (self.current_t % 1440.0))
        
        # update the congestion state
        if (self.is_congest == True) and np.random.uniform() > self.prob_c2c[(past_node, next_node)][time_idx]:
            self.is_congest = False
        elif (self.is_congest == False) and np.random.uniform() > self.prob_u2u[(past_node, next_node)][time_idx]:
            self.is_congest = True
        else:
            pass
        
        lb = self.travel_time_lb[self.is_congest][(past_node, next_node)][time_idx]
        ub = self.travel_time_ub[self.is_congest][(past_node, next_node)][time_idx]
        travel_time = np.random.uniform(lb, ub)
        
        if self.current_node == self.target_node:
            if self.chosen_graph_id == -1:
                try:
                    reward = (self.final_reward[0][self.start_node][self.target_node] - travel_time, travel_time)  # Large reward when reaching the target
                except:
                    reward = (- travel_time, travel_time)
            else:
                reward = (self.final_reward[self.chosen_graph_id][self.start_node][self.target_node] - travel_time, travel_time)  # Large reward when reaching the target
            done = True
        else:
            reward = (-travel_time, travel_time)  # Small penalty for each step            
            done = False
        
        self.current_t += travel_time # update the current time, 
        
        self.G_dgl.ndata['current_node'][past_node] = 0
        self.G_dgl.ndata['current_node'][self.current_node] = 1
        
        self.current_step += 1
        self.current_reward.append(reward[1])
        
        return self._get_state(), reward, done

    def _set_ndata(self):
        self.G_dgl.ndata['hop'] = dgl.shortest_dist(self.G_dgl, root=self.target_node).view(-1, 1)
        self.G_dgl.ndata['nid'] = self.G_dgl.nodes()
        
        self.G_dgl.ndata['curr2neigh'] = self.G_dgl.ndata['pos'] - self.G_dgl.ndata['pos'][self.current_node, :]
        # curr2target --> actually is neighbor to target
        self.G_dgl.ndata['curr2target'] = self.G_dgl.ndata['pos'][self.target_node, :] - self.G_dgl.ndata['pos'] 
        
        self.G_dgl.ndata['current_node'] = torch.zeros_like(self.G_dgl.ndata['nid'], dtype=torch.bool)
        self.G_dgl.ndata['current_node'][self.current_node] = 1
        
        # TODO:
        # -- try new feature --> include attention or RNN sequence in the routing
        # -- try random walk with different length --> multi-scale feature
        
        self.G_dgl.ndata['current_time'] = torch.ones_like(self.G_dgl.ndata['hop'], dtype=torch.float64) * self.current_t

    def _set_edata(self):
        pass
    
    def _get_state(self):
        # TODO
        # (done) 1. update all the graph data after each step.
        self._set_ndata()
        self._set_edata()
        
        # the state will be a subgraph of the whole graph. And the graph will be passed in the actor and critic.
        neighbors = self.G_dgl.successors(self.current_node).cpu().numpy().tolist()
        subgraph = dgl.node_subgraph(self.G_dgl, neighbors + [self.current_node] ) # [0,1,2] + [3] the order will determine the order in the subgraph
        
        # log.info(self)
        # log.info(subgraph.ndata['nid'])
        # log.info("expected_travel_time", subgraph.ndata['expected_travel_time'])
        # log.info("remaining distance", subgraph.ndata['remaining_distance'])

        log.info(self.current_node, self.target_node)

        return (subgraph, self.current_node)

    def __str__(self):
        return "start node {}, current node {}, target_node {}, current_t {}".format(self.start_node, \
            self.current_node, self.target_node, round(self.current_t, 2))
        
class GRAPH_ENV_MULTI:
    def __init__(self, args, env_idx=None):
        self.args = args
        self.env_idx = env_idx
        self.G_dgl_list, self.G_nx_list, self.node_pair_list = self.load_datasets() #, self.tdsp_list

        
        self.start_node = None
        self.target_node = None
        self.current_node = None
        self.is_congest = None     # initialize with a congested state
        
        self.final_reward = 0 #None # shortest path from src to dst
        
        # initialize the graph
        self.train_count = 0
        self.chosen_graph_id = None #random.randint(0, len(self.G_dgl_list)-1) # reset the graph_id every reset
        
        self.G_dgl = None #self.G_dgl_list[self.chosen_graph_id]
        self.G_nx = None #self.G_nx_list[self.chosen_graph_id]
        self.node_pair = None #self.node_pair_list[self.chosen_graph_id]
        
        
        self.time_tick = np.arange(0, 24*60., args.time_interval_list[self.env_idx]).round(2)
        
        
        self.softmax = torch.nn.Softmax(dim=0)
        self.normalize = torch.nn.functional.normalize
        
        # debug purpose
        self.current_step = 0
        self.current_reward = []

    def load_datasets(self):
        # create a dataloader and create separated DGL graphs for each nx graph in the dir
        dataset = MultiRoutingDataset(self.args, self.env_idx)
        graph_num = len(dataset)

        graph_dgl_list, graph_nx_list, node_pair_list = [], [], [] #, tdsp_list, []
        for graph_id in range(graph_num):
            graph_dgl, graph_nx, curriculum = dataset[graph_id]
            node_pairs = curriculum['curriculum_list']
            
            graph_dgl_list.append(graph_dgl)
            graph_nx_list.append(graph_nx)
            # tdsp_list.append(tdsp)
            node_pair_list.append(node_pairs)

        return graph_dgl_list, graph_nx_list, node_pair_list #, tdsp_list
    
    def reset(self, src = None, dst = None, current_t = None, train_env=True, is_congest=None):
        # change to a new graph every 20 iteration
        if train_env:
            # in training, update the training counter            
            if self.train_count % 20 == 0:
                self.chosen_graph_id = random.randint(0, len(self.G_dgl_list)-1)
                self.load_features()
            self.train_count += 1
        else:
            # in test, only choose the last graph
            self.chosen_graph_id = -1
            self.load_features()

        # reset the environment: source, destination, starting time.
        if src is None and dst is None:
            self.start_node, self.target_node = random.choice(self.node_pair)
            while self.start_node == self.target_node:
                self.start_node, self.target_node = random.choice(self.node_pair)
            self.current_t = random.sample(np.arange(6*60., 10*60., self.args.time_interval_list[self.env_idx]).round(2).tolist(), k=1)[0]
            self.is_congest = random.choice([True, False])
        else:
            self.start_node, self.target_node = src, dst
            self.current_t = current_t
            self.is_congest = is_congest
            
        self.current_node = self.start_node
        # self.final_reward = 0
        
        self._set_ndata()
        self._set_edata()

        self.current_step = 0
        self.current_reward = []
            
        return self._get_state()

    def load_features(self):
        self.G_dgl = self.G_dgl_list[self.chosen_graph_id]
        self.G_nx = self.G_nx_list[self.chosen_graph_id]
        self.node_pair = self.node_pair_list[self.chosen_graph_id]
        
        self.static_travel_time = nx.get_edge_attributes(self.G_nx, 'static_travel_time')
        self.travel_time_mean = {True: nx.get_edge_attributes(self.G_nx, 'congest_mean'),
                                 False: nx.get_edge_attributes(self.G_nx, 'uncongest_mean')}
        self.travel_time_lb = {True: nx.get_edge_attributes(self.G_nx, 'congest_lb'),
                               False: nx.get_edge_attributes(self.G_nx, 'uncongest_lb')}
        self.travel_time_ub = {True: nx.get_edge_attributes(self.G_nx, 'congest_ub'),
                               False: nx.get_edge_attributes(self.G_nx, 'uncongest_ub')}
        
        self.prob_c2c = nx.get_edge_attributes(self.G_nx, 'prob_c2c')
        self.prob_u2u = nx.get_edge_attributes(self.G_nx, 'prob_u2u')

    def get_test_node_pair(self, k = 100):
        # random.seed(42)
        # return random.sample(self.node_pair, k=10)
        pass
        
    def step(self, action):
        next_node = action
        past_node = self.current_node
        # You might want to check here if next_node is a valid action
        # (i.e., if it is connected to current_node)
        self.current_node = next_node

        time_idx = np.argmax(self.time_tick > self.current_t) # find the suitable time slot
        
        # update the congestion state
        if (self.is_congest == True) and np.random.uniform() > self.prob_c2c[(past_node, next_node)][time_idx]:
            self.is_congest = False
        elif (self.is_congest == False) and np.random.uniform() > self.prob_u2u[(past_node, next_node)][time_idx]:
            self.is_congest = True
        else:
            pass
        
        mean = self.travel_time_mean[self.is_congest][(past_node, next_node)][time_idx]
        std = (self.travel_time_ub[self.is_congest][(past_node, next_node)][time_idx] - \
            self.travel_time_lb[self.is_congest][(past_node, next_node)][time_idx]) / 6
        travel_time = np.random.normal(mean, std)
        
        # lb = self.travel_time_lb[(past_node, next_node)][time_idx]
        # ub = self.travel_time_ub[(past_node, next_node)][time_idx]
        # travel_time = np.random.uniform(lb, ub)
        
        if self.current_node == self.target_node:
            reward = (self.final_reward - travel_time, travel_time)  # Large reward when reaching the target
            done = True
        else:
            reward = (-travel_time, travel_time)  # Small penalty for each step            
            done = False
        
        self.current_t += travel_time # update the current time, 

        self.G_dgl.ndata['current_node'][past_node] = 0
        self.G_dgl.ndata['current_node'][self.current_node] = 1
        
        self.current_step += 1
        self.current_reward.append(reward[1])
        
        return self._get_state(), reward, done

    def _set_ndata(self):
        self.G_dgl.ndata['hop'] = dgl.shortest_dist(self.G_dgl, root=self.target_node).view(-1, 1)
        self.G_dgl.ndata['nid'] = self.G_dgl.nodes()
        
        self.G_dgl.ndata['curr2neigh'] = self.G_dgl.ndata['pos'] - self.G_dgl.ndata['pos'][self.current_node, :]
        self.G_dgl.ndata['curr2target'] = self.G_dgl.ndata['pos'][self.target_node, :] - self.G_dgl.ndata['pos'] 
        
        self.G_dgl.ndata['current_node'] = torch.zeros_like(self.G_dgl.ndata['nid'], dtype=torch.bool)
        self.G_dgl.ndata['current_node'][self.current_node] = 1
      
    def _set_edata(self):
        pass
    
    def _get_state(self):
        # TODO
        # (done) 1. update all the graph data after each step.
        self._set_ndata()
        self._set_edata()
        
        # the state will be a subgraph of the whole graph. And the graph will be passed in the actor and critic.
        neighbors = self.G_dgl.successors(self.current_node).cpu().numpy().tolist()
        subgraph = dgl.node_subgraph(self.G_dgl, neighbors + [self.current_node] ) # [0,1,2] + [3] the order will determine the order in the subgraph

        log.info(self.current_node, self.target_node)

        return (subgraph, self.current_node)

    def __str__(self):
        return "start node {}, current node {}, target_node {}, current_t {}".format(self.start_node, \
            self.current_node, self.target_node, round(self.current_t, 2))


