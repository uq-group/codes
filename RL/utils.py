import copy
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
import pickle
import random
from scipy.stats import lognorm
import contextlib

from utils_log import createLogger
log = createLogger(__name__)
log.info("in utils.py")

def load_whole_graph(args):
    if args.map_name == 'Manhattan':
        G = nx.read_edgelist("{}/osm/txt/edge_connect.txt".format(args.map_name), nodetype=int, data=(('length',float),))
        idx, x_coord, y_coord = np.loadtxt("{}/osm/txt/node_connect.txt".format(args.map_name), delimiter=' ', usecols=(0, 1, 2), unpack=True, \
            dtype={'names': ('id', 'lat', 'lon'), 'formats': ('i4', 'f8', 'f8')})
    else:
        G = nx.read_edgelist("{}/osm/txt/edge.txt".format(args.map_name), nodetype=int, data=(('length',float),))
        idx, x_coord, y_coord = np.loadtxt("{}/osm/txt/node.txt".format(args.map_name), delimiter=' ', usecols=(0, 1, 2), unpack=True, \
            dtype={'names': ('id', 'lat', 'lon'), 'formats': ('i4', 'f8', 'f8')})
    nx.set_node_attributes(G, {i:coord for i, coord in zip(idx, x_coord)}, "x_coord")
    nx.set_node_attributes(G, {i:coord for i, coord in zip(idx, y_coord)}, "y_coord")
    log.info("undirected", len(G.nodes()), len(G.edges()), nx.is_directed(G), l=1)
    
    H = G.to_directed()
    log.info("directed", len(H.nodes()), len(H.edges()), nx.is_directed(H), nx.is_strongly_connected(H), l=1)
    
    return H

def build_edge_feat(args, H):
    # generate time_tick based on time resolution
    time_tick = np.arange(0, 24*60., args.time_resolution).round(2)

    
    link_length_dict = nx.get_edge_attributes(H, 'length')
    static_travel_time = {link: length/args.travel_time_normalizer for link, length in link_length_dict.items()}
    nx.set_edge_attributes(H, static_travel_time, "static_travel_time")
    
    
    # build the travel time history: under congested and uncongested condition
    uncongest_mean_dict, uncongest_lb_dict, uncongest_ub_dict = build_link_travel_time_feat(time_tick, congest=False, keys=static_travel_time.keys())
    congest_mean_dict, congest_lb_dict, congest_ub_dict = build_link_travel_time_feat(time_tick, congest=True, keys=static_travel_time.keys())
    
    for k in static_travel_time.keys():
        uncongest_mean_dict[k] = static_travel_time[k] * uncongest_mean_dict[k]
        uncongest_lb_dict[k] = static_travel_time[k] * uncongest_lb_dict[k]
        uncongest_ub_dict[k] = static_travel_time[k] * uncongest_ub_dict[k]
        congest_mean_dict[k] = static_travel_time[k] * congest_mean_dict[k]
        congest_lb_dict[k] = static_travel_time[k] * congest_lb_dict[k]
        congest_ub_dict[k] = static_travel_time[k] * congest_ub_dict[k]
    nx.set_edge_attributes(H, uncongest_mean_dict, "uncongest_mean")
    nx.set_edge_attributes(H, uncongest_lb_dict, "uncongest_lb")
    nx.set_edge_attributes(H, uncongest_ub_dict, "uncongest_ub")
    nx.set_edge_attributes(H, congest_mean_dict, "congest_mean")
    nx.set_edge_attributes(H, congest_lb_dict, "congest_lb")
    nx.set_edge_attributes(H, congest_ub_dict, "congest_ub")
    
    
    # build the transirition probability
    # p(c|c) --> 1xT array
    # p(u|u) --> 1xT array
    # p(c|c) and p(u|u) is determined by a normal distribution
    prob_c2c_dict = build_transition_prob_feat(time_tick, congest=True, keys=static_travel_time.keys())
    prob_u2u_dict = build_transition_prob_feat(time_tick, congest=False, keys=static_travel_time.keys())
    nx.set_edge_attributes(H, prob_c2c_dict, "prob_c2c")
    nx.set_edge_attributes(H, prob_u2u_dict, "prob_u2u")

    return H

def build_link_travel_time_feat(time_tick, congest=True, keys=None):
    mean_dict, lb_dict, ub_dict = {}, {}, {}
    
    for k in keys:
        # bi-model travel time: one peak in 8 am, another peak in 6pm.
        s, scale = np.random.uniform(low=0.17, high=0.23), np.random.uniform(low=0.25, high=0.35)
        travel_time_1 = lognorm.pdf(time_tick/1440, s=s, loc=0.05, scale=scale)
        s, scale = np.random.uniform(low=0.17, high=0.23), np.random.uniform(low=0.25, high=0.35)
        travel_time_2 = lognorm.pdf(1-time_tick/1440, s=s, loc=0.05, scale=scale)
        
        participate_ratio = np.random.uniform(low=0.45, high=0.55)
        travel_time = participate_ratio*travel_time_1 + (1-participate_ratio)*travel_time_2
        
        # build the variation from the travel_time
        # amplified: uncongested: 1-1.5, congested: 1-2
        if congest:
            amplitude = np.random.uniform(low=1.9, high=2.1)
            bound = 0.2
        else:
            amplitude = np.random.uniform(low=1.4, high=1.5)
            bound = 0.15
            
        mean_travel_time = (amplitude - 1)*(travel_time - np.min(travel_time))/(np.max(travel_time)-np.min(travel_time)) + 1.0
        lb_travel_time = (amplitude - bound - 1)*(travel_time - np.min(travel_time))/(np.max(travel_time)-np.min(travel_time)) + 1.0
        ub_travel_time = (amplitude + bound - 1)*(travel_time - np.min(travel_time))/(np.max(travel_time)-np.min(travel_time)) + 1.0

        
        mean_dict[k] = mean_travel_time
        lb_dict[k] = lb_travel_time
        ub_dict[k] = ub_travel_time
          
    return mean_dict, lb_dict, ub_dict

def build_transition_prob_feat(time_tick, congest=True, keys=None):
    prob_dict = {}
    
    for k in keys:
        s, scale = np.random.uniform(low=0.17, high=0.23), np.random.uniform(low=0.25, high=0.35)
        transition_prob_1 = lognorm.pdf(time_tick/1440, s=s, loc=0.05, scale=scale)
        s, scale = np.random.uniform(low=0.17, high=0.23), np.random.uniform(low=0.25, high=0.35)
        transition_prob_2 = lognorm.pdf(1-time_tick/1440, s=s, loc=0.05, scale=scale)
        
        participate_ratio = np.random.uniform(low=0.45, high=0.55)
        transition_prob = participate_ratio*transition_prob_1 + (1-participate_ratio)*transition_prob_2
        
        # build the variation from the travel_time
        # amplified: uncongested: 1-1.5, congested: 1-2
        if congest:
            amplitude = 0.8
        else:
            amplitude = 0.75
            
        mean_transition_prob = (amplitude - 0.5)*(transition_prob - np.min(transition_prob))/(np.max(transition_prob)-np.min(transition_prob)) + 0.5
        transition_prob = np.clip(np.random.normal(mean_transition_prob, 0.05, size=mean_transition_prob.shape), 0.0, 1.0)

        
        prob_dict[k] = transition_prob
        
    return prob_dict

def extract_subgraph(args, H):
    if not os.path.exists('./{}/graph/{}'.format(args.map_name, args.n_node)):
        os.makedirs('./{}/graph/{}'.format(args.map_name, args.n_node))
    
    # return the subgraph with all the feature
    subgraphs = []
    for graph_id in range(args.n_graph):
        node = random.choice(list(H.nodes()))
        k = 3
        while True:
            subgraph = nx.ego_graph(H, node, radius=k)
            k += 1
            if len(subgraph.nodes()) > args.n_node: break

        # reorder the subgraph --> to 0, 1, 2, ...
        node_order = {old_node_id: new_node_id for new_node_id, old_node_id in enumerate(subgraph.nodes())}
        subgraph = nx.relabel_nodes(subgraph, node_order, copy=True)
        
        
        subgraphs.append(subgraph)
        print(len(subgraph.nodes()), nx.is_strongly_connected(subgraph))
        
        # save the subgraph into gpickle
        nx.write_gpickle(subgraph, "./{}/graph/{}/graph_{}_{}.gpickle".format(args.map_name, args.n_node, graph_id, args.time_resolution))
    
    return subgraphs

def extract_curriculum(args, subgraphs):
    from collections import defaultdict
    
    # return the subgraph with all the feature
    for graph_id, subgraph in enumerate(subgraphs):
        print(len(subgraph.nodes()))
        
        curriculum_list, nodes = extract_train_pairs(subgraph)

        
        res = dict()
        res['curriculum_list'] = curriculum_list # all the pair in a list
        res['nodes'] = nodes
        
        with open("./{}/graph/{}/curriculum_{}.pickle".format(args.map_name, args.n_node, graph_id), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def plot_graph(args, graph):
    x_coord = nx.get_node_attributes(graph, "x_coord")
    y_coord = nx.get_node_attributes(graph, "y_coord")
    
    plt.figure()
    for e in graph.edges():
        n1, n2 = e
        plt.plot([x_coord[n1], x_coord[n2]], [y_coord[n1], y_coord[n2]], 'r-')
    
    plt.savefig("subgraph.png")
    plt.close('all')

def cal_actual_travel_time(link_visited, graph, time_tick, curr_t):
    
    def argmin(a):
        return min(range(len(a)), key=lambda x : a[x])
    
    def cal_single_actual_travel_time(link_visited, graph, time_tick, curr_t):
        current_t = curr_t
        
        link_travel_time_list = []
        weight = nx.get_edge_attributes(graph, 'weight')
        travel_time_mean = nx.get_edge_attributes(graph, 'congest_mean')
        travel_time_lb = nx.get_edge_attributes(graph, 'congest_lb')
        travel_time_ub = nx.get_edge_attributes(graph, 'congest_ub')
        for link in link_visited:
            
            time_idx = np.argmax(time_tick > current_t) # find the suitable time slot
            
            link_time = np.random.uniform(travel_time_lb[link][time_idx], travel_time_ub[link][time_idx])
            
            link_travel_time_list.append(link_time)
            current_t += link_time

        total_time = current_t - curr_t
        return link_travel_time_list, total_time
    
    whole_l_list, t_list = [], []
    for _ in range(5):
        l_list, t = cal_single_actual_travel_time(link_visited, graph, time_tick, curr_t)
        whole_l_list.append(l_list)
        t_list.append(t)
    return whole_l_list[argmin(t_list)]    

def find_cloest_idx(time_history, coarse_t):
    if coarse_t == 0:
        cloest_idx = 0
    elif coarse_t >= np.max(time_history):
        cloest_idx = time_history.shape[0] - 1
    else:
        cloest_idx = np.where(time_history >= coarse_t)[0][0]
    return cloest_idx

def load_baseline_result(args):
    with open('./testing_history_multi/testing_tdsp_{}_{}.pickle'.format(args.map_name[-1], args.graph_size_list[-1]), 'rb') as handle:
        tdsp_result = pickle.load(handle)
    with open('./testing_history_multi/testing_static_{}_{}.pickle'.format(args.map_name[-1], args.graph_size_list[-1]), 'rb') as handle:
        static_result = pickle.load(handle)
    return tdsp_result, static_result
    
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
        
def extract_test_pairs(G, n_pair = 40):
    node_pairs = []
    with temp_seed(42):
        sorted_list = sorted(list(G.nodes()))
        nodes = np.random.choice(sorted_list, size=n_pair, replace=False)
        for node in nodes:
            try:
                # dest = np.random.choice(list(set(G.nodes()) - set(nx.bfs_tree(G, source=node, depth_limit=75).nodes())), size=1, replace=False)[0]
                dest = np.random.choice(list(set(nx.bfs_tree(G, source=node, depth_limit=60).nodes()) - \
                                             set(nx.bfs_tree(G, source=node, depth_limit=40).nodes())), size=1, replace=False)[0]
            except:
                dest = np.random.choice(sorted_list, size=1, replace=False)[0]
            node_pairs.append((node, dest))
    return node_pairs

def extract_train_pairs(subgraph):
    curriculum_list = []
    with temp_seed(42):
        try:
            nodes = np.random.choice(subgraph.nodes(), size=40, replace=False)
        except:
            nodes = np.random.choice(subgraph.nodes(), size=15, replace=False)

        for node in nodes:
            try:
                # dest = np.random.choice(list(set(G.nodes()) - set(nx.bfs_tree(G, source=node, depth_limit=75).nodes())), size=1, replace=False)[0]
                dest_list = np.array(list(set(nx.bfs_tree(subgraph, source=node, depth_limit=60).nodes()) - \
                                         set(nx.bfs_tree(subgraph, source=node, depth_limit=40).nodes())))
                assert(len(dest_list) > 0)
            except:
                dest_list = np.array(subgraph.nodes())

            for dest in dest_list:
                curriculum_list.append((node, dest))
    return curriculum_list, nodes
