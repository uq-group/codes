import pickle
import dgl
from dgl.data import DGLDataset
import numpy as np
import torch
from torch.nn import ParameterDict
from torch.utils.data import ConcatDataset
torch.set_printoptions(precision=2, threshold=20, edgeitems=5, linewidth=300, sci_mode=False)

class TrafficAssignmentDataset(DGLDataset):
    def __init__(self, num_sample, data_dir, n_node, map_name):
        self.num_sample = num_sample
        self.data_dir = data_dir
        self.n_node = n_node
        self.map_name = map_name
        self.od_ratio = {'Sioux':1000, 'EMA':1000, 'Anaheim':800, 'ANAHEIM':800, 'Braess':1000} # used for normalization
        self.cap_ratio = {'Sioux':1000, 'EMA':1000, 'Anaheim':800, 'ANAHEIM':800, 'Braess':500} # times the cap_ratio
        super(TrafficAssignmentDataset, self).__init__(name='bridge', hash_key={num_sample, data_dir})
     
    def process(self):
        self.graphs = []
        
        coord_all = np.loadtxt('./{}/coord.csv'.format(self.map_name), delimiter=' ')
        coord = coord_all[:, 3:]
        
        # load data from imp folder
        for graph_id in range(self.num_sample):
            # load each graph
            with open('../DataGeneration/{}/data_{}.pickle'.format(self.data_dir, graph_id+1), 'rb') as handle:
                graph_data = pickle.load(handle)
            print(graph_id)    
            # print(graph_data.keys())

            ######################### build graph #########################
            hetergraph_data = {}
            for v_id, ca_list in graph_data['ca_list'].items():
                # data is directed, make it undirected
                ncn = torch.tensor(ca_list[:,0:2].squeeze(), dtype=torch.int32)
                hetergraph_data[('node', 'connect_{}'.format(v_id), 'node')] = (ncn[:, 0], ncn[:, 1])
                
            for v_id, od_list in graph_data['od_list'].items():
                ## now nodn only connect edge that od > 0 --> considering add edge feature in the future
                nodn = torch.tensor(od_list[:,0:2].squeeze(), dtype=torch.int32)
                hetergraph_data[('node', 'od_{}'.format(v_id), 'node')] = (nodn[:, 0], nodn[:, 1])
                
            g = dgl.heterograph(hetergraph_data)
            
            ######################### build node input #########################
            # node_feat_np = [torch.tensor(graph_data['demand_matrix'][v_id], dtype=torch.float32).unsqueeze(2) \
            #     for v_id in graph_data['vehicle_class_list']]
            # od_node_feat = torch.cat(tuple(node_feat_np), dim=2)
            # g.nodes['node'].data['coord'] = torch.tensor(coord, dtype=torch.float32)
            # g.nodes['node'].data['feat'] = od_node_feat / self.od_ratio[self.map_name] # od vector
            
            g.nodes['node'].data['coord'] = torch.tensor(coord, dtype=torch.float32)
            
            for v_id in graph_data['vehicle_class_list']:
                g.nodes['node'].data['feat_{}'.format(v_id)] = \
                    torch.tensor(graph_data['demand_matrix'][v_id], dtype=torch.float32) / self.od_ratio[self.map_name]
                g.nodes['node'].data['OD_{}'.format(v_id)] = \
                    torch.tensor(graph_data['demand_matrix'][v_id], dtype=torch.float32)
                g.nodes['node'].data['OD_T_{}'.format(v_id)] = g.nodes['node'].data['OD_{}'.format(v_id)].T

            # g.nodes['node'].data['feat_T'] = od_node_feat.T / self.cap_ratio[self.map_name]
            
            ######################### build virtual edge input #########################
            
            ######################### build real edge input #########################
            for v_id in graph_data['vehicle_class_list']:
                connect_edge_feat = torch.tensor(graph_data['ca_list'][v_id][:, 2:], dtype=torch.float32)
                g.edges['connect_{}'.format(v_id)].data['feat'] = connect_edge_feat # feature
                g.edges['connect_{}'.format(v_id)].data['capacity'] = connect_edge_feat[:, 0:1]*self.cap_ratio[self.map_name] # capacity
                # normalize
                g.edges['connect_{}'.format(v_id)].data['feat'][:, 0] /= 10.0 # feature
                
            ######################### build real edge result #########################
            for v_id in graph_data['vehicle_class_list']:
                ratio_list, flow_list = [], []
                flow, ratio = graph_data['flow'][v_id], graph_data['ratio'][v_id]
                for k1, k2 in zip(graph_data['ca_list'][v_id][:,0].squeeze(), graph_data['ca_list'][v_id][:,1].squeeze()):
                    flow_list.append([k1, k2, flow[(int(k1)), int(k2)]])
                    ratio_list.append([k1, k2, ratio[(int(k1)), int(k2)]])
                ratio_list = np.vstack(ratio_list)
                flow_list = np.vstack(flow_list)
                
                assert(np.array_equal(ratio_list[:, 0].squeeze(), graph_data['ca_list'][v_id][:, 0].squeeze()))
                assert(np.array_equal(ratio_list[:, 1].squeeze(), graph_data['ca_list'][v_id][:, 1].squeeze()))
                assert(np.array_equal(flow_list[:, 0].squeeze(), graph_data['ca_list'][v_id][:, 0].squeeze()))
                assert(np.array_equal(flow_list[:, 1].squeeze(), graph_data['ca_list'][v_id][:, 1].squeeze()))
                
                g.edges['connect_{}'.format(v_id)].data['res_ratio'] = torch.tensor(ratio_list[:, 2:], dtype=torch.float32) # result_ratio
                g.edges['connect_{}'.format(v_id)].data['res_flow'] = torch.tensor(flow_list[:, 2:], dtype=torch.float32) # result_flow
                
            self.graphs.append(g)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)


def collate_hetero(samples, v_id_list):
    n = len(samples)
    ratio_dict, flow_dict = {}, {}
    for v_id in v_id_list:
        res_ratio = [samples[i].edges['connect_{}'.format(v_id)].data['res_ratio'] for i in range(n)]
        res_flow = [samples[i].edges['connect_{}'.format(v_id)].data['res_flow'] for i in range(n)]
        ratio_dict[v_id] = torch.vstack((*res_ratio,))
        flow_dict[v_id] = torch.vstack((*res_flow,))
    batched_graph = dgl.batch(samples)
    return batched_graph, ratio_dict, flow_dict

def load_dataset(num_sample_list, data_dir_list, map_name, n_node):
    all_data_batch = []
    for num_sample, data_dir in zip(num_sample_list, data_dir_list):
        dataset = TrafficAssignmentDataset(num_sample, data_dir, n_node, map_name)
        all_data_batch.append(dataset)
    data_batch = ConcatDataset(all_data_batch)
    return data_batch