import os
import pickle
import time
import warnings
warnings.simplefilter('ignore')

import dgl
from dgl.data import DGLDataset
import numpy as np
import networkx as nx

import torch
from torch.utils.data import DataLoader, Dataset

from utils_log import createLogger
log = createLogger(__name__)
log.info('in dataloading.py')

device = 'cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu'
torch.set_printoptions(precision=2, linewidth=400, sci_mode=True)

class TrafficAssignmentDataset(DGLDataset):
    def __init__(self, data_dir, n_sample, n_T):
        # self.num_sample = num_sample
        self.pickle_file = self.load_pickle(data_dir)
        self.preprocessing()
        self.n_T = n_T
        self.n_sample = n_sample
        
        # normalizer
        self.od_normalization = 300.0
        self.cap_normalization = 10000.0
        self.speed_normalization = 120.0
        self.length_normalization = 6.0
        self.free_time_normalization = self.length_normalization / self.speed_normalization
        # self.n_node = n_node
        super(TrafficAssignmentDataset, self).__init__(name='dta', hash_key={'dta', data_dir})
    
    def load_pickle(self, filename):
        t1 = time.time()
        with open('../data/{}.pickle'.format(filename), 'rb') as handle:
            pickle_file = pickle.load(handle)

        log.info("loading pickle time: ", time.time()-t1, l=1)
        return pickle_file
    
    def preprocessing(self):
        self.edge_list = self.pickle_file[0]['g'].edges()
        self.n_node = len(self.pickle_file[0]['g'].nodes())
        self.src_ncn = np.array(self.edge_list)[:, 0]
        self.dst_ncn = np.array(self.edge_list)[:, 1]
        log.debug(self.edge_list)
        log.debug(self.pickle_file[0]['g'])
        log.debug(self.src_ncn)
        log.debug(self.dst_ncn)
        
    def build_od_feat(self, od_matrix_list):
        # expand the od matrix in to the full matrix
        # to do: if only have od demand from few location --> dont create full od matrix
        od_matrix_list = np.dstack(od_matrix_list)
        log.debug(od_matrix_list.shape)
                
        log.debug(od_matrix_list[:, :, 0])
        log.debug(od_matrix_list.shape)
        
        return torch.tensor(od_matrix_list, dtype=torch.float32)
    
    def build_loc_feat(self, x_dict, y_dict):
        # expand the od matrix in to the full matrix
        # to do: if only have od demand from few location --> dont create full od matrix

        # Convert list to an array
        x_array = np.array(list(x_dict.values())).reshape(-1, 1)
        y_array = np.array(list(y_dict.values())).reshape(-1, 1)
        
        self.x_coord_min, self.x_coord_max = np.min(x_array), np.max(x_array)
        self.y_coord_min, self.y_coord_max = np.min(y_array), np.max(y_array)
        
        return torch.tensor(x_array, dtype=torch.float32), torch.tensor(y_array, dtype=torch.float32)
    
    def process(self):
        t1 = time.time()
        log.debug(list(self.pickle_file[0].keys()))
        self.graphs = []
        log.debug(self.src_ncn)
        
        for i in range(self.n_sample):            
            # build node features
            # od matrix: N x N x T
            # geocoordiante: N x 1
            od_feat = self.build_od_feat(self.pickle_file[i]['full_od_demand_list'])
            
            od_link = torch.nonzero(torch.sum(od_feat, dim=2))
            src_nodn, dst_nodn = od_link[:, 0], od_link[:, 1]
            
            # create directed graph
            hetergraph_data = {
                ('node', 'connect', 'node'): (self.src_ncn, self.dst_ncn),
                ('node', 'od', 'node'): (src_nodn, dst_nodn)}
            g = dgl.heterograph(hetergraph_data)
            
            x_loc_feat, y_loc_feat = self.build_loc_feat(self.pickle_file[0]['x_coord'],self.pickle_file[0]['y_coord'])
            g.nodes['node'].data['feat'] = od_feat
            g.nodes['node'].data['feat_sum'] = torch.sum(od_feat, dim=2)
            g.nodes['node'].data['feat_sum_T'] = torch.sum(od_feat, dim=2).T
            log.info(od_feat[:,:,1], l=10)
            log.info(od_feat[:,:,1].shape, l=10)
            log.info(x_loc_feat, l=10)
            log.info(y_loc_feat, l=10)
            g.nodes['node'].data['x_coord'] = x_loc_feat
            g.nodes['node'].data['y_coord'] = y_loc_feat

            log.debug(self.pickle_file[i]['g'].edges(data=True))
            log.debug(g.num_edges('od'))
            log.debug((self.src_ncn, self.dst_ncn))
            log.debug(self.pickle_file[i]['length'])
            
            # CA: number entering link, CD: number exit link
            res_density, res_ratio, res_CA, res_CD = [], [], [], []
            for src, dst in zip(self.src_ncn, self.dst_ncn):
                res_density.append(self.pickle_file[i]['whole_flows'][(src, dst)])
                res_ratio.append(self.pickle_file[i]['whole_ratio'][(src, dst)])
                res_CA.append(self.pickle_file[i]['whole_CA'][(src, dst)])
                res_CD.append(self.pickle_file[i]['whole_CD'][(src, dst)])
            res_density, res_ratio = np.vstack(res_density), np.vstack(res_ratio)
            res_CA, res_CD = np.vstack(res_CA), np.vstack(res_CD)
            
            aggr_density, aggr_ratio, aggr_CA, aggr_CD = [], [], [], []
            for src, dst in zip(self.src_ncn, self.dst_ncn):
                aggr_density.append(self.pickle_file[i]['aggr_flows'][(src, dst)])
                aggr_ratio.append(self.pickle_file[i]['aggr_ratio'][(src, dst)])
                aggr_CA.append(self.pickle_file[i]['aggr_CA'][(src, dst)])
                aggr_CD.append(self.pickle_file[i]['aggr_CD'][(src, dst)])
            aggr_density, aggr_ratio = np.vstack(aggr_density), np.vstack(aggr_ratio)
            aggr_CA, aggr_CD = np.vstack(aggr_CA), np.vstack(aggr_CD)

            # build edge features
            # edge capacity
            # edge length # km
            # edge speed # km/h
            # add flow/ratio into networkx graph
            dg = dgl.from_networkx(self.pickle_file[i]['g'], node_attrs=['x_coord', 'y_coord'], \
                edge_attrs=['length', 'free_speed', 'capacity'], idtype=torch.int32)
            g.edges['connect'].data['length_org'] = torch.tensor(dg.edata['length'], dtype=torch.float32)
            
            g.edges['connect'].data['capacity_org'] = torch.tensor(dg.edata['capacity'], dtype=torch.float32).reshape(-1, 1)
            g.edges['connect'].data['capacity_unnorm'] = torch.tensor(dg.edata['capacity'], dtype=torch.float32).reshape(-1, 1)
            g.edges['connect'].data['free_speed'] = torch.tensor(dg.edata['free_speed'], dtype=torch.float32)
            g.edges['connect'].data['free_time'] = torch.tensor(g.edges['connect'].data['length_org'] / (g.edges['connect'].data['free_speed']), dtype=torch.float32)
            log.info(g.edges['connect'].data['length_org'].shape, g.edges['connect'].data['capacity_org'].shape, g.edges['connect'].data['free_speed'].shape, l=10)
            
            # build result feature
            # log.debug(self.pickle_file[i]['result'].flows.shape)
            # log.debug({edge: flow.round(2) for flow, edge in zip(self.pickle_file[i]['result'].flows[0, :].squeeze(), self.pickle_file[i]['g'].edges())})
            g.edges['connect'].data['res_density'] = torch.tensor(res_density, dtype=torch.float32) # number of vehicle / length
            g.edges['connect'].data['res_ratio'] = torch.tensor(res_ratio, dtype=torch.float32)     # CD_t - CD_{t-1}
            g.edges['connect'].data['res_CA'] = torch.tensor(res_CA, dtype=torch.float32)           # vehicle enter the link
            g.edges['connect'].data['res_CD'] = torch.tensor(res_CD, dtype=torch.float32)           # vehicle leave the link
            
            g.edges['connect'].data['aggr_density'] = torch.tensor(aggr_density, dtype=torch.float32) # number of vehicle / length
            g.edges['connect'].data['aggr_ratio'] = torch.tensor(aggr_ratio, dtype=torch.float32)     # CD_t - CD_{t-1}
            g.edges['connect'].data['aggr_CA'] = torch.tensor(aggr_CA, dtype=torch.float32)           # vehicle enter the link
            g.edges['connect'].data['aggr_CD'] = torch.tensor(aggr_CD, dtype=torch.float32)           # vehicle leave the link
            
            
            log.info(g.edges['connect'].data['res_ratio'][75, :].T, l=10)
            log.info(g.edges['connect'].data['res_density'][75, :].T, l=10)
            log.debug(g.edges['connect'].data['res_density'].shape, g.edges['connect'].data['capacity_org'].shape, g.edges['connect'].data['res_ratio'].shape)
            log.debug(g.edges['connect'].data['res_density'][:, 10].squeeze(), g.edges['connect'].data['capacity_org'].squeeze())
            
            # normalize the feature
            g.nodes['node'].data['feat'] = g.nodes['node'].data['feat'] / self.od_normalization
            g.edges['connect'].data['capacity'] = g.edges['connect'].data['capacity_org'] / self.cap_normalization
            g.edges['connect'].data['free_speed'] = g.edges['connect'].data['free_speed'] / self.speed_normalization
            g.edges['connect'].data['length'] = g.edges['connect'].data['length_org'] / self.length_normalization
            g.edges['connect'].data['free_time'] = g.edges['connect'].data['free_time'] / self.free_time_normalization
            g.nodes['node'].data['x_coord'] = (g.nodes['node'].data['x_coord'] - self.x_coord_min) / (self.x_coord_max - self.x_coord_min)
            g.nodes['node'].data['y_coord'] = (g.nodes['node'].data['y_coord'] - self.y_coord_min) / (self.y_coord_max - self.y_coord_min)
            
            # build features used in training
            g.edges['connect'].data['curr_ratio'] = torch.zeros(g.edges['connect'].data['res_density'].shape[0], dtype=torch.float32)
            log.info(g.edges['connect'].data['res_density'].shape, g.edges['connect'].data['capacity_org'].shape)
            self.graphs.append(g)
            
        log.info("dgl process time: ", time.time()-t1, l=1)
    
    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)



