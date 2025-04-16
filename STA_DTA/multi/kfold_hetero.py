
import pickle
import time
import dgl
import dgl.function as fn
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import MultiStepLR
from functools import partial
from utils_data import collate_hetero, load_dataset
from modelzoo import *

from utils_parser import parse_args_hetero
args = parse_args_hetero()

from utils_log import createLogger
log = createLogger(__name__, args.log_save_path, args.log_level, args.log_file_name)
log.info('in kfold_hetero.py')

train_num_sample_list = args.train_num_sample_list
test_num_sample_list = args.test_num_sample_list
train_data_dir_list = args.train_data_dir_list
test_data_dir_list = args.test_data_dir_list
map_name = args.map_name

# dgl.seed(234)
# torch.manual_seed(234)
# np.random.seed(234)
torch.set_printoptions(linewidth=200)

if args.gpu >= 0:
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'
    
def train(train_dataloader, test_dataloader, model, args, fold_idx):
    time_total = 0
    min_test_loss = 1e9
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[1000], gamma=0.2)
    
    for e in range(args.epoch):
        t0 = time.time()
        train_loss, train_return_dict = run_epoch(train_dataloader, model, args, optimizer, is_train=True)
        time_total += (time.time() - t0)
        scheduler.step()
        
        if e % 5 == 0:
            test_loss, test_return_dict = run_epoch(test_dataloader, model, args, optimizer, is_train=False)
            
            if args.conservation_loss:
                log.info('In epoch {}, train_loss: {:.3f}, test_loss: {:.3f}, conservation_loss: {:.3f}'.format(e, \
                    train_return_dict['loss_ratio'], test_return_dict['loss_ratio'], test_return_dict['loss_conservation_{}'.format(args.base_vehicle_class)]))
            else:
                log.info('In epoch {}, train_loss: {:.3f}, test_loss: {:.3f}'.format(e, \
                    train_return_dict['loss_ratio'], test_return_dict['loss_ratio']))
    
            if torch.mean(test_loss) < min_test_loss:
                min_test_loss = np.mean(np.array(test_loss.detach().cpu().numpy()))
                torch.save(model.state_dict(), 'model/{}_{}_{}_{}.pth'.format(
                    '_'.join(args.train_data_dir_list), '_'.join(args.test_data_dir_list), args.model_idx, fold_idx))
    
    
    # save result in pickle
    eval_dict = {}
    for v_id in args.vehicle_class:
        eval_dict['pred_ratio_{}'.format(v_id)] = []
        eval_dict['edge_ratio_{}'.format(v_id)] = []
        eval_dict['pred_flow_{}'.format(v_id)] = []
        eval_dict['edge_flow_{}'.format(v_id)] = []
        
    model.eval()
    
    for _, (g, edge_ratio_dict, edge_flow_dict) in enumerate(test_dataloader):
        g = g.to(device)
            
        for v_id in args.vehicle_class:
            edge_ratio_dict[v_id] = edge_ratio_dict[v_id].to(device)
            edge_flow_dict[v_id] = edge_flow_dict[v_id].to(device)
            
        pred_ratio_dict, pred_flow_dict = model(g)
        
        for v_id in args.vehicle_class:
            eval_dict['pred_ratio_{}'.format(v_id)] = pred_ratio_dict[v_id].detach().cpu().numpy().tolist()
            eval_dict['edge_ratio_{}'.format(v_id)] = edge_ratio_dict[v_id].detach().cpu().numpy().tolist()
            eval_dict['pred_flow_{}'.format(v_id)] = pred_flow_dict[v_id].detach().cpu().numpy().tolist()
            eval_dict['edge_flow_{}'.format(v_id)] = edge_flow_dict[v_id].detach().cpu().numpy().tolist()
        
    with open('./result/{}_{}_{}_{}.pickle'.format(
        '_'.join(args.train_data_dir_list), '_'.join(args.test_data_dir_list), args.model_idx, fold_idx), 'wb') as handle:
        pickle.dump(eval_dict, handle)

def run_epoch(train_dataloader, model, args, optimizer, is_train):
    if is_train:    
        model.train()
    else:   
        model.eval()
    
    return_dict = {"loss_ratio": None,
                   "loss_flow": None,
                   "loss_residue": None,
                   "loss": None}
    
    for _, (g, edge_ratio_dict, edge_flow_dict) in enumerate(train_dataloader):
        g = g.to(device)
            
        for v_id in args.vehicle_class:
            edge_ratio_dict[v_id] = edge_ratio_dict[v_id].to(device)
            edge_flow_dict[v_id] = edge_flow_dict[v_id].to(device)
            
        # ratio
        pred_ratio_dict, pred_flow_dict = model(g)

        return_dict['loss_ratio'] = torch.cat([torch.abs(pred_ratio_dict[v_id] - edge_ratio_dict[v_id]) for v_id in args.vehicle_class]).mean()
        return_dict['loss_flow'] = torch.cat([torch.abs(pred_flow_dict[v_id] - edge_flow_dict[v_id]) for v_id in args.vehicle_class]).mean()
                        
        if args.conservation_loss:
            residue_dict = {}
            residue = 0
            for v_id in args.vehicle_class:
                g.edges['connect_{}'.format(v_id)].data['flow'] = pred_ratio_dict[v_id] * g.edges['connect_{}'.format(v_id)].data['capacity']
            rg = dgl.reverse(g, copy_ndata=False, copy_edata=True)
            
            for v_id in args.vehicle_class:
                g['connect_{}'.format(v_id)].update_all(fn.copy_e('flow', 'm'), fn.sum('m', f'in_flow_{v_id}'), etype=f'connect_{v_id}')
                rg['connect_{}'.format(v_id)].update_all(fn.copy_e('flow', 'm'), fn.sum('m', f'out_flow_{v_id}'), etype=f'connect_{v_id}')
                
                residue_dict[v_id] = rg.nodes['node'].data[f'out_flow_{v_id}'] - g.nodes['node'].data[f'in_flow_{v_id}'] - \
                    (torch.sum(g.nodes['node'].data[f'OD_{v_id}'], dim=1).view(-1, 1) - \
                    torch.sum(g.nodes['node'].data[f'OD_T_{v_id}'], dim=1).view(-1, 1))
                return_dict['loss_conservation_{}'.format(v_id)] = torch.mean(torch.abs(residue_dict[v_id]))
                residue += torch.mean(torch.abs(residue_dict[v_id]))
                
        if args.conservation_loss:
            loss = return_dict['loss_ratio'] + return_dict['loss_flow'] * 0.0001 + residue * 0.0001
        else:
            loss = return_dict['loss_ratio'] + return_dict['loss_flow'] * 0.0001
        
        if is_train:
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return loss, return_dict
        

if 'Braess' in map_name:    n_node = 4
if 'Sioux' in map_name:     n_node = 24

if args.model_idx == 1: 
    model = HetTransformer(in_feats=n_node, pre_reg_feats=64, h_feats=32, num_head=8, 
                           base_v_id=args.base_vehicle_class,
                           v_id_list=args.vehicle_class).to(device)
    
collate = partial(collate_hetero, v_id_list=args.vehicle_class)

train_data_dir_list.sort()
test_data_dir_list.sort()

print(train_data_dir_list)
print(test_data_dir_list)

if train_data_dir_list == test_data_dir_list:
    data_batch = load_dataset(train_num_sample_list, train_data_dir_list, map_name, n_node)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_index, test_index) in enumerate(kf.split(data_batch)):
        print(f"Fold {fold_idx}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        train_batch = Subset(data_batch, train_index)
        test_batch = Subset(data_batch, test_index)
        
        train_dataloader = DataLoader(train_batch, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=8)
        test_dataloader = DataLoader(test_batch, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=8)
        # Create the model. The output has three pred_ratio for three classes.
        train(train_dataloader, test_dataloader, model, args, fold_idx)
        break
else:
    train_batch = load_dataset(train_num_sample_list, train_data_dir_list, map_name, n_node)
    test_batch = load_dataset(test_num_sample_list, test_data_dir_list, map_name, n_node)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_fold, test_fold) in enumerate(zip(kf.split(train_batch), kf.split(test_batch))):
        (train_index, _) = train_fold
        (_, test_index) = test_fold
        print(f"Fold {fold_idx}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        
        train_batch_subset = Subset(train_batch, train_index)
        test_batch_subset = Subset(test_batch, test_index)
        
        train_dataloader = DataLoader(train_batch_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=8)
        test_dataloader = DataLoader(test_batch_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=8)
        # Create the model. The output has three pred_ratio for three classes.
        train(train_dataloader, test_dataloader, model, args, fold_idx)
        break
