import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

def extract_data(mat_contents):

    # structure the data
    xcoor_list = []
    ycoor_list = []
    u_list = []
    mu_list = []
    mask_list = []
    kk_list = []
    num_samples = len(mat_contents.keys())

    for i in mat_contents.keys():
        xcoor = mat_contents[i]['xcoor']   
        ycoor = mat_contents[i]['ycoor']    
        u = mat_contents[i]['U']    
        mu = mat_contents[i]['mu'] 
        mask = mat_contents[i]['mask'] 
        rho_omega_square = mat_contents[i]['omega_over_c']
        
        # mask the displacement
        for j in range(u.shape[0]):
            field = u[j]
            field[mask==0] = 0
            u[j,:,:] = field
        for j in range(mu.shape[0]):
            field = mu[j]
            field[mask==0] = 0
            mu[j,:,:] = field
        
        # store the data
        xcoor_list.append(xcoor)
        ycoor_list.append(ycoor)
        u_list.append(u)
        mu_list.append(mu)
        mask_list.append(mask)
        kk_list.append(rho_omega_square)

    '''
    all the data are assumed to be the same size
    '''
    xcoor = torch.from_numpy(np.array(xcoor_list))
    ycoor = torch.from_numpy(np.array(ycoor_list))
    u = torch.from_numpy(np.array(u_list))
    mu = torch.from_numpy(np.array(mu_list))
    mask = torch.from_numpy(np.array(mask_list))
    kk = torch.from_numpy(np.array(kk_list))
    print(xcoor.shape, ycoor.shape, u.shape, mu.shape, mask.shape, kk.shape)

    return [xcoor, ycoor, u, mu, mask, kk]

def create_data_driven_data_loader(data, split_ratio, bs, train_shuffle=True):
    # Extract displacement field

    disps = data[2].permute(0, 2, 3, 4, 1)
    mu = data[3][:, 0, :, :, :]
    mask = data[4]
    organized_data = [disps, mu, mask]

    # Calculate the number of 10-sample groups
    num_samples = data[2].shape[0]
    num_groups = num_samples // 10

    NS1, NS2 = split_ratio
    train_disps, val_disps, test_disps = [], [], []
    train_mu, val_mu, test_mu = [], [], []
    train_mask, val_mask, test_mask = [], [], []

    # Split each group of 10 samples
    for i in range(num_groups):
        start = i * 10
        end = start + 10

        # Select group of 10
        group_disps = organized_data[0][start:end]
        group_mu = organized_data[1][start:end]
        group_mask = organized_data[2][start:end]

        # Determine the split within the group
        train_end = int(NS1 * 10)
        val_end = int(NS2 * 10)

        # Append to corresponding lists
        train_disps.append(group_disps[:train_end])
        val_disps.append(group_disps[train_end:val_end])
        test_disps.append(group_disps[val_end:])

        train_mu.append(group_mu[:train_end])
        val_mu.append(group_mu[train_end:val_end])
        test_mu.append(group_mu[val_end:])

        train_mask.append(group_mask[:train_end])
        val_mask.append(group_mask[train_end:val_end])
        test_mask.append(group_mask[val_end:])

    # Concatenate all groups into single tensors
    train_disps = torch.cat(train_disps, dim=0)
    val_disps = torch.cat(val_disps, dim=0)
    test_disps = torch.cat(test_disps, dim=0)
    
    train_mu = torch.cat(train_mu, dim=0)
    val_mu = torch.cat(val_mu, dim=0)
    test_mu = torch.cat(test_mu, dim=0)
    
    train_mask = torch.cat(train_mask, dim=0)
    val_mask = torch.cat(val_mask, dim=0)
    test_mask = torch.cat(test_mask, dim=0)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_disps, train_mu, train_mask)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=train_shuffle)
    
    val_dataset = torch.utils.data.TensorDataset(val_disps, val_mu, val_mask)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    test_dataset = torch.utils.data.TensorDataset(test_disps, test_mu, test_mask)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
