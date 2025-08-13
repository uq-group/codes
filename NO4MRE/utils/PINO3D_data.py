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
    zcoor_list = []
    u_list = []
    mu_list = []
    mask_list = []
    kk_list = []
    num_samples = len(mat_contents.keys())

    for i in mat_contents.keys():
        xcoor = mat_contents[i]['xcoor']   
        ycoor = mat_contents[i]['ycoor']    
        zcoor = mat_contents[i]['zcoor']    
        u = mat_contents[i]['U']    
        mu = mat_contents[i]['mu'] 
        mask = mat_contents[i]['mask'] 
        rho_omega_square = mat_contents[i]['omega_over_c']
        
        # mask the displacement
        for j in range(u.shape[0]):
            field = u[j]
            field[mask==0] = 0
            u[j,:,:,:] = field
        for j in range(mu.shape[0]):
            field = mu[j]
            field[mask==0] = 0
            mu[j,:,:,:] = field
        
        # store the data
        xcoor_list.append(xcoor)
        ycoor_list.append(ycoor)
        zcoor_list.append(zcoor)
        u_list.append(u)
        mu_list.append(mu)
        mask_list.append(mask)
        kk_list.append(rho_omega_square)

    '''
    all the data are assumed to be the same size
    '''
    xcoor = torch.from_numpy(np.array(xcoor_list))
    ycoor = torch.from_numpy(np.array(ycoor_list))
    zcoor = torch.from_numpy(np.array(zcoor_list))
    u = torch.from_numpy(np.array(u_list))
    mu = torch.from_numpy(np.array(mu_list))
    mask = torch.from_numpy(np.array(mask_list))
    kk = torch.from_numpy(np.array(kk_list))
    print(xcoor.shape, ycoor.shape, zcoor.shape, u.shape, mu.shape, mask.shape, kk.shape)

    return [xcoor, ycoor, zcoor, u, mu, mask, kk]

def create_physics_informed_data_loader(data, split_ratio, bs, train_shuffle=True):
    # Extract input data
    xcoor = data[0]
    ycoor = data[1]
    zcoor = data[2]
    disps = data[3].permute(0, 2, 3, 4, 1)  # (N, H, W, D, C)
    mu = data[4][:, 0, :, :, :]
    mask = data[5]
    rho_omega_square = data[6]
    
    organized_data = [xcoor, ycoor, zcoor, disps, mu, mask, rho_omega_square]
    
    num_samples = data[3].shape[0]
    num_groups = num_samples // 10
    NS1, NS2 = split_ratio

    # Initialize lists for each split
    train_data = [[] for _ in range(len(organized_data))]
    val_data = [[] for _ in range(len(organized_data))]
    test_data = [[] for _ in range(len(organized_data))]

    for i in range(num_groups):
        start = i * 10
        end = start + 10

        group = [d[start:end] for d in organized_data]
        train_end = int(NS1 * 10)
        val_end = int(NS2 * 10)

        for j in range(len(group)):
            train_data[j].append(group[j][:train_end])
            val_data[j].append(group[j][train_end:val_end])
            test_data[j].append(group[j][val_end:])

    # Concatenate each list into a tensor
    train_data = [torch.cat(d, dim=0) for d in train_data]
    val_data = [torch.cat(d, dim=0) for d in val_data]
    test_data = [torch.cat(d, dim=0) for d in test_data]

    # Create datasets and loaders
    train_dataset = torch.utils.data.TensorDataset(*train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=train_shuffle)

    val_dataset = torch.utils.data.TensorDataset(*val_data)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    test_dataset = torch.utils.data.TensorDataset(*test_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader

def create_one_sample_data_loader(data, sample_id):

    # extract displacement field
    xcoor = data[0]
    ycoor = data[1]
    zcoor = data[2]
    disps = data[3].permute(0,2,3,4,1)
    mu = data[4][:,0,:,:,:]
    mask = data[5]
    rho_omega_square = data[6]
    organized_data = [xcoor, ycoor, zcoor, disps, mu, mask, rho_omega_square]

    # split the data
    train_data = [organized_data[i][sample_id:sample_id+1] for i in range(len(organized_data))]

    # create data loader
    train_dataset = torch.utils.data.TensorDataset(train_data[0], train_data[1], train_data[2], 
                                                  train_data[3], train_data[4], train_data[5], train_data[6])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    return train_loader 