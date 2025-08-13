import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.colors import ListedColormap

def train(args, model, device, loaders, optimizer):

    # extract the loaders
    train_loader, val_loader, test_loader = loaders

    model.train()
    mse = nn.MSELoss()
    err_list = []
    for ep in range(args.epochs):

        # training
        model.train()
        for (x, y, z, disp, mu, mask, rho_omega_square) in train_loader:

            # reset optimizer
            optimizer.zero_grad()
            
            # prepare the data
            disp_map = disp.float().to(device)   # (B, 6, S, S, S)
            mu = mu.float().to(device)    # (B, S, S, S)
            rho_omega_square = rho_omega_square.float().to(device)    # (B)
            mask = mask.float().to(device)    # (B, S, S, S)
            
            # model forward
            mu_pred, lam_pred = model(disp_map)

            # backward
            if args.data == 'heter':
                residual_u, residual_v, residual_w = pde_residual_compute_heter_3d(
                    disp_map[:,:,:,:,0], disp_map[:,:,:,:,1], disp_map[:,:,:,:,2],
                    mu_pred, lam_pred, x, y, z, rho_omega_square, mask)
            elif args.data == 'homo' or args.data == 'homo3d':
                residual_u, residual_v, residual_w = pde_residual_compute_homo_3d(
                    disp_map[:,:,:,:,0], disp_map[:,:,:,:,1], disp_map[:,:,:,:,2],
                    mu_pred, x, y, z, rho_omega_square, mask)
            
            # mask
            residual_u = residual_u * mask[:,2:-2,2:-2,2:-2]
            residual_v = residual_v * mask[:,2:-2,2:-2,2:-2]
            residual_w = residual_w * mask[:,2:-2,2:-2,2:-2]
            
            loss = mse(residual_u, torch.zeros_like(residual_u)) + \
                   mse(residual_v, torch.zeros_like(residual_v)) + \
                   mse(residual_w, torch.zeros_like(residual_w))
            loss.backward()
            optimizer.step()

        # check error
        if ep % 10 == 0:
            rel_err = test(args, model, device, val_loader)
            err_list.append(rel_err)
            print('current epoch:', ep)
            print('average relative error:', rel_err)
            if rel_err <= min(err_list):
                print('saved new model.')
                # Ensure the trained_models directory exists
                os.makedirs('./trained_models', exist_ok=True)
                torch.save(model.state_dict(), r'./trained_models/{}_{}_{}.pth'.format(args.model, args.data, args.train_method))

def tuning(args, model, device, loaders, optimizer):

    # extract the loaders
    train_loader, val_loader, test_loader = loaders

    model.train()
    mse = nn.MSELoss()
    err_list = []
    for ep in range(args.epochs):

        # training
        model.train()
        for (x, y, z, disp, mu, mask, rho_omega_square) in train_loader:

            # reset optimizer
            optimizer.zero_grad()
            
            # prepare the data
            disp_map = disp.float().to(device)   # (B, 6, S, S, S)
            mu = mu.float().to(device)    # (B, S, S, S)
            mask = mask.float().to(device)    # (B, S, S, S)
            rho_omega_square = rho_omega_square.float().to(device)    # (B)
            
            # model forward
            mu_pred, lam_pred = model(disp_map)

            # backward
            if args.data == 'heter':
                residual_u, residual_v, residual_w = pde_residual_compute_heter_3d(
                    disp_map[:,:,:,:,0], disp_map[:,:,:,:,1], disp_map[:,:,:,:,2],
                    mu_pred, lam_pred, x, y, z, rho_omega_square, mask)
            elif args.data == 'homo' or args.data == 'homo3d':
                residual_u, residual_v, residual_w = pde_residual_compute_homo_3d(
                    disp_map[:,:,:,:,0], disp_map[:,:,:,:,1], disp_map[:,:,:,:,2],
                    mu_pred, x, y, z, rho_omega_square, mask)
            
            # mask
            residual_u = residual_u * mask[:,2:-2,2:-2,2:-2]
            residual_v = residual_v * mask[:,2:-2,2:-2,2:-2]
            residual_w = residual_w * mask[:,2:-2,2:-2,2:-2]
            
            loss = mse(residual_u, torch.zeros_like(residual_u)) + \
                   mse(residual_v, torch.zeros_like(residual_v)) + \
                   mse(residual_w, torch.zeros_like(residual_w))
            loss.backward()
            optimizer.step()

        # check error
        if ep % 100 == 0:
            rel_err = test(args, model, device, val_loader)
            err_list.append(rel_err)
            print('current epoch:', ep)
            print('average relative error:', rel_err)
    
    return model

def test(args, model, device, val_loader):

    model.eval()
    avg_err = 0
    num_sample = 0
    for (x, y, z, disp, mu, mask, rho_omega_square) in val_loader:

        # prepare the data
        disp_map = disp.float().to(device)   # (B, 6, S, S, S)
        mu = mu.float().to(device)    # (B, S, S, S)
        mask = mask.float().to(device)    # (B, S, S, S)
        
        # model forward
        mu_pred, lam_pred = model(disp_map)

        # compute the error
        mu_pred = mu_pred * mask
        mu = mu * mask
        relative_err = torch.sqrt(torch.sum((mu_pred - mu)**2) / torch.sum(mu**2)).detach().cpu().item()
        avg_err += relative_err
        num_sample += 1
    
    return avg_err / num_sample

def plot_tuned(args, model, model_tuned, device, test_loader):

    model.eval()
    avg_err = 0
    num_sample = 0
    sid = 0
    for (x, y, z, disp, mu, mask, rho_omega_square) in test_loader:

        batchsize = disp.shape[0]
        assert batchsize == 1, 'plotting need batch size = 1'

        # prepare the data
        disp_map = disp.float().to(device)   # (B, 6, S, S, S)
        mu = mu.float().to(device)    # (B, S, S, S)
        mask = mask.float().to(device)    # (B, S, S, S)
        
        # model forward
        mu_pred, lam_pred = model(disp_map)
        mu_tuned_pred, _ = model_tuned(disp_map)

        # define color map
        jet = plt.cm.get_cmap('jet', 256)
        new_colors = jet(np.linspace(0, 1, 256))
        new_colors[0] = np.array([1, 1, 1, 1])  # RGBA for white
        custom_jet = ListedColormap(new_colors)

        # compute the error and plot middle slice
        mid_slice = mu.shape[-1] // 2
        mu_pred_slice = (mu_pred * mask).squeeze(0)[:,:,mid_slice].detach().cpu().numpy()
        mu_tuned_pred_slice = (mu_tuned_pred * mask).squeeze(0)[:,:,mid_slice].detach().cpu().numpy()
        mu_slice = (mu * mask).squeeze(0)[:,:,mid_slice].detach().cpu().numpy()
        
        plt.figure(figsize=(15, 5), dpi=300)
        plt.subplot(131)
        ax1 = sns.heatmap(mu_slice, cmap=custom_jet, vmin=0, vmax=5, cbar_kws={'label': 'Storage modulus (kPa)'})
        cbar = ax1.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15) 
        cbar.ax.set_ylabel('Storage modulus (kPa)', fontsize=16)
        plt.xticks([]), plt.yticks([])
        plt.title('(a) Exact', fontsize=20)

        plt.subplot(132)
        ax2 = sns.heatmap(mu_pred_slice, cmap=custom_jet, vmin=0, vmax=5, cbar_kws={'label': 'Storage modulus (kPa)'})
        cbar = ax2.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15) 
        cbar.ax.set_ylabel('Storage modulus (kPa)', fontsize=16)
        plt.xticks([]), plt.yticks([])
        plt.title('(b) Pretrained model', fontsize=20)

        plt.subplot(133)
        ax3 = sns.heatmap(mu_tuned_pred_slice, cmap=custom_jet, vmin=0, vmax=5, cbar_kws={'label': 'Storage modulus (kPa)'})
        cbar = ax3.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15) 
        cbar.ax.set_ylabel('Storage modulus (kPa)', fontsize=16)
        plt.xticks([]), plt.yticks([])
        plt.title('(c) Fine-tuned model', fontsize=20)

        save_dir = r'./visuals/{}/{}/{}/'.format(args.train_method, args.model, args.data)
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{sid}.png'))
        sid+=1
        if sid >= 10:
            break

def plot(args, model, device, test_loader):

    model.eval()
    avg_err = 0
    num_sample = 0
    sid = 0
    for (x, y, z, disp, mu, mask, rho_omega_square) in test_loader:

        batchsize = disp.shape[0]
        assert batchsize == 1, 'plotting need batch size = 1'

        # prepare the data
        disp_map = disp.float().to(device)   # (B, 6, S, S, S)
        mu = mu.float().to(device)    # (B, S, S, S)
        mask = mask.float().to(device)    # (B, S, S, S)
        
        # model forward
        mu_pred, lam_pred = model(disp_map)

        # define color map
        jet = plt.cm.get_cmap('jet', 256)
        new_colors = jet(np.linspace(0, 1, 256))
        new_colors[0] = np.array([1, 1, 1, 1])  # RGBA for white
        custom_jet = ListedColormap(new_colors)

        # compute the error and plot middle slice
        mid_slice = mu.shape[-1] // 2
        mu_pred_slice = (mu_pred * mask).squeeze(0)[:,:,mid_slice].detach().cpu().numpy()
        mu_slice = (mu * mask).squeeze(0)[:,:,mid_slice].detach().cpu().numpy()
        
        plt.figure(figsize=(10, 5), dpi=300)
        plt.subplot(121)
        ax1 = sns.heatmap(mu_slice, cmap=custom_jet, vmin=0, vmax=5, cbar_kws={'label': 'Exact storage modulus (kPa)'})
        cbar = ax1.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)  # Change tick font size
        cbar.ax.set_ylabel('Exact storage modulus (kPa)', fontsize=16)    # Label font size
        plt.xticks([]), plt.yticks([])
        # plt.title('(a) Exact', fontsize=20)
        plt.subplot(122)
        ax2 = sns.heatmap(mu_pred_slice, cmap=custom_jet, vmin=0, vmax=5, cbar_kws={'label': 'Predicted storage modulus (kPa)'})
        cbar = ax2.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15) 
        cbar.ax.set_ylabel('Predicted storage modulus (kPa)', fontsize=16)
        plt.xticks([]), plt.yticks([])
        # plt.title('(b) Prediction', fontsize=20)
        save_dir = r'./visuals/{}/{}/{}/'.format(args.train_method, args.model, args.data)
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{sid}.png'))
        sid+=1
        if sid >= 30:
            break

def pde_residual_compute_heter_3d(u, v, w, mu, lambda_, x, y, z, rho_omega_square, mask):
    """
    first dimension: batch
    second dimension : x 
    third dimension: y
    fourth dimension: z
    """

    # compute first order derivative
    dx = x[0,1,0,0] - x[0,0,0,0]
    dy = y[0,0,1,0] - y[0,0,0,0]
    dz = z[0,0,0,1] - z[0,0,0,0]
    
    u_x = (u[:, 2:, 1:-1, 1:-1] - u[:, :-2, 1:-1, 1:-1]) / (2 * dx)
    u_y = (u[:, 1:-1, 2:, 1:-1] - u[:, 1:-1, :-2, 1:-1]) / (2 * dy)
    u_z = (u[:, 1:-1, 1:-1, 2:] - u[:, 1:-1, 1:-1, :-2]) / (2 * dz)
    
    v_x = (v[:, 2:, 1:-1, 1:-1] - v[:, :-2, 1:-1, 1:-1]) / (2 * dx)
    v_y = (v[:, 1:-1, 2:, 1:-1] - v[:, 1:-1, :-2, 1:-1]) / (2 * dy)
    v_z = (v[:, 1:-1, 1:-1, 2:] - v[:, 1:-1, 1:-1, :-2]) / (2 * dz)
    
    w_x = (w[:, 2:, 1:-1, 1:-1] - w[:, :-2, 1:-1, 1:-1]) / (2 * dx)
    w_y = (w[:, 1:-1, 2:, 1:-1] - w[:, 1:-1, :-2, 1:-1]) / (2 * dy)
    w_z = (w[:, 1:-1, 1:-1, 2:] - w[:, 1:-1, 1:-1, :-2]) / (2 * dz)

    # process the stiff
    mu = mu[:, 1:-1, 1:-1, 1:-1]
    lambda_ = lambda_[:, 1:-1, 1:-1, 1:-1]

    # Compute the divergence of u
    div_uvw = u_x + v_y + w_z

    # Compute components of the stress tensor
    sigma_xx = 2 * mu * u_x + lambda_ * div_uvw
    sigma_yy = 2 * mu * v_y + lambda_ * div_uvw
    sigma_zz = 2 * mu * w_z + lambda_ * div_uvw
    sigma_xy = mu * (u_y + v_x)
    sigma_xz = mu * (u_z + w_x)
    sigma_yz = mu * (v_z + w_y)
    sigma_yx = sigma_xy
    sigma_zx = sigma_xz
    sigma_zy = sigma_yz

    # Compute the divergence of the stress tensor
    div_sigma_x = (sigma_xx[:, 2:, 1:-1, 1:-1] - sigma_xx[:, :-2, 1:-1, 1:-1]) / (2 * dx) +\
                  (sigma_xy[:, 1:-1, 2:, 1:-1] - sigma_xy[:, 1:-1, :-2, 1:-1]) / (2 * dy) +\
                  (sigma_xz[:, 1:-1, 1:-1, 2:] - sigma_xz[:, 1:-1, 1:-1, :-2]) / (2 * dz)

    div_sigma_y = (sigma_yx[:, 2:, 1:-1, 1:-1] - sigma_yx[:, :-2, 1:-1, 1:-1]) / (2 * dx) +\
                  (sigma_yy[:, 1:-1, 2:, 1:-1] - sigma_yy[:, 1:-1, :-2, 1:-1]) / (2 * dy) +\
                  (sigma_yz[:, 1:-1, 1:-1, 2:] - sigma_yz[:, 1:-1, 1:-1, :-2]) / (2 * dz)
    
    div_sigma_z = (sigma_zx[:, 2:, 1:-1, 1:-1] - sigma_zx[:, :-2, 1:-1, 1:-1]) / (2 * dx) +\
                  (sigma_zy[:, 1:-1, 2:, 1:-1] - sigma_zy[:, 1:-1, :-2, 1:-1]) / (2 * dy) +\
                  (sigma_zz[:, 1:-1, 1:-1, 2:] - sigma_zz[:, 1:-1, 1:-1, :-2]) / (2 * dz)
    
    # process the data
    u = u[:, 2:-2, 2:-2, 2:-2]
    v = v[:, 2:-2, 2:-2, 2:-2]
    w = w[:, 2:-2, 2:-2, 2:-2]

    # Compute the residuals
    residual_u = div_sigma_x + rho_omega_square * u
    residual_v = div_sigma_y + rho_omega_square * v
    residual_w = div_sigma_z + rho_omega_square * w
    
    return residual_u, residual_v, residual_w

def pde_residual_compute_homo_3d(u, v, w, mu, x, y, z, rho_omega_square, mask):
    """
    first dimension: batch
    second dimension : x 
    third dimension: y
    fourth dimension: z
    """

    # compute first order derivative
    dx = x[0,1,0,0] - x[0,0,0,0]
    dy = y[0,0,1,0] - y[0,0,0,0]
    dz = z[0,0,0,1] - z[0,0,0,0]
    
    u_x = (u[:, 2:, 1:-1, 1:-1] - u[:, :-2, 1:-1, 1:-1]) / (2 * dx)
    u_y = (u[:, 1:-1, 2:, 1:-1] - u[:, 1:-1, :-2, 1:-1]) / (2 * dy)
    u_z = (u[:, 1:-1, 1:-1, 2:] - u[:, 1:-1, 1:-1, :-2]) / (2 * dz)
    
    v_x = (v[:, 2:, 1:-1, 1:-1] - v[:, :-2, 1:-1, 1:-1]) / (2 * dx)
    v_y = (v[:, 1:-1, 2:, 1:-1] - v[:, 1:-1, :-2, 1:-1]) / (2 * dy)
    v_z = (v[:, 1:-1, 1:-1, 2:] - v[:, 1:-1, 1:-1, :-2]) / (2 * dz)
    
    w_x = (w[:, 2:, 1:-1, 1:-1] - w[:, :-2, 1:-1, 1:-1]) / (2 * dx)
    w_y = (w[:, 1:-1, 2:, 1:-1] - w[:, 1:-1, :-2, 1:-1]) / (2 * dy)
    w_z = (w[:, 1:-1, 1:-1, 2:] - w[:, 1:-1, 1:-1, :-2]) / (2 * dz)

    # process the stiff
    mu = mu[:, 1:-1, 1:-1, 1:-1]

    # Compute the divergence of u
    div_uvw = u_x + v_y + w_z

    # Compute components of the stress tensor
    sigma_xx = 2 * mu * u_x 
    sigma_yy = 2 * mu * v_y 
    sigma_zz = 2 * mu * w_z
    sigma_xy = mu * (u_y + v_x)
    sigma_xz = mu * (u_z + w_x)
    sigma_yz = mu * (v_z + w_y)
    sigma_yx = sigma_xy
    sigma_zx = sigma_xz
    sigma_zy = sigma_yz

    # Compute the divergence of the stress tensor
    div_sigma_x = (sigma_xx[:, 2:, 1:-1, 1:-1] - sigma_xx[:, :-2, 1:-1, 1:-1]) / (2 * dx) +\
                  (sigma_xy[:, 1:-1, 2:, 1:-1] - sigma_xy[:, 1:-1, :-2, 1:-1]) / (2 * dy) +\
                  (sigma_xz[:, 1:-1, 1:-1, 2:] - sigma_xz[:, 1:-1, 1:-1, :-2]) / (2 * dz)

    div_sigma_y = (sigma_yx[:, 2:, 1:-1, 1:-1] - sigma_yx[:, :-2, 1:-1, 1:-1]) / (2 * dx) +\
                  (sigma_yy[:, 1:-1, 2:, 1:-1] - sigma_yy[:, 1:-1, :-2, 1:-1]) / (2 * dy) +\
                  (sigma_yz[:, 1:-1, 1:-1, 2:] - sigma_yz[:, 1:-1, 1:-1, :-2]) / (2 * dz)
    
    div_sigma_z = (sigma_zx[:, 2:, 1:-1, 1:-1] - sigma_zx[:, :-2, 1:-1, 1:-1]) / (2 * dx) +\
                  (sigma_zy[:, 1:-1, 2:, 1:-1] - sigma_zy[:, 1:-1, :-2, 1:-1]) / (2 * dy) +\
                  (sigma_zz[:, 1:-1, 1:-1, 2:] - sigma_zz[:, 1:-1, 1:-1, :-2]) / (2 * dz)
    
    # process the data
    u = u[:, 2:-2, 2:-2, 2:-2]
    v = v[:, 2:-2, 2:-2, 2:-2]
    w = w[:, 2:-2, 2:-2, 2:-2]

    # Compute the residuals
    residual_u = div_sigma_x + rho_omega_square * u
    residual_v = div_sigma_y + rho_omega_square * v
    residual_w = div_sigma_z + rho_omega_square * w
    
    return residual_u, residual_v, residual_w 