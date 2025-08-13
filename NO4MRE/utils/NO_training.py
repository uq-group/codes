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
        for (disp, mu, mask) in train_loader:

            # plt.figure()
            # plt.subplot(131)
            # sns.heatmap(disp[0,:,:,0].detach().cpu().numpy())
            # plt.subplot(132)
            # sns.heatmap(mu[0,:,:].detach().cpu().numpy())
            # plt.subplot(133)
            # sns.heatmap(mask[0,:,:].detach().cpu().numpy())
            # plt.savefig('mask.png')
            # assert 1==2

            # reset optimizer
            optimizer.zero_grad()
            
            # prepare the data
            disp_map = disp.float().to(device)   # (B, 4, S, S)
            mu = mu.float().to(device)    # (B, S, S)
            mask = mask.float().to(device)    # (B, S, S)
            
            # model forward
            mu_pred = model(disp_map)

            # backward
            loss = mse(mu_pred * mask, mu * mask)
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
    

def test(args, model, device, val_loader):

    model.eval()
    avg_err = 0
    num_sample = 0
    for (disp, mu, mask) in val_loader:

        # prepare the data
        disp_map = disp.float().to(device)   # (B, 4, S, S)
        mu = mu.float().to(device)    # (B, S, S)
        mask = mask.float().to(device)    # (B, S, S)
        
        # model forward
        mu_pred = model(disp_map)

        # compute the error
        mu_pred = mu_pred * mask
        mu = mu * mask
        relative_err = torch.sqrt(torch.sum((mu_pred - mu)**2) / torch.sum(mu**2)).detach().cpu().item()
        avg_err += relative_err
        num_sample += 1
    
    return avg_err / num_sample

def plot(args, model, device, test_loader):

    model.eval()
    avg_err = 0
    num_sample = 0
    sid = 0
    for (disp, mu, mask) in test_loader:

        batchsize = disp.shape[0]
        assert batchsize == 1, 'plotting need batch size = 1'

        # prepare the data
        disp_map = disp.float().to(device)   # (B, 4, S, S)
        mu = mu.float().to(device)    # (B, S, S)
        mask = mask.float().to(device)    # (B, S, S)
        
        # model forward
        mu_pred = model(disp_map)

        # define color map
        jet = plt.cm.get_cmap('jet', 256)
        new_colors = jet(np.linspace(0, 1, 256))
        new_colors[0] = np.array([1, 1, 1, 1])  # RGBA for white
        custom_jet = ListedColormap(new_colors)

        # compute the error
        mu_pred = (mu_pred * mask).squeeze(0).detach().cpu().numpy()
        mu = (mu * mask).squeeze(0).detach().cpu().numpy()
        plt.figure(figsize=(10, 5), dpi=300)
        plt.subplot(121)
        ax1 = sns.heatmap(mu, cmap=custom_jet, vmin=0, vmax=5,  cbar_kws={'label': 'Exact storage modulus (kPa)'})
        cbar = ax1.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)  # Change tick font size
        cbar.ax.set_ylabel('Exact storage modulus (kPa)', fontsize=16)    # Label font size
        plt.xticks([]), plt.yticks([])
        # plt.title('(a) Exact', fontsize=20)
        plt.subplot(122)
        ax2 = sns.heatmap(mu_pred, cmap=custom_jet, vmin=0, vmax=5,  cbar_kws={'label': 'Predicted storage modulus (kPa)'})
        cbar = ax2.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15) 
        cbar.ax.set_ylabel('Predicted storage modulus (kPa)', fontsize=16)
        plt.xticks([]), plt.yticks([])
        # plt.title('(b) Prediction', fontsize=20)
        save_dir = r'./visuals/{}/{}/{}/'.format(args.train_method, args.model, args.data)
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{sid}.png'))
        sid += 1
        if sid >= 30:
            break

