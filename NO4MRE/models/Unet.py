import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

# define model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        
        # Decoder (upsampling)
        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.double_conv(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.double_conv(128, 64)
        
        # Output layer (NO GELU here)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)    # (1,64,100,100)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))    # (1,128,50,50)
        
        # Bottleneck
        bottleneck = self.enc3(F.max_pool2d(enc2, 2))     # (1,256,25,25)
        
        # Decoder
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.out_conv(dec1)

class data_driven_Unet_Model(nn.Module):
    def __init__(self):
        super(data_driven_Unet_Model, self).__init__()
        self.stiff_model = UNet(4, 1)
    
    def forward(self, u):
        u = u.permute(0,3,1,2)
        stiff = self.stiff_model(u)
        stiff = stiff.permute(0,2,3,1)
        return stiff.squeeze(-1)

class physics_informed_Unet_Model(nn.Module):
    def __init__(self):
        super(physics_informed_Unet_Model, self).__init__()
        self.stiff_model = UNet(4, 64)
        self.mu_map = nn.Linear(64, 1)
        self.lam_map = nn.Linear(64, 1)
    
    def forward(self, u):
        u = u.permute(0,3,1,2)
        stiff = self.stiff_model(u)
        stiff = stiff.permute(0,2,3,1)
        mu = self.mu_map(stiff).squeeze(-1)
        lam = self.lam_map(stiff).squeeze(-1)
        return mu, lam
