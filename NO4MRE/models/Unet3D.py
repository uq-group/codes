import torch
import torch.nn as nn
import torch.nn.functional as F

# define 3D UNet model
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
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
        self.out_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)    # (1,64,100,100,100)
        enc2 = self.enc2(F.max_pool3d(enc1, 2))    # (1,128,50,50,50)
        
        # Bottleneck
        bottleneck = self.enc3(F.max_pool3d(enc2, 2))     # (1,256,25,25,25)
        
        # Decoder with proper size handling
        dec2 = self.upconv2(bottleneck)
        # Ensure sizes match for concatenation
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='trilinear', align_corners=False)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        # Ensure sizes match for concatenation
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='trilinear', align_corners=False)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.out_conv(dec1)

class data_driven_Unet3D_Model(nn.Module):
    def __init__(self):
        super(data_driven_Unet3D_Model, self).__init__()
        self.stiff_model = UNet3D(6, 1)
    
    def forward(self, u):
        """
        Args:
            - u: (batch_size, x_grid, y_grid, z_grid, 6) - input features including spatial coordinates
        Returns:
            - stiff: (batch_size, x_grid, y_grid, z_grid) - stiffness field
        """
        u = u.permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        stiff = self.stiff_model(u)
        stiff = stiff.permute(0, 2, 3, 4, 1)  # (B, H, W, D, C)
        return stiff.squeeze(-1)

class physics_informed_Unet3D_Model(nn.Module):
    def __init__(self):
        super(physics_informed_Unet3D_Model, self).__init__()
        self.stiff_model = UNet3D(6, 64)
        self.mu_map = nn.Linear(64, 1)
        self.lam_map = nn.Linear(64, 1)
    
    def forward(self, u):
        """
        Args:
            - u: (batch_size, x_grid, y_grid, z_grid, 6) - input features including spatial coordinates
        Returns:
            - mu: (batch_size, x_grid, y_grid, z_grid) - shear modulus field
            - lam: (batch_size, x_grid, y_grid, z_grid) - Lame's first parameter field
        """
        u = u.permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        stiff = self.stiff_model(u)
        stiff = stiff.permute(0, 2, 3, 4, 1)  # (B, H, W, D, C)
        mu = self.mu_map(stiff).squeeze(-1)
        lam = self.lam_map(stiff).squeeze(-1)
        return mu, lam

# Example usage and testing
if __name__ == "__main__":
    # Test the 3D UNet model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a test input
    batch_size = 2
    grid_size = 32
    in_channels = 6
    
    x = torch.randn(batch_size, grid_size, grid_size, grid_size, in_channels).to(device)
    
    # Test data-driven model
    model_dd = data_driven_Unet3D_Model().to(device)
    output_dd = model_dd(x)
    print(f"Data-driven UNet3D model output shape: {output_dd.shape}")
    print(f"Data-driven UNet3D model parameters: {sum(p.numel() for p in model_dd.parameters()):,}")
    
    # Test physics-informed model
    model_pi = physics_informed_Unet3D_Model().to(device)
    mu, lam = model_pi(x)
    print(f"Physics-informed UNet3D model mu shape: {mu.shape}")
    print(f"Physics-informed UNet3D model lam shape: {lam.shape}")
    print(f"Physics-informed UNet3D model parameters: {sum(p.numel() for p in model_pi.parameters()):,}") 