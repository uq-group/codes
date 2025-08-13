import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_act(act):
    if act == 'tanh':
        func = F.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{act} is not supported')
    return func

def add_padding3(x, num_pad1, num_pad2, num_pad3):
    if max(num_pad1) > 0 or max(num_pad2) > 0 or max(num_pad3) > 0:
        res = F.pad(x, (num_pad3[0], num_pad3[1], num_pad2[0], num_pad2[1], num_pad1[0], num_pad1[1]), 'constant', 0.)
    else:
        res = x
    return res

def remove_padding3(x, num_pad1, num_pad2, num_pad3):
    if max(num_pad1) > 0 or max(num_pad2) > 0 or max(num_pad3) > 0:
        res = x[..., num_pad1[0]:-num_pad1[1], num_pad2[0]:-num_pad2[1], num_pad3[0]:-num_pad3[1]]
    else:
        res = x
    return res

def compl_mul3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
    res = torch.einsum("bixyz,ioxyz->boxyz", a, b)
    return res

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-3]
        size2 = x.shape[-2]
        size3 = x.shape[-1]
        
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, 
                            device=x.device, dtype=torch.cfloat)
        
        # Handle different frequency modes
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), dim=[2, 3, 4])
        return x

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

class UFNO3D_layer(nn.Module):
    def __init__(self, hidden_channels):
        super(UFNO3D_layer, self).__init__()

        self.unet = UNet3D(hidden_channels, hidden_channels)
        self.fno = SpectralConv3d(hidden_channels, hidden_channels, 16, 16, 4)
    
    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        xu = self.unet(x).permute(0, 2, 3, 4, 1)  # (B, H, W, D, C)
        xf = self.fno(x).permute(0, 2, 3, 4, 1)   # (B, H, W, D, C)

        return xu + xf

class UFNO3D(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(UFNO3D, self).__init__()

        self.encoder = nn.Linear(in_channel, hidden_channel)
        self.layer1 = UFNO3D_layer(hidden_channel)
        self.layer2 = UFNO3D_layer(hidden_channel)
        self.layer3 = UFNO3D_layer(hidden_channel)
        self.decoder = nn.Linear(hidden_channel, out_channel)
    
    def forward(self, x):
        x = self.encoder(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.decoder(x)

        return x

class data_driven_UFNO3D_Model(nn.Module):
    def __init__(self):
        super(data_driven_UFNO3D_Model, self).__init__()
        
        self.stiff_model = UFNO3D(6, 128, 1)
    
    def forward(self, u):
        """
        Args:
            - u: (batch_size, x_grid, y_grid, z_grid, 6) - input features including spatial coordinates
        Returns:
            - stiff: (batch_size, x_grid, y_grid, z_grid) - stiffness field
        """
        stiff = self.stiff_model(u)
        return stiff.squeeze(-1)

class physics_informed_UFNO3D_Model(nn.Module):
    def __init__(self):
        super(physics_informed_UFNO3D_Model, self).__init__()
        
        self.stiff_model = UFNO3D(6, 128, 64)
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
        stiff = self.stiff_model(u)
        mu = self.mu_map(stiff).squeeze(-1)
        lam = self.lam_map(stiff).squeeze(-1)

        return mu, lam

# Example usage and testing
if __name__ == "__main__":
    # Test the 3D UFNO model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a test input
    batch_size = 2
    grid_size = 32
    in_channels = 6
    
    x = torch.randn(batch_size, grid_size, grid_size, grid_size, in_channels).to(device)
    
    # Test data-driven model
    model_dd = data_driven_UFNO3D_Model().to(device)
    output_dd = model_dd(x)
    print(f"Data-driven UFNO3D model output shape: {output_dd.shape}")
    print(f"Data-driven UFNO3D model parameters: {sum(p.numel() for p in model_dd.parameters()):,}")
    
    # Test physics-informed model
    model_pi = physics_informed_UFNO3D_Model().to(device)
    mu, lam = model_pi(x)
    print(f"Physics-informed UFNO3D model mu shape: {mu.shape}")
    print(f"Physics-informed UFNO3D model lam shape: {lam.shape}")
    print(f"Physics-informed UFNO3D model parameters: {sum(p.numel() for p in model_pi.parameters()):,}") 