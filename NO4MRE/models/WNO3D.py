import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

""" Def: 3d Wavelet-inspired convolutional layer using 3D FFT """
class WaveConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, omega=8):
        super(WaveConv3d, self).__init__()

        """
        3D Wavelet-inspired layer using 3D FFT and spectral convolution.
        This is a 3D adaptation of the wavelet concept using frequency domain processing.
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        modes1       : scalar, number of modes in first dimension
        modes2       : scalar, number of modes in second dimension
        modes3       : scalar, number of modes in third dimension
        omega        : scalar, frequency scaling factor
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.omega = omega
        
        # Parameter initialization with wavelet-inspired scaling
        self.scale = (1 / (in_channels * out_channels))
        
        # Multiple weight tensors for different frequency bands (wavelet-inspired)
        self.weights_low = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1//2, self.modes2//2, self.modes3//2, dtype=torch.cfloat))
        self.weights_mid1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1//2, self.modes2//2, self.modes3//2, dtype=torch.cfloat))
        self.weights_mid2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1//2, self.modes2//2, self.modes3//2, dtype=torch.cfloat))
        self.weights_high = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1//2, self.modes2//2, self.modes3//2, dtype=torch.cfloat))

    # 3D Convolution
    def mul3d(self, input, weights):
        """
        Performs element-wise multiplication for 3D

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y * z)
                  3D frequency coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y * z)
                  kernel weights of corresponding frequency coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y * z)
        """
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
    
    # Spectral Convolution for 3D with wavelet-inspired frequency bands
    def spectralconv3d(self, waves):
        """
        Performs spectral convolution using 3D Fourier decomposition with wavelet-inspired frequency bands

        Input Parameters
        ----------
        waves : tensor, shape-[Batch * Channel * size(x) * size(y) * size(z)]
                signal to be convolved

        Returns
        -------
        convolved signal : tensor, shape-[batch * out_channel * size(x) * size(y) * size(z)]

        """
        # Get the frequency components using 3D FFT
        xw = torch.fft.rfftn(waves, dim=[2, 3, 4])
        
        # Initialize the output
        conv_out = torch.zeros(waves.shape[0], self.out_channels, waves.shape[-3], waves.shape[-2], waves.shape[-1]//2+1, 
                              dtype=torch.cfloat, device=waves.device)
        
        # Apply different weights to different frequency bands (wavelet-inspired)
        # Low frequency band
        conv_out[:,:,:self.modes1//2,:self.modes2//2,:self.modes3//2] = \
            self.mul3d(xw[:,:,:self.modes1//2,:self.modes2//2,:self.modes3//2], self.weights_low)
        
        # Mid frequency band 1
        conv_out[:,:,-self.modes1//2:,:self.modes2//2,:self.modes3//2] = \
            self.mul3d(xw[:,:,-self.modes1//2:,:self.modes2//2,:self.modes3//2], self.weights_mid1)
        
        # Mid frequency band 2
        conv_out[:,:,:self.modes1//2,-self.modes2//2:,:self.modes3//2] = \
            self.mul3d(xw[:,:,:self.modes1//2,-self.modes2//2:,:self.modes3//2], self.weights_mid2)
        
        # High frequency band
        conv_out[:,:,-self.modes1//2:,-self.modes2//2:,:self.modes3//2] = \
            self.mul3d(xw[:,:,-self.modes1//2:,-self.modes2//2:,:self.modes3//2], self.weights_high)
        
        return torch.fft.irfftn(conv_out, s=(waves.shape[-3], waves.shape[-2], waves.shape[-1]), dim=[2, 3, 4])

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y * z]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y * z]
        """      
        return self.spectralconv3d(x)


# %%
""" The forward operation """
class WNO3d(nn.Module):
    def __init__(self, out_channel, width, level, layers, size, wavelet, in_channel, grid_range, omega, padding=0):
        super(WNO3d, self).__init__()

        """
        The WNO3D network. It contains l-layers of the Wavelet-inspired integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y,z) = g(K.v + W.v)(x,y,z).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 6-channel tensor, Initial input and location (a(x,y,z), x,y,z)
              : shape: (batchsize * x=width * y=height * z=depth * c=6)
        Output: Solution of a later timestep (u(x,y,z))
              : shape: (batchsize * x=width * y=height * z=depth * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition levels (used for mode calculation)
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 3 elements (for 3D), image size
        wavelet: string, wavelet filter (kept for compatibility)
        in_channel: scalar, channels in input including grid
        grid_range: list with 3 elements (for 3D), right supports of 3D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.omega = omega
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        # Calculate modes based on level (wavelet-inspired)
        base_modes = 16 // (2 ** (level - 1))
        self.modes1 = max(base_modes, 4)
        self.modes2 = max(base_modes, 4)
        self.modes3 = max(base_modes, 2)
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel + 3, self.width) # input channel is 6: (a(x, y, z), x, y, z)
        for i in range(self.layers):
            self.conv.append(WaveConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3, omega=self.omega))
            self.w.append(nn.Conv3d(self.width, self.width, 1))
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channel)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    
        x = self.fc0(x)                      # Shape: Batch * x * y * z * Channel
        x = x.permute(0, 4, 1, 2, 3)         # Shape: Batch * Channel * x * y * z
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding, 0,self.padding]) 
        
        for index, (convl, wl) in enumerate(zip(self.conv, self.w)):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y * z
                
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding, :-self.padding]     
        x = x.permute(0, 2, 3, 4, 1)         # Shape: Batch * x * y * z * Channel
        x = F.mish(self.fc1(x))              # Shape: Batch * x * y * z * Channel
        x = self.fc2(x)                      # Shape: Batch * x * y * z * Channel
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, self.grid_range[2], size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


class data_driven_WNO3D_Model(nn.Module):
    def __init__(self):
        super(data_driven_WNO3D_Model, self).__init__()
        
        self.stiff_model = WNO3d(out_channel=1, width=40, level=2, layers=4, size=[100,100,50], 
            wavelet=['near_sym_b', 'qshift_b'], in_channel=6, grid_range=[1,1,1], omega=8, padding=0)
    
    def forward(self, u):
        """
        Args:
            - u: (batch_size, x_grid, y_grid, z_grid, 6) - input features including spatial coordinates
        Returns:
            - stiff: (batch_size, x_grid, y_grid, z_grid) - stiffness field
        """
        stiff = self.stiff_model(u)
        return stiff.squeeze(-1)

class physics_informed_WNO3D_Model(nn.Module):
    def __init__(self):
        super(physics_informed_WNO3D_Model, self).__init__()
        
        self.stiff_model = WNO3d(out_channel=64, width=40, level=2, layers=4, size=[100,100,50], 
            wavelet=['near_sym_b', 'qshift_b'], in_channel=6, grid_range=[1,1,1], omega=8, padding=0)
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
    # Test the 3D WNO model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a test input
    batch_size = 2
    grid_size = 32
    in_channels = 6
    
    x = torch.randn(batch_size, grid_size, grid_size, grid_size, in_channels).to(device)
    
    # Test data-driven model
    model_dd = data_driven_WNO3D_Model().to(device)
    output_dd = model_dd(x)
    print(f"Data-driven WNO3D model output shape: {output_dd.shape}")
    print(f"Data-driven WNO3D model parameters: {sum(p.numel() for p in model_dd.parameters()):,}")
    
    # Test physics-informed model
    model_pi = physics_informed_WNO3D_Model().to(device)
    mu, lam = model_pi(x)
    print(f"Physics-informed WNO3D model mu shape: {mu.shape}")
    print(f"Physics-informed WNO3D model lam shape: {lam.shape}")
    print(f"Physics-informed WNO3D model parameters: {sum(p.numel() for p in model_pi.parameters()):,}") 