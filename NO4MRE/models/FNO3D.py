from siren_pytorch import SirenNet
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
3D FNO model for MRE inversion
'''

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

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=4, out_dim=1,
                 act='gelu', 
                 pad_ratio=[0., 0., 0.]):
        super(FNO3d, self).__init__()
        """
        Args:
            - modes1: list of int, number of modes in first dimension in each layer
            - modes2: list of int, number of modes in second dimension in each layer
            - modes3: list of int, number of modes in third dimension in each layer
            - width: int, optional, if layers is None, it will be initialized as [width] * [len(modes1) + 1] 
            - in_dim: number of input channels
            - out_dim: number of output channels
            - act: activation function, {tanh, gelu, relu, leaky_relu}, default: gelu
            - pad_ratio: list of float, or float; portion of domain to be extended. If float, paddings are added to the right. 
            If list, paddings are added to all sides. pad_ratio[0] pads left in dim1, pad_ratio[1] pads right in dim2, pad_ratio[2] pads right in dim3. 
        """
        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 3, 'Cannot add padding in more than 3 directions'
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
    
        self.pad_ratio = pad_ratio
        # input channel is 4: (a(x, y, z), x, y, z)
        if layers is None:
            self.layers = [width] * (len(modes1) + 1)
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv3d(
            in_size, out_size, mode1_num, mode2_num, mode3_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, layers[-1])
        self.fc3 = nn.Linear(layers[-1], out_dim)
        self.act = _get_act(act)

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, z_grid, 4)
        Returns:
            - x: (batch size, x_grid, y_grid, z_grid, 1)
        '''
        size_1, size_2, size_3 = x.shape[1], x.shape[2], x.shape[3]
        if max(self.pad_ratio) > 0:
            num_pad1 = [round(i * size_1) for i in self.pad_ratio]
            num_pad2 = [round(i * size_2) for i in self.pad_ratio]
            num_pad3 = [round(i * size_3) for i in self.pad_ratio]
        else:
            num_pad1 = num_pad2 = num_pad3 = [0.]

        length = len(self.ws)
        batchsize = x.shape[0]
        x = self.fc0(x)    # (B, S, S, S, F)
        x = x.permute(0, 4, 1, 2, 3)   # (B, F, S, S, S)
        x = add_padding3(x, num_pad1, num_pad2, num_pad3)
        size_x, size_y, size_z = x.shape[-3], x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            
            # Fourier convolution
            x1 = speconv(x)

            # linear layer
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y, size_z)    # (B, F, S, S, S)
            
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)
                
        x = remove_padding3(x, num_pad1, num_pad2, num_pad3)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x

class data_driven_FNO3D_Model(nn.Module):
    def __init__(self):
        super(data_driven_FNO3D_Model, self).__init__()
        
        self.stiff_model = FNO3d(modes1=[16, 16, 16, 16], modes2=[16, 16, 16, 16], modes3=[4, 4, 4, 4],
                 width=64, fc_dim=128,
                 layers=[64, 64, 64, 64, 64],
                 in_dim=6, out_dim=1,
                 act='gelu')
    
    def forward(self, u):
        """
        Args:
            - u: (batch_size, x_grid, y_grid, z_grid, 4) - input features including spatial coordinates
        Returns:
            - stiff: (batch_size, x_grid, y_grid, z_grid) - stiffness field
        """
        stiff = self.stiff_model(u)
        return stiff.squeeze(-1)

class physics_informed_FNO3D_Model(nn.Module):
    def __init__(self):
        super(physics_informed_FNO3D_Model, self).__init__()
        
        self.stiff_model = FNO3d(modes1=[16, 16, 16, 16], modes2=[16, 16, 16, 16], modes3=[4, 4, 4, 4],
                 width=64, fc_dim=128,
                 layers=[64, 64, 64, 64, 64],
                 in_dim=6, out_dim=64,
                 act='gelu')
        self.mu_map = nn.Linear(64, 1)
        self.lam_map = nn.Linear(64, 1)
    
    def forward(self, u):
        """
        Args:
            - u: (batch_size, x_grid, y_grid, z_grid, 4) - input features including spatial coordinates
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
    # Test the 3D FNO model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a test input
    batch_size = 2
    grid_size = 32
    in_channels = 4
    
    x = torch.randn(batch_size, grid_size, grid_size, grid_size, in_channels).to(device)
    
    # Test data-driven model
    model_dd = data_driven_FNO3D_Model().to(device)
    output_dd = model_dd(x)
    print(f"Data-driven model output shape: {output_dd.shape}")
    print(f"Data-driven model parameters: {sum(p.numel() for p in model_dd.parameters()):,}")
    
    # Test physics-informed model
    model_pi = physics_informed_FNO3D_Model().to(device)
    mu, lam = model_pi(x)
    print(f"Physics-informed model mu shape: {mu.shape}")
    print(f"Physics-informed model lam shape: {lam.shape}")
    print(f"Physics-informed model parameters: {sum(p.numel() for p in model_pi.parameters()):,}") 