import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import numpy as np

""" Def: 2d Wavelet convolutional layer (slim continuous) """
class WaveConv2dCwt(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet1, wavelet2, omega=8):
        super(WaveConv2dCwt, self).__init__()

        """
        !! It is computationally expensive than the discrete "WaveConv2d" !!
        2D Wavelet layer. It does SCWT (Slim continuous wavelet transform),
                                linear transform, and Inverse dWT. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet1     : string, Specifies the first level biorthogonal wavelet filters
        wavelet2     : string, Specifies the second level quarter shift filters
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights0 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights- 15r, 45r, 75r, 105r, 135r, 165r : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for REAL wavelet coefficients at 15, 45, 75, 105, 135, 165 angles
        self.weights- 15c, 45c, 75c, 105c, 135c, 165c : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for COMPLEX wavelet coefficients at 15, 45, 75, 105, 135, 165 angles
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 2:
                raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
        self.wavelet_level1 = wavelet1
        self.wavelet_level2 = wavelet2        
        dummy_data = torch.randn( 1,1,*self.size ) 
        dwt_ = DTCWTForward(J=self.level, biort=self.wavelet_level1, qshift=self.wavelet_level2)
        mode_data, mode_coef = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        self.modes21 = mode_coef[-1].shape[-3]
        self.modes22 = mode_coef[-1].shape[-2]
        self.omega = omega
        self.effective_modes_x = self.modes1//self.omega+1
        self.effective_modes_y = self.modes2//self.omega+1
        self.effective_modes_xx = self.modes21//self.omega+1
        self.effective_modes_yy = self.modes22//self.omega+1
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights_01 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, dtype=torch.cfloat))
        self.weights_02 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, dtype=torch.cfloat))
        self.weights_15r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_15r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_15c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_15c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_45r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_45r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_45c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_45c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_75r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_75r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_75c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_75c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_105r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_105r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_105c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_105c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_135r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_135r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_135c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_135c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_165r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_165r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_165c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_165c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))

    # Convolution
    def mul2d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    # Spectral Convolution
    def spectralconv(self, waves, weights1, weights2):
        """
        Performs spectral convolution using Fourier decomposition

        Input Parameters
        ----------
        waves : tensor, shape-[Batch * Channel * size(x)]
                signal to be convolved, here the wavelet coefficients.
        weights : tensor, shape-[in_channel * out_channel * size(x)]
                The weights/kernel of the neural network.

        Returns
        -------
        convolved signal : tensor, shape-[batch * out_channel * size(x)]

        """
        # Get the frequency componenets
        modes1, modes2 = weights2.shape[-2], weights2.shape[-1]
        xw = torch.fft.rfft2(waves)
        
        # Initialize the output
        conv_out = torch.zeros(waves.shape[0], self.out_channels, waves.shape[-2], waves.shape[-1]//2+1, dtype=torch.cfloat, device=waves.device)
        
        # Perform Element-wise multiplication in spectral doamin
        conv_out[:,:,:modes1,:modes2] = self.mul2d(xw[:,:,:modes1,:modes2], weights1)
        conv_out[:,:,-modes1:,:modes2] = self.mul2d(xw[:,:,-modes1:,:modes2], weights2)
        return torch.fft.irfft2(conv_out, s=(waves.shape[-2], waves.shape[-1]))

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """      
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(J=self.level+factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(J=self.level-factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)            
        else:
            # Compute dual tree continuous Wavelet coefficients 
            cwt = DTCWTForward(J=self.level, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)
        
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.spectralconv(x_ft, self.weights_01, self.weights_02)
        # Multiply the final detailed wavelet coefficients        
        out_coeff[-1][:,:,0,:,:,0] = self.spectralconv(x_coeff[-1][:,:,0,:,:,0].clone(), self.weights_15r1, self.weights_15r2)
        out_coeff[-1][:,:,0,:,:,1] = self.spectralconv(x_coeff[-1][:,:,0,:,:,1].clone(), self.weights_15c1, self.weights_15c2)
        out_coeff[-1][:,:,1,:,:,0] = self.spectralconv(x_coeff[-1][:,:,1,:,:,0].clone(), self.weights_45r1, self.weights_45r2)
        out_coeff[-1][:,:,1,:,:,1] = self.spectralconv(x_coeff[-1][:,:,1,:,:,1].clone(), self.weights_45c1, self.weights_45c2)
        out_coeff[-1][:,:,2,:,:,0] = self.spectralconv(x_coeff[-1][:,:,2,:,:,0].clone(), self.weights_75r1, self.weights_75r2)
        out_coeff[-1][:,:,2,:,:,1] = self.spectralconv(x_coeff[-1][:,:,2,:,:,1].clone(), self.weights_75c1, self.weights_75c2)
        out_coeff[-1][:,:,3,:,:,0] = self.spectralconv(x_coeff[-1][:,:,3,:,:,0].clone(), self.weights_105r1, self.weights_105r2)
        out_coeff[-1][:,:,3,:,:,1] = self.spectralconv(x_coeff[-1][:,:,3,:,:,1].clone(), self.weights_105c1, self.weights_105c2)
        out_coeff[-1][:,:,4,:,:,0] = self.spectralconv(x_coeff[-1][:,:,4,:,:,0].clone(), self.weights_135r1, self.weights_135r2)
        out_coeff[-1][:,:,4,:,:,1] = self.spectralconv(x_coeff[-1][:,:,4,:,:,1].clone(), self.weights_135c1, self.weights_135c2)
        out_coeff[-1][:,:,5,:,:,0] = self.spectralconv(x_coeff[-1][:,:,5,:,:,0].clone(), self.weights_165r1, self.weights_165r2)
        out_coeff[-1][:,:,5,:,:,1] = self.spectralconv(x_coeff[-1][:,:,5,:,:,1].clone(), self.weights_165c1, self.weights_165c2)        
        
        # Reconstruct the signal
        icwt = DTCWTInverse(biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
        x = icwt((out_ft, out_coeff))
        return x


# %%
""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, out_channel, width, level, layers, size, wavelet, in_channel, grid_range, omega, padding=0):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 2 elements (for 2D), image size
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet1 = wavelet[0]
        self.wavelet2 = wavelet[1]
        self.omega = omega
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range( self.layers ):
            self.conv.append( WaveConv2dCwt(self.width, self.width, self.level, size=self.size,
                                            wavelet1=self.wavelet1, wavelet2=self.wavelet2, omega=self.omega) )
            self.w.append( nn.Conv2d(self.width, self.width, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channel)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
                
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]     
        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        x = F.mish( self.fc1(x) )            # Shape: Batch * x * y * Channel
        x = self.fc2(x)                      # Shape: Batch * x * y * Channel
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class data_driven_WNO_Model(nn.Module):
    def __init__(self):
        super(data_driven_WNO_Model, self).__init__()
        
        self.stiff_model = WNO2d(out_channel=1, width=40, level=2, layers=4, size=[200,200], 
            wavelet=['near_sym_b', 'qshift_b'], in_channel=6, grid_range=[1,1], omega=8, padding=0)
    
    def forward(self, u):

        stiff = self.stiff_model(u)

        return stiff.squeeze(-1)

class physics_informed_WNO_Model(nn.Module):
    def __init__(self):
        super(physics_informed_WNO_Model, self).__init__()
        
        self.stiff_model =  WNO2d(out_channel=64, width=40, level=2, layers=4, size=[200,200], 
            wavelet=['near_sym_b', 'qshift_b'], in_channel=6, grid_range=[1,1], omega=8, padding=0)
        self.mu_map = nn.Linear(64, 1)
        self.lam_map = nn.Linear(64, 1)
    
    def forward(self, u):

        stiff = self.stiff_model(u)
        mu = self.mu_map(stiff).squeeze(-1)
        lam = self.lam_map(stiff).squeeze(-1)

        return mu, lam