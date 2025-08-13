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

def add_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = F.pad(x, (num_pad2[0], num_pad2[1], num_pad1[0], num_pad1[1]), 'constant', 0.)
    else:
        res = x
    return res

def remove_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = x[..., num_pad1[0]:-num_pad1[1], num_pad2[0]:-num_pad2[1]]
    else:
        res = x
    return res

def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    res =  torch.einsum("bixy,ioxy->boxy", a, b)
    return res

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device,
                                dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        return x

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

class UFNO_layer(nn.Module):
    def __init__(self, hidden_channels):
        super(UFNO_layer, self).__init__()

        self.unet = UNet(hidden_channels, hidden_channels)
        self.fno = SpectralConv2d(hidden_channels, hidden_channels, 20, 20)
    
    def forward(self, x):

        x = x.permute(0,3,1,2)
        xu = self.unet(x).permute(0,2, 3, 1)
        xf = self.fno(x).permute(0,2, 3, 1)

        return xu + xf

class UFNO(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(UFNO, self).__init__()

        self.encoder = nn.Linear(in_channel, hidden_channel)
        self.layer1 = UFNO_layer(hidden_channel)
        self.layer2 = UFNO_layer(hidden_channel)
        self.layer3 = UFNO_layer(hidden_channel)
        self.decoder = nn.Linear( hidden_channel, out_channel)
    
    def forward(self, x):

        x = self.encoder(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.decoder(x)

        return x

class data_driven_UFNO_Model(nn.Module):
    def __init__(self):
        super(data_driven_UFNO_Model, self).__init__()
        
        self.stiff_model = UFNO(4, 128, 1)
    
    def forward(self, u):

        stiff = self.stiff_model(u)

        return stiff.squeeze(-1)

class physics_informed_UFNO_Model(nn.Module):
    def __init__(self):
        super(physics_informed_UFNO_Model, self).__init__()
        
        self.stiff_model = UFNO(4, 128, 64)
        self.mu_map = nn.Linear(64, 1)
        self.lam_map = nn.Linear(64, 1)
    
    def forward(self, u):

        stiff = self.stiff_model(u)
        mu = self.mu_map(stiff).squeeze(-1)
        lam = self.lam_map(stiff).squeeze(-1)

        return mu, lam


