import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # we use group norm because it's better for small batch sizes
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # if the number of input channels is equal to the number of output channels, we don't need to change the shape
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, x):
        # x: (batch_size, in_channels, H, W)
        residual = x
        # (batch_size, in_channels, H, W) -> (batch_size, in_channels, H, W)
        x = self.groupnorm_1(x)
        x = F.silu(x)
        # (batch_size, in_channels, H, W) -> (batch_size, out_channels, H, W)
        x = self.conv1(x)
        # (batch_size, out_channels, H, W) -> (batch_size, out_channels, H, W)
        x = self.groupnorm_2(x)
        x = F.silu(x)   
        # (batch_size, out_channels, H, W) -> (batch_size, out_channels, H, W)
        x = self.conv2(x)
    
        return x + self.residual_layer(residual)
      

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        # we use only one head, channels is gonna be our d_model
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x):
        # x: (batch_size, channels, H, W)
        residue = x
        
        x = self.groupnorm(x)
        batch_size, channels, H, W = x.shape
        # (batch_size, channles, H, W) -> (batch_size, channels, H*W)
        x = x.view((batch_size, channels, H*W))
        # (batch_size, channels, H*W) -> (batch_size, H*W, channels)
        x = x.transpose(-1, -2)
        # Apply self-attention:
        # (batch_size, H*W, channels) -> (batch_size, H*W, channels)
        x = self.attention(x)
        # (batch_size, H*W, channels) -> (batch_size, channels, H*W)
        x = x.transpose(-1, -2)
        # (batch_size, channels, H*W) -> (batch_size, channels, H, W)
        x = x.view((batch_size, channels, H, W))
        # (batch_size, channels, H, W) + (batch_size, channels, H, W) -> (batch_size, channels, H, W)
        x = x + residue
        return x    
      
        
class VAE_encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
        # (batch_size, channels=3, H=512, W=512) -> (batch_size, channels=128, H=512, W=512)
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
        # (batch_size, channels=128, H=512, W=512) -> (batch_size, channels=128, H=512, W=512)
        VAE_ResidualBlock(in_channels=128, out_channels=128),
        # (batch_size, 128, 512, 512) -> (batch_size, 128, 512, 512)
        VAE_ResidualBlock(in_channels=128, out_channels=128),
        # (batch_size, 128, 512, 512) -> (batch_size, 128, 256, 256)
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
        # (batch_size, 128, 256, 256) -> (batch_size, 256, 256, 256)
        VAE_ResidualBlock(128, 256),
        # (batch_size, 256, 256, 256) -> (batch_size, 256, 256, 256)
        VAE_ResidualBlock(256, 256),
        # (batch_size, 256, 256, 256) -> (batch_size, 256, 128, 128)
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
        # (batch_size, 256, 128, 128) -> (batch_size, 512, 128, 128)
        VAE_ResidualBlock(256, 512),
        # (batch_size, 512, 128, 128) -> (batch_size, 512, 128, 128)
        VAE_ResidualBlock(512, 512),
        # (batch_size, 512, 128, 128) -> (batch_size, 512, 64, 64)
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
        # (batch_size, 512, 64, 64) -> (batch_size, 512, 64, 64)
        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),
        VAE_ResidualBlock(512, 512),
        # (batch_size, 512, 64, 64) -> (batch_size, 512, 64, 64)
        VAE_AttentionBlock(512),
        # (batch_size, 512, 64, 64) -> (batch_size, 512, 64, 64)
        VAE_ResidualBlock(512, 512),
    
        nn.GroupNorm(32, 512),
        nn.SiLU(),
        # (batch_size, 512, 64, 64) -> (batch_size, 8, 64, 64)
        nn.Conv2d(512, 8, kernel_size=3, padding=1),
        # (batch_size, 8, 64, 64) -> (batch_size, 8, 64, 64)
        nn.Conv2d(8, 8, kernel_size=8, padding=0)
        )
        
    def forward(self, x, noise):
        # x: (batch_size, channel, H ,W)
        # noise: (batch_size, 4, H/8, W/8)
        
        # (batch_size, 3, 512 ,512) -> (batch_size, 8, 64, 64) 
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        # (batch_size, 8, 64, 64) -> 2 Tensors of shape (batch_size, 4, 64, 64)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        standard_div = variance.sqrt()
        
        x = mean + standard_div * noise
        x *= 0.18125
        
        return x
        
        
class VAE_decoder(nn.Sequential):
    def __init__(self):
        super.__init__(
            # (batch_size, 4, 64, 64) -> (batch_size, 4, 64, 64)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            # (batch_size, 4, 64, 64) -> (batch_size, 512, 64, 64)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            # (batch_size, 512, 64, 64) -> (batch_size, 512, 64, 64)
            VAE_ResidualBlock(512, 512), 
            VAE_AttentionBlock(512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            # (batch_size, 512, 64, 64) -> (batch_size, 512, 128, 128)
            nn.Upsample(scale_factor=2),
            # (batch_size, 512, 128, 128) -> (batch_size, 512, 128, 128)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            # (batch_size, 512, 128, 128) -> (batch_size, 512, 256, 256)
            nn.Upsample(scale_factor=2), 
            # (batch_size, 512, 256, 256) -> (batch_size, 512, 256, 256)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            # (batch_size, 512, 256, 256) -> (batch_size, 256, 256, 256)
            VAE_ResidualBlock(512, 256), 
            # (batch_size, 256, 256, 256) -> (batch_size, 256, 256, 256)
            VAE_ResidualBlock(256, 256), 
            VAE_ResidualBlock(256, 256), 
            # (batch_size, 256, 256, 256) -> (batch_size, 256, 512, 512)
            nn.Upsample(scale_factor=2), 
            # (batch_size, 256, 512, 512) -> (batch_size, 256, 512, 512)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            # (batch_size, 256, 512, 512) -> (batch_size, 128, 512, 512)
            VAE_ResidualBlock(256, 128), 
            # (batch_size, 128, 512, 512) -> (batch_size, 128, 512, 512)
            VAE_ResidualBlock(128, 128), 
            VAE_ResidualBlock(128, 128), 
            nn.GroupNorm(32, 128), 
            nn.SiLU(), 
            # (batch_size, 128, 512, 512) -> (batch_size, 3, 512, 512)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),      
        )
        
    def forward(self, x):
        # x: (batch_size, 4, 64, 64)
        x /= 0.18125
        
        # (batch_size, 4, 64, 64) -> (batch_size, 3, 512, 512)
        for module in self:
            x = module(x)
        
        return x