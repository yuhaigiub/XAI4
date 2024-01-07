import torch
import torch.nn.functional as F
from torch import nn, Tensor

import torchdiffeq

from beatsODE2.layers import CGPBlock, dilated_inception

class BeatsODE2(nn.Module):
    def __init__(self,
                 device,
                 adj,
                 time=1.0,
                 step_size=0.25):
        super(BeatsODE2, self).__init__()
        self.time = time
        self.step_size = step_size
        self.stacks = nn.ModuleList()
        self.blocks = nn.ModuleList()
        
        n_stacks = 2 
        n_blocks = 2
        for i in range(n_stacks):
            blocks = nn.ModuleList()
            for j in range(n_blocks):
                blocks.append(BeatsBlock(device, adj=adj))
            self.stacks.append(blocks)
    
    def forward(self, backcast: Tensor):
        forecast = torch.zeros_like(backcast).type_as(backcast)
        for blocks in self.stacks:
            for block in blocks:
                b, f = block(backcast)
            
                forecast = forecast + f.transpose(1, 3)
                backcast = backcast - b.transpose(1, 3)
        
        backcast = backcast.transpose(1, 3)
        forecast = forecast.transpose(1, 3)
        
        return forecast
        
class BeatsBlock(nn.Module):
    def __init__(self,
                 device,
                 in_dim=2,
                 out_dim=12,
                 adj=None,
                 seq_len=12,
                 time=1.2, step_size=0.4,
                 conv_dim=32,
                 end_dim=128):
        super(BeatsBlock, self).__init__()
        self.seq_len = seq_len
        self.adj = torch.tensor(adj, dtype=torch.float32).to(device)
        
        self.start_conv = nn.Conv2d(in_dim, conv_dim, kernel_size=(1, 1))
        
        self.nfe = round(time / step_size)
        max_kernel_size = 7
        self.receptive_field = self.nfe * (max_kernel_size - 1) + 2
        
        self.ODE = ODEBlock(STBlock(self.receptive_field, 1, conv_dim), 
                            time, 
                            step_size)
        
        # self.end_conv_0 = nn.Conv2d(conv_dim, end_dim // 2, kernel_size=(1, 1))
        # self.end_conv_1 = nn.Conv2d(end_dim // 2, end_dim, kernel_size=(1, 1))
        # self.end_conv_2 = nn.Conv2d(end_dim, out_dim, kernel_size=(1, 1))
        
        self.backcast_decoder = nn.Sequential(nn.Conv2d(conv_dim, end_dim, kernel_size=(1, 1)),
                                              nn.ReLU(),
                                              nn.Conv2d(end_dim, out_dim, kernel_size=(1, 1)))
        
        self.forecast_decoder = nn.Sequential(nn.Conv2d(conv_dim, end_dim, kernel_size=(1, 1)),
                                              nn.ReLU(),
                                              nn.Conv2d(end_dim, out_dim, kernel_size=(1, 1)))
        
        self.backcast = None
    
    def set_backcast(self, backcast: Tensor):
        if backcast is not None:
            self.backcast = backcast.clone().detach()
        else:
            self.backcast = None
        
    def forward(self, x: Tensor):
        if self.seq_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - self.seq_len, 0))
        
        x = self.start_conv(x)
        
        self.ODE.odefunc.set_adj(self.adj)
        x = self.ODE(x)
        self.ODE.odefunc.set_intermediate(1)
        
        x = x[..., -2:]
        x = F.layer_norm(x, tuple(x.shape[1:]), weight=None, bias=None, eps=1e-5)
        
        # decoder
        backcast = self.backcast_decoder(x)
        forecast = self.forecast_decoder(x)
        
        return backcast, forecast

class ODEBlock(nn.Module):
    def __init__(self, odefunc, time, step_size):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        
        self.time = time
        self.step_size = step_size
    
    def forward(self, x: Tensor):
        self.integration_time = torch.tensor([0, self.time]).float().type_as(x)
        out = torchdiffeq.odeint(self.odefunc, 
                                 x, 
                                 self.integration_time,
                                 method="euler",
                                 options=dict(step_size=self.step_size))
        return out[-1]

class STBlock(nn.Module):
    def __init__(self, 
                 receptive_field,
                 dilation,
                 hidden_dim):
        super(STBlock, self).__init__()
        self.receptive_field = receptive_field
        self.intermediate_seq_len = receptive_field
        self.new_dilation = 1
        self.dilation_factor = dilation
        
        self.inception_1 = dilated_inception(hidden_dim, hidden_dim)
        self.inception_2 = dilated_inception(hidden_dim, hidden_dim)
        
        self.gconv1 = CGPBlock(hidden_dim, hidden_dim)
        self.gconv2 = CGPBlock(hidden_dim, hidden_dim)
        
        self.adj = None
    
    def set_adj(self, adj: Tensor):
        self.adj = adj
    
    def set_intermediate(self, dilation):
        self.new_dilation = dilation
        self.intermediate_seq_len = self.receptive_field
    
    def forward(self, t, x: Tensor):
        x = x[..., -self.intermediate_seq_len:]
        
        for tconv in self.inception_1.tconv:
            tconv.dilation = (1, self.new_dilation)
        for tconv in self.inception_2.tconv:
            tconv.dilation = (1, self.new_dilation)
        
        _filter = self.inception_1(x)
        _filter = torch.tanh(_filter)
        
        _gate = self.inception_2(x)
        _gate = torch.sigmoid(_gate)
        
        x = _filter * _gate
        
        self.new_dilation *= self.dilation_factor
        self.intermediate_seq_len = x.size(3)
        
        x = self.gconv1(x, self.adj) + self.gconv2(x, self.adj.T)
        
        x = F.pad(x, (self.receptive_field - x.size(3), 0))
        return x
