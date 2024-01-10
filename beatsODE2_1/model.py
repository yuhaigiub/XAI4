import torch
import torch.nn.functional as F
from torch import nn, Tensor

import torchdiffeq

from beatsODE2_1.layers import CGPODEBlock, dilated_inception

class BeatsODE2(nn.Module):
    def __init__(self,
                 in_dim=2,
                 out_dim=2,
                 seq_len=12,
                 time_1=1.0, step_size_1=0.25,
                 time_2=1.2, step_size_2=0.4,
                 rtol=1e-4, atol=1e-3,
                 share_weight_in_stack=True):
        super(BeatsODE2, self).__init__()
        print('BeatsODE2_1')
        if share_weight_in_stack:
            print('BeatsODE with share_stack_weight')
        
        self.time = time_1
        self.step_size = step_size_1
        self.stacks = nn.ModuleList()
        
        n_stacks = 3
        n_blocks = 3
        for stack_id in range(n_stacks):
            blocks = nn.ModuleList()
            for block_id in range(n_blocks):
                if share_weight_in_stack and block_id != 0:
                    block = blocks[-1]
                else:
                    block = BeatsBlock(in_dim, 
                                       out_dim, 
                                       seq_len, 
                                       time_1, step_size_1, 
                                       time_2, step_size_2, 
                                       rtol, atol)
                blocks.append(block)
            
            self.stacks.append(blocks)
    
    def forward(self, backcast: Tensor, adj_mx: Tensor):
        forecast = torch.zeros_like(backcast).type_as(backcast)
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast, adj_mx)
            
                forecast = forecast + f.transpose(1, 3)
                backcast = backcast - b.transpose(1, 3)
        
        backcast = backcast.transpose(1, 3)
        forecast = forecast.transpose(1, 3)
        
        return forecast

class BeatsBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 seq_len,
                 time_1, step_size_1,
                 time_2, step_size_2,
                 rtol, atol,
                 conv_dim=32,
                 end_dim=128):
        super(BeatsBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.seq_len = seq_len
        self.nfe = round(time_1 / step_size_1)
        max_kernel_size = 7
        self.receptive_field = self.nfe * (max_kernel_size - 1) + out_dim
        
        self.start_conv = nn.Conv2d(in_dim, conv_dim, kernel_size=(1, 1))
        
        st_block = STBlock(self.receptive_field, 1, conv_dim, time_2, step_size_2, rtol, atol)
        self.ODE = ODEBlock(st_block, time_1, step_size_1, rtol, atol)
        
        self.backcast_decoder = nn.Sequential(nn.Conv2d(conv_dim, end_dim, kernel_size=(1, 1)),
                                              nn.ReLU(),
                                              nn.Conv2d(end_dim, seq_len, kernel_size=(1, 1)))
        
        self.forecast_decoder = nn.Sequential(nn.Conv2d(conv_dim, end_dim, kernel_size=(1, 1)),
                                              nn.ReLU(),
                                              nn.Conv2d(end_dim, seq_len, kernel_size=(1, 1)))
    
    def forward(self, x: Tensor, adj: Tensor):
        if self.seq_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - self.seq_len, 0))
        
        x = self.start_conv(x)
        
        self.ODE.odefunc.set_adj(adj)
        x = self.ODE(x)
        self.ODE.odefunc.set_intermediate(1)
        
        x = x[..., -self.out_dim:]
        x = F.layer_norm(x, tuple(x.shape[1:]), weight=None, bias=None, eps=1e-5)
        
        # decoder
        backcast = self.backcast_decoder(x)
        forecast = self.forecast_decoder(x)
        
        return backcast, forecast

class ODEBlock(nn.Module):
    def __init__(self, odefunc, time_1, step_size_1, rtol, atol):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        
        self.time = time_1
        self.step_size = step_size_1
        
        self.rtol = rtol
        self.atol = atol
    
    def forward(self, x: Tensor):
        self.integration_time = torch.tensor([0, self.time]).float().type_as(x)
        out = torchdiffeq.odeint(self.odefunc, 
                                 x, 
                                 self.integration_time,
                                 method="euler",
                                 rtol=self.rtol, atol=self.atol,
                                 options=dict(step_size=self.step_size))
        return out[-1]

class STBlock(nn.Module):
    def __init__(self, 
                 receptive_field,
                 dilation,
                 hidden_dim,
                 time_2, step_size_2,
                 rtol, atol,
                 dropout=0.3):
        super(STBlock, self).__init__()
        self.dropout = dropout
        self.receptive_field = receptive_field
        self.intermediate_seq_len = receptive_field
        self.new_dilation = 1
        self.dilation_factor = dilation
        
        self.inception_1 = dilated_inception(hidden_dim, hidden_dim)
        self.inception_2 = dilated_inception(hidden_dim, hidden_dim)
        
        self.gconv1 = CGPODEBlock(hidden_dim, hidden_dim, time_2, step_size_2, rtol, atol)
        self.gconv2 = CGPODEBlock(hidden_dim, hidden_dim, time_2, step_size_2, rtol, atol)
        
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
        
        x = F.dropout(x, self.dropout)
        
        x = self.gconv1(x, self.adj) + self.gconv2(x, self.adj.T)
        
        x = F.pad(x, (self.receptive_field - x.size(3), 0))
        
        del _filter, _gate
        return x
