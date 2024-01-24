import torch
import torch.nn.functional as F
from torch import nn, Tensor

import torchdiffeq

from beatsODE3_2.layers import CGPODEBlock, dilated_inception

class BeatsODE3(nn.Module):
    def __init__(self,
                 in_dim=2,
                 out_dim=2,
                 input_seq_len=12,
                 seq_lens=[3, 6, 12],
                 time_0=1.2, step_size_0=0.4,
                 time_1=1.0, step_size_1=0.25,
                 time_2=1.2, step_size_2=0.4,
                 rtol=1e-4, atol=1e-3, perturb=False,
                 share_weight_in_stack=False):
        super(BeatsODE3, self).__init__()
        print('BeatsODE3_2')
        if share_weight_in_stack:
            print('BeatsODE with share_stack_weight')
        
        self.time = time_0
        self.step_size = step_size_0
        self.seq_lens = seq_lens
        
        self.rtol = rtol
        self.atol = atol
        self.perturb = perturb
        
        n_stacks = 3
        self.stacks = nn.ModuleList()
        
        for stack_id in range(n_stacks):
            self.stacks.append(BeatsODEBlock(in_dim, out_dim, 
                                             input_seq_len,
                                             seq_lens[stack_id], 
                                             time_1, step_size_1, 
                                             time_2, step_size_2,
                                             rtol, atol, perturb))
    
    def forward(self, backcast: Tensor, adj: Tensor):
        self.integration_time = torch.tensor([-self.time, 0]).float().type_as(backcast)
        
        batch_size, channels, num_nodes, backcast_seq_len = backcast.size()
        outs = []
        for stack_id in range(len(self.stacks)):
            forecast = torch.zeros(batch_size, channels, num_nodes, self.seq_lens[stack_id]).type_as(backcast)
            stack: BeatsODEBlock = self.stacks[stack_id]
            
            stack.set_adj(adj)
            stack.forecast = forecast
            
            backcast = torchdiffeq.odeint(stack,
                                          backcast,
                                          self.integration_time,
                                          method="euler",
                                          rtol=self.rtol, atol=self.atol,
                                          options=dict(step_size=self.step_size, perturb=self.perturb))[-1]
            outs.append(stack.forecast.transpose(1, 3))
            stack.forecast = None
            
        backcast = backcast.transpose(1, 3)
        forecast = forecast.transpose(1, 3)
        
        return outs

class BeatsODEBlock(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 input_seq_len,
                 output_seq_len,
                 time_1, step_size_1,
                 time_2, step_size_2,
                 rtol, atol, perturb,
                 conv_dim=32,
                 end_dim=128):
        super(BeatsODEBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.seq_len = input_seq_len
        self.nfe = round(time_1 / step_size_1)
        max_kernel_size = 7
        self.receptive_field = self.nfe * (max_kernel_size - 1) + out_dim
        
        self.start_conv = nn.Conv2d(in_dim, conv_dim, kernel_size=(1, 1))
        st_block = STBlock(self.receptive_field, 1, conv_dim, time_2, step_size_2, rtol, atol, perturb)
        self.ODE = ODEBlock(st_block, time_1, step_size_1, rtol, atol, perturb)
        
        self.backcast_decoder = nn.Sequential(nn.Conv2d(conv_dim, end_dim, kernel_size=(1, 1)),
                                              nn.ReLU(),
                                              nn.Conv2d(end_dim, input_seq_len, kernel_size=(1, 1)))
        
        self.forecast_decoder = nn.Sequential(nn.Conv2d(conv_dim, end_dim, kernel_size=(1, 1)),
                                              nn.ReLU(),
                                              nn.Conv2d(end_dim, output_seq_len, kernel_size=(1, 1)))
        
        self.forecast = None
        self.adj = None
    
    def set_adj(self, adj: Tensor):
        self.adj = adj
    
    def forward(self, t, x: Tensor):
        if self.seq_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - self.seq_len, 0))
        
        x = self.start_conv(x)
        
        self.ODE.odefunc.set_adj(self.adj)
        x = self.ODE(x)
        self.ODE.odefunc.set_intermediate(1)
        
        x = x[..., -self.out_dim:]
        x = F.layer_norm(x, tuple(x.shape[1:]), weight=None, bias=None, eps=1e-5)
        
        # x = F.dropout(x, 0.3)
        
        # decoder
        backcast = self.backcast_decoder(x).transpose(1, 3)
        forecast = self.forecast_decoder(x).transpose(1, 3)
        
        self.forecast = self.forecast + forecast
        
        return backcast

class ODEBlock(nn.Module):
    def __init__(self, odefunc, time_1, step_size_1, rtol, atol, perturb):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        
        self.time = time_1
        self.step_size = step_size_1
        
        self.rtol = rtol
        self.atol = atol
        self.perturb = perturb
    
    def forward(self, x: Tensor):
        self.integration_time = torch.tensor([0, self.time]).float().type_as(x)
        out = torchdiffeq.odeint(self.odefunc, 
                                 x, 
                                 self.integration_time,
                                 method="euler",
                                 rtol=self.rtol, atol=self.atol,
                                 options=dict(step_size=self.step_size, perturb=self.perturb))
        return out[-1]

class STBlock(nn.Module):
    def __init__(self, 
                 receptive_field,
                 dilation,
                 hidden_dim,
                 time_2, step_size_2,
                 rtol, atol, perturb,
                 dropout=0.3):
        super(STBlock, self).__init__()
        self.dropout = dropout
        self.receptive_field = receptive_field
        self.intermediate_seq_len = receptive_field
        self.new_dilation = 1
        self.dilation_factor = dilation
        
        self.inception_1 = dilated_inception(hidden_dim, hidden_dim)
        self.inception_2 = dilated_inception(hidden_dim, hidden_dim)
        
        self.gconv1 = CGPODEBlock(hidden_dim, hidden_dim, time_2, step_size_2, rtol, atol, perturb)
        self.gconv2 = CGPODEBlock(hidden_dim, hidden_dim, time_2, step_size_2, rtol, atol, perturb)
        
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
