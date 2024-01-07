import torch
from torch import nn, Tensor
import torch.nn.functional as F

import torchdiffeq

from mtgode.layers import dilated_inception, CGP

class MTGODE(nn.Module):
    def __init__(self, 
                 device,
                 in_dim,
                 out_dim,
                 adj,
                 seq_len,
                 time_1=1.2, step_size_1=0.4,
                 time_2=1.0, step_size_2=0.25,
                 conv_dim=32,
                 end_dim = 128,
                 dilation_exponential=1):
        super(MTGODE, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.adj = torch.tensor(adj, dtype=torch.float32).to(device)
        
        self.start_conv = nn.Conv2d(in_dim, conv_dim, kernel_size=(1, 1))
        
        self.estimated_nfe = round(time_1 / (step_size_1))
        
        max_kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(1 + (max_kernel_size - 1) * (dilation_exponential**self.estimated_nfe - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = self.estimated_nfe * (max_kernel_size - 1) + 1
        
        # print('receptive_field', self.receptive_field)
        
        odefunc = STBlock(self.receptive_field, dilation_exponential, conv_dim, time_2, step_size_2)
        self.ODE = ODEBlock(odefunc, time_1, step_size_1)
        
        self.end_conv_0 = nn.Conv2d(conv_dim, end_dim // 2, kernel_size=(1, 1))
        self.end_conv_1 = nn.Conv2d(end_dim // 2, end_dim, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_dim, out_dim, kernel_size=(1, 1))
    
    def forward(self, x: Tensor):
        if self.seq_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - self.seq_len, 0))
        # print('F.pad:', x.shape)
        
        x = self.start_conv(x)
        # print('start_conv:', x.shape)
        
        self.ODE.odefunc.set_adj(self.adj)
        x = self.ODE(x)
        self.ODE.odefunc.set_intermediate(dilation=1)
        # print('self.ODE:', x.shape)
        
        x = x[..., -1:]
        x = F.layer_norm(x, tuple(x.shape[1:]), weight=None, bias=None, eps=1e-5)
        # print('F.layer_norm', x.shape)
        
        x = F.relu(self.end_conv_0(x))
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        
        return x

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
                 hidden_dim,
                 time,
                 step_size):
        super(STBlock, self).__init__()
        self.receptive_field = receptive_field
        self.intermediate_seq_len = receptive_field
        self.new_dilation = 1
        self.dilation_factor = dilation
        
        self.inception_1 = dilated_inception(hidden_dim, hidden_dim, dilation_factor=1)
        self.inception_2 = dilated_inception(hidden_dim, hidden_dim, dilation_factor=1)
        
        self.gconv1 = CGP(hidden_dim, hidden_dim, time, step_size)
        self.gconv2 = CGP(hidden_dim, hidden_dim, time, step_size)
        
        self.adj = None
    
    def set_adj(self, adj: Tensor):
        self.adj = adj
    
    def set_intermediate(self, dilation):
        self.new_dilation = dilation
        self.intermediate_seq_len = self.receptive_field
    
    def forward(self, t, x: Tensor):
        # print('start STBlock-----')

        x = x[..., -self.intermediate_seq_len:]
        # print('indexing', x.shape)
        
        for tconv in self.inception_1.tconv:
            tconv.dilation = (1, self.new_dilation)
        for tconv in self.inception_2.tconv:
            tconv.dilation = (1, self.new_dilation)
        
        _filter = self.inception_1(x)
        _filter = torch.tanh(_filter)
        
        _gate = self.inception_2(x)
        _gate = torch.sigmoid(_gate)
        
        x = _filter * _gate
        # print('rnn', x.shape)
        
        self.new_dilation *= self.dilation_factor
        self.intermediate_seq_len = x.size(3)
        
        # x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.gconv1(x, self.adj) + self.gconv2(x, self.adj.T)
        
        x = F.pad(x, (self.receptive_field - x.size(3), 0))
        # print('F.pad', x.shape)
        
        return x