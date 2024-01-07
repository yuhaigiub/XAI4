import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import init
import torch.nn.functional as F

from torchdiffeq import odeint

from beatsODE.layers import GraphConv

class Block(nn.Module):
    def __init__(self, 
                 device, 
                 in_dim,
                 out_dim,
                 time_steps, 
                 theta_dim):
        super(Block, self).__init__()
        self.device = device
        
        self.fc = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1))
        self.block_ode = BlockODE(out_dim)
        
        self.backcast_linspace = linear_space(time_steps)
        self.forecast_linspace = linear_space(time_steps)    
        
        self.theta_f_fc = nn.Conv2d(time_steps, theta_dim, kernel_size=(1, 1))
        self.theta_b_fc = nn.Conv2d(time_steps, theta_dim, kernel_size=(1, 1))
        
        self.adj = None
        self.forecast = None
        self.backcast = None
    
    def set_backcast(self, backcast: Tensor):
        if backcast is not None:
            self.backcast = backcast.clone().detach()
        else:
            self.backcast = None
    
    def initialize_forecast(self, forecast: Tensor):
        self.forecast = forecast
    
    def set_adj(self, adj: Tensor):
        self.block_ode.set_adj(adj)
    
    def forward(self, x: Tensor):
        time = torch.tensor([0, 1], dtype=torch.float32).to(self.device)
        if self.backcast is not None:
            x = x - self.backcast
        
        x = self.fc(x)
        x = odeint(self.block_ode, x, time, method='euler', options=dict(step_size=0.3))[-1]
        
        return x

class BlockODE(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super(BlockODE, self).__init__()
        self.dropout = dropout
        self.gcn1 = GraphConv(hidden_dim, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, hidden_dim)
        
        initialize_gcn_parameters(self)
        
        self.adj = None
    
    def set_adj(self, adj: Tensor):
        self.adj = adj
    
    def forward(self, t, x: Tensor):
        # [batch_size, time_steps, num_nodes, channels]
        x = x.transpose(-1, -3)
        x = F.dropout(x, self.dropout)
        
        x = F.relu(self.gcn1(x, self.adj))
        x = F.relu(self.gcn2(x, self.adj))
        
        # [batch_size, channels, num_nodes, time_steps]
        x = x.transpose(-3, -1)
        return x

class TrendBlock(Block):
    def __init__(self, device, in_dim, out_dim, time_steps, theta_dim, hidden_dim=32):
        super(TrendBlock, self).__init__(device, in_dim, hidden_dim, time_steps, theta_dim)
        self.conv = nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, 1))
        
        initialize_gcn_parameters(self)
    
    def forward(self, t, x: Tensor):
        x = super(TrendBlock, self).forward(x)
        
        x = self.conv(x)
        
        x = x.transpose(1, 3)
        theta_b = self.theta_b_fc(x).transpose(-3, -1)
        theta_f = self.theta_f_fc(x).transpose(-3, -1)
        
        backcast = trend_model(theta_b, self.backcast_linspace, self.device)
        forecast = trend_model(theta_f, self.forecast_linspace, self.device)
        
        self.forecast = self.forecast + forecast
        self.set_backcast(backcast)
        
        return backcast

class SeasonalityBlock(Block):
    def __init__(self, device, in_dim, out_dim, time_steps, theta_dim, hidden_dim=32):
        super(SeasonalityBlock, self).__init__(device, in_dim, hidden_dim, time_steps, theta_dim)
        self.conv = nn.Conv2d(hidden_dim, out_dim, kernel_size=(1, 1))
        
        initialize_gcn_parameters(self)
    
    def forward(self, t, x: Tensor):
        x = super(SeasonalityBlock, self).forward(x)
        
        x = self.conv(x)
        
        x = x.transpose(1, 3)
        theta_b = self.theta_b_fc(x).transpose(-3, -1)
        theta_f = self.theta_f_fc(x).transpose(-3, -1)
        
        backcast = seasonality_model(theta_b, self.backcast_linspace, self.device)
        forecast = seasonality_model(theta_f, self.backcast_linspace, self.device)
        
        self.forecast = self.forecast + forecast
        self.set_backcast(backcast)
        
        return backcast

class GenericBlock(Block):
    def __init__(self, device, in_dim, out_dim, time_steps, theta_dim, hidden_dim=32):
        super(GenericBlock, self).__init__(device, in_dim, hidden_dim, time_steps, theta_dim)
        self.backcast_out = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=(1, 1)),
                                          nn.ReLU(),
                                          nn.Conv2d(hidden_dim // 2, out_dim, kernel_size=(1, 1)),
                                          nn.ReLU())
        
        self.forecast_out = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=(1, 1)),
                                          nn.ReLU(),
                                          nn.Conv2d(hidden_dim // 2, out_dim, kernel_size=(1, 1)),
                                          nn.ReLU())
        
        initialize_gcn_parameters(self)
    
    def forward(self, t, x: Tensor):
        x = super(GenericBlock, self).forward(x)
        
        x = F.layer_norm(x, tuple(x.shape[1:]), weight=None, bias=None, eps=1e-5)
        
        backcast = self.backcast_out(x)
        forecast = self.forecast_out(x)
        
        self.forecast = self.forecast + forecast
        self.set_backcast(backcast)
        
        return backcast

class ODEStack(nn.Module):
    def __init__(self, device, in_dim, out_dim, time_steps, theta_dim, stack_type):
        super(ODEStack, self).__init__()
        self.device = device
        
        if stack_type == "GENERIC":
            self.block_ode = GenericBlock(device, in_dim, out_dim, time_steps, theta_dim)
        elif stack_type == "TREND":
            self.block_ode = TrendBlock(device, in_dim, out_dim, time_steps, theta_dim)
        elif stack_type == "SEASONALITY":
            self.block_ode = SeasonalityBlock(device, in_dim, out_dim, time_steps, theta_dim)
        else:
            raise Exception("Invalid Block Type")
    
    def set_adj(self, adj: Tensor):
        self.block_ode.set_adj(adj)
    
    def initialize_forecast(self, forecast: Tensor):
        self.block_ode.initialize_forecast(forecast)
    
    def clear_backcast(self):
        self.block_ode.set_backcast(None)
    
    def forward(self, x: Tensor):
        time = torch.tensor([0, 1], dtype=torch.float32).to(self.device)
        
        backcast = odeint(self.block_ode, x, time, method="euler", options=dict(step_size=0.3))[-1]
        forecast = self.block_ode.forecast
        
        return backcast, forecast

class BeatsODE(nn.Module):
    def __init__(self, 
                 device, 
                 in_dim, out_dim, 
                 time_steps,
                 n_stacks=3,
                 theta_dims=[4, 8, 4], 
                 stack_types=['TREND', 'SEASONALITY', 'GENERIC']):
        super(BeatsODE, self).__init__()
        self.device = device
        
        if n_stacks != len(theta_dims) or n_stacks != len(stack_types):
            raise Exception("check theta_dims or stack_types")
        
        self.stacks = nn.ModuleList()
        for i in range(n_stacks):
            self.stacks.append(ODEStack(device, in_dim, out_dim, time_steps, theta_dims[i], stack_types[i]))
    
    def forward(self, backcast: Tensor, adj: Tensor):
        forecast = torch.zeros_like(backcast, requires_grad=True).to(self.device)
        
        for stack in self.stacks:
            stack.set_adj(adj)
            stack.initialize_forecast(forecast)
            backcast, forecast = stack(backcast)
            
            stack.clear_backcast()
        
        # [batch_size, time_steps, num_nodes, channels]
        backcast = backcast.transpose(1, 3)
        forecast = forecast.transpose(1, 3)
        
        return backcast, forecast

def trend_model(thetas: Tensor, t: Tensor, device):
    p = thetas.size()[-1] # theta_dim
    assert p <= 4, 'theta_dim is too big'
    
    T = torch.tensor(np.array([t ** i for i in range(p)])).float()
    
    return torch.einsum('abcd, dy -> abcy', thetas, T.to(device))

def seasonality_model(thetas: Tensor, t: Tensor, device):
    p = thetas.size()[-1] # theta_dim
    
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    
    s1 = torch.Tensor(np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])).float() # H/2 - 1
    s2 = torch.Tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
    S = torch.cat([s1, s2])
    
    return torch.einsum('abcd, dy -> abcy',thetas, S.to(device))

def linear_space(time_steps):
    t = np.arange(0, time_steps) / time_steps
    return t

def initialize_gcn_parameters(model):
    for m in model.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=init.calculate_gain('relu'))
                
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)