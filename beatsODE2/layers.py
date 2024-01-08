import torch
import torch.nn.functional as F
from torch import nn, Tensor

import torchdiffeq

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self, x: Tensor, A: Tensor):
        # x.shape = (batch, channels, nodes, time_steps)
        # A.shape = (node, node)
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(linear,self).__init__()
        self.mlp = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)
    
    def forward(self, x: Tensor):
        return self.mlp(x)

class dilated_inception(nn.Module):
    def __init__(self, in_dim, out_dim, dilation_factor=1):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernels = [2, 3, 6, 7]
        out_dim = int(out_dim / len(self.kernels))
        for kern in self.kernels:
            self.tconv.append(nn.Conv2d(in_dim, out_dim, (1, kern), dilation=(1, dilation_factor)))
    
    def forward(self, input: Tensor):
        x = []
        for i in range(len(self.kernels)):
            output = self.tconv[i](input)
            x.append(output)
        
        for i in range(len(self.kernels)):
            x[i] = x[i][..., -x[-1].size(3):]
        
        
        x = torch.cat(x, dim=1)
        return x

class CGPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1.0):
        super(CGPBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.conv = nconv()
    
    def forward(self, x: Tensor, adj: Tensor):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        _d = torch.diag(torch.pow(d, -0.5))
        adj_norm = torch.mm(torch.mm(_d, adj), _d)
        
        ax = self.conv(x, adj_norm)
        f = 0.5 * self.alpha * (ax - x)
        
        return f

class CGPODEBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time, step_size, alpha=1.0):
        super(CGPODEBlock, self).__init__()
        self.time = time
        self.step_size = step_size
        
        self.estimated_nfe = round(time / step_size)
        
        self.odefunc = CGPFunc(in_dim, out_dim, alpha)
        
        self.mlp = linear((self.estimated_nfe + 1) * in_dim, out_dim)
    
    def forward(self, x: Tensor, adj: Tensor):
        self.odefunc.set_x0(x)
        self.odefunc.set_adj(adj)
        
        self.integration_time = torch.tensor([0, self.time]).float().type_as(x)
        out = torchdiffeq.odeint(self.odefunc, 
                                 x, 
                                 self.integration_time,
                                 method="euler",
                                 options=dict(step_size=self.step_size))
        
        outs = self.odefunc.out
        self.odefunc.out = []
        outs.append(out[-1])
        
        h_out = torch.cat(outs, dim=1)
        h_out = self.mlp(h_out)
        
        return h_out

class CGPFunc(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=1.0):
        super(CGPFunc, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.conv = nconv()
        
        self.out = []
        
        self.x0 = None
        self.adj = None
    
    def set_x0(self, x0: Tensor):
        self.x0 = x0.clone().detach()
    
    def set_adj(self, adj: Tensor):
        self.adj = adj
    
    def forward(self, t, x: Tensor):
        adj = self.adj + torch.eye(self.adj.size(0)).to(x.device)
        d = adj.sum(1)
        _d = torch.diag(torch.pow(d, -0.5))
        adj_norm = torch.mm(torch.mm(_d, adj), _d)
        
        self.out.append(x)
        
        ax = self.conv(x, adj_norm)
        f = 0.5 * self.alpha * (ax - x)
        
        return f
