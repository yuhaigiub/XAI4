import torch
import torch.nn.functional as F
from torch import nn, Tensor

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
        self.alpha = alpha
        self.conv = nconv()
        self.out = []
    
    def forward(self, x: Tensor, adj: Tensor):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        _d = torch.diag(torch.pow(d, -0.5))
        adj_norm = torch.mm(torch.mm(_d, adj), _d)
        
        self.out.append(x)
        ax = self.conv(x, adj_norm)
        f = 0.5 * self.alpha * (ax - x)
        
        return f
