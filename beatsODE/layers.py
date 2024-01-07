import torch
from torch import nn, Tensor
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout=0.0,
                 bias=True,
                 normalize_embedding=True):
        super(GraphConv, self).__init__()
        self.dropout = dropout
        
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        
        self.normalize_embedding = normalize_embedding
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.bias = None
    
    def forward(self, x: Tensor, adj: Tensor):
        '''
        x: [batch_size, time_steps, num_nodes, channels]
        adj: [num_nodes, num_nodes]
        '''
        batch_size, time_steps, num_nodes = x.size()[:-1]
        x = x.reshape(batch_size * time_steps, *x.size()[-2:])
        
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        
        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        
        if self.bias is not None:
            y = y + self.bias
        
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=1)
        
        y = y.reshape(batch_size, time_steps, num_nodes, self.out_dim)
        
        return y

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1))
    
    def forward(self, x: Tensor):
        x = x.transpose(-1, -3)
        x = self.conv(x)
        x = x.transpose(-1, -3)
        
        return x

