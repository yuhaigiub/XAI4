import torch
import torch.optim as optim

import util

class Engine2():
    def __init__(self, scaler, model, num_nodes, lrate, wdecay, device, adj_mx, seq_lens=[3, 6, 12]):
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        print('number of parameters:', sum(p.numel() for p in model.parameters()))
        print('number of trainable parameters:', sum(p.numel() 
                                                     for p in self.model.parameters() if p.requires_grad))
        self.seq_lens = seq_lens
        self.coeffs = [1., 1., 1.]
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        

        self.edge_index = [[], []]
        self.edge_weight = []

        # The adjacency matrix is converted into an edge_index list
        # in accordance with PyG API
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_mx.item((i, j)) != 0:
                    self.edge_index[0].append(i)
                    self.edge_index[1].append(j)
                    self.edge_weight.append(adj_mx.item((i, j)))

        self.adj_mx = torch.tensor(adj_mx).to(device)
        self.edge_index = torch.tensor(self.edge_index).to(device)
        self.edge_weight = torch.tensor(self.edge_weight).to(device)

    def train(self, input, real_val):
        '''
        input: [batch_size, channels, num_nodes, time_steps]
        '''
        self.model.train()
        self.optimizer.zero_grad()
        
        outs = self.model(input, self.adj_mx)
        real = torch.unsqueeze(real_val, dim=1)
        
        loss = 0
        mape = 0
        rmse = 0
        L, MAPE, RMSE = [], [], []
        for i, out in enumerate(outs):
            predict = self.scaler.inverse_transform(out[:, 0:1, : ,:].transpose(-3, -1))
            r = real[:, :, :, 0: self.seq_lens[i]]
            
            l_1 = self.loss(predict, r, 0.0)
            l_2 = util.masked_mape(predict, r, 0.0).item()
            l_3 = util.masked_rmse(predict, r, 0.0).item()
            loss = loss + self.coeffs[i] * l_1
            mape = mape + self.coeffs[i] * l_2
            rmse = rmse + self.coeffs[i] * l_3
            
            L.append(round(l_1.item(), 4))
            MAPE.append(round(l_2, 4))
            RMSE.append(round(l_3, 4))
            
        loss.backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        
        return loss.item(), mape, rmse, L, MAPE, RMSE

    def eval(self, input, real_val):
        '''
        input: [batch_size, channels, num_nodes, time_steps]
        '''
        self.model.eval()
        
        # output = self.model(input, self.edge_index, self.edge_weight)
        outs = self.model(input, self.adj_mx)
        real = torch.unsqueeze(real_val, dim=1)
        
        loss = 0
        mape = 0
        rmse = 0
        L, MAPE, RMSE = [], [], []
        for i, out in enumerate(outs):
            predict = self.scaler.inverse_transform(out[:, 0:1, : ,:].transpose(-3, -1))
            r = real[:, :, :, 0: self.seq_lens[i]]
            
            l_1 = self.loss(predict, r, 0.0)
            l_2 = util.masked_mape(predict, r, 0.0).item()
            l_3 = util.masked_rmse(predict, r, 0.0).item()
            loss = loss + l_1
            mape = mape + l_2
            rmse = rmse + l_3
            
            L.append(round(l_1.item(), 4))
            MAPE.append(round(l_2, 4))
            RMSE.append(round(l_3, 4))
        
        return loss.item(), mape, rmse, L, MAPE, RMSE