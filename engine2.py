import torch
from torch import nn
import torch.optim as optim

import util

class Engine2():
    def __init__(self,
                 model,  
                 scaler, 
                 lrate, 
                 wdecay, 
                 device):
        self.model = model
        self.model.to(device)
        
        _, _, self.adj_mx = util.load_adj('store/adj_mx.pkl')
        self.adj_mx = torch.tensor(self.adj_mx, dtype=torch.float32).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        
        # # [batch_size, time_step, num_nodes, channels]
        # input = input.transpose(-3, -1)

        _, forecast = self.model(input, self.adj_mx)
        forecast = forecast.transpose(-3, -1) # [16, 2, 207, 12]
        real = torch.unsqueeze(real_val, dim=1) # [16, 1, 207, 12]
        predict = self.scaler.inverse_transform(forecast)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        
        # input = input.transpose(-3, -1)
        _, forecast = self.model(input, self.adj_mx)
        
        forecast = forecast.transpose(-3, -1)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(forecast)
        
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        
        return loss.item(), mape, rmse