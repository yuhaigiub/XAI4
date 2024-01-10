import argparse
import os
import time

import numpy as np
import torch
import util
# from graphwavenet.model import GraphWaveNet
# from beatsODE.model import BeatsODE
# from mtgode.model import MTGODE
# from beatsODE2.model import BeatsODE2
from beatsODE2_1.model import BeatsODE2
# from beatsODE2_2.model import BeatsODE2
# from beatsODE3.model import BeatsODE3

from engine import Engine
# from engine2 import Engine2

parser = argparse.ArgumentParser()

"""
"""

parser.add_argument('--device', type=str, default='cuda', help='device to run the model on')
parser.add_argument('--data', type=str, default='store/METR-LA', help='data path')
parser.add_argument('--adjdata', type=str, default='store/adj_mx.pkl', help='adj data path')

parser.add_argument('--epochs', type=int, default=5, help='')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')

parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='saved_models/nbeatsODE', help='save path')

parser.add_argument('--context_window', type=int, default=12, help='sequence length')
parser.add_argument('--target_window', type=int, default=12, help='predict length')
parser.add_argument('--patch_len', type=int, default=1, help='patch length')
parser.add_argument('--stride', type=int, default=1, help='stride')
parser.add_argument('--blackbox_file', type=str, default='save_blackbox/G_T_model_1.pth', help='blackbox .pth file')
parser.add_argument('--iter_epoch', type=str, default=1, help='using for save pth file')

parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--timestep', type=str, default=12, help='time step')
parser.add_argument('--input_dim', type=str, default=2, help='channels')
parser.add_argument('--output_dim', type=str, default=2, help='channels')
parser.add_argument('--hidden', type=str, default=64, help='hidden layers')
parser.add_argument('--num_layer', type=str, default=4, help='number layers')

args = parser.parse_args()

def main():
    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adjdata)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    
    # Mean / std dev scaling is performed to the model output
    scaler = dataloader['scaler']
    
    # model = GraphWaveNet(args.num_nodes, args.input_dim, args.output_dim, args.timestep)
    # model = BeatsODE(device, args.input_dim, args.output_dim, args.timestep)
    # model = MTGODE(device, args.input_dim, args.timestep, adj_mx, args.timestep)
    model = BeatsODE2(in_dim=2, out_dim=2, seq_len=12)
    # model = BeatsODE3(in_dim=2, out_dim=2, seq_len=12)
    
    engine = Engine(scaler,
                    model,
                    args.num_nodes, 
                    args.learning_rate,
                    args.weight_decay, 
                    device, 
                    adj_mx)
    
    # engine = Engine2(model, scaler, args.learning_rate, args.weight_decay, device)
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    log_file_train = open('loss_train_log.txt', 'w')
    log_file_val = open('loss_val_log.txt', 'w')
    log_file_test = open('loss_test_log.txt', 'w')

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    best_epoch = 0

    for i in range(args.iter_epoch, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device) 
            trainy = torch.Tensor(y).to(device)
            
            trainx = trainx.transpose(1, 3)
            trainy = trainy.transpose(1, 3)
            
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: ' + '{:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
                
        t2 = time.time()
        
        train_time.append(t2 - t1)
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).to(device)
            testx = testx.transpose(1, 3)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        
        test_loss = []
        test_mape = []
        test_rmse = []

        
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx_ = torch.Tensor(x).to(device)
            testy_ = torch.Tensor(y).to(device)
            testx_ = testx_.transpose(1, 3)
            testy_ = testy_.transpose(1, 3)
            
            metrics = engine.eval(testx_, testy_[:, 0, :, :])
            test_loss.append(metrics[0])
            test_mape.append(metrics[1])
            test_rmse.append(metrics[2])
            
            
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        
        mtest_loss = np.mean(test_loss)
        mtest_mape = np.mean(test_mape)
        mtest_rmse = np.mean(test_rmse)
        
        his_loss.append(mvalid_loss)
        
        torch.save(engine.model.state_dict(), args.save + "/G_T_model_" + str(i) + ".pth")
        if np.argmin(his_loss) == len(his_loss) - 1:
            best_epoch = i

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, ' + \
              'Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, ' + \
              'Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'

        print(log.format(i, 
                         mtrain_loss, 
                         mtrain_mape, 
                         mtrain_rmse, 
                         mvalid_loss,
                         mvalid_mape, 
                         mvalid_rmse, 
                         mtest_loss,
                         mtest_mape,
                         mtest_rmse,
                         (t2 - t1)))
        
        log_file_train.write(f'Epoch {i}, Training Loss: {mtrain_loss:.4f}, Training MAPE: {mtrain_mape:.4f}, Training RMSE: {mtrain_rmse:.4f} \n')
        log_file_train.flush()
        log_file_val.write(f'Epoch {i}, Val Loss: {mvalid_loss:.4f}, Val MAPE: {mvalid_mape:.4f}, Val RMSE: {mvalid_rmse:.4f} \n')
        log_file_val.flush()
        log_file_test.write(f'Epoch {i}, Test Loss: {mtest_loss:.4f}, Test MAPE: {mtest_mape:.4f}, Test RMSE: {mtest_rmse:.4f} \n')
        log_file_test.flush()
       
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    engine.model.load_state_dict(torch.load(args.save + "/G_T_model_" + str(best_epoch) + ".pth"))
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        with torch.no_grad():
            back, preds = engine.model(testx, adj_mx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred= scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE:' + '{:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: ' + '{:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))