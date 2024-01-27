import argparse
import os
import time

import numpy as np
import torch
import util

from beatsODE3_2.model import BeatsODE3
# from beatsODE3_3.model import BeatsODE3
# from beatsODE3_4.model import BeatsODE3

from engine2 import Engine2

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
parser.add_argument('--iter_epoch', type=str, default=-1, help='using for save pth file')

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
    
    seq_lens = [3, 6, 12]
    model = BeatsODE3(in_dim=2, out_dim=2, input_seq_len=12, seq_lens=seq_lens)
    
    engine = Engine2(scaler,
                     model,
                     args.num_nodes, 
                     args.learning_rate,
                     args.weight_decay, 
                     device, 
                     adj_mx)
    adj_mx = torch.tensor(adj_mx).to(device)
    # load checkpoints
    if args.iter_epoch != -1:
        print('loading epoch {}'.format(args.iter_epoch))
        model.load_state_dict(torch.load(args.save + '/G_T_model_{}.pth'.format(args.iter_epoch)))
    
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

    for i in range(args.iter_epoch + 1, args.iter_epoch + 1 + args.epochs):
        print('training epoch {} ***'.format(i))
        train_loss_list = []
        train_mape_list = []
        train_rmse_list = [] 
        
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device) 
            trainy = torch.Tensor(y).to(device)
            
            trainx = trainx.transpose(1, 3)
            trainy = trainy.transpose(1, 3)
            
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            
            train_loss_list.append(metrics[0])
            train_mape_list.append(metrics[1])
            train_rmse_list.append(metrics[2])
            
            if iter % args.print_every == 0:
                print('Epoch: {}; Iter: {}'.format(i, iter))
                print('- MAE:  {}'.format(train_loss_list[-1]))
                print('- MAPE: {}'.format(train_mape_list[-1]))
                print('- RMSE: {}'.format(train_rmse_list[-1]))
                
                
                
            # if iter == 50:
            #     break
        
        t2 = time.time()
        train_time.append(t2 - t1)
        
        mtrain_loss = np.mean(train_loss_list, axis=0).round(4)
        mtrain_mape = np.mean(train_mape_list, axis=0).round(4)
        mtrain_rmse = np.mean(train_rmse_list, axis=0).round(4)
        
        del train_loss_list, train_mape_list, train_rmse_list
        
        # val
        
        valid_loss_list = []
        valid_mape_list = []
        valid_rmse_list = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testy = torch.Tensor(y).to(device)
            
            testx = testx.transpose(1, 3)
            testy = testy.transpose(1, 3)
            
            metrics = engine.eval(testx, testy[:, 0, :, :])
            
            valid_loss_list.append(metrics[0])
            valid_mape_list.append(metrics[1])
            valid_rmse_list.append(metrics[2])
            
            # if iter == 50:
            #     break

        
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        
        mvalid_loss = np.mean(valid_loss_list, axis=0).round(4)
        mvalid_mape = np.mean(valid_mape_list, axis=0).round(4)
        mvalid_rmse = np.mean(valid_rmse_list, axis=0).round(4)
        
        del valid_loss_list, valid_mape_list, valid_rmse_list
        
        # test
        
        test_loss_list = []
        test_mape_list = []
        test_rmse_list = []

        
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx_ = torch.Tensor(x).to(device)
            testy_ = torch.Tensor(y).to(device)
            
            testx_ = testx_.transpose(1, 3)
            testy_ = testy_.transpose(1, 3)
            
            metrics = engine.eval(testx_, testy_[:, 0, :, :])
            test_loss_list.append(metrics[0])
            test_mape_list.append(metrics[1])
            test_rmse_list.append(metrics[2])
            
            # if iter == 50:
            #     break
  
        
        mtest_loss = np.mean(test_loss_list, axis=0).round(4)
        mtest_mape = np.mean(test_mape_list, axis=0).round(4)
        mtest_rmse = np.mean(test_rmse_list, axis=0).round(4)
        
        del test_loss_list, test_mape_list, test_rmse_list
        
        his_loss.append(mvalid_loss[-1]) # long-term loss
        
        torch.save(engine.model.state_dict(), args.save + "/G_T_model_" + str(i) + ".pth")
        if np.argmin(his_loss) == len(his_loss) - 1:
            best_epoch = i

        # log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, ' + \
        #       'Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, ' + \
        #       'Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'

        mtrain_loss = list(mtrain_loss)[-1]
        mtrain_mape = list(mtrain_mape)[-1]
        mtrain_rmse = list(mtrain_rmse)[-1]
        mvalid_loss = list(mvalid_loss)[-1]
        mvalid_mape = list(mvalid_mape)[-1]
        mvalid_rmse = list(mvalid_rmse)[-1]
        mtest_loss = list(mtest_loss)[-1]
        mtest_mape = list(mtest_mape)[-1]
        mtest_rmse = list(mtest_rmse)[-1]

        # print(log.format(i, 
        #                  mtrain_loss, 
        #                  mtrain_mape, 
        #                  mtrain_rmse, 
        #                  mvalid_loss,
        #                  mvalid_mape, 
        #                  mvalid_rmse, 
        #                  mtest_loss,
        #                  mtest_mape,
        #                  mtest_rmse,
        #                  (t2 - t1)))
        print(f'Epoch: {i}, Train Loss: {mtrain_loss:.4f}, Train MAPE: {mtrain_mape:.4f}, Train RMSE: {mtrain_rmse:.4f}, Val Loss: {mvalid_loss:.4f}, Val MAPE: {mvalid_mape:.4f}, Val RMSE: {mvalid_rmse:.4f},Test Loss: {mtest_loss:.4f}, Test MAPE: {mtest_mape:.4f}, Test RMSE: {mtest_rmse:.4f} \n')
        #print(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss,mvalid_mape, mvalid_rmse, mtest_loss, mtest_mape, mtest_rmse)
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
    
    # Temp: test on long-term sequence only
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        #testx = torch.Tensor(x).to(device)
        testx = torch.FloatTensor(x).transpose(1, 3).to(device)
        with torch.no_grad():
            outs = engine.model(testx, adj_mx)
            pred = outs[-1].transpose(1, 3)[:, 0:1, :, :]
        outputs.append(pred.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
 

    print("Training finished")

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        print(pred.shape, real.shape)
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE:' + '{:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: ' + '{:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae, axis=0), np.mean(amape, axis=0), np.mean(armse, axis=0)))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))