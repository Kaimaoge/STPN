import torch
import util
import numpy as np
import argparse

from baseline_methods import test_error, StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='China',help='data type')
parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], help='hidden layer dimension',type=float)
parser.add_argument('--in_len',type=int,default=36,help='input time series length')
parser.add_argument('--out_len',type=int,default=12,help='output time series length')
parser.add_argument('--period',type=int,default=36,help='periodic for temporal embedding')
parser.add_argument('--period1',type=int,default=7,help='the input sequence is longer than one day, we use this periodicity to allocate a unique index to each time point')

args = parser.parse_args()

def main():
    device = torch.device(args.device)
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)
    supports = [torch.tensor(i).to(device) for i in adj]
    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(), training_data[~np.isnan(training_data)].std())
    test_index = list(range(test_data.shape[1] - (args.in_len + args.out_len)))
    label = []
    for i in range(len(test_index)):
        label.append(np.expand_dims(test_data[:, test_index[i] + args.in_len:test_index[i] + args.in_len + args.out_len, :], axis = 0))
    label = np.concatenate(label)
    model = torch.load("spdpn" + args.data +".pth")
    
    outputs = []
    model.eval()
    for i in range(len(test_index)):
        testx = np.expand_dims(test_data[:, test_index[i]: test_index[i] + args.in_len, :], axis = 0)
        testx = scaler.transform(testx)
        testw = np.expand_dims(test_w[:, test_index[i]: test_index[i] + args.in_len], axis = 0)
        testw = torch.LongTensor(testw).to(device)
        testx[np.isnan(testx)] = 0
        ti_add = (np.arange(int(training_data.shape[1] + val_data.shape[1])+test_index[i], int(training_data.shape[1] + val_data.shape[1])+test_index[i]+ args.in_len)// args.period % args.period1)/args.period1
        to_add = (np.arange(int(training_data.shape[1] + val_data.shape[1])+test_index[i] + args.in_len, int(training_data.shape[1] + val_data.shape[1])+test_index[i] + args.in_len + args.out_len)// args.period % args.period1)/args.period1
        testti = (np.arange(int(training_data.shape[1] + val_data.shape[1])+test_index[i], int(training_data.shape[1]+ val_data.shape[1])+test_index[i]+ args.in_len) % args.period) * np.ones([1, args.in_len])/(args.period - 1) + ti_add
        testto = (np.arange(int(training_data.shape[1] + val_data.shape[1])+test_index[i] + args.in_len, int(training_data.shape[1]+ val_data.shape[1])+test_index[i] + args.in_len + args.out_len) % args.period) * np.ones([1, args.out_len])/(args.period - 1) + to_add
        testx = torch.Tensor(testx).to(device)
        testx= testx.permute(0, 3, 1, 2)
        testti = torch.Tensor(testti).to(device)
        testto = torch.Tensor(testto).to(device)
        output = model(testx, testti, supports, testto, testw)
        output = output.permute(0, 2, 3, 1)
        output = output.detach().cpu().numpy()
        output = scaler.inverse_transform(output)
        outputs.append(output)
    yhat = np.concatenate(outputs)

    log = '3 step ahead arrival delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:,:,2,0],label[:,:,2,0])
    print(log.format(MAE, R2, RMSE))
    
    log = '6 step ahead arrival delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:,:,5,0],label[:,:,5,0])
    print(log.format(MAE, R2, RMSE))
    
    log = '12 step ahead arrival delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:,:,11,0],label[:,:,11,0])
    print(log.format(MAE, R2, RMSE))
    
    log = '3 step ahead departure delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:,:,2,1],label[:,:,2,1])
    print(log.format(MAE, R2, RMSE))
    
    log = '6 step ahead departure delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:,:,5,1],label[:,:,5,1])
    print(log.format(MAE, R2, RMSE))
    
    log = '12 step ahead departure delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:,:,11,1],label[:,:,11,1])
    print(log.format(MAE, R2, RMSE))
    
if __name__ == "__main__":   
    main()  
