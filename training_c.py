# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:25:29 2022

@author: AA
"""

import torch
import util
import argparse
import baseline_methods
import random
import copy
import torch.optim as optim
import numpy as np

from baseline_methods import test_error
from model import STPN

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='China',help='data type')
parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], help='hidden layer dimension', type=float)
parser.add_argument('--h_layers',type=int,default=2,help='number of hidden layer')
parser.add_argument('--in_channels',type=int,default=2,help='input variable')
parser.add_argument("--hidden_channels", nargs="+", default=[128, 64, 32], help='hidden layer dimension', type=int)
parser.add_argument('--out_channels',type=int,default=2,help='output variable')
parser.add_argument('--emb_size',type=int,default=16,help='time embedding size')
parser.add_argument('--dropout',type=float,default=0,help='dropout rate')
parser.add_argument('--wemb_size',type=int,default=4,help='covairate embedding size')
parser.add_argument('--time_d',type=int,default=4,help='normalizing factor for self-attention model')
parser.add_argument('--heads',type=int,default=4,help='number of attention heads')
parser.add_argument('--support_len',type=int,default=3,help='number of spatial adjacency matrix')
parser.add_argument('--order',type=int,default=2,help='order of diffusion convolution')
parser.add_argument('--num_weather',type=int,default=7,help='number of weather condition')
parser.add_argument('--use_se', type=str, default=True,help="use SE block")
parser.add_argument('--use_cov', type=str, default=True,help="use Covariate")
parser.add_argument('--decay', type=float, default=1e-5, help='decay rate of learning rate ')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate ')
parser.add_argument('--in_len',type=int,default=36,help='input time series length')      # a relatively long sequence can handle missing data
parser.add_argument('--out_len',type=int,default=12,help='output time series length')
parser.add_argument('--batch',type=int,default=32,help='training batch size')
parser.add_argument('--episode',type=int,default=50,help='training episodes')
parser.add_argument('--period',type=int,default=36,help='periodic for temporal embedding')
parser.add_argument('--period1',type=int,default=7,help='the input sequence is longer than one day, we use this periodicity to allocate a unique index to each time point')

args = parser.parse_args()
def main():
    device = torch.device(args.device)
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)
    model = STPN(args.h_layers, args.in_channels, args.hidden_channels, args.out_channels, args.emb_size, 
                 args.dropout, args.wemb_size, args.time_d, args.heads, args.support_len,
                 args.order, args.num_weather, args.use_se, args.use_cov).to(device)
    supports = [torch.tensor(i).to(device) for i in adj]
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scaler = baseline_methods.StandardScaler(training_data[~np.isnan(training_data)].mean(), training_data[~np.isnan(training_data)].std())
    training_data = scaler.transform(training_data)
    training_data[np.isnan(training_data)] = 0
    
    MAE_list = []
    batch_index = list(range(training_data.shape[1] - (args.in_len + args.out_len)))
    val_index = list(range(val_data.shape[1] - (args.in_len + args.out_len)))
    label = []
    for i in range(len(val_index)):
        label.append(np.expand_dims(val_data[:, val_index[i] + args.in_len:val_index[i] + args.in_len + args.out_len, :], axis = 0))
    label = np.concatenate(label)
    
    print("start training...",flush=True)
    
    for ep in range(1,1+args.episode):
        random.shuffle(batch_index)
        for j in range(len(batch_index) // args.batch - 1):
            trainx = []
            trainy = []
            trainti = []
            trainto = []
            trainw = []
            for k in range(args.batch):
                trainx.append(np.expand_dims(training_data[:, batch_index[j * args.batch +k]: batch_index[j * args.batch +k] + args.in_len, :], axis = 0))
                trainy.append(np.expand_dims(training_data[:, batch_index[j * args.batch +k] + args.in_len:batch_index[j * args.batch +k] + args.in_len + args.out_len, :], axis = 0))
                trainw.append(np.expand_dims(training_w[:, batch_index[j * args.batch +k]: batch_index[j * args.batch +k] + args.in_len], axis = 0))
                ti_add = (np.arange(batch_index[j * args.batch +k], batch_index[j * args.batch +k] + args.in_len) // args.period % args.period1)/args.period1
                trainti.append((np.arange(batch_index[j * args.batch +k], batch_index[j * args.batch +k] + args.in_len) % args.period) * np.ones([1, args.in_len])/(args.period - 1) + ti_add)
                to_add = (np.arange(batch_index[j * args.batch +k] + args.in_len, batch_index[j * args.batch +k] + args.in_len + args.out_len)// args.period % args.period1)/args.period1
                trainto.append((np.arange(batch_index[j * args.batch +k] + args.in_len, batch_index[j * args.batch +k] + args.in_len + args.out_len) % args.period) * np.ones([1, args.out_len])/(args.period - 1) + to_add)
            trainx = np.concatenate(trainx)
            trainti = np.concatenate(trainti)
            trainto = np.concatenate(trainto)
            trainy = np.concatenate(trainy)
            trainw = np.concatenate(trainw)
            trainw = torch.LongTensor(trainw).to(device)
            trainx = torch.Tensor(trainx).to(device)
            trainx= trainx.permute(0, 3, 1, 2)
            trainy = torch.Tensor(trainy).to(device)
            trainy = trainy.permute(0, 3, 1, 2)
            trainti = torch.Tensor(trainti).to(device)
            trainto = torch.Tensor(trainto).to(device)
            model.train()
            optimizer.zero_grad()
            output = model(trainx, trainti, supports, trainto, trainw)
            loss = util.masked_rmse(output, trainy, 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            
        outputs = []
        model.eval()
        for i in range(len(val_index)):
            testx = np.expand_dims(val_data[:, val_index[i]: val_index[i] + args.in_len, :], axis = 0)
            testx = scaler.transform(testx)
            testw = np.expand_dims(val_w[:, val_index[i]: val_index[i] + args.in_len], axis = 0)
            testw = torch.LongTensor(testw).to(device)
            testx[np.isnan(testx)] = 0
            ti_add = (np.arange(int(training_data.shape[1])+val_index[i], int(training_data.shape[1])+val_index[i]+ args.in_len)// args.period % args.period1)/args.period1
            testti = (np.arange(int(training_data.shape[1])+val_index[i], int(training_data.shape[1])+val_index[i]+ args.in_len) % args.period) * np.ones([1, args.in_len])/(args.period - 1) + ti_add
            to_add = (np.arange(int(training_data.shape[1])+val_index[i] + args.in_len, int(training_data.shape[1])+val_index[i] + args.in_len + args.out_len)// args.period % args.period1)/args.period1
            testto = (np.arange(int(training_data.shape[1])+val_index[i] + args.in_len, int(training_data.shape[1])+val_index[i] + args.in_len + args.out_len) % args.period) * np.ones([1, args.out_len])/(args.period - 1) + to_add
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
         
        amae = []
        ar2 = []
        armse = []
        for i in range(12):
            metrics = test_error(yhat[:,:,i,:],label[:,:,i,:])
            amae.append(metrics[0])
            ar2.append(metrics[2])
            armse.append(metrics[1])
         
        log = 'On average over all horizons, Test MAE: {:.4f}, Test R2: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(amae),np.mean(ar2),np.mean(armse)))
     
        MAE_list.append(np.mean(amae))
        if np.mean(amae) == min(MAE_list):
            best_model = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model)
    torch.save(model, "spdpn" + args.data +".pth")
    
if __name__ == "__main__":   
    main() 
