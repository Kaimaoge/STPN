# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:24:24 2023

@author: AA
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from PeMS_Process import ForecastDataset,ForecastTestDataset, de_normalized
from torch.utils.data import DataLoader
from stpn_pems import STPN
import torch.optim as optim
import torch
import util
import torch.nn as nn
from PeMS_evaluate import evaluate
import copy
import pandas as pd


if __name__ ==  '__main__':
    dataset = '08'
    
    data_file = os.path.join('PeMS/PEMS' + dataset + '.npz')
    print('data file:',data_file)
    data = np.load(data_file,allow_pickle=True)
    data = data['data'][:,:,0]
    train_ratio = 6 / (6 + 2 + 2)
    valid_ratio = 2 / (6 + 2 + 2)
    test_ratio = 1 - train_ratio - valid_ratio
    train_data = data[:int(train_ratio * len(data))]
    valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
    test_data = data[int((train_ratio + valid_ratio) * len(data)):]
    begin_time = int((train_ratio) * len(data))
    train_mean = np.mean(train_data, axis=0)
    train_std = np.std(train_data, axis=0)
    train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    val_mean = np.mean(valid_data, axis=0)
    val_std = np.std(valid_data, axis=0)
    val_normalize_statistic = {"mean": val_mean.tolist(), "std": val_std.tolist()}
    test_mean = np.mean(test_data, axis=0)
    test_std = np.std(test_data, axis=0)
    test_normalize_statistic = {"mean": test_mean.tolist(), "std": test_std.tolist()} # data_leakage
    train_set = ForecastDataset(train_data, window_size=12, horizon=12,
                            normalize_method='z_score', norm_statistic=train_normalize_statistic)
    val_set = ForecastTestDataset(valid_data, window_size=12, horizon=12, begin_time = begin_time,
                                normalize_method='z_score', norm_statistic=test_normalize_statistic)
    
    train_loader = DataLoader(train_set, batch_size=32, drop_last=False, shuffle=True,
                                        num_workers=1)
    test_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=1)
    
    data_d = np.reshape(data, [data.shape[0]//288, 288, data.shape[1]])
    daily_trend = np.mean(data_d, axis = 0)
    daily_diff = daily_trend[1:, :] - daily_trend[:-1, :] 
    cov = np.corrcoef(daily_trend.transpose())
    cov[cov < 0.95] = 0
    cov_d = np.corrcoef(daily_diff.transpose())
    cov_d[cov_d < 0.6] = 0
    con = pd.read_csv('PeMS/adj_PEMS' + dataset + '.csv', header=None)
    con = np.array(con)
    adj = [util.asym_adj(cov), util.asym_adj(cov_d), util.asym_adj(con)]
    supports = [torch.tensor(i).to('cuda') for i in adj]
    
    h1 = 128
    h2 = 64
    h3 = 32
    l = 2
    model = STPN(l, 1, [h1, h2, h3], 1, 16, 0, support_len = 3)
    model.to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    forecast_loss = nn.L1Loss().cuda()
    MAE_list = []
    for epoch in range(1, 201):
        model.train()
        loss_total = 0
        loss_total_F = 0
        loss_total_M = 0
        for i, (inputs, inputs_t, target, target_t) in enumerate(train_loader):
            inputs = inputs.cuda()  # torch.Size([32, 12, 228])
            target = target.cuda()
            inputs_t = inputs_t.cuda()
            target_t = target_t.cuda()
            inputs = torch.unsqueeze(inputs, 1).permute(0, 1, 3, 2)
            target = torch.unsqueeze(target, 1).permute(0, 1, 3, 2)
            model.zero_grad()
            output = model(inputs, inputs_t, supports, target_t)
            loss = forecast_loss(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
        
        forecast_set = []
        target_set = []
        model.eval()
        with torch.no_grad():
            for i, (inputs, inputs_t, target, target_t) in enumerate(test_loader):
                inputs = inputs.cuda()  # torch.Size([32, 12, 228])
                target = target.cuda()
                inputs_t = inputs_t.cuda()
                target_t = target_t.cuda()
                inputs = torch.unsqueeze(inputs, 1).permute(0, 1, 3, 2)
                target = torch.unsqueeze(target, 1).permute(0, 1, 3, 2)
                output = model(inputs, inputs_t, supports, target_t)
                forecast_set.append(output.detach().cpu().numpy())
                target_set.append(target.detach().cpu().numpy())
            result_save = np.concatenate(forecast_set, axis=0)
            target_save = np.concatenate(target_set, axis=0)
        
        result = de_normalized(result_save[:, 0, :, :].transpose(0, 2, 1), 'z_score', test_normalize_statistic)
        target = de_normalized(target_save[:, 0, :, :].transpose(0, 2, 1), 'z_score', test_normalize_statistic)
        MAPE, MAE, RMSE = evaluate(result, target)
        log = 'On average, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(MAE,MAPE,RMSE))
        MAE_list.append(MAE)
        if MAE == min(MAE_list):
            best_model = copy.deepcopy(model.state_dict())
        
        model.load_state_dict(best_model)
        torch.save(model, "spdn_pems08" + str(l) + str(h1) + str(h2) +".pth")
