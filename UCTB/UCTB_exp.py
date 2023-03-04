# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 18:04:15 2023

@author: AA
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from UCTB_process import ForecastDataset,ForecastTestDataset,Normalizer,merge_data,rmse
from torch.utils.data import DataLoader
from stpn_pems import STPN
import torch.optim as optim
import torch
import util
import torch.nn as nn
import copy

if __name__ ==  '__main__':
    Merge = 12
    dataset = 'UCTB/Bike_NYC.pkl'
    data, AM, CM, IM = merge_data(dataset, Merge)
    train_ratio = 0.9
    test_ratio = 1 - train_ratio
    train_data = data[:int(0.8* len(data))]
    test_data = data[int((0.8) * len(data)):int(train_ratio * len(data))]
    begin_time = int((train_ratio) * len(data))
    normalize_method = Normalizer(train_data)
    train_set = ForecastDataset(train_data, window_size = 6, horizon = 1, normalize_method=normalize_method, interval=1, MergeIndex = Merge)
    test_set = ForecastTestDataset(test_data, window_size= 6, horizon= 1, begin_time = begin_time,
                                normalize_method = normalize_method, interval=1, MergeIndex = Merge)
    
    train_loader = DataLoader(train_set, batch_size=32, drop_last=False, shuffle=True,
                                        num_workers=1)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=1)
    
    adj = [util.asym_adj(AM), util.asym_adj(IM), util.asym_adj(CM)]
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
        
        result = normalize_method.min_max_denormal(result_save[:, 0, :, :].transpose(0, 2, 1))
        target = normalize_method.min_max_denormal(target_save[:, 0, :, :].transpose(0, 2, 1))
        RMSE = rmse(result, target)
        log = 'On average, Test RMSE: {:.4f}'
        print(log.format(RMSE))
        MAE_list.append(RMSE)
        if RMSE == min(MAE_list):
            best_model = copy.deepcopy(model.state_dict())
        
    model.load_state_dict(best_model)
    torch.save(model, "bike_nyc60" + str(l) + str(h1) + str(h2) +".pth")