# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:42:10 2022

@author: AA
"""
import numpy as np
import scipy.sparse as sp
import torch


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def load_data(data_name, ratio = [0.7, 0.1]):
    if data_name == 'US':
        adj_mx = np.load('udata/adj_mx.npy')
        od_power = np.load('udata/od_pair.npy')
        od_power = od_power/(1.5*od_power.max())
        od_power[od_power < 0.1] = 0
        for i in range(70):
            od_power[i, i] = 1
        adj = [asym_adj(adj_mx), asym_adj(od_power), asym_adj(od_power.T)]
        data = np.load('udata/udelay.npy')
        wdata = np.load('udata/weather2016_2021.npy')  
    if data_name == 'China':
        adj_mx = np.load('cdata/dist_mx.npy')
        od_power = np.load('cdata/od_mx.npy')
        od_power = od_power/(1.5*od_power.max())
        od_power[od_power < 0.1] = 0
        for i in range(50):
            od_power[i, i] = 1
        adj = [asym_adj(adj_mx), asym_adj(od_power), asym_adj(od_power.T)]
        data = np.load('cdata/delay.npy')
        data[data<-15] = -15
        wdata = np.load('cdata/weather_cn.npy')
    training_data = data[:, :int(ratio[0]*data.shape[1]) ,:]
    val_data = data[:,int(ratio[0]*data.shape[1]):int((ratio[0] + ratio[1])*data.shape[1]),:]
    test_data = data[:, int((ratio[0] + ratio[1])*data.shape[1]):, :]
    training_w = wdata[:, :int(ratio[0]*data.shape[1])]
    val_w = wdata[:,int(ratio[0]*data.shape[1]):int((ratio[0] + ratio[1])*data.shape[1])]
    test_w = wdata[:, int((ratio[0] + ratio[1])*data.shape[1]):]    
    return adj, training_data, val_data, test_data, training_w, val_w, test_w

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_wmae(preds, labels, weights, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask * weights
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
        
