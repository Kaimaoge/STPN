# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 13:36:31 2023

@author: AA
"""
import numpy as np
from scipy.stats import entropy
from scipy.special import kl_div

def rmse(truth, pred):
    return np.sqrt(((truth-pred)**2).mean())

def mae(truth, pred):
    return np.abs(truth-pred).mean()

def wape(truth,pred):
    return np.abs(np.subtract(pred,truth)).sum()/np.sum(truth)
    
def mape(truth,pred):
    return np.mean( np.abs( (np.subtract(pred,truth)+1e-5)/(truth+1e-5) ) )

def true_zeros(truth,pred):
    idx = truth == 0
    return np.sum(pred[idx]==0)/np.sum(idx)

def KL_DIV(truth,pred):
    return np.sum( pred*np.log( (pred+1e-5)/(truth+1e-5) ) )

def KL_DIV_divide(truth,pred):
    return np.sum( pred*np.log( (pred+1e-5)/(truth+1e-5) ) )/np.prod(truth.shape)

def F1_SCORE(truth,pred):
    true_zeros = truth == 0
    pred_zeros = pred == 0
    precision = np.sum(pred_zeros & true_zeros ) / np.sum(pred_zeros)
    recall = np.sum(pred_zeros)/np.sum(true_zeros)
    return 2*(precision*recall)/(precision+recall)

def print_errors(truth,pred,string=None):
    print(string,' RMSE %.4f MAE %.4f F1_SCORE %.4f KL-Div: %.4f, KL-Div-divide: %.4f, true_zeros_rate %.4f : '%(
        rmse(truth,pred),mae(truth,pred),F1_SCORE(truth,pred),KL_DIV(truth,pred),KL_DIV_divide(truth,pred),true_zeros(truth,pred)
    ))