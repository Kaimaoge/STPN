# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:29:31 2023

@author: AA
"""
from __future__ import division
import os
import pickle
import torch
import numpy as np
import torch.utils.data as torch_data

from math import radians, cos, sin, asin, sqrt
from scipy.stats import pearsonr

def rmse(prediction, target, threshold=None):
    """
    Args:
        prediction(ndarray): prediction with shape [batch_size, ...]
        target(ndarray): same shape with prediction, [batch_size, ...]
        threshold(float): data smaller or equal to threshold in target
            will be removed in computing the rmse
    """
    if threshold is None:
        return np.sqrt(np.mean(np.square(prediction - target)))
    else:
        return np.sqrt(np.dot(np.square(prediction - target).reshape([1, -1]),
                              target.reshape([-1, 1]) > threshold) / np.sum(target > threshold))[0][0]

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371

    return c * r * 1000

def distance_adjacent(lat_lng_list, threshold):
    '''
    Calculate distance graph based on geographic distance.
    Args:
        lat_lng_list(list): A list of geographic locations. The format of each element
                in the list is [latitude, longitude].
        threshold(float): (meters) nodes with geographic distacne smaller than this 
            threshold will be linked together.
    '''
    adjacent_matrix = np.zeros([len(lat_lng_list), len(lat_lng_list)])
    for i in range(len(lat_lng_list)):
        for j in range(len(lat_lng_list)):
            adjacent_matrix[i][j] = haversine(lat_lng_list[i][0], lat_lng_list[i][1],
                                                            lat_lng_list[j][0], lat_lng_list[j][1])
    adjacent_matrix = (adjacent_matrix <= threshold).astype(np.float32)
    return adjacent_matrix

def correlation_adjacent(traffic_data, threshold):
    '''
    Calculate correlation graph based on pearson coefficient.
    Args:
        traffic_data(ndarray): numpy array with shape [sequence_length, num_node].
        threshold(float): float between [-1, 1], nodes with Pearson Correlation coefficient
            larger than this threshold will be linked together.
    '''
    adjacent_matrix = np.zeros([traffic_data.shape[1], traffic_data.shape[1]])
    for i in range(traffic_data.shape[1]):
        for j in range(traffic_data.shape[1]):
            r, p_value = pearsonr(traffic_data[:, i], traffic_data[:, j])
            adjacent_matrix[i, j] = 0 if np.isnan(r) else r
    adjacent_matrix = (adjacent_matrix >= threshold).astype(np.float32)
    return adjacent_matrix


class Normalizer(object):
    '''
    This class can help normalize and denormalize data by calling min_max_normal and min_max_denormal method.
    '''
    def __init__(self, X):
        self._min = np.min(X)
        self._max = np.max(X)

    def min_max_normal(self, X):
        '''
        Input X, return normalized results.
        :type: numpy.ndarray
        '''
        return (X - self._min) / (self._max - self._min)

    def min_max_denormal(self, X):
        '''
        Input X, return denormalized results.
        :type: numpy.ndarray
        '''
        return X * (self._max - self._min) + self._min

def merge_data(pkl, MergeIndex, train_ratio = 0.9, threshold_distance = 1000, c_threshold = 0.5, a_threshold = 40, daily_slots = 288):   
    with open(pkl, 'rb') as f:
        data_all = pickle.load(f)
    data = data_all['Node']['TrafficNode'] 
    traffic_data_index = np.where(np.mean(data, axis=0) * daily_slots // MergeIndex > 1)[0]
    data = data[:, traffic_data_index].astype(np.float32)
    func = np.sum
    new = np.zeros((data.shape[0]//MergeIndex,data.shape[1]),dtype=np.float32)
    for new_ind,ind in enumerate(range(0,data.shape[0],MergeIndex)):
        new[new_ind,:] = func(data[ind:ind+MergeIndex,:],axis=0)
    lat_lng_list = np.array([[float(e1) for e1 in e[2:4]]
                         for e in data_all['Node']['StationInfo']])
    AM = distance_adjacent(lat_lng_list[traffic_data_index],
                                    threshold=float(threshold_distance))
    train_data = new[:int(0.9*new.shape[0])]
    CM = correlation_adjacent(train_data[-30 * int(daily_slots)//MergeIndex:],
                                       threshold=c_threshold)
    monthly_interaction = data_all['Node']['TrafficMonthlyInteraction'][:, traffic_data_index, :][:, :, traffic_data_index]
    monthly_interaction = monthly_interaction[:int(0.9*monthly_interaction.shape[0])]
    annually_interaction = np.sum(monthly_interaction[-12:], axis=0)
    annually_interaction = annually_interaction + annually_interaction.transpose()
    IM = (annually_interaction >= a_threshold).astype(np.float32)    
    return new, AM, CM, IM

class ForecastDataset(torch_data.Dataset):
    def __init__(self, data, window_size, horizon, normalize_method=None, interval=1, MergeIndex = 12):
        self.window_size = window_size # 12
        self.interval = interval  #1
        self.horizon = horizon
        self.data = data
        self.df_length = data.shape[0]
        self.x_end_idx = self.get_x_end_idx()
        self.MergeIndex = MergeIndex
        if normalize_method:
            self.data = normalize_method.min_max_normal(self.data)

    def __getitem__(self, index):
        hi = self.x_end_idx[index] #12
        lo = hi - self.window_size #0
        train_data = self.data[lo: hi] #0:12
        train_time = np.arange(lo, hi) % (288 // self.MergeIndex)
        target_data = self.data[hi:hi + self.horizon] #12:24
        target_time = np.arange(hi, hi + self.horizon) % (288 // self.MergeIndex)
        x = torch.from_numpy(train_data).type(torch.float)
        xt = torch.from_numpy(train_time).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        yt = torch.from_numpy(target_time).type(torch.float)
        return x,xt,y,yt

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx
    
class ForecastTestDataset(torch_data.Dataset):
    def __init__(self, data, window_size, horizon, begin_time, normalize_method=None, interval=1, MergeIndex = 12):
        self.window_size = window_size # 12
        self.interval = interval  #1
        self.horizon = horizon
        self.data = data
        self.df_length = data.shape[0]
        self.x_end_idx = self.get_x_end_idx()
        self.begin_time = begin_time
        self.MergeIndex = MergeIndex
        if normalize_method:
            self.data = normalize_method.min_max_normal(self.data)

    def __getitem__(self, index):
        hi = self.x_end_idx[index] #12
        lo = hi - self.window_size #0
        train_data = self.data[lo: hi] #0:12
        train_time = np.arange(lo + self.begin_time, hi + self.begin_time) % (288 // self.MergeIndex)
        target_data = self.data[hi:hi + self.horizon] #12:24
        target_time = np.arange(hi + self.begin_time, hi + self.begin_time + self.horizon) % (288 // self.MergeIndex)
        x = torch.from_numpy(train_data).type(torch.float)
        xt = torch.from_numpy(train_time).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        yt = torch.from_numpy(target_time).type(torch.float)
        return x,xt,y,yt

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * 12] for j in range((len(x_index_set)) // 12)]
        return x_end_idx