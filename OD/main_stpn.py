# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:23:34 2023

@author: AA
"""
from __future__ import division
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
import torch.optim as optim
from torch import nn
from utils import generate_dataset, get_normalized_adj, calculate_random_walk_matrix
from STPN_ZINB import STPN
import copy

import os
from final_eval import *
# Parameters
device = torch.device('cuda') 
A = np.load('ny_data_full_15min/adj_rand0.npy') # change the loading folder
X = np.load('ny_data_full_15min/cta_samp_rand0.npy')

num_timesteps_output = 4 
num_timesteps_input = num_timesteps_output
 
h1 = 128
h2 = 64
h3 = 32
l = 2
STmodel = STPN(1, [h1, h2, h3], 16, 0, support_len = 1).to(device=device)

epochs = 500
batch_size = 32
# Load dataset

X = X.T
X = X.astype(np.float32)
X = X.reshape((X.shape[0],1,X.shape[1]))

split_line1 = int(X.shape[2] * 0.6)
split_line2 = int(X.shape[2] * 0.7)
print(X.shape,A.shape)

# normalization
max_value = np.max(X[:, :, :split_line1])

train_original_data = X[:, :, :split_line1]
val_original_data = X[:, :, split_line1:split_line2]
test_original_data = X[:, :, split_line2:]
training_input, training_target = generate_dataset(train_original_data,
                                                    num_timesteps_input=num_timesteps_input,
                                                    num_timesteps_output=num_timesteps_output)

val_input, val_target = generate_dataset(val_original_data,
                                            num_timesteps_input=num_timesteps_input,
                                            num_timesteps_output=num_timesteps_output)

V_in = []
V_out = []
for j in range(val_input.shape[0]):
    in_temp = torch.arange(split_line1 + j, (split_line1 + j + num_timesteps_input))
    out_temp = torch.arange((split_line1 + j + num_timesteps_input), (split_line1 + j + num_timesteps_input + num_timesteps_output) )
    V_in.append((in_temp % 96).unsqueeze(0))
    V_out.append((out_temp % 96).unsqueeze(0))
V_in = torch.concat(V_in).to(device=device)
V_out = torch.concat(V_out).to(device=device)
#print('input shape: ',training_input.shape,val_input.shape,test_input.shape)

A_wave = get_normalized_adj(A)
A_q = torch.from_numpy((calculate_random_walk_matrix(A_wave)).astype('float32'))
A_q = A_q.to(device=device)


supports = [A_q]


optimizer = optim.Adam(STmodel.parameters(), lr=1e-3)
training_nll   = []
validation_nll = []
validation_mae = []
forecast_loss = nn.SmoothL1Loss().cuda()
#forecast_loss = nn.MSELoss().cuda()

for epoch in range(epochs):
    ## Step 1, training
    """
    # Begin training, similar training procedure from STGCN
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    """
    permutation = torch.randperm(training_input.shape[0])
    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        STmodel.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=device)
        y_batch = y_batch.to(device=device)
        y_batch = y_batch.unsqueeze(1)
        X_batch = X_batch.permute([0, 3, 1, 2])
        T_in = []
        T_out = []
        for k in range(len(indices)):
            in_temp = torch.arange(indices[k], (indices[k] + num_timesteps_input))
            out_temp = torch.arange((indices[k] + num_timesteps_input), (indices[k] + num_timesteps_input + num_timesteps_output) )
            T_in.append((in_temp % 96).unsqueeze(0))
            T_out.append((out_temp % 96).unsqueeze(0))
        #y_batch = y_batch.permute([0, 3, 1, 2])
        T_in = torch.concat(T_in).to(device=device)
        T_out = torch.concat(T_out).to(device=device)
        p = STmodel(X_batch, T_in, supports, T_out)
        loss = forecast_loss(y_batch, p) 
        loss.backward()
       # torch.nn.utils.clip_grad_norm_(STmodel.parameters(), 3)
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    training_nll.append(sum(epoch_training_losses)/len(epoch_training_losses))
    ## Step 2, validation
    STmodel.eval()
    with torch.no_grad():
        p_val = []

        val_target = val_target.to(device=device)
        for j in range(val_input.shape[0]):
            t_input = val_input[j:j+1].to(device=device)
            t_input = t_input.permute([0, 3, 1, 2])
            p_temp = STmodel(t_input,V_in[j:j+1],supports,V_out[j:j+1])
            p_val.append(p_temp)
        p_val = torch.concat(p_val)
        p_val[p_val < 0] = 0
        val_pred = torch.round(p_val).detach().cpu().numpy()
        # Calculate the expectation value        

        MAE = mae(val_pred, val_target.unsqueeze(1).detach().cpu().numpy())
        validation_mae.append(MAE)
        print_errors(val_target.unsqueeze(1).detach().cpu().numpy(), val_pred)

        #val_input = val_input.to(device="cpu")
        val_target = val_target.to(device="cpu")

    if np.asscalar(validation_mae[-1]) == min(validation_mae):
        best_model = copy.deepcopy(STmodel.state_dict())
    checkpoint_path = "checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if np.isnan(training_nll[-1]):
        break
STmodel.load_state_dict(best_model)
torch.save(STmodel,'pth/STZINB_ny_full_15min.pth')
