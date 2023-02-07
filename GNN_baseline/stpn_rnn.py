# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:55:54 2023

@author: AA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        return h
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class STGLSTM_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, V, T_in)` format
        - Input[1]: Input random walk matrix in a list :math:`(V, V)` format
        - INput[2]: Input time label :math:`(N, T)`
        - Output[0]: Output graph sequence in :math:`(N, out_channels, V, T_out)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """
    def __init__(self, in_channels, out_channels, dropout, support_len = 1, order = 2):
        super(STGLSTM_layer,self).__init__()   
        self.gcn = gcn(in_channels,support_len=support_len,order=order)
        gc_in = (order*support_len+1)*in_channels
        self.gout = linear(gc_in, 4*in_channels)
        self.temb = nn.ModuleList()
        self.out = linear(in_channels, out_channels)
        if in_channels != out_channels: 
            self.residual=nn.Sequential(linear(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
            )                   
        else:
            self.residual=nn.Identity()       
        #self.prelu = nn.PReLU()
        self.dropout = dropout
        self.in_channels = in_channels
        
    def forward(self, x, supports, out_len = None, final_layer = False):
        outputs = []
        B, C, S, T = x.shape
        if out_len != None:
            T = out_len
        hx = torch.zeros(B, C, S, 1).to(x.device)
        cx = torch.zeros(B, C, S, 1).to(x.device)
        for t in range(T):
            if final_layer:
                combined = x[:,:,:,-1:] + hx
            else:
                combined = x[:,:,:,t:t+1] + hx
            gates = self.gcn(combined, supports)
            gates = self.gout(gates)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.in_channels, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate) 
            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy
        outputs = torch.cat(outputs, dim = -1)
        outputs = self.out(outputs)
        outputs = F.dropout(outputs, self.dropout, training=self.training)
        return outputs
    
class STPN(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, V, T_in)` format
        - Input[1]: Input time label :math:`(N, T_in)` format   
        - Input[2]: Input random walk matrix in a list :math:`(V, V)` format
        - Input[3]: Output time label :math:`(N, T_out)`
        - Input[4]: Input covariate sequence in :math:`(N, V, T_out)`
        - Output[0]: Output graph sequence in :math:`(N, out_channels, V, T_out)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels= dimension of coordinates
            :out_channels= dimension of coordinates
            +
    """
    def __init__(self, out_len, h_layers, in_channels, hidden_channels, out_channels, dropout, wemb_size = 4, support_len = 3, order = 2, num_weather = 8):
        super(STPN,self).__init__()
        """
        Params:
            - h_layers: number of hidden STS GCNs
            - in_channels: num = 2 for delay prediction
            - hidden_channels: a list of dimensions for hidden features
            - out_channels: num = 2 for delay prediction
            - emb_size: embedding size for self attention model
            - dropout: dropout rate
            - wemb_size: covariate embedding size
            - time_d: d for self attention model
            - heads: number of attention head
            - supports_len: number of spatial adjacency matrix
            - order: order of diffusion convolution
            - num_weather: number of weather condition
            - use_se: whether use SE block
            - use_cov: whether use weather information
        """
      
        self.h_layers = h_layers
        self.convs = nn.ModuleList()
        self.se = nn.ModuleList()
        self.convs.append(STGLSTM_layer(in_channels+ wemb_size, hidden_channels[0], dropout, support_len, order))
        self.w_embedding = nn.Embedding(num_weather, wemb_size)
        for i in range(h_layers):
            self.se.append(SELayer(hidden_channels[i]))
            self.convs.append(STGLSTM_layer(hidden_channels[i], hidden_channels[i+1], dropout, support_len, order))  
        #self.final_conv = STMH_GCNN_layer(hidden_channels[h_layers] , out_channels, emb_size, dropout, time_d, heads, support_len, order, True)
        self.out_conv = STGLSTM_layer(hidden_channels[h_layers] , out_channels, dropout, support_len, order)
        self.out_len = out_len
    def forward(self, x, supports, w_type):
        w_vec = self.w_embedding(w_type)
        w_vec = w_vec.permute(0, 3, 1, 2)
        x = torch.cat([x, w_vec], 1)
        for i in range(self.h_layers + 1):
            x = self.convs[i](x, supports)
            if i < self.h_layers:
                x = self.se[i](x)
        out = self.out_conv(x, supports, out_len = self.out_len, final_layer = True)
        return out
    
