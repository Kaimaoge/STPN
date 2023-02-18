# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:07:14 2023

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
    
class learnEmbedding(nn.Module):
    def __init__(self, d_model):
        super(learnEmbedding, self).__init__()
        self.factor = nn.parameter.Parameter(torch.randn(1,),requires_grad=True).to('cuda')
        self.d_model = d_model
    
    def forward(self, x):
        div = torch.arange(0, self.d_model, 2).to('cuda')
        div_term = torch.exp(div * self.factor)
        if len(x.shape) == 2:
            v1 = torch.sin(torch.einsum('bt, f->btf', x, div_term))
            v2 = torch.cos(torch.einsum('bt, f->btf', x, div_term))
        else:
            v1 = torch.sin(torch.einsum('bvz, f->bvzf', x, div_term))
            v2 = torch.cos(torch.einsum('bvz, f->bvzf', x, div_term))
        return torch.cat([v1, v2], -1)
    
class ATT(nn.Module):
    def __init__(self,c_in, d = 16, device = 'cuda'):
        super(ATT,self).__init__()
        self.d = d
        self.qm = nn.Linear(in_features = c_in, out_features = d, bias = False)
        self.km = nn.Linear(in_features = c_in, out_features = d, bias = False)
        self.device = device
    
    def forward(self,x, y):
        if len(x.shape) == 3:
            query = self.qm(y)
            key = self.km(x)
            attention = torch.einsum('btf,bpf->btp', query, key)
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=-1)
        else:
            query = self.qm(y)
            key = self.km(x)
            attention = torch.einsum('bvzf,buzf->bvu', query, key)
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=2)
        return attention
    

class STMH_GCNN_layer(nn.Module):
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
    def __init__(self, in_channels, out_channels, emb_size, dropout, time_d = 16, heads = 4, support_len = 1, order = 2, final_layer = False):
        super(STMH_GCNN_layer,self).__init__()   
        self.gcn = gcn(in_channels,support_len=support_len,order=order)
        gc_in = (order*support_len+1)*in_channels
        self.out = linear(gc_in, out_channels)
        self.temb = nn.ModuleList()
        self.tgraph = nn.ModuleList()
        for i in range(heads):
            self.temb.append(learnEmbedding(emb_size))
            self.tgraph.append(ATT(emb_size, time_d))
        if in_channels != out_channels: 
            self.residual=nn.Sequential(linear(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
            )                   
        else:
            self.residual=nn.Identity()       
        self.prelu = nn.PReLU()
        self.final_layer = final_layer
        self.dropout = dropout
        self.heads = heads
        
    def forward(self, x, t_in, supports, t_out = None):
        t_att = []
        for i in range(self.heads):
            k_emb = self.temb[i](t_in)
            if t_out == None:
                q_emb = k_emb         
            else:
                q_emb = self.temb[i](t_out)        
            t_att.append(self.tgraph[i](k_emb, q_emb))
        res=self.residual(x)
        xt = torch.einsum('ncvt,npt->ncvp', (x, t_att[0]))
        for i in range(self.heads - 1):
            xt += torch.einsum('ncvt,npt->ncvp', (x, t_att[i+1])) 
        x = self.gcn(xt, supports)
        x = self.out(x)
        if not self.final_layer:
            x = x+res
            x = self.prelu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        return x
    
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
    def __init__(self, h_layers, hidden_channels, emb_size, dropout,  time_d = 4, heads = 4, support_len = 3, order = 2, use_se = True):
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
        self.use_se = use_se
        self.convs.append(STMH_GCNN_layer(1, hidden_channels[0], emb_size, dropout, time_d, heads, support_len, order, False))
        for i in range(h_layers):
            if self.use_se:
                self.se.append(SELayer(hidden_channels[i]))
            self.convs.append(STMH_GCNN_layer(hidden_channels[i], hidden_channels[i+1], emb_size, dropout, time_d, heads, support_len, order, False))  
        self.n_conv = STMH_GCNN_layer(hidden_channels[h_layers] ,1, emb_size, dropout, time_d, heads, support_len, order, True)
        self.p_conv = STMH_GCNN_layer(hidden_channels[h_layers] ,1, emb_size, dropout, time_d, heads, support_len, order, True) 
        self.pi_conv = STMH_GCNN_layer(hidden_channels[h_layers] ,1, emb_size, dropout, time_d, heads, support_len, order, True) 
    def forward(self, x, t_in, supports, t_out):
        for i in range(self.h_layers + 1):
            x = self.convs[i](x, t_in, supports)
            if i < self.h_layers and self.use_se:
                x = self.se[i](x)
        p = (self.p_conv(x, t_in, supports, t_out))
        return p
