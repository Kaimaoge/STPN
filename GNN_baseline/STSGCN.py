# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 17:18:26 2023

@author: AA
"""

import torch
import torch.nn as nn
import math


class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 time_dim,
                 joints_dim, in_channels
    ):
        super(ConvTemporalGraphical,self).__init__()

        self.T=nn.Parameter(torch.FloatTensor(joints_dim , time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv,stdv)
      
    def forward(self, x):
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        return x.contiguous() 

class ConvTemporalGraphical2(nn.Module):
    
    def __init__(self,
                 time_dim,
                 joints_dim, in_channels
    ):
        super(ConvTemporalGraphical2,self).__init__()
        
        self.A=nn.Parameter(torch.FloatTensor(time_dim, joints_dim,joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv,stdv)

      
    def forward(self, x):
        x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        return x.contiguous() 


class ST_GCNN_layer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True):
        
        super(ST_GCNN_layer,self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        
        
        self.gcn=ConvTemporalGraphical(time_dim,joints_dim, in_channels) # the convolution layer
        self.gcn2=ConvTemporalGraphical2(time_dim,joints_dim, in_channels)

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        
        self.tcn2 = nn.Sequential(
            nn.Conv2d(
                joints_dim,
                joints_dim,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        ) 

        if stride != 1 or in_channels != out_channels: 

            self.residual=nn.Sequential(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )
            
            
        else:
            self.residual=nn.Identity()
        
        
        self.prelu = nn.PReLU()

        

    def forward(self, x):
     #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res=self.residual(x)
        x=self.gcn(x) 
        x=self.tcn(x)
        x=self.prelu(x)
        x=self.gcn2(x)
        x=x.permute(0,2,1,3)
        x=self.tcn2(x)
        x=x.permute(0,2,1,3)
        x=x+res
        x=self.prelu(x)
        return x



class CNN_layer(nn.Module): # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
        
        super(CNN_layer,self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        
        
        self.block= [nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
                     ,nn.BatchNorm2d(out_channels),nn.Dropout(dropout, inplace=True)] 



            
        
        self.block=nn.Sequential(*self.block)
        

    def forward(self, x):
        
        output= self.block(x)
        return output



class Model(nn.Module):

    def __init__(self,
                 input_channels,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 joints_to_consider,
                 n_txcnn_layers,
                 txc_kernel_size,
                 txc_dropout,
                 bias=True):
        
        super(Model,self).__init__()
        self.input_time_frame=input_time_frame
        self.output_time_frame=output_time_frame
        self.joints_to_consider=joints_to_consider
        self.st_gcnns=nn.ModuleList()
        self.n_txcnn_layers=n_txcnn_layers
        self.txcnns=nn.ModuleList()
        
      
        self.st_gcnns.append(ST_GCNN_layer(input_channels,64,[1,1],1,input_time_frame,
                                           joints_to_consider,st_gcnn_dropout))
        self.st_gcnns.append(ST_GCNN_layer(64,32,[1,1],1,input_time_frame,
                                               joints_to_consider,st_gcnn_dropout))
            
        self.st_gcnns.append(ST_GCNN_layer(32,64,[1,1],1,input_time_frame,
                                               joints_to_consider,st_gcnn_dropout))
                                               
        self.st_gcnns.append(ST_GCNN_layer(64,input_channels,[1,1],1,input_time_frame,
                                               joints_to_consider,st_gcnn_dropout))                                               
                
                
                # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)           
        self.txcnns.append(CNN_layer(input_time_frame,output_time_frame,txc_kernel_size,txc_dropout)) # with kernel_size[3,3] the dimensinons of C,V will be maintained       
        for i in range(1,n_txcnn_layers):
            self.txcnns.append(CNN_layer(output_time_frame,output_time_frame,txc_kernel_size,txc_dropout))
        
            
        self.prelus = nn.ModuleList()
        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())


        

    def forward(self, x):
        #for gcn in (self.st_gcnns):
         #   x = gcn(x)

        x=self.st_gcnns[0].gcn(x)
        x=self.st_gcnns[0].gcn2(x)

        x=self.st_gcnns[1].gcn(x)
        x=self.st_gcnns[1].gcn2(x)

        x=self.st_gcnns[2].gcn(x)
        x=self.st_gcnns[2].gcn2(x)

        x=self.st_gcnns[3].gcn(x)
        x=self.st_gcnns[3].gcn2(x)

        x= x.permute(0,2,1,3) # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)
        
        x=self.prelus[0](self.txcnns[0](x))
        
        for i in range(1,self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) +x # residual connection
        x = x.permute(0,2,1,3)   
        return x
