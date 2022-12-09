#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:17:36 2021

@author: sned

Architecture classes
"""
import torch
from torch import nn, optim
from torch.nn import functional as F
import pdb
class DAE(nn.Module):
    def __init__(self, LD, num_features):
        super(DAE, self).__init__()

        latent_dim = LD
        io_layer = num_features


        ## eGE.. Func
        self.fe1 = nn.Linear(io_layer, 40)
        self.fe6 = nn.Linear(40, 20)
        self.fe61 = nn.Linear(20, 10)
        self.fe7 = nn.Linear(10, latent_dim)

        self.fd8 = nn.Linear(latent_dim, 10)
        self.fd9 = nn.Linear(10, 20)
        self.fd10 = nn.Linear(20, 40)
        self.fd16 = nn.Linear(40, io_layer)


    #DAE encoder
    def encode(self, x):

        if (self.fe1.weight != self.fe1.weight).any():
            pdb.set_trace()  
        ## eGE
        h1 = F.relu (self.fe1(x))
        h6 = F.relu(self.fe6(h1))
        h61 = F.relu(self.fe61(h6))
        # h1 = F.softplus(self.fe1(x))
        # h6 = F.softplus(self.fe6(h1))
        # h61 = F.softplus(self.fe61(h6))
        latent_rep = (self.fe7(h61))

        return latent_rep

    #DAE decoder
    def decode(self, z):

        ## eGE
        h8 = (self.fd8(z))
        h9 = F.relu(self.fd9(h8))
        # h9 = F.softplus(self.fd9(h8))
        h10 = (self.fd10(h9))
        h16 = (self.fd16(h10))

        return h16


    def forward(self, x):
        latent_rep = self.encode(x)
        if (self.fe1.weight != self.fe1.weight).any():
            pdb.set_trace()   
        x_hat = self.decode(latent_rep)
        return x_hat, latent_rep, torch.tensor(0), torch.tensor(0), torch.tensor(0)
    
    
    

def isAnyNNWeightNan(NNmodel):
    print('hi')
    
# class DAE_LSTM(nn.Module):
#     def __init__(self, LD, num_features):
#         super(DAE_LSTM, self).__init__()

#         latent_dim = LD
#         i_channels = num_features
#         o_channels = i_channels
#         kernel_size = 100
#         kernel_size_last = 8
#         stride = int(kernel_size/2)

#         ## eGE.. Func
#         self.cnn1 = nn.Conv1d(i_channels, o_channels, kernel_size, stride=stride)
#         self.cnn2 = nn.Conv1d(i_channels, o_channels, kernel_size_last, stride=int(kernel_size_last/2))
#         self.fe1 = nn.Linear(o_channels, 10)
#         self.fe6 = nn.Linear(13, 6)
#         self.fe61 = nn.Linear(6, 3)
#         self.fe7 = nn.Linear(3, latent_dim)

#         self.fd8 = nn.Linear(latent_dim, 3)
#         self.fd9 = nn.Linear(3, 6)
#         self.fd10 = nn.Linear(6, 13)
#         self.fd16 = nn.Linear(10, i_channels)
        
#         self.decnn2 = nn.ConvTranspose1d(i_channels, o_channels, kernel_size, stride=stride)
#         self.decnn1 = nn.ConvTranspose1d(i_channels, o_channels, 1, stride=5)


#     #DAE encoder
#     def encode(self, x):

#         if (self.fe1.weight != self.fe1.weight).any():
#             pdb.set_trace()  
#         ## eGE
#         h1_cnn = self.cnn1(x)
#         h2_cnn = self.cnn2(h1_cnn)
#         x_input = torch.flatten(h2_cnn, start_dim=1)
#         h1 = F.relu (self.fe6(x_input))
#         # h6 = F.relu(self.fe6(h1))
#         h61 = F.relu(self.fe61(h1))
#         latent_rep = (self.fe7(h61))
        
#         if (latent_rep != latent_rep).any():
#             pdb.set_trace()   

#         return latent_rep

#     #DAE decoder
#     def decode(self, z):

#         ## eGE
        
#         h8 = (self.fd8(z))
#         h9 = F.relu(self.fd9(h8))
#         h10 = (self.fd10(h9))
#         # h16 = (self.fd16(h10))

#         x_output = torch.unsqueeze(h10, 2)
#         h1_cnn = self.decnn2(x_output)
#         h2_cnn = self.decnn1(h1_cnn)

        
#         return h2_cnn


#     def forward(self, x):
#         latent_rep = self.encode(x)
#         if (latent_rep != latent_rep).any():
#             pdb.set_trace()   
#         x_hat = self.decode(latent_rep)
#         if (x_hat != x_hat).any():
#             pdb.set_trace()   
#         return x_hat, latent_rep

class DAE_LSTM(nn.Module):
    def __init__(self, LD, num_features):
        super(DAE_LSTM, self).__init__()

        latent_dim = LD
        i_channels = num_features
        o_channels = i_channels
        kernel_size = 50
        kernel_size_last = 8
        stride = int(kernel_size/2)

        ## eGE.. Func
        self.cnn1 = nn.Conv1d(i_channels, o_channels, kernel_size, stride=stride)
        self.cnn2 = nn.Conv1d(i_channels, o_channels, kernel_size_last, stride=int(kernel_size_last/2))
        self.fe1 = nn.Linear(o_channels, 32)
        self.fe6 = nn.Linear(32, 16)
        self.fe61 = nn.Linear(16, 8)
        self.fe7 = nn.Linear(8, latent_dim)

        self.fd8 = nn.Linear(latent_dim, 8)
        self.fd9 = nn.Linear(8, 16)
        self.fd10 = nn.Linear(16, 32)
        self.fd16 = nn.Linear(32, i_channels)
        
        self.decnn2 = nn.ConvTranspose1d(i_channels, o_channels, kernel_size, stride=stride)
        self.decnn1 = nn.ConvTranspose1d(i_channels, o_channels, 1, stride=5)


    #DAE encoder
    def encode(self, x):

        if (self.fe1.weight != self.fe1.weight).any():
            pdb.set_trace()  
        ## eGE
        h1_cnn = F.relu(self.cnn1(x))
        h2_cnn = F.relu(self.cnn2(h1_cnn))
        x_input = F.relu(torch.flatten(h2_cnn, start_dim=1))
        h1 = F.relu(self.fe1(x_input))
        h6 = F.relu(self.fe6(h1))
        h61 = F.relu(self.fe61(h6))
        latent_rep = (self.fe7(h61))
        
        if (latent_rep != latent_rep).any():
            pdb.set_trace()   

        return latent_rep

    #DAE decoder
    def decode(self, z):

        ## eGE
        
        h8 = (self.fd8(z))
        h9 = F.relu(self.fd9(h8))
        h10 = (self.fd10(h9))
        h16 = (self.fd16(h10))

        x_output = F.relu(torch.unsqueeze(h16, 2))
        h1_cnn = F.relu(self.decnn2(x_output))
        h2_cnn = F.relu(self.decnn1(h1_cnn))

        
        return h2_cnn


    def forward(self, x):
        latent_rep = self.encode(x)
        if (latent_rep != latent_rep).any():
            pdb.set_trace()   
        x_hat = self.decode(latent_rep)
        if (x_hat != x_hat).any():
            pdb.set_trace()   
        return x_hat, latent_rep
    
    
class VAE(nn.Module):
    def __init__(self, LD, num_features):
        super(VAE, self).__init__()

        latent_dim = LD
        io_layer = num_features


        ## eGE.. Func
        self.fe1 = nn.Linear(io_layer, 40)
        self.fe6 = nn.Linear(40, 20)
        self.fe61 = nn.Linear(20, 10)
        self.fe7_mu = nn.Linear(10, latent_dim)
        self.fe7_var = nn.Linear(10, latent_dim)

        self.fd8 = nn.Linear(latent_dim, 10)
        self.fd9 = nn.Linear(10, 20)
        self.fd10 = nn.Linear(20, 40)
        self.fd16 = nn.Linear(40, io_layer)
        
        self.dropout = nn.Dropout(0.2)
        
        self.single_enc = nn.Linear(io_layer, latent_dim)
        self.single_dec = nn.Linear(latent_dim, io_layer)


    #DAE encoder
    def encode(self, x):

        # eGE
        h1 = F.relu(self.fe1(x))
        #h1 = self.dropout(h1)
        h6 = F.relu(self.fe6(h1))
        #h6 = self.dropout(h6)
        h61 = F.relu(self.fe61(h6))
        mu = (self.fe7_mu(h61))
        var = self.fe7_var(h61)
        return mu, var
        
        # z = self.single_enc(x)
        # return z

    #DAE decoder
    def decode(self, z):

        ## eGE
        h8 = (self.fd8(z))
        #h8 = self.dropout(h8)
        h9 = F.relu(self.fd9(h8))
        #h9 = self.dropout(h9)
        h10 = (self.fd10(h9))
        h16 = (self.fd16(h10))
        return h16
        
        # recon = self.single_dec(z)
        # return recon
    
    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # Sampling from standard normal dist
        return eps*std+mu # scaling the samples from standard normal dist.

    # ######################################################################
    # # Original training and testing
    # #####################################################################
    # def forward(self, x):
    #     mu, var = self.encode(x)
    #     z = self.reparametrization(mu, var)
    #     # z = mu
    #     x_hat = self.decode(z)
    #     return x_hat, z, x, mu, var
    
    ######################################################################
    # Feature importance study
    #####################################################################
    def forward(self, x):
        mu, var = self.encode(x)
        z = self.reparametrization(mu, var)
        # z = mu
        x_hat = self.decode(z)
        return x_hat

        

        
        