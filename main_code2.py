#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 11:06:40 2022

@author: sned

joint optim act val
"""


from architectures_TER import DAE
import pdb
import torch
from torch import nn, optim
import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
import readFeatureFilesAllDatasets as RFFall
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
from matplotlib import pyplot as plt
from audtorch.metrics import ConcordanceCC as ccc
from sklearn.manifold import TSNE

def clusteringLoss(z, labels):
    
    unique_labels = np.unique(labels)
    count = 0    
    D_c = 0
    eps = 1e-10
    centroid = torch.empty([len(unique_labels), z.shape[1]], dtype=torch.float64)
    for label_id in unique_labels:
        centroid[count, :] = torch.mean(z[np.where(labels==label_id), :], axis=1)
        tmp1 = ((z[np.where(labels==label_id), 0]-centroid[count, 0])**2)+((z[np.where(labels==label_id), 1]-centroid[count, 1])**2)
        D_c += torch.sum(torch.sqrt(tmp1+eps))
        # print('Dc Label: '+str(label_id)+'; '+str(tmp1.detach()))
        count += 1
 
    D_r = 0
    for centroid_idx in range(centroid.shape[0]-1):
        tmp2 = torch.sqrt(((centroid[centroid_idx+1:, 0]-centroid[centroid_idx, 0])**2)+((centroid[centroid_idx+1:, 1]-centroid[centroid_idx, 1])**2))
        D_r += torch.sum(torch.sqrt(tmp2+eps))
        # print('Dr centroid: '+str(centroid_idx)+'; '+str(tmp2.detach()))

    loss_cluster = (D_c+eps)/(D_r+eps)
    # print('Dc: '+str(D_c)+'; D_r: '+str(D_r))

    return loss_cluster

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    mse = 1
    if mse:
        mse_loss = nn.MSELoss()
        MSE = (mse_loss(recon_x, x))
 

    return  MSE


def getDeltaVectors(z, labels):
    eps = 1e-6
    simple = 0
    if simple:
        x = torch.empty([len(z)-1, 1])
        n = torch.empty([len(z)-1, 1])
        for idx in range(z.shape[0]-1):
            z1 = z[idx]
            z2 = z[idx+1]
            l1 = labels[idx]
            l2 = labels[idx+1]

            # num = torch.sqrt(sum((z1-z2)**2)+eps)/z.shape[1]
            # den = torch.sqrt(((l1-l2)**2)+eps)
            
            num = (sum(z1-z2)+eps)/z.shape[1]
            den = ((l1-l2)+eps)
    
            x[idx] = den
            n[idx] = num       
        
    else:
        
        vec_len = len(z)
        vec_tmp1 = np.arange(1, vec_len)
        vec_tmp2 = int(vec_len)*torch.ones([len(vec_tmp1)])
        vec_tmp3 = vec_tmp2-vec_tmp1
        vec_size = torch.sum(vec_tmp3, dtype=int)
        x = torch.empty([vec_size, 1])
        n = torch.empty([vec_size, 1])
        count = 0
        
        for idx in range(len(z)-1):
           
            z2_tmp = z[idx+1:]
            z1_tmp = z[idx]*torch.ones(len(z2_tmp), 1)
            # n[count:count+len(z2_tmp)] = torch.unsqueeze(torch.sqrt(torch.sum((z1_tmp-z2_tmp)**2, dim=1)+eps)/z.shape[1], dim=1)
            n[count:count+len(z2_tmp)] = torch.unsqueeze((torch.sum((z1_tmp-z2_tmp), dim=1)+eps)/z.shape[1], dim=1)            

            l2_tmp = labels[idx+1:]
            if len(l2_tmp.shape)<2:
                l2_tmp = torch.unsqueeze(l2_tmp, dim=1)
            l1_tmp = labels[idx]*torch.ones(len(l2_tmp), 1)
            x[count:count+len(l2_tmp)] = torch.unsqueeze((torch.sum((l1_tmp-l2_tmp), dim=1)+eps), dim=1)
            count = count+len(l2_tmp)
            
    return x, n
            

    
def customDistLoss2(z, labels):

    x, n = getDeltaVectors(z, labels)
    b = torch.inverse(torch.matmul(torch.transpose(x, 0, 1), x))*torch.matmul(torch.transpose(x, 0, 1), n)
    
    l0 = torch.min(x)
    l1 = torch.max(x)
    
    n_est = b*x
    res = torch.mean(((n_est-n))**2)
    
    n0 = b*l0
    n1 = b*l1
    
    slope_c = (n1-n0)/(l1-l0)
    
    loss_slope = torch.abs(slope_c-1)
    loss_res = res
    
    # loss =  loss_slope
    loss = loss_slope+loss_res
    return loss, loss_slope, loss_res

def customLossMetricActVal1(z, labels_act, labels_val):
    z_act = z[:, :int(z.shape[1]/2)]
    z_val = z[:, int(z.shape[1]/2):]

    loss_act, __, __ = customDistLoss2(z_act, labels_act)
    loss_val, __, __ = customDistLoss2(z_val, labels_val)
    
    loss = loss_act+loss_val
    
    return loss, loss_act, loss_val

def customDistLoss3(z, labels):

    l, e = getDeltaVectors(z, labels)
    l = l-torch.mean(l)
    e = e-torch.mean(e)
    matrix = np.cov(l.detach().numpy().T, e.detach().numpy().T)
    sl_comp = torch.tensor((abs(matrix[1, 1]/matrix[0, 0])-1)**2)
    var_comp = torch.tensor((abs(matrix[1, 0])/abs(matrix[0, 0])-1)**2)
  
       
    total_loss = sl_comp+var_comp
    return total_loss, total_loss, total_loss   

def customDistLoss4(z, labels):
    l, e = getDeltaVectors(z, labels)
    ccc_metric = ccc(batch_first=False)
    ccc_loss = ccc_metric(e, l)
    loss_to_min = torch.abs(1-torch.abs(ccc_loss))
    return loss_to_min, loss_to_min, loss_to_min

def embedding2Labelratio(z, labels):
    l, e = getDeltaVectors(z, labels)
    ELR = torch.mean(torch.log(torch.abs(torch.div(e, l)-1)))
    return ELR, ELR, ELR

   

def train_DAE(epoch, mse_gain, metric_gain, loss_name, metric):
    modelDAE.train()
    train_loss_epoch = 0
    mse_loss_epoch = 0
    res_loss_epoch = 0
    sl_loss_epoch = 0
    metric_loss_epoch = 0 
    elr_loss_epoch = 0
    batch_idx = 0
    count = 0
    for data, class_labels, val, act, __ in train_loader:
        data_noisy = data + NOISE_FACTOR * torch.randn(data.shape)
        data_noisy = data_noisy.to(device)
        data = data.to(device)
        optimizer_DAE.zero_grad()
        recon_batch, latent_rep_batch = modelDAE(data_noisy)
        mse_loss_batch = loss_function(recon_batch, data)

        if metric_gain == 1:
            if loss_name == 'Metric-cluster':
                metric_loss_batch = clusteringLoss(latent_rep_batch, class_labels)
            elif loss_name == 'Metric-act':
                metric_loss_batch, metric_loss_slope, metric_loss_res = customDistLoss2(latent_rep_batch, act)
                elr_loss_batch, __, __ = embedding2Labelratio(latent_rep_batch, act)
            elif loss_name == 'Metric-val':
                metric_loss_batch, metric_loss_slope, metric_loss_res = customDistLoss2(latent_rep_batch, val)
                elr_loss_batch, __, __ = embedding2Labelratio(latent_rep_batch, val)
            elif loss_name== 'ELR':
                metric_loss_batch, metric_loss_slope, metric_loss_res = embedding2Labelratio(latent_rep_batch, act)
            elif loss_name == 'Metric-ActVal':
                metric_loss_batch, metric_loss_slope, metric_loss_res = customLossMetricActVal1(latent_rep_batch, act, val)
                elr_loss_batch = 'Nan'
            else:
                print('ERROR')
            loss_batch = torch.log((mse_gain*mse_loss_batch) + (metric_gain*metric_loss_batch))
            metric_loss_epoch += metric_loss_batch.item()
        else:
            loss_batch = (mse_gain*mse_loss_batch) 
            metric_loss_epoch = np.nan
        loss_batch.backward()
        optimizer_DAE.step()
        batch_idx = batch_idx + 1
        count += 1
        train_loss_epoch += loss_batch.item()
        if mse_gain != 0:
            mse_loss_epoch += mse_loss_batch.item()
        if metric_gain == 'Nan':
            continue
        else:
            res_loss_epoch += metric_loss_res.item()
            sl_loss_epoch += metric_loss_slope.item()
        # elr_loss_epoch += elr_loss_batch.item()


    # print('====> DAE Train Epoch: {} Average loss: {:.6f}\tMSE: {:.6f}\tMetric: {:.6f}'.format(
    #       epoch, train_loss_epoch / len(train_loader.dataset), mse_loss_epoch/len(train_loader.dataset), metric_loss_epoch/len(train_loader.dataset)))
    # return train_loss_epoch/len(train_loader.dataset), mse_loss_epoch/len(train_loader.dataset), metric_loss_epoch/len(train_loader.dataset), res_loss_epoch/len(train_loader.dataset), sl_loss_epoch/len(train_loader.dataset), elr_loss_epoch/len(train_loader.dataset)

    print('====> DAE Train Epoch: {} Average loss: {:.6f}\tMSE: {:.6f}\tMetric: {:.6f}'.format(
          epoch, train_loss_epoch / count, mse_loss_epoch/count, metric_loss_epoch/count))
    return train_loss_epoch/count, mse_loss_epoch/count, metric_loss_epoch/count, res_loss_epoch/count, sl_loss_epoch/count, elr_loss_epoch/count

def test_DAE(epoch, mse_gain, metric_gain, loss_name, metric, path_to_save_recon):
    modelDAE.eval()
    test_loss_epoch = 0
    mse_loss_epoch = 0
    metric_loss_epoch = 0 

    with torch.no_grad():
        count = 0
        for data, class_labels, val, act, __ in valid_loader:
            data_noisy = data + NOISE_FACTOR * torch.randn(data.shape)
            data_noisy = data_noisy.to(device)
            data = data.to(device)
            recon_batch, z_test = modelDAE(data_noisy)
            mse_loss_batch = loss_function(recon_batch, data)
            if metric_gain == 1:
                if loss_name == 'Metric-cluster':
                    metric_loss_batch = clusteringLoss(z_test, class_labels)
                elif loss_name == 'Metric-act':
                    metric_loss_batch, __, __ = customDistLoss2(z_test, act)
                    elr_loss_batch = embedding2Labelratio(z_test, act)
                elif loss_name == 'Metric-val':
                    metric_loss_batch, __, __ = customDistLoss2(z_test, val)
                    elr_loss_batch = embedding2Labelratio(z_test, val)
                elif loss_name== 'ELR':
                    metric_loss_batch, __, __ = embedding2Labelratio(z_test, act)
                elif loss_name == 'Metric-ActVal':
                    metric_loss_batch, __, __ = customLossMetricActVal1(z_test, act, val)
                else:
                    print('ERROR')
                loss_batch = torch.log((mse_gain*mse_loss_batch) + (metric_gain*metric_loss_batch))
                metric_loss_epoch += metric_loss_batch.item()
            else:
                loss_batch = (mse_gain*mse_loss_batch) 
                metric_loss_epoch = np.nan
            test_loss_epoch += (loss_batch)
            mse_loss_epoch += mse_loss_batch
            count += 1
        
    # test_loss = test_loss_epoch.item()/len(valid_loader.dataset)
    test_loss = test_loss_epoch.item()/count
    if metric_gain == 1:
        # loss_metric =  metric_loss_epoch/len(valid_loader.dataset)
        loss_metric =  metric_loss_epoch/count
    else:
        loss_metric = np.nan
    # mse_loss = mse_loss_epoch.item()/len(valid_loader.dataset)
    mse_loss = mse_loss_epoch.item()/count

    plot_reconSigs = 1
    if plot_reconSigs:
        plot_every_x = 5
        if (epoch+1)%plot_every_x == 0:
            n = np.random.randint(0, len(data), 1)
            plot_data_true = data[n]
            plot_data_recon = recon_batch[n]
            t = np.arange(1, num_features+1, 1)
            plt.figure(figsize=(20, 10))
            plt.plot(t, plot_data_true.numpy()[0], t, plot_data_recon.numpy()[0])
            save_path = path_to_save_recon
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path+ str(epoch) + '.png')
            plt.close()
    
    print('====> DAE Test Epoch: {} Average loss: {:.6f}\tMSE: {:.6f}\tMetric: {:.6f}'.format(
    epoch, test_loss, mse_loss, loss_metric))

    return test_loss, mse_loss, loss_metric



def getLossEntry(dict_losses, fold, epoch, method, loss_name, loss_component, weight, loss, random_state_val):
    dict_losses['Loss-value'].append(loss)
    dict_losses['Fold'].append(fold)
    dict_losses['Epoch'].append(epoch)
    dict_losses['Method'].append(method)
    dict_losses['Loss-name'].append(loss_name)
    dict_losses['Loss-component'].append(loss_component)
    dict_losses['Random'].append(random_state_val)
    
    return dict_losses

def getLossesDataFrame(dict_losses, mse_weight, rank_weight, loss_name, method, fold, epoch, train_loss_epoch, valid_loss_epoch, mse_loss_train_epoch, mse_loss_valid_epoch, rank_loss_train_epoch, rank_loss_valid_epoch, random_state_val, res_loss_train_epoch, sl_loss_train_epoch, elr_loss_train_epoch):
    
   
    dict_loses = getLossEntry(dict_losses, fold, epoch, method, loss_name, 'Train', np.nan, train_loss_epoch, random_state_val)
    dict_loses = getLossEntry(dict_losses, fold, epoch, method, loss_name, 'Valid', np.nan, valid_loss_epoch, random_state_val)
    
    dict_loses = getLossEntry(dict_losses, fold, epoch, method, loss_name, 'MSE (T)', mse_weight, mse_loss_train_epoch, random_state_val)
    dict_loses = getLossEntry(dict_losses, fold, epoch, method, loss_name, 'MSE (V)', mse_weight, mse_loss_valid_epoch, random_state_val)
    
    dict_loses = getLossEntry(dict_losses, fold, epoch, method, loss_name, 'Metric (T)', rank_weight, rank_loss_train_epoch, random_state_val)
    dict_loses = getLossEntry(dict_losses, fold, epoch, method, loss_name, 'Metric (V)', rank_weight, rank_loss_valid_epoch, random_state_val)    
    
    dict_loses = getLossEntry(dict_losses, fold, epoch, method, loss_name, 'Residual (T)', rank_weight, res_loss_train_epoch, random_state_val)
    dict_loses = getLossEntry(dict_losses, fold, epoch, method, loss_name, 'Slope (T)', rank_weight, sl_loss_train_epoch, random_state_val)    
    dict_loses = getLossEntry(dict_losses, fold, epoch, method, loss_name, 'ELR (T)', rank_weight, elr_loss_train_epoch, random_state_val)    

    return dict_losses


def getPlots(z, labels, labels_val, num_epoch, save_in_folder, model_idx):
    
    if model_idx < 5:
        x, n = getDeltaVectors(z, labels)
        # x = x-torch.mean(x)
        # n = n-torch.mean(n)
        # b = torch.inverse(torch.matmul(torch.transpose(x, 0, 1), x))*torch.matmul(torch.transpose(x, 0, 1), n)
        # l0 = torch.min(x)
        # l1 = torch.max(x)    
        # n_est = b*x
        # n0 = b*l0
        # n1 = b*l1
        # slope_c = (n1-n0)/(l1-l0)
        # loss_slope = torch.abs(slope_c-1)
        # res = torch.mean((n_est-n)**2)
        
        plt.figure(figsize=(12, 3))
        plt.subplot(121)
        plt.scatter(x.detach(), n.detach())
        # plt.title('S: '+str(loss_slope)+'; R: '+str(res))
    
        x, n = getDeltaVectors(z, labels_val)
        # x = x-torch.mean(x)
        # n = n-torch.mean(n)
        # b = torch.inverse(torch.matmul(torch.transpose(x, 0, 1), x))*torch.matmul(torch.transpose(x, 0, 1), n)
        # l0 = torch.min(x)
        # l1 = torch.max(x)    
        # n_est = b*x
        # n0 = b*l0
        # n1 = b*l1
        # slope_c = (n1-n0)/(l1-l0)
        # loss_slope = torch.abs(slope_c-1)
        # res = torch.mean((n_est-n)**2)
        plt.subplot(122)
        plt.scatter(x.detach(), n.detach())
        # plt.title('S: '+str(loss_slope)+'; R: '+str(res))
    
    
        plt.savefig(save_in_folder+'/'+str(num_epoch)+'.png')
        plt.close()
    else:
        z_act = z[:, :int(z.shape[1]/2)]
        z_val = z[:, int(z.shape[1]/2):]
        
        x_act, n_act = getDeltaVectors(z_act, labels)
        # b = torch.inverse(torch.matmul(torch.transpose(x, 0, 1), x))*torch.matmul(torch.transpose(x, 0, 1), n)
        # l0 = torch.min(x)
        # l1 = torch.max(x)    
        # n_est = b*x
        # n0 = b*l0
        # n1 = b*l1
        # slope_c = (n1-n0)/(l1-l0)
        # loss_slope = torch.abs(slope_c-1)
        # res = torch.mean((n_est-n)**2)
        
        plt.figure(figsize=(12, 3))
        plt.subplot(121)
        plt.scatter(x_act.detach(), n_act.detach())
        # plt.title('S: '+str(loss_slope)+'; R: '+str(res))
    
        x_val, n_val = getDeltaVectors(z_val, labels_val)
        # b = torch.inverse(torch.matmul(torch.transpose(x, 0, 1), x))*torch.matmul(torch.transpose(x, 0, 1), n)
        # l0 = torch.min(x)
        # l1 = torch.max(x)    
        # n_est = b*x
        # n0 = b*l0
        # n1 = b*l1
        # slope_c = (n1-n0)/(l1-l0)
        # loss_slope = torch.abs(slope_c-1)
        # res = torch.mean((n_est-n)**2)
        plt.subplot(122)
        plt.scatter(x_val.detach(), n_val.detach())
        # plt.title('S: '+str(loss_slope)+'; R: '+str(res))
    
    
        plt.savefig(save_in_folder+'/'+str(num_epoch)+'.png')
        plt.close()
    

rand_state_list = [4, 13, 6, 47, 59]


parser = argparse.ArgumentParser(description='DAE for SER')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--device_local', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--latent_dim', type=int, default=2)
parser.add_argument('--classes', type=int, default= [4, 5], nargs='*') # , [0, 2, 4, 5], [0, 1, 2, 3, 4, 5, 6, 7, 9]
parser.add_argument('--num_features', type=int, default=88)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--folds', type=int, default=5)
parser.add_argument('--noise', type=float, default=1.0)

args = parser.parse_args()
args.cuda = not args.no_cuda
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Saving train and test set
feature = 'eGE'
if feature == 'eGE':
    tot_features = 88
    num_features = 88
elif feature == 'compare':
    tot_features = 6373
    num_features = 2000

num_labels = 4
feature_list = range(num_features)
fields_feature = [i for i in range(0, len(feature_list))]
fields_labels = [i for i in range(tot_features, tot_features+num_labels)]
fields = fields_feature + fields_labels
NOISE_FACTOR = 1.0

device_local = args.device_local
if device_local:
    base_folder_path = '/home/sned/work/WORK_NNF/speechPara/Transferable_emotion_Rep/concept_1/'
    tmp_var1 = pd.read_csv('/home/sned/work/WORK_NNF/speechPara/feature_files_global/allfunc_'+feature+'_iemocap_allLabels.csv', usecols=fields)
else:
    base_folder_path = '/zhome/6b/5/160076/paraspeech_global/Transferable_emotion_Rep/concept_1/'
    tmp_var1 = pd.read_csv('/zhome/6b/5/160076/paraspeech_global/feature_files_global/allfunc_'+feature+'_iemocap_allLabels.csv', usecols=fields)

    
y = tmp_var1[tmp_var1.columns[-4]].astype(int)
S = 15
emo_str = ''
for emo_cls in args.classes:
    emo_str = emo_str+str(emo_cls)

mse_weight_list = [1, 1, 1, 1, 1, 1]
metric_weight_list = ['Nan', 1, 1, 1, 1, 1]
metric_type_list = ['Nan', 'clus', 'act', 'val', 'elr', 'ActVal']
loss_name_list = ['Unsupervised', 'Metric-cluster', 'Metric-act', 'Metric-val', 'ELR', 'Metric-ActVal']
dict_losses = {'Loss-value':[], 'Method':[], 'Fold':[], 'Epoch':[], 'Loss-name':[], 'Loss-component':[], 'Random':[]}
model_idx_list = [0]  #[0, 1, 2, 3]
# for random_state_idx in range(len(rand_state_list)):
for random_state_idx in range(0,1):
 
    random_state_val = rand_state_list[random_state_idx]
    for fold, (train_idx_kfold, test_idx_kfold) in enumerate(StratifiedKFold(n_splits=args.folds, random_state=random_state_val, shuffle=True).split(tmp_var1, y)):
        idx_train = train_idx_kfold
        idx_test = test_idx_kfold
    
        folder_path = base_folder_path+'ld'+str(args.latent_dim)+'_emo'+emo_str+'/rand'+str(random_state_idx)+'/fold'+str(fold)+'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
        np.savetxt(folder_path+'fold'+str(fold)+'_train_file.csv', tmp_var1.iloc[idx_train, :], delimiter=',')
        np.savetxt(folder_path+'fold'+str(fold)+'_valid_file.csv', tmp_var1.iloc[idx_test, :], delimiter=',')
    
        feature_set_train = RFFall.ReadFeatureFileTrain(csv_file=folder_path+'fold'+str(fold)+'_train_file.csv',
                                                        class_labels=args.classes)
        train_loader = DataLoader(feature_set_train, batch_size = 64, shuffle=True, **kwargs)
    
        feature_set_valid = RFFall.ReadFeatureFileValid(csv_file=folder_path+'fold'+str(fold)+'_valid_file.csv',
                                                        mean_train=feature_set_train.mean,
                                                        std_train=feature_set_train.std,
                                                        class_labels=args.classes)
        valid_loader = DataLoader(feature_set_valid, batch_size=64, shuffle=True, **kwargs)
        
        # for model_idx in range(len(mse_weight_list)):
        for model_idx in model_idx_list:
            

            
            ### Model 1: Unsupervised
            mse_weight = mse_weight_list[model_idx]
            metric_weight = metric_weight_list[model_idx]
            metric_type = metric_type_list[model_idx]

            metric_loss_name = loss_name_list[model_idx]
            modelName = metric_loss_name             
            
            # Checking if already trained
            model_exist = folder_path+'models/'+modelName+'.pt'
            if os.path.exists(model_exist):
                continue
    
        
            modelDAE = DAE(LD=args.latent_dim, num_features=len(feature_list)).to('cpu')
            optimizer_DAE = optim.Adam(modelDAE.parameters(), lr=1e-4)  
            
            dict_losses = {'Loss-value':[], 'Method':[], 'Fold':[], 'Epoch':[], 'Loss-name':[], 'Loss-component':[], 'Random':[]}
            if __name__ == '__main__':
                for epoch in range(0, args.epochs):
                    
                    train_loss_epoch, mse_loss_train_epoch, metric_loss_train_epoch, res_loss_train_epoch, sl_loss_train_epoch, elr_loss_train_epoch = train_DAE(epoch, mse_weight, metric_weight, metric_loss_name, metric_type)
                
                    path_to_save_recon = folder_path+'recon/'+modelName+'/'
                    valid_loss_epoch, mse_loss_valid_epoch, metric_loss_valid_epoch = test_DAE(epoch, mse_weight, metric_weight, metric_loss_name, metric_type, path_to_save_recon)
                
                
                    
                    dict_losses = getLossesDataFrame(dict_losses, mse_weight, 
                                                     metric_weight, metric_loss_name, modelName, 
                                                     fold, epoch, train_loss_epoch, 
                                                     valid_loss_epoch, mse_loss_train_epoch, 
                                                     mse_loss_valid_epoch, metric_loss_train_epoch, 
                                                     metric_loss_valid_epoch, random_state_val, 
                                                     res_loss_train_epoch, sl_loss_train_epoch, elr_loss_train_epoch)  
                    
                    ##############  
                    
                    ### Plotting: plot every x
                    plot_every_x = 5
                    if (epoch+1)%plot_every_x == 0:
                        with torch.no_grad():
                            
                            if model_idx_list[0]<5:
                            
                                ### Train DAE
                                
                                data_z_train = np.zeros([args.num_features, len(feature_set_train.feature_vectors)])
                                label_train = np.zeros([len(feature_set_train.feature_vectors), 1])
                                val_train = np.zeros([len(feature_set_train.feature_vectors), 1])
                                act_train = np.zeros([len(feature_set_train.feature_vectors), 1])
                                dom_train = np.zeros([len(feature_set_train.feature_vectors), 1])
                                z_recon_train = np.zeros([args.latent_dim, len(feature_set_train.feature_vectors)])
                
                                count = 0
                                for idx in range(len(feature_set_train.feature_vectors)):
                                    data_tmp, label_tmp, val_tmp, act_tmp, dom_tmp = feature_set_train.__getitem__(idx)
                                    data_z_train[:, count] = data_tmp.numpy()
                                    label_train[count] = label_tmp.numpy()
                                    val_train[count] = val_tmp.numpy()
                                    act_train[count] = act_tmp.numpy()
                                    dom_train[count] = dom_tmp.numpy()
                                    
                                    z_recon_train[:, count] = modelDAE.encode(data_tmp+NOISE_FACTOR * torch.randn(data_tmp.shape))
                                    count += 1
                                
                                if args.latent_dim < 3:
                                    Z_embedding_train = z_recon_train.T
                                else:
                                    Z_embedding_train = TSNE(n_components=2, init='random', random_state=4).fit_transform(z_recon_train.T)
                
                                
                
                                fig = plt.figure(figsize=(12, 10))
                
                                ax = plt.subplot2grid((3, 2), (0, 0))
                                scatter = ax.scatter(Z_embedding_train[:, 0], Z_embedding_train[:,1], c=label_train, s=S, marker='*')
                                legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
                                ax = plt.subplot2grid((3, 2), (1, 0))
                                scatter = ax.scatter(Z_embedding_train[:, 0], Z_embedding_train[:,1], c=val_train, s=S, marker='*')
                                legend1 = ax.legend(*scatter.legend_elements(), title="Val")
                                ax = plt.subplot2grid((3, 2), (2, 0))
                                scatter = ax.scatter(Z_embedding_train[:, 0], Z_embedding_train[:,1], c=act_train, s=S, marker='*')
                                legend1 = ax.legend(*scatter.legend_elements(), title="Act")
                                
                                
                
                                #### Valid DAE
                                data_z_valid = np.zeros([args.num_features, len(feature_set_valid.feature_vectors)])
                                label_valid = np.zeros([len(feature_set_valid.feature_vectors), 1])
                                val_valid = np.zeros([len(feature_set_valid.feature_vectors), 1])
                                act_valid = np.zeros([len(feature_set_valid.feature_vectors), 1])
                                dom_valid = np.zeros([len(feature_set_valid.feature_vectors), 1])
                                z_recon_valid = np.zeros([args.latent_dim, len(feature_set_valid.feature_vectors)])
                                
                                count = 0
                                for idx in range(len(feature_set_valid.feature_vectors)):
                                    del data_tmp, label_tmp, val_tmp, act_tmp, dom_tmp
                                    data_tmp, label_tmp, val_tmp, act_tmp, dom_tmp = feature_set_valid.__getitem__(idx)
                                    data_z_valid[:, count] = data_tmp.numpy()
                                    label_valid[count] = label_tmp.numpy()
                                    val_valid[count] = val_tmp.numpy()
                                    act_valid[count] = act_tmp.numpy()
                                    dom_valid[count] = dom_tmp.numpy()
                                    
                                    z_recon_valid[:, count] = modelDAE.encode(data_tmp+ NOISE_FACTOR * torch.randn(data_tmp.shape))
                
                                    count += 1
                
                                Z_embedding_valid = z_recon_valid.T
                                if args.latent_dim < 3:
                                    Z_embedding_valid = z_recon_valid.T
                                else:
                                    Z_embedding_valid = TSNE(n_components=2, init='random', random_state=4).fit_transform(z_recon_valid.T)

                
                
                                ax = plt.subplot2grid((3, 2), (0, 1))
                                scatter = ax.scatter(Z_embedding_valid[:, 0], Z_embedding_valid[:,1], c=label_valid, s=S, marker='*')
                                legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
                                ax = plt.subplot2grid((3, 2), (1, 1))
                                scatter = ax.scatter(Z_embedding_valid[:, 0], Z_embedding_valid[:,1], c=val_valid, s=S, marker='*')
                                legend1 = ax.legend(*scatter.legend_elements(), title="Val")
                                ax = plt.subplot2grid((3, 2), (2, 1))
                                scatter = ax.scatter(Z_embedding_valid[:, 0], Z_embedding_valid[:,1], c=act_valid, s=S, marker='*')
                                legend1 = ax.legend(*scatter.legend_elements(), title="Act")
                            
                            else:
                                ### Train DAE
                                
                                data_z_train = np.zeros([args.num_features, len(feature_set_train.feature_vectors)])
                                label_train = np.zeros([len(feature_set_train.feature_vectors), 1])
                                val_train = np.zeros([len(feature_set_train.feature_vectors), 1])
                                act_train = np.zeros([len(feature_set_train.feature_vectors), 1])
                                dom_train = np.zeros([len(feature_set_train.feature_vectors), 1])
                                z_recon_train = np.zeros([args.latent_dim, len(feature_set_train.feature_vectors)])
                
                                count = 0
                                for idx in range(len(feature_set_train.feature_vectors)):
                                    data_tmp, label_tmp, val_tmp, act_tmp, dom_tmp = feature_set_train.__getitem__(idx)
                                    data_z_train[:, count] = data_tmp.numpy()
                                    label_train[count] = label_tmp.numpy()
                                    val_train[count] = val_tmp.numpy()
                                    act_train[count] = act_tmp.numpy()
                                    dom_train[count] = dom_tmp.numpy()
                                    
                                    z_recon_train[:, count] = modelDAE.encode(data_tmp+NOISE_FACTOR * torch.randn(data_tmp.shape))
                                    z_act_train = z_recon_train[:int(z_recon_train.shape[0]/2), :]
                                    z_val_train = z_recon_train[int(z_recon_train.shape[0]/2):, :]
                                    count += 1
                                
                                if args.latent_dim < 3:
                                    Z_embedding_train = z_recon_train.T
                                else:
                                    Z_embedding_train = TSNE(n_components=2, init='random', random_state=4).fit_transform(z_recon_train.T)
                                
                                fig = plt.figure(figsize=(12, 10))
                
                                ax = plt.subplot2grid((3, 2), (0, 0))
                                scatter = ax.scatter(Z_embedding_train[:, 0], Z_embedding_train[:,1], c=label_train, s=S, marker='*')
                                legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
                                ax = plt.subplot2grid((3, 2), (1, 0))
                                if args.latent_dim > 2:
                                    scatter = ax.scatter(z_val_train[0, :].T, z_val_train[1, :].T, c=val_train, s=S, marker='*')
                                else:
                                    scatter = ax.scatter(val_train, z_val_train.T, c=val_train, s=S, marker='*')
                                legend1 = ax.legend(*scatter.legend_elements(), title="Val")
                                ax = plt.subplot2grid((3, 2), (2, 0))
                                if args.latent_dim > 2:
                                    scatter = ax.scatter(z_act_train[0, :].T, z_act_train[1, :].T, c=act_train, s=S, marker='*')
                                else:
                                    scatter = ax.scatter(act_train, z_act_train.T, c=act_train, s=S, marker='*')
                                legend1 = ax.legend(*scatter.legend_elements(), title="Act")
                                
                                
                                #### Valid DAE
                                data_z_valid = np.zeros([args.num_features, len(feature_set_valid.feature_vectors)])
                                label_valid = np.zeros([len(feature_set_valid.feature_vectors), 1])
                                val_valid = np.zeros([len(feature_set_valid.feature_vectors), 1])
                                act_valid = np.zeros([len(feature_set_valid.feature_vectors), 1])
                                dom_valid = np.zeros([len(feature_set_valid.feature_vectors), 1])
                                z_recon_valid = np.zeros([args.latent_dim, len(feature_set_valid.feature_vectors)])
                                
                                count = 0
                                for idx in range(len(feature_set_valid.feature_vectors)):
                                    del data_tmp, label_tmp, val_tmp, act_tmp, dom_tmp
                                    data_tmp, label_tmp, val_tmp, act_tmp, dom_tmp = feature_set_valid.__getitem__(idx)
                                    data_z_valid[:, count] = data_tmp.numpy()
                                    label_valid[count] = label_tmp.numpy()
                                    val_valid[count] = val_tmp.numpy()
                                    act_valid[count] = act_tmp.numpy()
                                    dom_valid[count] = dom_tmp.numpy()
                                    
                                    z_recon_valid[:, count] = modelDAE.encode(data_tmp+ NOISE_FACTOR * torch.randn(data_tmp.shape))
                                    z_act_valid = z_recon_valid[:int(z_recon_valid.shape[0]/2), :]
                                    z_val_valid = z_recon_valid[int(z_recon_valid.shape[0]/2):, :]               
                                    count += 1
                
                                Z_embedding_valid = z_recon_valid.T
                                if args.latent_dim < 3:
                                    Z_embedding_valid = z_recon_valid.T
                                else:
                                    Z_embedding_valid = TSNE(n_components=2, init='random', random_state=4).fit_transform(z_recon_valid.T)

                
                
                                ax = plt.subplot2grid((3, 2), (0, 1))
                                scatter = ax.scatter(Z_embedding_valid[:, 0], Z_embedding_valid[:,1], c=label_valid, s=S, marker='*')
                                legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
                                ax = plt.subplot2grid((3, 2), (1, 1))
                                if args.latent_dim > 2:
                                    scatter = ax.scatter(z_val_valid[0, :].T, z_val_valid[1, :].T, c=val_valid, s=S, marker='*')
                                else:
                                    scatter = ax.scatter(val_valid, z_val_valid.T, c=val_valid, s=S, marker='*')
                                legend1 = ax.legend(*scatter.legend_elements(), title="Val")
                                ax = plt.subplot2grid((3, 2), (2, 1))
                                if args.latent_dim > 2:
                                    scatter = ax.scatter(z_act_valid[0, :].T, z_act_valid[1, :].T, c=act_valid, s=S, marker='*')
                                else:
                                    scatter = ax.scatter(act_valid, z_act_valid.T, c=act_valid, s=S, marker='*')
                                legend1 = ax.legend(*scatter.legend_elements(), title="Act")
                          
            
                            save_path = folder_path+'results_im/'+modelName+'/'
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            plt.savefig(save_path+'latentDim_'+str(epoch)+'.png')
                            plt.close()
                            
                            if 1:
                                path_to_save_scatter = folder_path+'LvsZ/'+modelName+'/'    
                                if not os.path.exists(path_to_save_scatter):
                                    os.makedirs(path_to_save_scatter)
                                getPlots(torch.from_numpy(Z_embedding_valid), torch.from_numpy(act_valid), torch.from_numpy(val_valid), epoch, path_to_save_scatter, model_idx)
                                        
                            ##############
                    
                save_model = 1
                if save_model:
                    save_path = folder_path+'models/'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(modelDAE, save_path+modelName+'.pt')      
                    
                loss_folder_path = folder_path+modelName
                df_losses = pd.DataFrame(dict_losses)
                df_losses.to_csv(loss_folder_path+'_Losses.csv')

                
                del modelDAE, optimizer_DAE, dict_losses
                

        
        


