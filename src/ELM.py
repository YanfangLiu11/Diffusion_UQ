#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:05:59 2023

@author: yij
"""


import time
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from matplotlib.colors import LogNorm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)



eps_alpha = 0.05

def cond_alpha(t):
    return 1-(1-eps_alpha)*t

def cond_beta2(t):
    return eps_alpha + (1-eps_alpha)*t

def b(t):
    return -(1-eps_alpha)/cond_alpha(t)

def sigma_sq(t):
    sigma2=(1-eps_alpha) -2*b(t)*cond_beta2(t)
    return sigma2

def ODE_fun(t,zt,x_sample,log_weight_likelihood):
    log_weight_gauss= np.sum(-1.0*(zt[:,None,:]-cond_alpha(t)*x_sample[None,:,:])**2/(2*cond_beta2(t)),axis=2,keepdims=False)
    score_gauss = -1.0*(zt[:,None,:]-cond_alpha(t)*x_sample[None,:,:])/cond_beta2(t)
    weight_temp = np.exp(log_weight_gauss+log_weight_likelihood)
    weight = weight_temp/ np.sum(weight_temp,axis=1, keepdims=True)
    
    score = np.sum(score_gauss*weight[:,:,None],axis=1, keepdims= False) 
    return b(t)*zt-0.5*sigma_sq(t)*score

T=1.0
time_steps = 100
t_vec = np.linspace(T,0, time_steps+1)
def ODE_solver(t_vec,zt,obs_sample,x_sample,obs_train,obs_var,time_steps):
    log_weight_likelihood = np.sum(-1.0*(obs_train[:,None,:]-obs_sample[None,:,:])**2/(2*obs_var[None,None,:]),axis=2, keepdims=False)
    dt = t_vec[1] - t_vec[0]
    for j in range(time_steps): 
        t=t_vec[j]
        k1 = ODE_fun(t, zt, x_sample, log_weight_likelihood)
        k2 = ODE_fun(t+0.5*dt, zt+0.5*dt*k1, x_sample, log_weight_likelihood)
        k3 = ODE_fun(t+0.5*dt, zt+0.5*dt*k2, x_sample, log_weight_likelihood)
        k4 = ODE_fun(t+dt, zt+dt*k3, x_sample, log_weight_likelihood)
        zt= zt + (k1+2*k2+2*k3+k4)*dt/6
        # print(j)
    return zt


# Define the architecture of the  neural network
class FN_Net(nn.Module):
    
    def __init__(self, input_dim, output_dim, hid_size):
        super(FN_Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_size = hid_size
        
        self.input = nn.Linear(self.input_dim,self.hid_size)
        self.fc1 = nn.Linear(self.hid_size,self.hid_size)
        self.output = nn.Linear(self.hid_size,self.output_dim)

        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)
    
    def forward(self,x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = self.output(x)
        return x

    def update_best(self):

        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)

    def final_update(self):

        self.input.weight.data = self.best_input_weight 
        self.input.bias.data = self.best_input_bias
        self.fc1.weight.data = self.best_fc1_weight
        self.fc1.bias.data = self.best_fc1_bias
        self.output.weight.data = self.best_output_weight
        self.output.bias.data = self.best_output_bias




#================================================
# ----------------------------------------
#             Dan's example
# ----------------------------------------
inx = 974

Npar = 8
Nout = 5
outinx = 18
Ns = 1000

file_path = "/path/to/ELM/"
# Check if the directory at file_path exists
if os.path.exists(file_path):
    print("Directory exists.")
    
    # Load 'Ex3OU_train.npy' from the specified directory
    data = np.loadtxt(os.path.join(file_path, 'ELM_out15.dat'))



obs_sample = np.asarray(data[:,outinx:outinx+Nout])
x_sample = np.asarray(data[:,0:Npar])

#----------------------------------------
#           Data augmentation
#----------------------------------------
data_sample = np.concatenate((x_sample, obs_sample), axis=1)
x_mean = np.mean(x_sample, axis= 0)
x_std = np.std(x_sample, axis= 0)
x_normalized = (x_sample - x_mean)/x_std

obs_mean = np.mean(obs_sample, axis= 0)
obs_std = np.std(obs_sample, axis= 0)
obs_normalized = (obs_sample - obs_mean)/obs_std
data_dim = Npar + Nout




t1_LD_sample = time.time()
#================================================
train_size = 10000
obs_var = (1/obs_std)**2
zT=np.random.randn(train_size,Npar)
obs_normalized_train = np.random.randn(train_size,Nout)


x_normalized_hat = ODE_solver(t_vec,zT,obs_normalized,x_normalized,obs_normalized_train,obs_var,time_steps)

t2_LD_sample = time.time()
print('Labeled data generation time: ', (t2_LD_sample-t1_LD_sample)/60 , 'mins')

# x_hat = x_normalized_hat*x_std+x_mean
# for jj in range(Npar):
#     plt.hist(x_sample[:,jj], bins =20)
#     # plt.hist(x_hat[:,jj], bins =5)
#     plt.show()    




# plt.figure(figsize=(4*Npar, 4))
# plt.subplots_adjust(hspace=0.06)
# from scipy.stats import gaussian_kde
# from scipy.stats import entropy
# for jj in range(Npar):
#     ax = plt.subplot(1, Npar, jj + 1)

#     # Plot fit curve (PDF)
#     kde_train = gaussian_kde(x_normalized[:, jj])
#     kde_observed = gaussian_kde(x_normalized_hat[:, jj])

#     x_vals = np.linspace(min(x_normalized[:, jj].min(), x_normalized_hat[:, jj].min()),
#                           max(x_normalized[:, jj].max(), x_normalized_hat[:, jj].max()),
#                           100)

#     ax.plot(x_vals, kde_train(x_vals), color='blue', label='Train')
#     ax.plot(x_vals, kde_observed(x_vals), color='red', label='Observed')

#     ax.legend()
    
#     kl_divergence = entropy(kde_train(x_vals), qk = kde_observed(x_vals))
#     print("KL Divergence: ", kl_divergence)
# plt.show()



xTrain = np.hstack((obs_normalized_train,zT))
yTrain = x_normalized_hat
xTrain = torch.tensor(xTrain).to(device, dtype=torch.float32)
yTrain = torch.tensor(yTrain).to(device, dtype=torch.float32)


t1_FN_sample = time.time()

# Generate random indices for shuffling
indices = torch.randperm(train_size)

# Shuffle xTrain and yTrain using the generated indices
xTrain_shuffled = xTrain[indices,:]
yTrain_shuffled = yTrain[indices,:]


#==============================================================================
# train NN to learn F(x_0,z)=hat(x_{0.1}) or say F(x_train,z)=y_hat
#==============================================================================
NTrain = int(train_size* 0.8)
NValid = int(train_size * 0.2)

xValid_normal = xTrain_shuffled [NTrain:,:]
yValid_normal = yTrain_shuffled [NTrain:,:]

xTrain_normal = xTrain_shuffled [:NTrain,:]
yTrain_normal = yTrain_shuffled [:NTrain,:]

learning_rate = 0.01
FN_new = FN_Net(Npar + Nout, Npar,5).to(device)
FN_new.zero_grad()
optimizer = optim.Adam(FN_new.parameters(), lr = learning_rate, weight_decay=1e-6)
criterion = nn.MSELoss()

best_valid_err = 5.0
n_iter = 2000
for j in range(n_iter):
    optimizer.zero_grad()
    pred = FN_new(xTrain_normal)
    loss = criterion(pred,yTrain_normal)
    loss.backward()
    optimizer.step()

    pred1 = FN_new(xValid_normal)
    valid_loss = criterion(pred1,yValid_normal)
    if valid_loss < best_valid_err:
        FN_new.update_best()
        best_valid_err = valid_loss

    if j%100==0:
        print(j,loss,valid_loss)

FN_new.final_update()

t2_FN_sample = time.time()
print('Training time: ', (t2_FN_sample-t1_FN_sample)/60 , 'mins')

FN_path = os.path.join(file_path, 'FN_new.pth')
torch.save(FN_new.state_dict(), FN_path)


#=======================================
# result of NN

inx = 974 
# inx= np.random.randint(1000)
print('index is ', inx)
x_true = x_sample[inx,:]
print('synthetic true parameter values are: ', x_true)

obs_test = obs_sample[inx,:]
print('synthetic true obervations are: ', obs_test)
obs_normalized_test = obs_normalized[inx,:]

t1_NN_sample = time.time()

Npath = 2000
test = torch.tensor(obs_normalized_test*np.ones((Npath,Nout))).to(device, dtype=torch.float32)
test_py= torch.hstack((test,torch.randn(Npath,Npar).to(device, dtype=torch.float32)))
prediction = FN_new(test_py).to('cpu').detach().numpy()

t2_NN_sample = time.time()
print('NN data generation time: ', (t2_NN_sample-t1_NN_sample) , 'seconds')

#=======================================

plt.figure(figsize=(9*4,4))
plt.subplots_adjust(hspace=0.1)
# plt.suptitle("marginal pdf", fontsize=18, y=0.95)
labels = ['rootb_par','slatop','flnr','frootcn','froot_leaf','br_mr','crit_dayl','crit_onset_gdd', 'slatop']

N_interp = 200
map_estimator=np.zeros((1,Npar))
for ii, label in enumerate(labels):

    ax = plt.subplot(1,9,ii+1)
    
    if ii <=7 :
        y_pred_ii = prediction[:,ii] * x_std[ii]+ x_mean[ii]
        kernel = stats.gaussian_kde(y_pred_ii)
        grid_plot = np.linspace(y_pred_ii.min()*0.6,y_pred_ii.max()*1.2,N_interp)
        grid_pdf = kernel(grid_plot)
        ax.fill_between(grid_plot,grid_pdf,alpha=0.4 , color = 'green')
        ax.axvline(x_true[ii], color = 'r',linewidth=2, label = ' "True" ' )
        
        map_estimator_temp = grid_plot[np.argmax(grid_pdf)]
        ax.axvline(map_estimator_temp, linewidth=2, color= 'blue', label= 'MAP')
        
        map_estimator[0,ii] = map_estimator_temp
        ax.set_xlabel(label)
        ax.set_ylim(0,None)
        ax.set_yticks([])
        
        # ax.axvline(np.mean(y_pred_ii) , linestyle='-.',linewidth=2 , color= 'green', label= 'Mean')
    if ii == 8:
        y_pred_slatop = prediction[:,1] * x_std[1]+ x_mean[1]
        y_pred_flnr = prediction[:,2] * x_std[2]+ x_mean[2]
        slatop_vals = np.linspace(min(y_pred_slatop), max(y_pred_slatop), N_interp)
        flnr_vals = np.linspace(min(y_pred_flnr), max(y_pred_flnr), N_interp)
        slapton_grid, flnr_grid = np.meshgrid(slatop_vals,flnr_vals)
        
        kernel = stats.gaussian_kde(np.vstack([y_pred_slatop,y_pred_flnr]))
        sf_grid = np.vstack([slapton_grid.ravel(), flnr_grid.ravel()])
        grid_pdf=kernel(sf_grid).reshape(slapton_grid.shape)
        

        # Create a custom colormap with one color (white)
        cmap =  plt.cm.viridis
        min_draw_value= 65.0
        max_draw_value=600.0
        
        # Use levels to ensure there's no contour line
        levels = np.linspace(min_draw_value, max_draw_value, 20)
        contour = ax.contour(slapton_grid, flnr_grid, grid_pdf, cmap=cmap, linewidths=1, alpha=0.4, levels=levels, norm=LogNorm())
        contourf = ax.contourf(slapton_grid, flnr_grid, grid_pdf, cmap=cmap, alpha=0.4, levels=levels , norm=LogNorm())
        
        ax.set_xlabel('slatop')
        ax.set_ylabel('flnr')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
       
        ax.scatter(x_true[1], x_true[2], marker='*',color='red', s=100, label=' "True" ')
        ax.scatter(map_estimator[0,1], map_estimator[0,2], color='blue',s=50,  label='MAP')
        ax.set_xticks([0.025, 0.05])
        ax.set_yticks([0.1, 0.2, 0.3])
        ax.legend(loc= 'upper left')
        
        ax.set_xlim([0, 0.075])
        ax.set_ylim([0, 0.4])
    
    
    if ii == 0:
        ax.legend(loc= 'upper right')

plt.savefig(f'./marginal_new_inx{inx}.png', bbox_inches="tight", dpi=300)




#========================================
# performance evaluation 
nrmse = np.sqrt(np.sum((x_true-map_estimator)**2)/np.sum(x_true**2))
print('NRMSE: ', nrmse)

rsq = 1- np.sum( (x_true - map_estimator)**2)/ np.sum((x_true - x_mean)**2 )
print("R$^2$: ", rsq)


# NRMSE:  0.005683375328618241
# R$^2$:  0.9119104469588628




#=======================================
# result of NN

inx = 988
# inx= np.random.randint(1000)
print('index is ', inx)
x_true = x_sample[inx,:]
print('synthetic true parameter values are: ', x_true)

obs_test = obs_sample[inx,:]
print('synthetic true obervations are: ', obs_test)
obs_normalized_test = obs_normalized[inx,:]

Npath = 2000
test = torch.tensor(obs_normalized_test*np.ones((Npath,Nout))).to(device, dtype=torch.float32)
test_py= torch.hstack((test,torch.randn(Npath,Npar).to(device, dtype=torch.float32)))
prediction = FN_new(test_py).to('cpu').detach().numpy()


#=======================================

plt.figure(figsize=(9*4,4))
plt.subplots_adjust(hspace=0.1)
# plt.suptitle("marginal pdf", fontsize=18, y=0.95)
labels = ['rootb_par','slatop','flnr','frootcn','froot_leaf','br_mr','crit_dayl','crit_onset_gdd', 'slatop']

N_interp = 200
map_estimator=np.zeros((1,Npar))
for ii, label in enumerate(labels):

    ax = plt.subplot(1,9,ii+1)
    
    if ii <=7 :
        y_pred_ii = prediction[:,ii] * x_std[ii]+ x_mean[ii]
        kernel = stats.gaussian_kde(y_pred_ii)
        grid_plot = np.linspace(y_pred_ii.min()*0.6,y_pred_ii.max()*1.2,N_interp)
        grid_pdf = kernel(grid_plot)
        ax.fill_between(grid_plot,grid_pdf,alpha=0.4 , color = 'green')
        ax.axvline(x_true[ii], color = 'r',linewidth=2, label = ' "True" ' )
        
        map_estimator_temp = grid_plot[np.argmax(grid_pdf)]
        ax.axvline(map_estimator_temp, linewidth=2, color= 'blue', label= 'MAP')
        
        map_estimator[0,ii] = map_estimator_temp
        ax.set_xlabel(label)
        ax.set_ylim(0,None)
        ax.set_yticks([])
        
        # ax.axvline(np.mean(y_pred_ii) , linestyle='-.',linewidth=2 , color= 'green', label= 'Mean')
    if ii == 8:
        y_pred_slatop = prediction[:,1] * x_std[1]+ x_mean[1]
        y_pred_flnr = prediction[:,2] * x_std[2]+ x_mean[2]
        slatop_vals = np.linspace(min(y_pred_slatop), max(y_pred_slatop), N_interp)
        flnr_vals = np.linspace(min(y_pred_flnr), max(y_pred_flnr), N_interp)
        slapton_grid, flnr_grid = np.meshgrid(slatop_vals,flnr_vals)
        
        kernel = stats.gaussian_kde(np.vstack([y_pred_slatop,y_pred_flnr]))
        sf_grid = np.vstack([slapton_grid.ravel(), flnr_grid.ravel()])
        grid_pdf=kernel(sf_grid).reshape(slapton_grid.shape)
        

        # Create a custom colormap with one color (white)
        cmap =  plt.cm.viridis
        min_draw_value= 65.0
        max_draw_value=600.0
        
        # Use levels to ensure there's no contour line
        levels = np.linspace(min_draw_value, max_draw_value, 20)
        contour = ax.contour(slapton_grid, flnr_grid, grid_pdf, cmap=cmap, linewidths=1, alpha=0.4, levels=levels, norm=LogNorm())
        contourf = ax.contourf(slapton_grid, flnr_grid, grid_pdf, cmap=cmap, alpha=0.4, levels=levels , norm=LogNorm())
        
        ax.set_xlabel('slatop')
        ax.set_ylabel('flnr')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
       
        ax.scatter(x_true[1], x_true[2], marker='*',color='red', s=100, label=' "True" ')
        ax.scatter(map_estimator[0,1], map_estimator[0,2], color='blue',s=50,  label='MAP')
        ax.set_xticks([0.025, 0.05])
        ax.set_yticks([0.1, 0.2, 0.3])
        ax.legend(loc= 'upper left')
        
        ax.set_xlim([0, 0.075])
        ax.set_ylim([0, 0.4])
    
    
    if ii == 0:
        ax.legend(loc= 'upper right')

plt.savefig(f'./marginal_new_inx{inx}.png', bbox_inches="tight", dpi=300)




#========================================
# performance evaluation 
nrmse = np.sqrt(np.sum((x_true-map_estimator)**2)/np.sum(x_true**2))
print('NRMSE: ', nrmse)

rsq = 1- np.sum( (x_true - map_estimator)**2)/ np.sum((x_true - x_mean)**2 )
print("R$^2$: ", rsq)

test_data= np.loadtxt(os.path.join(file_path, 'ELM_obs.dat'))
test_obs = test_data[:,4]
test_std = test_data[:,5]