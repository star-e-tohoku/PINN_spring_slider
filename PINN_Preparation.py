import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.autograd as autograd
import torch.nn as nn
import scipy.io
import matplotlib.pyplot as plt
from numpy import format_float_scientific as ffs

device = 'cpu'

class Training_Tensor:

    def __init__(self, time, vpl, statepl, v_ini, state_ini):
        self.time = time
        self.vpl = vpl
        self.statepl = statepl
        self.v_ini = v_ini
        self.state_ini = state_ini


    def parameters(self):
        Nt = self.time.shape[1]

        T = np.meshgrid(self.time)
        T = np.array(T) 
        t_test = np.hstack((T[0,:]))
        t_test = t_test.reshape([t_test.size,1]) 

        return t_test, Nt

    def trainingdata_spring(self, N_f):
        #Initial Condition t = 0
        initial_t = np.array(self.time[:, 0])[None]       

        initial_u = np.hstack((np.log(np.array([self.v_ini]) / self.vpl), np.log(np.array([self.state_ini]) / self.statepl)))
        initial_u = initial_u.reshape([1,2])

        t_ini_train = initial_t
        u_ini_train = initial_u

        t_f_train = self.time.T

        return t_ini_train, u_ini_train, t_f_train

    def Output(self):
        t_test, Nt = self.parameters()
        lb = t_test[0] # [0]
        ub = t_test[-1]# [T]
        t_ini_train_np, u_ini_train_np,t_f_train_np = self.trainingdata_spring(Nt)

        t_ini_train = torch.from_numpy(t_ini_train_np).double().to(device)
        u_ini_train = torch.from_numpy(u_ini_train_np).double().to(device)
        t_f_train =  torch.from_numpy(t_f_train_np).double().to(device)
        t_test_tensor = torch.from_numpy(t_test).double().to(device)
        f_hat = torch.zeros(t_f_train.shape[0],1).to(device)

        return Nt, ub, lb, t_ini_train, u_ini_train, t_f_train, t_test_tensor, f_hat

