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
from Inverse_Parameter import vpl

def Make_Synthetic(obs_error, obs_start, obs_end, obs_span, True_NN):
   
    t_datan = np.arange(obs_start, obs_end , obs_span)
    t_data = torch.from_numpy(t_datan.reshape([-1, 1])).double().to(device)
    u = True_NN.forward(t_data)
    p_true = torch.reshape(u[:,0],(-1 ,1)) #v
    v_true = vpl * torch.exp(p_true)

    np.random.seed(1234)
    noise = 1 + np.random.randn(len(v_true)) * obs_error
    noise = torch.from_numpy(noise.reshape([-1, 1])).double().to(device)    
    v_data = torch.mul(v_true, noise)
    
    return t_data, v_data

def Load_Solution(Sequentialmodel):
    layers = np.array([1,20,20,20,20,20,20,20,20,2])
    PINN = Sequentialmodel(layers).to(device)
    name = "PINN_Interpolated_Numerical_Solution"
    PINN.load_state_dict(torch.load(name + '.pth'))

    return layers, PINN