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
from Inverse_Parameter import a, a_b, dc

def Save_Loss_forward(PINN, loss_list):
    loss_list["total"].append(PINN.loss_hist[-1])
    loss_list["ini"].append(PINN.lossini_hist[-1]) 
    loss_list["ode"].append(PINN.lossf_hist[-1])

def Save_PINN_forward(PINN, name, loss_list):
    model = PINN.to(device)
    torch.save(model.state_dict(), name + '.pth')

    f = h5py.File(name + '_loss.h5', 'w')
    f["loss"] = loss_list["total"]
    f["loss_ini"] = loss_list["ini"]
    f["loss_f"] = loss_list["ode"]
    f.close()

    print("NN is saved")

def Save_Loss_inverse(PINN, loss_list, fp_list, Ests):
    ap = torch.exp(Ests[0])
    a_bp = - torch.exp(Ests[1])
    dcp = torch.exp(Ests[2])

    fp_list["a"].append(ap.item() / a)
    fp_list["a-b"].append(a_bp.item() / a_b)
    fp_list["dc"].append(dcp.item() / dc)

    loss_list["total"].append(PINN.loss_hist[-1])
    loss_list["ini"].append(PINN.lossini_hist[-1]) 
    loss_list["ode"].append(PINN.lossf_hist[-1])
    loss_list["data"].append(PINN.lossd_hist[-1])

def Save_PINN_inverse(PINN, name, loss_list, fp_list):
    model = PINN.to(device)
    torch.save(model.state_dict(), name + '.pth')

    f = h5py.File(name + '_loss.h5', 'w')
    f["loss"] = loss_list["total"]
    f["loss_ini"] = loss_list["ini"]
    f["loss_f"] = loss_list["ode"]
    f["loss_data"] = loss_list["data"]

    f["a_list"] = fp_list["a"]
    f["a-b_list"] = fp_list["a-b"]
    f["dc_list"] = fp_list["dc"]
    f.close()

    print("NN is saved")