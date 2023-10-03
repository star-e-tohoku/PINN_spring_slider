import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn as nn
import scipy.io
import matplotlib.pyplot as plt
from numpy import format_float_scientific as ffs

from Inverse_Parameter import t1, t2, vpl, statepl, a, a_b, dc

device = 'cpu'

class Result:
    
    def __init__(self, NN, loss_list, fp_list, name):
        self.NN = NN
        self.loss_list = loss_list
        self.fp_list = fp_list
        self.name = name
        
    def Input(self):
        PINN = self.NN
        output_span = 10 * 3600
        t4outp = np.arange(t1, t2, output_span) - t1
        t4out = t4outp.reshape([t4outp.size, 1])
        t4out = torch.from_numpy(t4out).double()

        upred = PINN.forward(t4out)
        upred = upred.cpu().detach().numpy()
        Nt = len(t4outp)
        upred = np.reshape(upred,(Nt,2),order='F')

        t4outp = t4outp / (3600 * 24)
        p_out = upred[:, 0]
        q_out = upred[:, 1]
        v_out = vpl * np.exp(p_out)
        state_out = statepl * np.exp(q_out)

        return t4outp, v_out, state_out

    def plot_vtheta(self, t_data, v_data, save = False):
        PINN = self.NN
        t_datan = t_data.detach().numpy()[:, 0]
        v_datan = v_data.detach().numpy()[:, 0]        
        t4outp, v_out, state_out = self.Input()

        fig = plt.figure(figsize = (10, 10 / 3), dpi = 300)
        fig.patch.set_facecolor('white')

        ax = fig.add_subplot(121)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y",useMathText=True)
        ax.scatter(t_datan / (3600 * 24), v_datan, s = 5, color = "tab:blue", label = "Data", zorder = 4, alpha = 1)        
        ax.plot(t4outp, v_out, label = "PINN", color = "tab:red", linewidth = 3)
        ax.set_yscale('log')
        ax.set_xlabel("t [day]")
        ax.set_ylabel(r"$v$ [m/s]")
        ax.legend(fontsize = "large")

        ax2 = fig.add_subplot(122)
        ax2.ticklabel_format(style="sci", scilimits=(0,0), axis="y",useMathText=True)
        ax2.plot(t4outp, state_out, label = "PINN", color = "tab:red", linewidth = 3)
        ax2.set_yscale('log')
        ax2.set_xlabel("t [day]")
        ax2.set_ylabel(r"$\theta$ [s]")

        figname = self.name + "_vtheta.png"
        plt.tight_layout()

        if(save):
            fig.savefig(figname ,bbox_inches="tight", pad_inches=0.05)
        else:
            plt.show()

    def plot_loss(self, save = False):
        loss = self.loss_list["total"]
        loss_ini = self.loss_list["ini"]
        loss_f = self.loss_list["ode"]
        loss_data = self.loss_list["data"]
        
        print("L    : ", ffs(loss[-1]  , 3, 2))
        print("Lode : ", ffs(loss_f[-1], 3, 2))
        print("Lini : ", ffs(loss_ini[-1], 3,2))
        print("Ldata: ", ffs(loss_data[-1], 3, 2))
        print("Iteration : ", len(loss))

        fig = plt.figure(figsize = (5, 10/3), dpi = 300)
        fig.patch.set_facecolor('white')

        ax = fig.add_subplot(1, 1, 1)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y",useMathText=True)
        
        xiter = np.arange(0, len(loss_ini))

        ax.plot(xiter, loss_ini, label = r"$L_{ini}$", linewidth = 3, color = "tab:blue")
        ax.plot(xiter, loss_f, label = r"$ L_{ode}$", linewidth = 3, color = "tab:red")
        ax.plot(xiter, loss_data, label = r"$ \: L_{data}$", linewidth = 3, color = "tab:green")
        ax.set_yscale("log")

        ax.set_xlabel("Iteration", fontsize = 14)
        ax.set_ylabel("Loss", fontsize = 14)
        ax.legend(fontsize = "large")

        figname = self.name + '_loss.png'
        if(save):
            fig.savefig(figname ,bbox_inches="tight", pad_inches=0.05)
        else:
            plt.show()

        return
    
    def plot_parameter(self, save = False):
        fp_list = self.fp_list
        precision = 3
        esta   = ffs(a   * fp_list["a"][-1]  , precision, 1)
        esta_b = ffs(a_b * fp_list["a-b"][-1], precision, 1)
        estdc  = ffs(dc  * fp_list["dc"][-1] , precision, 1)
        truea  =  ffs(a, 3, 1);   truea_b = ffs(a_b, 3, 1);    truedc = ffs(dc, 3, 1)
        inia = ffs(a   * fp_list["a"][0]  , 3, 1)
        inia_b = ffs(a_b * fp_list["a-b"][0], 3, 1)
        inidc  = ffs(dc  * fp_list["dc"][0] , 3, 1)
        
        print("Estimated a  : ", esta, " True a  : ", truea, " Ini a  : ", inia)
        print("Estimated a-b: ", esta_b,"True a-b: ",truea_b,"Ini a-b: ", inia_b)
        print("Estimated dc : ", estdc," True dc : ", truedc," Ini dc : ", inidc) 
        
        fig = plt.figure(figsize = (5, 10/3), dpi = 300)
        fig.patch.set_facecolor('white')

        ax = fig.add_subplot(1, 1, 1)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y",useMathText=True)
        
        xiter = np.arange(0, len(fp_list["a"]))
        xx1 = np.array([0, xiter[-1]])
        yy1 = np.array([1, 1])
        
        ax.plot(xx1, yy1, linestyle = "dashed")
        ax.plot(xiter, fp_list["a"], label = r"$a$", linewidth = 3, color = "tab:red")
        ax.plot(xiter, fp_list["a-b"], label = r"$a - b$", linewidth = 3, color = "tab:blue")
        ax.plot(xiter, fp_list["dc"], label = r"$d_{c}$",  linewidth = 3, color = "tab:green")
        ax.set_yscale("log")

        ax.set_xlabel("Iteration", fontsize = 14)
        ax.set_ylabel("Estimated Value / True Value", fontsize = 14)
        ax.legend(fontsize = "medium", loc = 'upper right')

        figname = self.name + '_estimation.png'
        if(save):
            fig.savefig(figname ,bbox_inches="tight", pad_inches=0.05)
        else:
            plt.show()