import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn as nn
import scipy.io
import matplotlib.pyplot as plt
from numpy import format_float_scientific as ffs

from Forward_Parameter import t1, t2, vpl, statepl

device = 'cpu'

class Result:
    
    def __init__(self, NN, loss_list, name):
        self.NN = NN
        self.loss_list = loss_list
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
 
    def plot_vtheta(self, save = False):
        PINN = self.NN
        
        t4outp, v_out, state_out = self.Input()

        fig = plt.figure(figsize = (10, 10 / 3), dpi = 300)
        fig.patch.set_facecolor('white')

        ax = fig.add_subplot(121)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y",useMathText=True)
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
        
        print("L    : ", ffs(loss[-1]  , 3, 2))
        print("Lode : ", ffs(loss_f[-1], 3, 2))
        print("Lini : ", ffs(loss_ini[-1], 3,2))
        print("Iteration : ", len(loss))

        fig = plt.figure(figsize = (5, 10/3), dpi = 300)
        fig.patch.set_facecolor('white')

        ax = fig.add_subplot(1, 1, 1)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y",useMathText=True)
        
        xiter = np.arange(0, len(loss_ini))

        ax.plot(xiter, loss_ini, label = r"$L_{ini}$", linewidth = 3, color = "tab:blue")
        ax.plot(xiter, loss_f, label = r"$ L_{ode}$", linewidth = 3, color = "tab:red")
        ax.set_yscale("log")

        ax.set_xlabel("Iteration", fontsize = 14)
        ax.set_ylabel("Loss", fontsize = 14)
        ax.legend(fontsize = "large") #loc = 'upper right'

        figname = self.name + '_loss.png'
        if(save):
            fig.savefig(figname ,bbox_inches="tight", pad_inches=0.05)
        else:
            plt.show()

        return
 