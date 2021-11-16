import sys
import os, errno
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#
#plt.ioff()

from scipy import optimize, stats
from src_py.metrics_utils import  calculate_deltas_signed

pathIN  = "temp_results/nn_a1rho_Variant-All_soft_weights_hits_c0s_Unweighted_False_NO_NUM_CLASSES_21_SIZE_1000_LAYERS_6/monit_npy/"
pathOUT = "figures_a1rho/"

calc_w  = np.load(pathIN+'softmax_calc_w.npy')
preds_w = np.load(pathIN+'softmax_preds_w.npy')

k2PI = 6.28
#----------------------------------------------------------------------------------
                           
i = 1
filename = "soft_wt_calc_preds_a1rho_Variant-All_nc_21_event_1"
x = np.arange(1,22)
plt.plot(x,calc_w[i]/sum(calc_w[i]), label='Generated')
plt.step(x,preds_w[i], label='Classification: wt')
plt.legend()
plt.ylim([0.0, 0.125])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
#plt.title('Features list: Variant-All')
    
if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()
plt.clf()
#----------------------------------------------------------------------------------
                           
i = 10
filename = "soft_wt_calc_preds_a1rho_Variant-All_nc_21_event_10"
x = np.arange(1,22)
plt.plot(x,calc_w[i]/sum(calc_w[i]), 'o', label='Generated')
plt.plot(x,preds_w[i], 'd', label='predicted')
plt.legend()
plt.ylim([0.0, 0.125])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
#plt.title('Features list: Variant-All')
    
if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()
plt.clf()

#----------------------------------------------------------------------------------
                           
i = 1000
filename = "soft_wt_calc_preds_a1rho_Variant-All_nc_21_event_1000"
x = np.arange(1,22)
x2 = np.arange(0,22)
plt.plot(x,calc_w[i]/sum(calc_w[i]), label='Generated')
plt.step(x2+0.5,np.append(preds_w[i][0],preds_w[i]), label='Classification: wt')
plt.legend()
plt.ylim([0.0, 0.1])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
#plt.title('Features list: Variant-All')
    
if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()
plt.clf()


#----------------------------------------------------------------------------------
                           
i = 2000
filename = "soft_wt_calc_preds_a1rho_Variant-All_nc_21_event_2000"
x = np.arange(1,22)
x2 = np.arange(0,22)
plt.plot(x,calc_w[i]/sum(calc_w[i]), label='Generated')
plt.step(x2+0.5,np.append(preds_w[i][0],preds_w[i]), label='Classification: wt')
plt.legend()
plt.ylim([0.0, 0.1])
plt.xlabel('Class index')
plt.xticks(x)
plt.ylabel(r'$wt^{norm}$')
#plt.title('Features list: Variant-All')
    
if filename:
    try:
        os.makedirs(pathOUT)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig(pathOUT + filename+".eps")
    print('Saved '+pathOUT + filename+".eps")
    plt.savefig(pathOUT + filename+".pdf")
    print('Saved '+pathOUT + filename+".pdf")
else:
    plt.show()
plt.clf()

