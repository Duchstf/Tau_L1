
'''
Plot the efficiency of a given model.
'''

#Import everything needed
import uproot4
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential, load_model

from keras import optimizers
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv1D
from qkeras.utils import load_qmodel

import hist
from hist import Hist

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import mplhep as hep
plt.style.use(hep.style.ROOT)

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

#line thickness
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 5
    
def main():
    
    model_path = '../../../models/quantized_merged_pruned_gamma30.h5'
    data_path = '../../samples/14_0_0_pre3.root'
    global plot_name
    plot_name='plots/signal_efficiency_pt_14_0_0_pre3.pdf'
    TreeName= 'ntuplePupSingle'
    corrected_pt_cut= 0
    cut_point = 0.22
    
    #Load signal data
    sig = uproot4.open(data_path)
    
    #Load model and select out the tau
    #0 index is id, 1 is pT regression
    model = load_qmodel(model_path)
    
    
    sig_input = sig[TreeName]['tree']['m_inputs'].array()
    tau_pt = sig[TreeName]['tree']['pt'].array()
    tau_eta = sig[TreeName]['tree']['eta'].array()
    tau_phi = sig[TreeName]['tree']['phi'].array()
    
    CMSSW_id_output =  sig[TreeName]['tree']['chargedIso'].array()
    nn_output = model.predict(np.asarray(sig_input))
    nn_id_output = nn_output[0].flatten()
    nn_pt_output = nn_output[1].flatten()
    
    for i in range(10):
        print("Event {}".format(i))
        print("Tau pT: {}, eta: {}, phi: {}".format(tau_pt[i], tau_eta[i], tau_phi[i]))
        print("CMSSW ID : ", CMSSW_id_output[i])
        print("NN ID: {}. NN pT correction: {}".format(nn_id_output[i], nn_pt_output[i]))
        print("-----------------")
    
    
main()