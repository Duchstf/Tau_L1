'''
Plot the pt resoultion in bins of pt for a given model.
'''

import uproot4
import numpy as np
import awkward as ak
from scipy.stats import norm
from scipy.optimize import curve_fit
import os
import copy

import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential, load_model

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

import optparse
import importlib
import pathlib
from keras import optimizers
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv1D
from qkeras.utils import load_qmodel

import hist
from hist import Hist

import random

import matplotlib.pyplot as plt
import matplotlib.colors as colors
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

def average(number1, number2):
  return (number1 + number2) / 2.0

def main():
    
    #Load the data
    test_index = 400000
    data_path = '../ntuples/Jan_25_2023/test_sig_v12_emseed.root'
    model_path = '../models/quantized_merged_pruned_gamma30.h5'
    
    #Load the data
    data = uproot4.open(data_path)
    
    inputs = data['ntuplePupSingle']['tree']['m_inputs'].array()[test_index:]
    
    truth_pt = data['ntuplePupSingle']['tree']['genpt1'].array()[test_index:]
    reco_pt = data['ntuplePupSingle']['tree']['pt'].array()[test_index:]
    deltaR = data['ntuplePupSingle']['tree']['gendr1'].array()[test_index:]
    eta = data['ntuplePupSingle']['tree']['geneta1'].array()[test_index:]
    
    selection = (reco_pt > 0.) & (abs(deltaR) < 0.4) & (abs(eta) < 2.4)
    
    truth_pt_selected = np.asarray(truth_pt[selection])
    reco_pt_selected = np.asarray(reco_pt[selection])
    X_test = np.asarray(inputs[selection])
    
    #Load the model and make pt correction
    model = load_qmodel(model_path)
    y_pred = model.predict(X_test)[1].flatten()
    
    nn_pt = np.multiply(reco_pt_selected, y_pred)
    residual_nn = nn_pt - truth_pt_selected
    residual_puppi = reco_pt_selected - truth_pt_selected

    def calculate_mean_rms(min_pt, max_pt):
        
        #Calculate puppi resolution mean & rms
        puppi_mean = np.mean(residual_puppi[(truth_pt_selected > min_pt) & (truth_pt_selected < max_pt)])
        puppi_rms = np.std(residual_puppi[(truth_pt_selected > min_pt) & (truth_pt_selected < max_pt)])
        
        #Calculate nn resolution mean & rms
        nn_mean = np.mean(residual_nn[(truth_pt_selected > min_pt) & (truth_pt_selected < max_pt)])
        nn_rms = np.std(residual_nn[(truth_pt_selected > min_pt) & (truth_pt_selected < max_pt)])
        
        return puppi_mean, puppi_rms, nn_mean, nn_rms
    
    
    #Select te
    inputs = data['ntuplePupSingle']['tree']['m_inputs'].array()[test_index:]
    truth_pt = data['ntuplePupSingle']['tree']['genpt1'].array()[test_index:]
    reco_pt = data['ntuplePupSingle']['tree']['pt'].array()[test_index:]
    
    pT_egdes = [0,10,15,20,25,30,35,40,45,50,55,60,70,80,100,125,150]
    
    mean_puppi = []
    rms_puppi = []
    mean_nn = []
    rms_nn = []
    pt_bins = []
    
    for i in range(len(pT_egdes)-1):
        
        res_result = calculate_mean_rms(pT_egdes[i], pT_egdes[i+1])
        pt_bins.append(average(pT_egdes[i], pT_egdes[i+1]))
        
        local_mean_puppi = res_result[0]
        local_rms_puppi = res_result[1]
        mean_puppi.append(local_mean_puppi)
        rms_puppi.append(local_rms_puppi)
        
        local_mean_nn = res_result[2]
        local_rms_nn = res_result[3]
        mean_nn.append(local_mean_nn)
        rms_nn.append(local_rms_nn)
        

    #Plot the mean and rms over the pt bins
    plt.figure()
    plt.scatter(pt_bins, mean_puppi, label=r'PUPPI $\mu$')
    plt.scatter(pt_bins, mean_nn, label=r'NN Corrected $\mu$')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.ylabel(r'(Truth $p_{T}$ - $Reco p_{T}$) $\mu$ [GeV]')
    plt.legend()
    plt.show()
    plt.savefig('plots/mean_vs_pt.pdf', bbox_inches='tight')
    plt.close()
    
    plt.figure()
    plt.scatter(pt_bins, rms_puppi, label=r'PUPPI $\sigma$')
    plt.scatter(pt_bins, rms_nn, label=r'NN Corrected $\sigma$')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.ylabel(r'(Truth $p_{T}$ - $Reco p_{T}$) $\sigma$ [GeV]')
    plt.legend()
    plt.show()
    plt.savefig('plots/rms_vs_pt.pdf', bbox_inches='tight')
    plt.close()
    
    
main()