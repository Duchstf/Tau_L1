import hls4ml

# Standard Stuff
import h5py
import numpy as np
from numpy import expand_dims
import tensorflow as tf

from qkeras.utils import load_qmodel
from keras.models import load_model

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

#Plot settings
import matplotlib.pyplot as plt
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

import os
os.environ['PATH'] += os.pathsep + '/data/Xilinx/Vivado/2022.2/bin'

MODEL_PATH = '../models/merged_11_gamma33_weights.h5'

def merged_model(gamma=0.33):

    inputs = tf.keras.layers.Input(shape=(80,), name='puppi_inputs')
    
    main_branch = tf.keras.layers.Dense(25, activation = "LeakyReLU", name='Dense1')(inputs)
    main_branch = tf.keras.layers.Dense(25, activation = "LeakyReLU", name = 'Dense2')(main_branch)
    main_branch = tf.keras.layers.Dense(15, activation = "LeakyReLU", name = 'Dense3')(main_branch)
    main_branch = tf.keras.layers.Dense(15, activation = "LeakyReLU", name = 'Dense4')(main_branch)
    main_branch = tf.keras.layers.Dense(10, activation = "LeakyReLU", name = 'Dense5')(main_branch)
    
    jetID_branch = tf.keras.layers.Dense(1, activation='sigmoid', name='jetID_output')(main_branch)
    pT_branch = tf.keras.layers.Dense(1, name='pT_output')(main_branch)

    model = tf.keras.Model(inputs = inputs, outputs = [jetID_branch, pT_branch])
    
    return model

#----------------------------------------------
tau_model = merged_model(MODEL_PATH)
tau_model.load_weights()
    
config = hls4ml.utils.config_from_keras_model(tau_model, granularity='name')

for Layer in config['LayerName'].keys():          
    if "Dense" in Layer:

        config['LayerName'][Layer]['Strategy'] = 'Latency'
        config['LayerName'][Layer]['ReuseFactor'] = 1

    else:
        config['LayerName'][Layer]['Strategy'] = 'Latency'
        config['LayerName'][Layer]['ReuseFactor'] = 1

    hls_model = hls4ml.converters.convert_from_keras_model(tau_model,
                                                           backend='Vitis',
                                                           project_name='tau_nn', #I'm not very creative
                                                           clock_period=2.8, #1/360MHz = 2.8ns
                                                           hls_config=config,
                                                           output_dir='TauID',
                                                           part='xcvu9p-flga2104-2L-e')

    hls_model.compile()

    hls_model.build(csim=False, reset = True)