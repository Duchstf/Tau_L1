import uproot4
import numpy as np
import awkward as ak
from scipy.stats import norm
from scipy.optimize import curve_fit
import os
import copy

import tensorflow.keras as keras
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential, load_model
import tensorflow_model_optimization as tfmot

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

%matplotlib inline
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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
    except RuntimeError as e:
        print(e)

import hls4ml
import hls4ml.utils
import hls4ml.converters

os.environ['PATH'] += os.pathsep + '/tools/Xilinx/Vitis_HLS/2022.1/bin'

model = load_qmodel("models/quantized_merged_pruned_gamma30.h5")


config = hls4ml.utils.config_from_keras_model(model, granularity='name')

for Layer in config['LayerName'].keys():
        if "Dense" in Layer:

            config['LayerName'][Layer]['Strategy'] = 'Latency'
            config['LayerName'][Layer]['ReuseFactor'] = 1

        else:
            config['LayerName'][Layer]['Strategy'] = 'Latency'
            config['LayerName'][Layer]['ReuseFactor'] = 1

hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       backend='Vitis',
                                                       project_name='tau_nn', #I'm not very creative
                                                       clock_period=2.8, #1/360MHz = 2.8ns
                                                       hls_config=config,
                                                       output_dir='hardwareII_COM/testing60_pruned/hls4ml_prj',
                                                       part='xcvu9p-flga2104-2L-e')


# hls_model = hls4ml.converters.convert_from_keras_model(
#     model, hls_config=config, output_dir='model_3/hls4ml_prj', part='xcu250-figd2104-2L-e'
# )
hls_model.compile()

# y_qkeras = model.predict(np.ascontiguousarray(X_test))
# y_hls = hls_model.predict(np.ascontiguousarray(X_test))

hls_model.build(csim=False)
hls4ml.converters.convert_from_keras_model