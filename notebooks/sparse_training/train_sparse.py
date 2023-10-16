#Import stuff
import uproot4
import numpy as np
import awkward as ak
from scipy.stats import norm
from scipy.optimize import curve_fit
import os
import copy

import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
import tensorflow_model_optimization as tfmot
from keras import optimizers

import tempfile

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
    except RuntimeError as e:
        print(e)
#-----------
#Create training data
def make_weights(sig_pt, reco_bkg_pt, pt_edges = [20,25,30,35,40,45,50,55,60,70,80,100,125,150,2000]):
    
    weight_pt_bkg = np.zeros(reco_bkg_pt.shape)
    
    sig_hist, sig_bin_edges = np.histogram(sig_pt, bins = pt_edges)
    bkg_hist, bkg_bin_edges = np.histogram(reco_bkg_pt, bins = pt_edges)
    
    scale_factor = [sig_hist[i]/bkg_hist[i] for i in range(len(sig_bin_edges)-1)]
    
    #Assign the weights to qcd jets
    for i in range(len(sig_bin_edges)-1):
        weight_pt_bkg[(reco_bkg_pt >= sig_bin_edges[i]) & (reco_bkg_pt < sig_bin_edges[i+1])] = scale_factor[i]
    
    #weight_pt_bkg[reco_bkg_pt > 100] = weight_pt_bkg[reco_bkg_pt > 100]/2. #Give less weights for high pT background (v4)
    
    weight_pt_sig = np.ones(sig_pt.shape)
    weight_pt_sig[sig_pt > 100.] = 2 #Give more importance to high pT signal (v2 + v3)
#     weight_pt_sig[sig_pt > 120.] = 5. #Give more importance to high pT signal (v3)
    all_weight = np.concatenate([weight_pt_sig, weight_pt_bkg])
    
    return all_weight

def create_training_data(dir_path):
    
    pt_edges = [20,25,30,35,40,45,50,55,60,70,80,100,125,150,2000]
    
    #Might have to change the version for other ntuple files
    sig = uproot4.open(dir_path+"/test_sig_v12_emseed.root")
    bkg = uproot4.open(dir_path+"/test_bkg_v12_emseed.root")
    qcd = uproot4.open(dir_path+"/test_qcd_v12_emseed.root")
    
    sig_input = sig['ntuplePupSingle']['tree']['m_inputs'].array()[:400000]
    bkg_input = bkg['ntuplePupSingle']['tree']['m_inputs'].array()[:400000]
    qcd_input = qcd['ntuplePupSingle']['tree']['m_inputs'].array()
    
    truth_pt_sig = sig['ntuplePupSingle']['tree']['genpt1'].array()[:400000]
    reco_pt_sig = sig['ntuplePupSingle']['tree']['pt'].array()[:400000]
    deltaR_sig = sig['ntuplePupSingle']['tree']['gendr1'].array()[:400000]
    eta_sig = sig['ntuplePupSingle']['tree']['geneta1'].array()[:400000]
    selection_sig = (truth_pt_sig > 20.) & (abs(deltaR_sig) < 0.4) & (abs(eta_sig) < 2.4)
    
    reco_pt_bkg = bkg['ntuplePupSingle']['tree']['pt'].array()[:400000]
    selection_bkg = reco_pt_bkg > 20.
    
    reco_pt_qcd = qcd['ntuplePupSingle']['tree']['pt'].array()
    selection_qcd = reco_pt_qcd > 20.
    
    #Maybe better to use 2.4 for eta
    
    #Inputs: pt, eta, phi, particle id(one hot encoded)
    X_sig = np.nan_to_num(np.asarray(sig_input[selection_sig]))
    y_sig = np.full(X_sig.shape[0], 1.)
    sig_pt = np.asarray(reco_pt_sig[selection_sig])
    
    X_bkg = np.nan_to_num(np.asarray(bkg_input)[selection_bkg])
    y_bkg = np.full(X_bkg.shape[0], 0.)
    bkg_pt = np.asarray(reco_pt_bkg[selection_bkg])
    
    X_qcd = np.nan_to_num(np.asarray(qcd_input)[selection_qcd])
    y_qcd = np.full(X_qcd.shape[0], 0.)
    qcd_pt = np.asarray(reco_pt_qcd[selection_qcd])
    
    background_pt = np.concatenate([bkg_pt, qcd_pt])
    
    X_train = np.concatenate([X_sig, X_bkg, X_qcd])
    y_train = np.concatenate([y_sig, y_bkg, y_qcd])
    pt_array = np.concatenate([sig_pt, bkg_pt, qcd_pt])
    sample_weight = make_weights(sig_pt, background_pt, pt_edges)
    
    X_train[abs(X_train) > 1e+4] = 0.
    
    assert not np.any(np.isnan(X_train))
    assert not np.any(np.isnan(y_train))
    
    return X_train, y_train, sample_weight

X_train, y_train, sample_weight = create_training_data("../../ntuples/Jan_25_2023")


#Define the final sparsity you want to reach.
f_sparsity_list = [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

for f_sparsity in f_sparsity_list:
    
    #Reset the model
    model = None

    #Define the model
    model = Sequential()
    model.add(Dense(25, activation = 'relu', name='Dense1', input_shape=X_train.shape[1:]))
    model.add(Dense(10, activation = 'relu', name='Dense2'))
    model.add(Dense(10, activation = 'relu', name='Dense3'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = optimizers.Adam()
    model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['binary_accuracy'], weighted_metrics=[])

    print(model.summary())
    
    #
    batch_size = 1024
    epochs = 50
    validation_split = 0.1 # 10% of training set will be used for validation set. 

    # sparsity parameters
    i_sparsity = 0.0

    num_samples = X_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_samples / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=i_sparsity,
                                                                final_sparsity=f_sparsity,
                                                                begin_step=0,
                                                                end_step=end_step)
    }

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    
    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model_for_pruning.summary()
    
    logdir = tempfile.mkdtemp()

    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)
    ]

    model_for_pruning.fit(X_train, y_train,
                    batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                    callbacks=callbacks)

    # Save model
    model.save('models/{}_pruned.h5'.format(f_sparsity))