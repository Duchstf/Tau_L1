
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

def calculate_eff_pt(model_path,
              data_path,
              TreeName='ntuplePupSingle',
              test_indx=400000,
              corrected_pt_cut=34,
              cut_point = 0.22):
    '''
    Do the rate calculation while taking into account the nn predictions
    
    test_indx: index above which we use for predictions (the model was not trained on this).
    corrected_pt_cut: cutting on pt after correction
    cut_point: nn prediction cutting points
    '''
    
    #Load signal data
    sig = uproot4.open(data_path)
    
    #Load model and select out the tau
    #0 index is id, 1 is pT regression
    model = load_qmodel(model_path)
    
    truth_tau_pt = sig[TreeName]['tree']['genpt1'].array()[test_indx:]
    reco_tau_pt = sig[TreeName]['tree']['pt'].array()[test_indx:]
    
    #Get all the inputs
    sig_input = sig[TreeName]['tree']['m_inputs'].array()[test_indx:]

    reco_pt_sig = sig[TreeName]['tree']['pt'].array()[test_indx:]
    deltaR_sig = sig[TreeName]['tree']['gendr1'].array()[test_indx:]
    eta_sig = sig[TreeName]['tree']['geneta1'].array()[test_indx:]
    
    selection_deno = (truth_tau_pt > 1.) & (abs(eta_sig) < 2.1) #"genpt1 > 1. && abs(geneta1) < 2.1 "
    
    corrected_pt_sig = reco_pt_sig
    selection_num_corrected = selection_deno & (abs(deltaR_sig) < 0.4) & (corrected_pt_sig>corrected_pt_cut) # " abs(gendr1) < 0.4 && pt > 1."
    

    selection_num = selection_deno & (abs(deltaR_sig) < 0.4) & (reco_pt_sig>corrected_pt_cut) # " abs(gendr1) < 0.4 && pt > 1."
    
    #PUPPI only pT
    puppi_pT = truth_tau_pt[selection_num]
    
    #Corrected pT
    truth_tau_pt_select = truth_tau_pt[selection_num_corrected]

    X_sig = np.asarray(sig_input[selection_num_corrected])
    y_sig = model.predict(np.nan_to_num(X_sig))

    nn_cut = y_sig.flatten() > cut_point
    tau_pt_nn = np.asarray(truth_tau_pt_select)[nn_cut]
    
     #Fill two plots and divide the two.
    pT_egdes = [0,10,15,20,25,30,35,40,45,50,55,60,70,80,100,125,150] #200?
    pT_egdes = list(pT_egdes)
    pT_axis = hist.axis.Variable(pT_egdes, name = r"$ \tau_h$ $p_T^{gen}$")

    hist_all_tau = Hist(pT_axis)
    hist_selected_puppi = Hist(pT_axis)
    hist_selected_tau = Hist(pT_axis)

    hist_all_tau.fill(truth_tau_pt[selection_deno])
    hist_selected_puppi.fill(puppi_pT)
    hist_selected_tau.fill(tau_pt_nn)
    
    return hist_all_tau, hist_selected_puppi, hist_selected_tau
    
def plot_eff_pt(hist_all_tau,
                hist_selected_puppi,
                hist_selected_tau,
                corrected_pt_cut,
                cut_point):
    
    #Just to get the division objects
    fig = plt.figure(figsize=(10, 12))
    _, eff_new_nn_artists = hist_selected_tau.plot_ratio(
        hist_all_tau,
        rp_num_label=r"Selected Taus (New NN Score > {})".format(cut_point),
        rp_denom_label=r"All Taus",
        rp_uncert_draw_type="bar",
        rp_uncertainty_type="efficiency",
    )


    fig = plt.figure(figsize=(10, 12))
    _, eff_puppi_artists = hist_selected_puppi.plot_ratio(
        hist_all_tau,
        rp_num_label=r"Selected Taus (PUPPI)",
        rp_denom_label=r"All Taus",
        rp_uncert_draw_type="bar",
        rp_uncertainty_type="efficiency",
    )

    #The real efficiency plot
    fig = plt.figure()

    eff_new_nn_x = [eff_new_nn_artists.bar.patches[i].get_x() for i in range(len(eff_new_nn_artists.bar.patches))]
    eff_new_nn_y = [eff_new_nn_artists.bar.patches[i].get_y() for i in range(len(eff_new_nn_artists.bar.patches))]
    eff_new_nn_err = [eff_new_nn_artists.bar.patches[i].get_height() for i in range(len(eff_new_nn_artists.bar.patches))]

    eff_puppi_x = [eff_puppi_artists.bar.patches[i].get_x() for i in range(len(eff_puppi_artists.bar.patches))]
    eff_puppi_y = [eff_puppi_artists.bar.patches[i].get_y() for i in range(len(eff_puppi_artists.bar.patches))]
    eff_puppi_err =  [eff_puppi_artists.bar.patches[i].get_height() for i in range(len(eff_puppi_artists.bar.patches))]
    
    plt.plot([],[], 'none', label=r'$p_T^{{corrected}}$ cut: {} [GeV]'.format(round(corrected_pt_cut)))
    plt.errorbar(eff_new_nn_x, eff_new_nn_y, yerr=eff_new_nn_err,
                fmt='o',
                linewidth=2,
                label = 'Merged Tau NN (NN Score > {})'.format(cut_point))

    plt.errorbar(eff_puppi_x, eff_puppi_y, yerr=eff_puppi_err,
                fmt='o',
                linewidth=2,
                label = r'PUPPI $\epsilon$ (no $p_T$ correction)')

    #Plot formatting
    plt.hlines(1, 0, 150, linestyles='dashed', color='black', linewidth=3)
    plt.ylim([0.,1.1])
    plt.xlim([0,150])
    hep.cms.text("Phase 2 Simulation")
    hep.cms.lumitext("PU 200 (14 TeV)")
    plt.xlabel(r"$\tau_h$ $p_T^{gen}$ [GeV]")
    plt.ylabel(r"$\epsilon$(Di-$\tau_h$ trigger rate at 28 kHz)")
    plt.legend(loc = 'best', fontsize=20)
    plt.show()
    plt.savefig('plots/old_signal_efficiency_pt.pdf', bbox_inches='tight')
    plt.close()
    
def main():
    
    model_path = '../models/Feb_4_2023_JetMetTalk_v1_pTShape_EMSeed.h5'
    data_path = '../ntuples/14_0_0_pre1/tuples_vbf.root'
    TreeName= 'ntuplePupSingle'
    test_indx= 0
    corrected_pt_cut= 0
    cut_point = 0.05
    
    #Do the calculation for selected data and output the filled histograms
    hist_all_tau, hist_selected_puppi, hist_selected_tau = calculate_eff_pt(model_path,
                                                                           data_path,
                                                                           TreeName,
                                                                           test_indx,
                                                                           corrected_pt_cut,
                                                                           cut_point)
    
    plot_eff_pt(hist_all_tau,
                hist_selected_puppi,
                hist_selected_tau,
                corrected_pt_cut,
                cut_point)
    
    

main()