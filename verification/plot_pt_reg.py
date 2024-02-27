'''
Plot all the pt regression results.
'''
from imports import *

#Set plotting style
plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)
mpl.rcParams['lines.linewidth'] = 5

def plot_pt_dist(data, model, test_index):
    
    '''
    Plot distributions of PUPPI, corrected, and truth pt
    '''
    
    #Might have to change the version for other ntuple files
    inputs = data['ntuplePupSingle']['tree']['m_inputs'].array()[test_index:]
    
    truth_pt = data['ntuplePupSingle']['tree']['genpt1'].array()[test_index:]
    reco_pt = data['ntuplePupSingle']['tree']['pt'].array()[test_index:]
    deltaR = data['ntuplePupSingle']['tree']['gendr1'].array()[test_index:]
    eta = data['ntuplePupSingle']['tree']['geneta1'].array()[test_index:]

    selection = (reco_pt > 0.) & (abs(deltaR) < 0.4) & (abs(eta) < 2.4)
    
    truth_pt_selected = np.asarray(truth_pt[selection])
    reco_pt_selected = np.asarray(reco_pt[selection])
    
    X_test = np.asarray(inputs[selection])
    y_pred = model.predict(X_test)[1].flatten()
    pred_pt = np.multiply(reco_pt_selected, y_pred)
   
    hist_pred_pt, edges_pred_pt = np.histogram(pred_pt, bins=20, range = (0,300), density = True)
    errors_pred_pt = np.sqrt(hist_pred_pt*len(pred_pt)) / len(pred_pt)
    bin_centers_pred_pt = 0.5 * (edges_pred_pt[:-1] + edges_pred_pt[1:])
    
    hist_reco, edges_reco = np.histogram(reco_pt_selected, bins=20, range = (0,300), density = True)
    errors_reco = np.sqrt(hist_reco*len(reco_pt_selected)) / len(reco_pt_selected)
    bin_centers_reco = 0.5 * (edges_reco[:-1] + edges_reco[1:])
    
    hist_truth, edges_truth = np.histogram(truth_pt_selected, bins=20, range = (0,300), density = True)
    errors_truth = np.sqrt(hist_truth*len(truth_pt_selected)) / len(truth_pt_selected)
    bin_centers_truth = 0.5 * (edges_truth[:-1] + edges_truth[1:])

    #Plot the pts
    plt.hist(truth_pt_selected, bins = 20, range = (0,300), density=True, histtype = 'step', label = 'Gen', linewidth=5)
    plt.hist(reco_pt_selected, bins = 20, range = (0,300), density=True, histtype = 'step', label = 'PUPPI Reco', linewidth=5)
    plt.hist(pred_pt, bins = 20, range = (0,300), density=True, histtype = 'step', label = 'NN Corrected', linewidth=5)
    plt.errorbar(bin_centers_pred_pt, hist_pred_pt, yerr=errors_pred_pt, fmt='o', color = "#2ca02c", linewidth = 2, alpha = 0.5)
    plt.errorbar(bin_centers_reco, hist_reco, yerr=errors_pred_pt, fmt='o', color = '#ff7f0e', linewidth = 2, alpha = 0.5)
    plt.errorbar(bin_centers_truth, hist_truth, yerr=errors_truth, fmt='o', color = '#1f77b4', linewidth = 2, alpha = 0.5)
    plt.ylim(0, 0.018)

    plt.xlabel(r'$p_T$ [GeV]')
    plt.ylabel('a.u.')
    plt.legend()  
    plt.show()
    plt.savefig('plots/pt_distributions.pdf', bbox_inches='tight')
    plt.close()

def plot_mean_rms_pt(data, model, test_index):
    '''
    Plot the mean and rms w.r.t pt
    '''
    
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

def main():
    
    model_path = '../models/quantized_merged_pruned_gamma30.h5'
    data_path = '../ntuples/Jan_25_2023/test_sig_v12_emseed.root'
    test_index= 0
    
    data = uproot4.open(data_path)
    model = load_qmodel(model_path)
    
    #
    plot_pt_dist(data, model, test_index)
    plot_mean_rms_pt(data, model, test_index)

main()