'''
Check the distributions of input candidates between two versions
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

def read_inputs(data_path):
    
    data = uproot4.open(data_path)
    
    inputs = data['ntuplePupSingle']['tree']['m_inputs'].array()
    reco_pt = data['ntuplePupSingle']['tree']['pt'].array()
    deltaR = data['ntuplePupSingle']['tree']['gendr1'].array()
    eta = data['ntuplePupSingle']['tree']['geneta1'].array()

    selection = (reco_pt > 0.) & (abs(deltaR) < 0.4) & (abs(eta) < 2.4)
    
    return inputs[selection]
    

def main():
    
    data_path_old = '../ntuples/Jan_25_2023/test_sig_v12_emseed.root'
    data_path_new = '../ntuples/14_0_0_pre1/tuples_vbf.root'
    
    inputs_old = read_inputs(data_path_old)
    inputs_new = read_inputs(data_path_new)
    
    H, bins, _ = plt.hist(inputs_old[:,0][inputs_old[:,0] != 0], density=True, label = 'Winter 2020', alpha = 0.5)
    H2, bins, _ = plt.hist(inputs_new[:,0][inputs_new[:,0] != 0], bins=bins, density=True, label = 'Spring 2023', alpha=0.5)
    
    print(inputs_new[:,0])
    plt.xlabel(r"Leading candidate $p_T^{reco}$ [GeV]")
    plt.ylabel(r"a.u.")
    
    plt.legend()  
    plt.show()
    plt.savefig('plots/leading_pt_compare.pdf', bbox_inches='tight')
    plt.close()
    
    #Second candidate
    H, bins, _ = plt.hist(inputs_old[:,8], density=True, label = 'Winter 2020', alpha = 0.5)
    H2, bins, _ = plt.hist(inputs_new[:,8], bins=bins, density=True, label = 'Spring 2023', alpha=0.5)
    
    print(inputs_new[:,0])
    plt.xlabel(r"Second candidate $p_T^{reco}$ [GeV]")
    plt.ylabel(r"a.u.")
    
    plt.legend()  
    plt.show()
    plt.savefig('plots/second_pt_compare.pdf', bbox_inches='tight')
    plt.close()
    
    #Plot eta
    print(inputs_old[:,1])
    H, bins, _ = plt.hist(inputs_old[:,33][inputs_old[:,33] != 0],bins=50, density=True, label = 'Winter 2020', alpha = 0.5, range=(-0.5,0.5))
    H2, bins, _ = plt.hist(inputs_new[:,33][inputs_new[:,33] != 0], bins = bins, density=True, label = 'Spring 2023', alpha=0.5)
    
    plt.xlabel(r"4th candidate $\eta^{reco}$ [GeV]")
    plt.ylabel(r"a.u.")
    
    plt.legend()  
    plt.show()
    plt.savefig('plots/leading_eta_compare.pdf', bbox_inches='tight')
    plt.close()
    

main()