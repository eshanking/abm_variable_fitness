# URGENT TO DO:
# REMOVE DRUGLESS GROWTH RATES AND IC50 GLOBAL VARIABLES!

# persistant local maxima in fitness landscapes
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

conc_step = 10
min_prominence = 0.1

###############################################################################
drugless_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\ogbunugafor_drugless.csv"
ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\cycloguanil_ic50.csv"
#ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\pyrimethamine_ic50.csv"

def is_local_max(allele, conc,
                 n_alleles=16,
                 min_prominence=.01):
    ###########################################################################
    # generate the nearest neighbors in base-10
    bin_allele = int_to_binary(allele)
    bin_allele = list(bin_allele)
    bin_allele = np.array(bin_allele)
    bin_allele = bin_allele.astype(int)
    
    larger_indx = np.nonzero(bin_allele==0)
    smaller_indx = np.nonzero(bin_allele==1)
    
    larger_indx = 3-larger_indx[0]
    smaller_indx = 3-smaller_indx[0]
    
    kk = 0
    neighbors = np.zeros(4)
#    print(str(larger_indx))
    for power in larger_indx:
#        print(str(power))
        neighbors[kk] = allele+2**power
#        print(str(power))
        kk+=1
    for power in smaller_indx:
#        print(str(power))
        neighbors[kk] = allele-2**power
        kk+=1
#    return neighbors
    ###########################################################################
    # Determine if the current allele is a local maxima with a prominance 
    # greater than the min prominance
    
    fit_land = gen_fit_land(conc)
    
    neighbors = neighbors.astype(int)
    
    local_land = fit_land[neighbors]
    cur_allele = fit_land[allele]
    prominence = cur_allele-local_land
    
    if all(prominence >= min_prominence):
        return True, prominence
    else:
        return False, prominence

def int_to_binary(num, pad=4):
    return bin(num)[2:].zfill(pad)

# generate the entire fitness landscape for a given drug conc.
def gen_fit_land(conc):
    fit_land = np.zeros(16)
    for kk in range(16):
        fit_land[kk]=gen_fitness(kk,conc)
    return fit_land

def load_fitness(data_path):
    # also use to load ic50 and drugless growth rate
    fitness = pd.read_csv(data_path)
    cols = list(fitness.columns)
    fit_array = np.array(cols)
    fit_array = fit_array.astype(np.float)
    return fit_array

def gen_fitness(allele,conc):
    # input: allele (integer from 0 to 15)
    # conc: current drug concentration
    # output: allele fitness
    c = -.6824968
#    c = -100
    # logistic equation from Ogbunugafor 2016
    conc = conc/10**6;
    
    # ic50 is already log-ed in the dataset
    log_eqn = lambda d,i: d/(1+np.exp((i-np.log10(conc))/c))
    
    fitness = log_eqn(drugless_rate[allele],ic50[allele])
#    fitness = 0
    return fitness

drugless_rate = load_fitness(drugless_path)
ic50 = load_fitness(ic50_path)

conc = np.logspace(-3,5,9)
conc = np.concatenate(([0],conc))
alleles = np.linspace(0,15,16)
alleles = alleles.astype(int)

binary_peaks = np.zeros((16,10))
#n_checks = 0

for conc_indx in range(0,10):
#    print(str(conc_indx))
    for allele_indx in range(0,16):
        check_local_max = is_local_max(alleles[allele_indx],conc[conc_indx],min_prominence=0)
#        print(str(alleles[allele_indx]) + ' ' + str(conc[conc_indx]))
#        print(str(check_local_max[1]))
#        n_checks+=1
        if check_local_max[0]:
            binary_peaks[allele_indx,conc_indx] = 1

#x_ticks = ['0','10^-3','10^-2','10^-1','10^0','10^1','10^2','10^3','10^4','10^5']
y_ticks = []
for allele in range(16):
    y_ticks.append(str(int_to_binary(allele)))
    
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(binary_peaks,cmap='magma',cbar=False,linewidth=0.5,square=False)
ax.set_xticklabels(['0','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$',
                         '$10^1$','$10^2$','$10^3$','$10^4$','$10^5$'])
ax.set_yticklabels(y_ticks)

ax.yaxis.set_tick_params(rotation=0)
ax.tick_params(axis='y',labelsize=15)
ax.tick_params(axis='x',labelsize=15)

plt.xlabel('Concentration (uM)',fontsize=20)
plt.ylabel('Allele',fontsize=20)
ax.set_title('Cycloguanil',fontsize=20)