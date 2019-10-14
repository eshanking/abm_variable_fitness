# persistant local maxima in fitness landscapes
import numpy as np
import pandas as pd

def is_local_max(allele, 
                 conc,
                 ic50,
                 drugless_rate,
                 n_alleles=16,
                 min_prominence=.01):
###############################################################################
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
###############################################################################
    # Determine if the current allele is a local maxima with a prominance 
    # greater than the min prominance
    
    fit_land = gen_fit_land(conc,ic50,drugless_rate)
    
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
def gen_fit_land(conc,ic50,drugless_rate):
    fit_land = np.zeros(16)
    for kk in range(16):
        fit_land[kk]=gen_fitness(kk,conc,drugless_rate,ic50)
    return fit_land

def load_fitness(data_path):
    # also use to load ic50 and drugless growth rate
    fitness = pd.read_csv(data_path)
    cols = list(fitness.columns)
    fit_array = np.array(cols)
    fit_array = fit_array.astype(np.float)
    return fit_array

def gen_fitness(allele,
                conc,
                drugless_rate,
                ic50):
    # input: allele (integer from 0 to 15)
    # conc: current drug concentration
    # output: allele fitness
    c = -.6824968
#    c = -100
    # logistic equation from Ogbunugafor 2016
    conc = conc/10**6;
#    print(str(conc))
    # ic50 is already log-ed in the dataset
    if conc == 0:
        fitness = drugless_rate[allele]
    else:
        log_eqn = lambda d,i: d/(1+np.exp((i-np.log10(conc))/c))
        
        fitness = log_eqn(drugless_rate[allele],ic50[allele])
#    fitness = 0
    return fitness