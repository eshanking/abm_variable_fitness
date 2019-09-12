import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.gridspec as gridspec

from moviepy.video.io.bindings import mplfig_to_npimage
import  moviepy.editor as mpy
import seaborn as sns
from scipy.stats import entropy
import pandas as pd

# converts decimals to binary
def int_to_binary(num, pad=4):
    return bin(num)[2:].zfill(pad)

# computes hamming distance between two genotypes
def hammingDistance(s1, s2):
    assert len(s1) == len(s2)
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

# converts an integer to a genotype and padding to the left by 0s
def convertIntToGenotype(anInt, pad):
	offset = 2**pad
	return [int(x) for x in bin(offset+anInt)[3:]]

# generates transition matrix for random mutations
def random_mutations(N):
    trans_mat = np.zeros([N,N])
    for mm in range(N):
        for nn in range(N):
            trans_mat[mm, nn] = hammingDistance( int_to_binary(mm) , int_to_binary(nn))

    trans_mat[trans_mat>1] = 0
    trans_mat = trans_mat/trans_mat.sum(axis=1)

    return trans_mat

def load_fitness(data_path):
    # also use to load ic50 and drugless growth rate
    fitness = pd.read_csv(data_path)
    cols = list(fitness.columns)
    fit_array = np.array(cols)
    fit_array = fit_array.astype(np.float)
    return fit_array

def gen_fitness(allele,conc,drugless_rate,ic50):
    # input: allele (integer from 0 to 15)
    # conc: current drug concentration
    # output: allele fitness
    c = -.6824968
#    c = -100
    # logistic equation from Ogbunugafor 2016
    conc = conc/10**4;
    
    # ic50 is already log-ed in the dataset
    log_eqn = lambda d,i: d/(1+np.exp((i-np.log(conc))/c))
    
    fitness = log_eqn(drugless_rate[allele],ic50[allele])
#    fitness = 0
    return fitness

# generates drug concentration curves
def calc_conc(step,
              curve_type='linear',
              const_dose = 0):
    if curve_type == 'linear':
#        print('here')
        if step <= 500:
            slope = (10**3-10**(-3))/500
            conc = slope*step+10**-3
        else:
            step = 500
            slope = (10**3-10**(-3))/500
            conc = slope*step+10**-3
    elif curve_type == 'constant':
        conc = const_dose
    else:
        conc = np.log(step)
        # output in uM
    return conc

def var_fit_automaton(n_gen=40,  # Number of simulated generations
                  mut_rate=0.1,  # probability of mutation per generation
                  max_cells=10**5,  # Max number of cells
                  death_rate=0.3,  # Death rate
                  init_counts=None,
                  carrying_cap=True,
                  plot = True,
                  curve_type = 'linear',
                  const_dose = 0
                  ):

    drugless_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\ogbunugafor_drugless.csv"
    ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\cycloguanil_ic50.csv"
    
    drugless_rates = load_fitness(drugless_path)
    ic50 = load_fitness(ic50_path)
    
    # Number of different alleles
    n_allele = len(drugless_rates)
    # Obtain transition matrix for mutations
    P = random_mutations( n_allele )
   
    
    # Keeps track of cell counts at each generation
    counts = np.zeros([n_gen+1, n_allele], dtype=int)
    drug_curve = np.zeros(n_gen+1)

    if init_counts is None:
        counts[0] = 10*np.ones(n_allele)
    else:
        counts[0] = init_counts

    for mm in range(n_gen):
        
        if curve_type == 'constant':
            # dose is in uM
#            conc = calc_conc(mm,curve_type,const_dose)
            conc = const_dose
#            print('here')
        else:
            conc = calc_conc(mm,curve_type)
        
        drug_curve[mm] = conc
        
        fit_land = np.zeros(n_allele)
        
        for kk in range(n_allele):
            fit_land[kk] = gen_fitness(kk,conc,drugless_rates,ic50)
#            print(str(conc))
        
        n_cells = np.sum( counts[mm] )

        # Scale division rates based on carrying capacity
        if carrying_cap:
            division_scale = 1 / (1+(2*np.sum(counts[mm])/max_cells)**4)
        else:
            division_scale = 1

        if counts[mm].sum()>max_cells:
            division_scale = 0

        div_rate = np.repeat( fit_land*division_scale, counts[mm] )
        cell_types = np.repeat( np.arange(n_allele) , counts[mm] )

        # Death of cells
        death_rates = np.random.rand(n_cells)
        surv_ind = death_rates > death_rate
        div_rate = div_rate[surv_ind]
        cell_types = cell_types[surv_ind]
        n_cells = len(cell_types)

        counts[mm+1] = np.bincount(cell_types, minlength=n_allele)

        #Divide and mutate cells
        div_ind = np.random.rand(n_cells) < div_rate

        # Mutate cells
        # initial state of allele types
        daughter_types = cell_types[div_ind].copy()

        # Generate random numbers to check for mutation
        daughter_counts = np.bincount( daughter_types , minlength=n_allele)

        # Mutate cells of each allele type
        for allele in np.random.permutation(np.arange(n_allele)):
            n_mut = np.sum( np.random.rand( daughter_counts[allele] ) < mut_rate )

            # note that columns in P are normalized to probability densities (columns sum to 1)
            mutations = np.random.choice(n_allele, size=n_mut, p=P[:,allele]).astype(np.uint8)

            #Add mutating cell to their final types
            counts[mm+1] +=np.bincount( mutations , minlength=n_allele)

            #Substract mutating cells from that allele
            daughter_counts[allele] -=n_mut

        counts[mm+1] += daughter_counts
    if plot:
        plot_timecourse(counts,drug_curve)
#        print('im here!')
#    return counts
    return counts

def plot_timecourse(counts, drug_curve,
                    fig_title = ''):
    fig, ax = plt.subplots(figsize = (10,7))
#    plt.rcParams.update({'font.size': 22})

    for allele in range(0,counts.shape[1]-1):
        plt.plot(counts[:,allele],linewidth=3.0)
        
    plt.xlim(0,counts.shape[0])
    ax.set_facecolor(color='w')
    ax.grid(False)

    plt.xlabel('Generations',fontsize=20)
    plt.ylabel('Cells',fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    ax2 = ax.twinx()
    color = 'k'
    ax2.set_ylabel('Drug Concentration (uM)', color=color,fontsize=20)  # we already handled the x-label with ax1
    ax2.plot(drug_curve, color=color, linewidth=3.0, linestyle = 'dashed')
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax2.legend(['Drug Concentration'],loc=(.7,.9))
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(fig_title,fontsize=20)
    plt.show()
    return ax

def plot_fitness_curves():
    
    drugless_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\ogbunugafor_drugless.csv"
    ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\cycloguanil_ic50.csv"
    
    drugless_rates = load_fitness(drugless_path)
    ic50 = load_fitness(ic50_path)
    
    fig, ax = plt.subplots(figsize = (10,7))
    
    powers = np.linspace(-3,5,9)
    conc = np.power(10*np.ones(9),powers)
    fit = np.zeros(conc.shape[0])
    
    for i in range(15):
        for j in range(conc.shape[0]):
            fit[j] = gen_fitness(i,conc[j],drugless_rates,ic50)
        plt.plot(fit)
    ind = np.arange(9)
    plt.xticks(ind,['10^-3','10^-2','10^-1','10^0','10^1','10^2','10^3','10^4','10^5'])
    
    return ax