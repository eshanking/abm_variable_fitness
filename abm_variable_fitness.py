import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from cycler import cycler

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
    # Don't include mutant no. 4
    
    trans_mat[trans_mat>1] = 0
    trans_mat = trans_mat/trans_mat.sum(axis=1)
    
#    trans_mat[3,:] = 0
#    trans_mat[:,3] = 0
#    print(str(trans_mat))
    return trans_mat

# load fitness data from a csv file
def load_fitness(data_path):
    # also use to load ic50 and drugless growth rate
    fitness = pd.read_csv(data_path)
    cols = list(fitness.columns)
    fit_array = np.array(cols)
    fit_array = fit_array.astype(np.float)
    return fit_array

# calculate fitness given a genotype and drug concentration (Ogbunugafor 2016)
def gen_fitness(allele,conc,drugless_rate,ic50):
    # input: allele (integer from 0 to 15)
    # conc: current drug concentration
    # output: allele fitness
    c = -.6824968
#    c = -100
    # logistic equation from Ogbunugafor 2016
    conc = conc/10**6;
    
    # ic50 is already log-ed in the dataset
    log_eqn = lambda d,i: d/(1+np.exp((i-np.log10(conc))/c))
    if conc <= 0:
        fitness = drugless_rate[allele]
    else:
        fitness = log_eqn(drugless_rate[allele],ic50[allele])
#    fitness = 0
    return fitness

# generates drug concentration curves
def calc_conc(step,
              curve_type='linear',
              const_dose = 0,
              steepness=100,
              max_dose = 10,
              h_step = 100,
              min_dose=1,
              K_elim=0.01,
              K_abs=0.1
              ):
#    steepness = 100
    if curve_type == 'linear':
#        print('here')
        if step <= steepness:
            slope = (max_dose-10**(-3))/steepness
            conc = slope*step+10**-3
        else:
            step = steepness
            slope = (max_dose-10**(-3))/steepness
            conc = slope*step+10**-3
    elif curve_type == 'constant':
        conc = const_dose
    elif curve_type == 'log':
        conc = np.log(step)
        # output in uM
    elif curve_type == 'heaviside':
        if step <= h_step:
            conc = min_dose
        else:
            conc = max_dose 
    elif curve_type == 'pharm':
        conc = np.exp(-K_elim*step)-np.exp(-K_abs*step)
        t_max = np.log(K_elim/K_abs)/(K_elim-K_abs)
        conc = conc/(np.exp(-K_elim*t_max)-np.exp(-K_abs*t_max))
        conc = conc*max_dose
    return conc

# Equation for a simple pharmacokinetic model
def pharm_eqn(t,k_elim=0.01,k_abs=0.1,max_dose=1):
    conc = np.exp(-k_elim*t)-np.exp(-k_abs*t)
    t_max = np.log(k_elim/k_abs)/(k_elim-k_abs)
    conc = conc/(np.exp(-k_elim*t_max)-np.exp(-k_abs*t_max))
    conc = conc*max_dose
    return conc

# Convolve the arbitrary curve u with the pharmacokinetic model
def convolve_pharm(u,t_max,
                   k_elim=0.01,
                   k_abs=0.1,
                   max_dose=1):
    
    # k_lim and k_abs are the absorption and elimination rate constants for the pharmacokinetic model
    # t_max is the max length of the output curve
    # algorithm is at best O(n^2)...
    conv = np.zeros(t_max)
    for t in range(t_max):
        for tau in range(t_max):
            if t-tau >= 0 and t-tau<u.shape[0]:
                conv[t] += u[t-tau]*pharm_eqn(tau,k_elim=k_elim,k_abs=k_abs,max_dose=max_dose)
    return conv

# Generates an impulse train to input to convolve_pharm()
def gen_impulses(n_impulse,t_max,
                 pad_right=True):
    gap = np.floor(t_max/n_impulse)
    u = np.zeros(t_max)
    if pad_right:
        impulse_indx = np.arange(n_impulse)*gap
    else:
        impulse_indx = np.arange(n_impulse+1)*gap-1
    impulse_indx = impulse_indx.astype(int)
    u[impulse_indx]=1 
    return u

def var_fit_automaton(drugless_rates,
                    ic50,
                    n_gen=40,  # Number of simulated generations
                    mut_rate=0.001,  # probability of mutation per generation
                    max_cells=10**5,  # Max number of cells
                    death_rate=0.3,  # Death rate
                    init_counts=None,
                    carrying_cap=True,
                    plot = True,
                    curve_type = 'linear',
                    const_dose = 0,
                    slope=100,
                    max_dose = 10,
                    min_dose=1,
                    h_step=100,
                    k_elim=0.01,
                    k_abs=0.1,
                    pharm_impulse_response=0,
                    div_scale = 1
                    ):

#    drugless_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\ogbunugafor_drugless.csv"
##    ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\cycloguanil_ic50.csv"
#    ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\pyrimethamine_ic50.csv"
#    drugless_rates = load_fitness(drugless_path)
#    ic50 = load_fitness(ic50_path)
    
    # Number of different alleles
    n_allele = len(drugless_rates)
#    print(str(drugless_rates.shape))
    # Obtain transition matrix for mutations
    P = random_mutations( n_allele )
   
    
    # Keeps track of cell counts at each generation
    counts = np.zeros([n_gen, n_allele], dtype=int)
    drug_curve = np.zeros(n_gen)

    if init_counts is None:
        counts[0] = 10*np.ones(n_allele)
    else:
        counts[0] = init_counts

    for mm in range(n_gen-1):
        
        if curve_type == 'constant':
            # dose is in uM
            conc = const_dose
        elif curve_type == 'impulse-response':
            if mm>pharm_impulse_response.shape[0]-1:
                conc=0
            else:
                conc = pharm_impulse_response[mm]
        else:
            conc = calc_conc(mm,curve_type,steepness=slope,max_dose=max_dose,
                             h_step=h_step,
                             min_dose=min_dose,
                             K_elim=k_elim,
                             K_abs=k_abs)
        
        drug_curve[mm] = conc
        
        fit_land = np.zeros(n_allele)
        

        for kk in range(n_allele):
            if kk == 3:
                fit_land[kk] = 0
            elif kk < 3:
                fit_land[kk] = gen_fitness(kk,conc,drugless_rates,ic50)
            elif kk > 3:
                fit_land[kk] = gen_fitness(kk,conc,drugless_rates,ic50)
        
        fit_land = fit_land*div_scale
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
            counts[:,3] =  0
            #Substract mutating cells from that allele
            daughter_counts[allele] -=n_mut

        counts[mm+1] += daughter_counts
    if plot:
        print(str('here!'))
        plot_timecourse(counts,drug_curve)
#        print('im here!')
#    return counts
    return counts, drug_curve

def vectorized_abm(drugless_rates,
                  ic50,
                  n_gen=40,  # Number of simulated generations
                  mut_rate=0.001,  # probability of mutation per generation
                  mut_noise=0.05,
                  max_cells=10**5,  # Max number of cells
                  death_rate=0.3,  # Death rate
                  death_noise=0.1,
                  init_counts=None,
                  carrying_cap=True,
                  plot = True,
                  curve_type = 'constant',
                  const_dose = 0,
                  slope=100,
                  max_dose = 10,
                  min_dose=1,
                  h_step=100,
                  k_elim=0.01,
                  k_abs=0.1,
                  pharm_impulse_response = 0,
                  div_scale = 1
                  ):

    # Obtain transition matrix for mutations

    # Number of different alleles
    n_allele = len(drugless_rates)
    P = random_mutations(n_allele)
#    print(str(n_allele))
    
    # Keeps track of cell counts at each generation
    counts = np.zeros([n_gen, n_allele])
    drug_curve = np.zeros(n_gen)

    if init_counts is None:
        counts[0, :] = 10
    else:
        counts[0, :] = init_counts


    for mm in range(n_gen):
        if curve_type == 'constant':
            # dose is in uM
            conc = const_dose
        elif curve_type == 'impulse-response':
            if mm>pharm_impulse_response.shape[0]-1:
                conc=0
            else:
                conc = pharm_impulse_response[mm]
        else:
            conc = calc_conc(mm,curve_type,steepness=slope,max_dose=max_dose,
                             h_step=h_step,
                             min_dose=min_dose,
                             K_elim=k_elim,
                             K_abs=k_abs)
        
        drug_curve[mm] = conc
        
        fit_land = np.zeros(n_allele)
        

        for kk in range(n_allele):
            if kk == 3:
                fit_land[kk] = 0
            elif kk < 3:
                fit_land[kk] = gen_fitness(kk,conc,drugless_rates,ic50)
            elif kk > 3:
                fit_land[kk] = gen_fitness(kk,conc,drugless_rates,ic50)

        fit_land = fit_land*div_scale # scale division rate
        # Death of cells
#        n_cells = np.sum(counts[mm])

        dead_cells = np.random.normal(death_rate, death_noise, n_allele)
        dead_cells =  counts[mm]* dead_cells

        counts[mm] = counts[mm] - np.int_(dead_cells)
        counts[mm, counts[mm] < 0] = 0

        # Divide and mutate
        # Scale division rates based on carrying capacity
        if carrying_cap:
            division_scale = 1 / (1+(2*np.sum(counts[mm])/max_cells)**4)
        else:
            division_scale = 1

        if counts[mm].sum()>max_cells:
            division_scale = 0

        dividing_cells = np.int_(counts[mm]*fit_land*division_scale)

#         mutating_cells = dividing_cells*mut_rate

        mutating_cells = np.random.normal(mut_rate, mut_noise, n_allele)
        mutating_cells =  dividing_cells* mutating_cells
        mutating_cells = np.int_(mutating_cells)

        final_types = np.zeros(n_allele)

        # Mutate cells of each allele type
        for allele in np.random.permutation(np.arange(n_allele)):
            if mutating_cells[allele] > 0:
                mutations = np.random.choice(
                    n_allele, size=mutating_cells[allele], p=P[allele])

                final_types += np.bincount(mutations, minlength=n_allele)

        # Add final types to the cell counts
        new_counts = counts[mm] + dividing_cells - mutating_cells + final_types

        counts[mm] = new_counts
        counts[mm, counts[mm] < 0] = 0

        if mm < n_gen-1:
            counts[mm+1] = counts[mm]
#    if plot:
#        plot_timecourse(counts,drug_curve)
#        print('im here!')
#    return counts
    return counts, drug_curve

def plot_timecourse(counts, drug_curve,
                    fig_title = '',
                    drug_log_scale=False,
                    counts_log_scale=False,
                    normalize=False):
    
    fig, ax = plt.subplots(figsize = (6,4))
#    plt.rcParams.update({'font.size': 22})
    counts_total = np.sum(counts,axis=0)
    
    sorted_index = counts_total.argsort()
    sorted_index_big = sorted_index[8:]
    
#    colors = sns.color_palette('Set1')
#    colors = sns.color_palette("hls", 8)
    colors = sns.color_palette('bright')
    colors = np.concatenate((colors[0:9],colors[0:7]),axis=0)
    # shuffle colors
#    c1 = colors[15]
#    c2 = colors[14]
#    colors[15] = c2
#    colors[14] = c1
    colors[[14,15]] = colors[[15,14]]
    
    cc = (cycler(color=colors) + 
          cycler(linestyle=['-', '-','-','-','-','-','-','-','-',
                            '--','--','--','--','--','--','--']))
    
    ax.set_prop_cycle(cc)
#    if counts_log_scale:
#        counts[counts==0] = np.nan
#        counts = np.log10(counts)
#        counts[counts==np.nan] = 0
#    for allele in range(counts.shape[1]):
#        if allele in sorted_index_big:
#            ax.plot(counts[:,allele],linewidth=3.0,label=str(int_to_binary(allele)))
##            print(str(allele))
#        else:
#            ax.plot(counts[:,allele],linewidth=3.0,label=None)
#    ax.legend(loc=(1.25,.03),frameon=False,fontsize=15)
##    ax.legend(frameon=False,fontsize=15)
##        ax.legend([str(int_to_binary(allele))])
#        
#    ax.set_xlim(0,counts.shape[0])
#    ax.set_facecolor(color='w')
#    ax.grid(False)
#
#    ax.set_xlabel('Time',fontsize=20)
#    ax.set_ylabel('Cells',fontsize=20)
#    ax.tick_params(labelsize=20)
#    if counts_log_scale:
#        ax.set_yscale('log')
#        ax.set_ylim(10,5*10**5)
#    else:
#        ax.set_yticks([0,20000,40000,60000,80000,100000])
#        ax.set_yticklabels(['0','$2x10^{5}$','$4x10^{5}$','$6x10^{5}$',
#                            '$8x10^{5}$','$10x10^{5}$'])
#        ax.set_ylim(0,100000)
#    plt.yticks(fontsize=18)
#    ax.set_ylim(0,ax.get_ylim()[1])
#    ax.set_ylim(0,80000)
    
    
#    ylabel = ax.yaxis.get_label()
#    ylabel_xpos = ylabel.get_position()[0]
    
    ax2 = ax.twinx()
#    color = 'k'
    color = [0.5,0.5,0.5]
    ax2.set_ylabel('Drug Concentration (uM)', color=color,fontsize=20) # we already handled the x-label with ax1
    
    if drug_log_scale:
        if all(drug_curve>0):
            drug_curve = np.log10(drug_curve)
        yticks = np.log10([10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3])    
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(['0','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$',
                         '$10^1$','$10^2$','$10^3$'])
        ax2.set_ylim(-4,3)
    else:
        ax2.set_ylim(0,1.1*max(drug_curve))

#    ax2.plot(drug_curve, color=color, linewidth=3.0, linestyle = 'dashed')
    ax2.plot(drug_curve, color=color, linewidth=2.0)
    ax2.tick_params(axis='y', labelcolor=color)
    
#    ax2.legend(['Drug Concentration'],loc=(.7,.9))
#    leg_loc = ax.get_legend().get_window_extent().p1 # in pixels
#    leg_loc = ax.get_legend()
#    leg_loc = leg_loc.get_window_extent()
#    leg_loc = leg_loc.p1
    
#    ax_ll = ax2.get_window_extent().p0 # lower left pos of axes
#    ax_ur = ax2.get_window_extent().p1 # upper right pos of axes
#    ax_width = ax_ur[0]-ax_ll[0]
#    ax_height = ax_ur[1]-ax_ll[1]
#    
#    leg_loc[0] = leg_loc[0]/ax_width
#    leg_loc[1] = leg_loc[1]/ax_height

    ax2.legend(['Drug Conc.'],loc=(1.25,0.85),frameon=False,fontsize=15)
#    ax2.set_ylim(ymin=0,ymax=10**3)
#    ax2.set_yscale('log')
#    ax2.set_ylim(0,10**3)
 
#    t_0 = np.array([0])
#    yticks = np.log10([10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3])
##    yticks = np.concatenate((t_0,t_1),axis=None)
#    
#    ax2.set_yticks(yticks)
#    ax2.set_yticklabels(['0','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$',
#                         '$10^1$','$10^2$','$10^3$'])
#    ax.set_yticks
#    plt.ylim(0,10**3)
#    plt.ylim(0,max(drug_curve)+0.1*max(drug_curve))
    
    ax2.tick_params(labelsize=18)
#    plt.yticks(fontsize=18)
    ax2.set_title(fig_title,fontsize=20)
    if normalize:
        counts = counts/np.max(counts)
        
    for allele in range(counts.shape[1]):
        if allele in sorted_index_big:
            ax.plot(counts[:,allele],linewidth=3.0,label=str(int_to_binary(allele)))
#            print(str(allele))
        else:
            ax.plot(counts[:,allele],linewidth=3.0,label=None)
    ax.legend(loc=(1.25,.03),frameon=False,fontsize=15)
#    ax.legend(frameon=False,fontsize=15)
#        ax.legend([str(int_to_binary(allele))])
        
    ax.set_xlim(0,counts.shape[0])
    ax.set_facecolor(color='w')
    ax.grid(False)

    ax.set_xlabel('Time',fontsize=20)
    ax.set_ylabel('Cells',fontsize=20)
    ax.tick_params(labelsize=20)
    
    if counts_log_scale:
        ax.set_yscale('log')
        ax.set_ylim(1,5*10**5)
    else:
#        ax.set_yticks([0,20000,40000,60000,80000,100000])
#        ax.set_yticklabels(['0','$2x10^{5}$','$4x10^{5}$','$6x10^{5}$',
#                            '$8x10^{5}$','$10x10^{5}$'])
        ax.set_ylim(0,np.max(counts))

#    fig.tight_layout()
    plt.show()
    return ax

def plot_fitness_curves(fig_title=''):
    
    drugless_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\ogbunugafor_drugless.csv"
    ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\pyrimethamine_ic50.csv"
#    ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\pyrimethamine_ic50.csv"
    drugless_rates = load_fitness(drugless_path)
    ic50 = load_fitness(ic50_path)
    
    fig, ax = plt.subplots(figsize = (8,6))
    
    powers = np.linspace(-3,5,20)
    conc = np.power(10*np.ones(powers.shape[0]),powers)
    fit = np.zeros(conc.shape[0])
    
    colors = sns.color_palette('bright')
    colors = np.concatenate((colors[0:9],colors[0:7]),axis=0)
    colors[[14,15]] = colors[[15,14]]
    
    cc = (cycler(color=colors) + 
          cycler(linestyle=['-', '-','-','-','-','-','-','-','-',
                            '--','--','--','--','--','--','--']))
    ax.set_prop_cycle(cc) 
    
    for allele in range(16):
        if allele == 3:
            fit = np.zeros(conc.shape[0])
        if allele > 3:
            for j in range(conc.shape[0]):
                fit[j] = gen_fitness(allele,conc[j],drugless_rates,ic50)
        else:
            for j in range(conc.shape[0]):
                fit[j] = gen_fitness(allele,conc[j],drugless_rates,ic50)
        ax.plot(powers,fit,linewidth=3,label=str(int_to_binary(allele)))
#    ind = np.arange(9)
    ax.legend(fontsize=15,frameon=False,loc=(1.05,-.30))
    ax.set_xticks([-3,-2,-1,0,1,2,3,4,5])
    ax.set_xticklabels(['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$',
                         '$10^1$','$10^2$','$10^3$','$10^4$','$10^5$'])
    
    plt.title(fig_title,fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.xlabel('Drug concentration ($\mathrm{\mu}$M)',fontsize=20)
    plt.ylabel('Growth Rate',fontsize=20)
    ax.set_frame_on(False)
    return ax

def plot_2d_fitness(cmap='magma_r'):
    curve_path = 'C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\pyr_2d_fitness_curve.csv'
    
    curve = pd.read_csv(curve_path,header=None)
    curve = np.array(curve)
    fig, ax = plt.subplots(figsize=(10,8))
    x_ticks = ['0','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$',
                         '$10^1$','$10^2$','$10^3$','$10^4$','$10^5$']
    
    y_ticks = []
    for allele in range(16):
        y_ticks.append(str(int_to_binary(allele)))
    sns.heatmap(curve,linewidth = 0.5,cmap=cmap,square=False,yticklabels=y_ticks,xticklabels=x_ticks,ax=ax)
#    ax.
    ax.yaxis.set_tick_params(rotation=0)
    ax.tick_params(axis='y',labelsize=15)
    ax.tick_params(axis='x',labelsize=15)
    ax.set_xlabel('Concentration (uM)',fontsize=20)
    ax.set_ylabel('Allele',fontsize=20)
    plt.title('Cycloguanil',fontsize=20)
    return