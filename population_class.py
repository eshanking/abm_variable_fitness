# TODO: check fitness landscape computation in abm model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns
import scipy as sp

class Population:
###############################################################################    
    # Initializer
    def __init__(self,
                 drugless_path = None,
                 ic50_path = None,
                 n_gen=1000,
                 max_cells = 10**6,
                 init_counts = None,
                 steepness = None,
                 max_dose = 1,
                 n_impulse = 2,
                 pad_right = False,
                 h_step = 500,
                 min_dose = 0,
                 mut_rate = 0.01,
                 death_rate = 0.15,
                 death_noise = 0.01,
                 mut_noise = 0.005,
                 carrying_cap = True,
                 thresh = 5000,
                 k_elim = 0.001,
                 k_abs = 0.07,
                 div_scale = 1,
                 normalize = False,
                 n_sims = 1,
                 mode='ab',
                 curve_type='constant',
                 plot=True,
                 drug_log_scale = False,
                 counts_log_scale = False,
                 constant_pop = False,
                 fig_title = '',
                 drug_curve = None,
                 v2=True,
                 prob_drop=0,
                 debug=False):
                
        # Evolutionary parameters
        
        # Number of generations (time steps)
        self.n_gen = n_gen
        self.max_cells = max_cells
        
        # Initial number of cells (default = 10,000 at 0000)
        if init_counts is None:
            self.init_counts = np.zeros(16)
            self.init_counts[0] = 10**4
        else:
            self.init_counts = init_counts
        
        # ABM parameters
        self.mut_rate = mut_rate
        self.death_rate = death_rate
        
        # Vectorized model parameters
        self.death_noise = death_noise
        self.mut_noise = mut_noise
        
        # Timecouse (set after running self.simulate)
        self.counts = np.zeros([self.n_gen,16])
        self.counts_extinct = np.zeros([self.n_gen,16])
        self.counts_survive = np.zeros([self.n_gen,16])
                                            
        # Model parameters
        
        self.mode = mode # ABM vs vectorized vs hybrid (not yet implemented)
        self.carrying_cap = True
        self.thresh = thresh # Threshold for hybrid model (not yet implemented)
        self.div_scale = div_scale # Scale the division rate to simulate different organisms
        self.n_sims = n_sims # number of simulations to average together in self.simulate
        self.constant_pop = constant_pop
        self.v2 = v2
        self.debug = debug
        
        # Data paths
        if drugless_path is None:
            self.drugless_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\ogbunugafor_drugless.csv"
        else:
            self.drugless_path = drugless_path
            
        if ic50_path is None:
            self.ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\pyrimethamine_ic50.csv"
        else:
            self.ic50_path = ic50_path
        
        # load the data
        self.drugless_rates = self.load_fitness(self.drugless_path)
        self.ic50 = self.load_fitness(self.ic50_path)
        
        self.timestep_scale = max(self.drugless_rates)

        
        # determine number of alleles from data (not yet implemented)
        self.n_allele = self.drugless_rates.shape[0]
        
        # Dose parameters
        self.curve_type = curve_type # linear, constant, heaviside, pharm, pulsed
        
        # Pharmacological paramters
        self.k_elim = k_elim
        self.k_abs = k_abs 
        self.pad_right = pad_right
        
        if steepness is None:
            self.steepness = self.n_gen # Ramped parameter (what time step to reach maximum dose, determines slope)
        else:
            self.steepness = steepness
        
        self.max_dose = max_dose
        self.n_impulse = n_impulse # number of impulses for a pulsed dose
        self.prob_drop = prob_drop # probability of dropping a dose
        self.h_step = h_step # when to turn on heaviside function
        self.min_dose = min_dose 
        
        # Generate drug dosage curves if one is not specified
        if drug_curve is None:
            self.drug_curve = self.gen_curves()
        else:
            self.drug_curve = drug_curve
        
        # Visualization parameters
        self.plot = plot # boolean
        self.drug_log_scale = drug_log_scale # plot drugs on log scale
        self.counts_log_scale = counts_log_scale # plot counts on log scale
        self.fig_title = fig_title
        self.normalize = normalize
        self.counts_log_scale = counts_log_scale
###############################################################################       
    
    # Load data
    def load_fitness(self,data_path):
            # also use to load ic50 and drugless growth rate
            fitness = pd.read_csv(data_path)
            cols = list(fitness.columns)
            fit_array = np.array(cols)
            fit_array = fit_array.astype(np.float)
            return fit_array
        
###############################################################################
    # ABM helper methods
    
    # converts decimals to binary
    def int_to_binary(self,num, pad=4):
        return bin(num)[2:].zfill(pad)
    
    # computes hamming distance between two genotypes
    def hammingDistance(self,s1,s2):
        assert len(s1) == len(s2)
        return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))
    
    # converts an integer to a genotype and padding to the left by 0s
    def convertIntToGenotype(self,anInt,pad):
    	offset = 2**pad
    	return [int(x) for x in bin(offset+anInt)[3:]]
    
    def random_mutations(self,N):
        trans_mat = np.zeros([N,N])
        for mm in range(N):
            for nn in range(N):
                trans_mat[mm, nn] = self.hammingDistance( self.int_to_binary(mm) , self.int_to_binary(nn))
        # Don't include mutant no. 4
        
        trans_mat[trans_mat>1] = 0
        trans_mat = trans_mat/trans_mat.sum(axis=1)
        
    #    trans_mat[3,:] = 0
    #    trans_mat[:,3] = 0
    #    print(str(trans_mat))
        return trans_mat
    
    # compute fitness given a drug concentration
    def gen_fitness(self,allele,conc,drugless_rate,ic50):
        # input: allele (integer from 0 to 15)
        # conc: current drug concentration
        # output: allele fitness
        c = -.6824968 # empirical curve fit
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
    
    def gen_fit_land(self,conc):
        
        fit_land = np.zeros(16)
        
        for allele in range(16):
            fit_land[allele] = self.gen_fitness(allele,conc,self.drugless_rates,self.ic50)
        
        return fit_land
    
###############################################################################
    # Methods for generating drug curves
    
    # Equation for a simple 1 compartment pharmacokinetic model
    def pharm_eqn(self,t,k_elim=0.01,k_abs=0.1,max_dose=1):
        conc = np.exp(-k_elim*t)-np.exp(-k_abs*t)
        t_max = np.log(k_elim/k_abs)/(k_elim-k_abs)
        conc = conc/(np.exp(-k_elim*t_max)-np.exp(-k_abs*t_max))
        conc = conc*max_dose
        return conc
    
    # Old convolution algorithm (very slow)
    # Convolve the arbitrary curve u with the pharmacokinetic model
    # def convolve_pharm(self,u):
    #                    # k_elim=0.01,
    #                    # k_abs=0.1,
    #                    # max_dose=1):
    #     k_elim = self.k_elim
    #     k_abs = self.k_abs
    #     max_dose = self.max_dose
        
    #     # k_lim and k_abs are the absorption and elimination rate constants for the pharmacokinetic model
    #     # t_max is the max length of the output curve
    #     # algorithm is at best O(n^2)...
        
    #     conv = np.zeros(self.n_gen)
    #     for t in range(self.n_gen):
    #         for tau in range(self.n_gen):
    #             if t-tau >= 0 and t-tau<u.shape[0]:
    #                 conv[t] += u[t-tau]*self.pharm_eqn(tau,k_elim=k_elim,k_abs=k_abs,max_dose=max_dose)
    #     # conv = np.convolve()
    #     return conv

    # New convolution method (much faster with numpy)
    def convolve_pharm(self,u):
                       # k_elim=0.01,
                       # k_abs=0.1,
                       # max_dose=1):
        k_elim = self.k_elim
        k_abs = self.k_abs
        max_dose = self.max_dose
        
        # k_lim and k_abs are the absorption and elimination rate constants for the pharmacokinetic model
        # t_max is the max length of the output curve
        # algorithm is at best O(n^2)...
        
        # conv = np.zeros(self.n_gen)
        # for t in range(self.n_gen):
        #     for tau in range(self.n_gen):
        #         if t-tau >= 0 and t-tau<u.shape[0]:
        #             conv[t] += u[t-tau]*self.pharm_eqn(tau,k_elim=k_elim,k_abs=k_abs,max_dose=max_dose)
        
        pharm = np.zeros(self.n_gen)
        for i in range(self.n_gen):
            pharm[i] = self.pharm_eqn(i,k_elim=k_elim,k_abs=k_abs,max_dose=max_dose)
        
        # using FFT turns out to be much faster for convolutions!
        conv = np.convolve(u,pharm)
        conv = conv[0:self.n_gen]
        return conv
    
    # Generates an impulse train to input to convolve_pharm()
    def gen_impulses(self):
        gap = np.floor(self.n_gen/self.n_impulse)
        u = np.zeros(self.n_gen)
        if self.pad_right:
            impulse_indx = np.arange(self.n_impulse)*gap
        else:
            impulse_indx = np.arange(self.n_impulse+1)*gap-1
            
        # eliminate random doses
        keep_indx = np.random.rand(len(impulse_indx)) > self.prob_drop
        
        impulse_indx = impulse_indx[keep_indx]
        
        impulse_indx = impulse_indx.astype(int)
        u[impulse_indx]=1 
        return u
    
    # generates drug concentration curves
    def gen_curves(self):
        curve = np.zeros(self.n_gen)
        # print('hi')
        if self.curve_type == 'linear': # aka ramp linearly till timestep defined by steepness
            for i in range(self.n_gen):
                if i <= self.steepness:
                    slope = (self.max_dose-10**(-3))/self.steepness
                    conc = slope*i+10**-3
                else:
                    # step = self.steepness
                    slope = (self.max_dose-10**(-3))/self.steepness
                    conc = slope*i+10**-3
                curve[i]=conc
                
        elif self.curve_type == 'constant':
            curve[:] = self.max_dose
            # print('here')

        elif self.curve_type == 'heaviside':
            for i in range(self.n_gen):
                if i <= self.h_step:
                    curve[i] = self.min_dose
                else:
                    curve[i] = self.max_dose 
        
        # Two compartment pharmacokinetic model
        elif self.curve_type == 'pharm':
            t_max = np.log(self.k_elim/self.k_abs)/(self.k_elim-self.k_abs)
            for i in range(self.n_gen):
                conc = np.exp(-self.k_elim*i)-np.exp(-self.k_abs*i)
                conc = conc/(np.exp(-self.k_elim*t_max)-np.exp(-self.k_abs*t_max))
                conc = conc*self.max_dose
                curve[i] = conc
        
        # Pulsed convolves an impulse train with the 1-compartment model (models patient taking a maintenence dose)
        elif self.curve_type == 'pulsed':
            u = self.gen_impulses()
            curve = self.convolve_pharm(u)
        return curve

###############################################################################
    # Run one abm simulation (ignores n_sim)
    def run_abm(self):
        
        n_allele = len(self.drugless_rates)

        # Obtain transition matrix for mutations
        P = self.random_mutations( n_allele )

        # Keeps track of cell counts at each generation
        counts = np.zeros([self.n_gen, n_allele], dtype=int)
        # drug_curve = np.zeros(self.n_gen)

        counts[0,:] = self.init_counts
    
        for mm in range(self.n_gen-1):
            # Normalize to constant population
                            
            conc = self.drug_curve[mm]
            
            fit_land = np.zeros(self.n_allele)
            
            # fitness of allele 0010 is not quantified in the dataset - set to zero
            for kk in range(self.n_allele):
                if kk == 3:
                    fit_land[kk] = 0 # fitness of allele 0010 is not quantified in the dataset
                elif kk < 3:
                    fit_land[kk] = self.gen_fitness(kk,conc,self.drugless_rates,self.ic50)
                elif kk > 3:
                    fit_land[kk] = self.gen_fitness(kk,conc,self.drugless_rates,self.ic50)
            
            fit_land = fit_land*self.div_scale
            n_cells = np.sum( counts[mm] )
    
            # Scale division rates based on carrying capacity
            if self.carrying_cap:
                division_scale = 1 / (1+(2*np.sum(counts[mm])/self.max_cells)**4)
            else:
                division_scale = 1
    
            if counts[mm].sum()>self.max_cells:
                division_scale = 0
    
            div_rate = np.repeat( fit_land*division_scale, counts[mm] )
            cell_types = np.repeat( np.arange(n_allele) , counts[mm] )
    
            # Death of cells
            death_rates = np.random.rand(n_cells)
            surv_ind = death_rates > self.death_rate
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
                n_mut = np.sum( np.random.rand( daughter_counts[allele] ) < self.mut_rate )
    
                # note that columns in P are normalized to probability densities (columns sum to 1)
                mutations = np.random.choice(n_allele, size=n_mut, p=P[:,allele]).astype(np.uint8)
    
                #Add mutating cell to their final types
                counts[mm+1] +=np.bincount( mutations , minlength=n_allele)
                counts[:,3] =  0
                
                #Substract mutating cells from that allele
                daughter_counts[allele] -=n_mut
    
            counts[mm+1] += daughter_counts
            
            # Normalize to constant population            
            if self.constant_pop:
                cur_size = np.sum(counts[mm+1])
                counts[mm+1] = counts[mm+1]*self.init_counts[0]/cur_size
                counts[mm+1] = np.floor(counts[mm+1])

        return counts
    
    def run_abm_v2(self):
        
        n_allele = len(self.drugless_rates)

        # Obtain transition matrix for mutations
        P = self.random_mutations( n_allele )

        # Keeps track of cell counts at each generation
        counts = np.zeros([self.n_gen, n_allele], dtype=int)
        # drug_curve = np.zeros(self.n_gen)

        counts[0,:] = self.init_counts
    
        for mm in range(self.n_gen-1):
            
            if self.debug:
                if np.mod(mm,10) == 0:
                    print(str(mm))
                            
            conc = self.drug_curve[mm]
            
            fit_land = np.zeros(self.n_allele)
            
            # fitness of allele 0010 is not quantified in the dataset - set to zero
            for kk in range(self.n_allele):
                if kk == 3:
                    fit_land[kk] = 0 # fitness of allele 0010 is not quantified in the dataset
                elif kk < 3:
                    fit_land[kk] = self.gen_fitness(kk,conc,self.drugless_rates,self.ic50)
                elif kk > 3:
                    fit_land[kk] = self.gen_fitness(kk,conc,self.drugless_rates,self.ic50)
            
            fit_land = fit_land*self.div_scale
    
            # Scale division rates based on carrying capacity
            if self.carrying_cap:
                division_scale = 1 / (1+(2*np.sum(counts[mm])/self.max_cells)**4)
            else:
                division_scale = 1
    
            if counts[mm].sum()>self.max_cells:
                division_scale = 0
            
            fit_land = fit_land*division_scale
            
            # fit_land = fit_land/self.timestep_scale # ensures fitness is never greater than 1
            # death_rate = self.death_rate/self.timestep_scale # scales all other rates proportionally
            # mut_rate = self.mut_rate/self.timestep_scale
            
            death_rate = self.death_rate
            mut_rate = self.mut_rate
            
            counts[mm+1] = counts[mm]
    
            # Kill cells
            
            counts[mm+1] = counts[mm+1]-np.random.binomial(counts[mm],death_rate)
    
            # Divide cells
            
            divide = np.random.binomial((counts[mm+1]*self.timestep_scale).astype(int),fit_land/self.timestep_scale)
            
            # Mutate cells
            
            daughter_types = np.repeat( np.arange(n_allele) , divide )
            daughter_counts = np.bincount( daughter_types , minlength=n_allele )
    
            # Mutate cells of each allele type
            for allele in np.random.permutation(np.arange(n_allele)):
                n_mut = np.random.binomial(daughter_counts[allele],mut_rate)
    
                mutations = np.random.choice(n_allele, size=n_mut, p=P[:,allele]).astype(np.uint8)
    
                # Add mutating cell to their final types
                counts[mm+1] +=np.bincount( mutations , minlength=n_allele)
                counts[:,3] =  0
                # Substract mutating cells from that allele
                daughter_counts[allele] -=n_mut
    
            counts[mm+1] += daughter_counts
            
            # Normalize to constant population            
            if self.constant_pop:
                cur_size = np.sum(counts[mm+1])
                counts[mm+1] = counts[mm+1]*self.init_counts[0]/cur_size
                counts[mm+1] = np.floor(counts[mm+1])

        return counts

    # Runs abm simulation n_sim times and averages results. Then sets self.counts to final result. Also quantifies survival number
    def simulate(self):
        
        counts_t = np.zeros([self.n_gen,16])
        counts = np.zeros([self.n_gen,16])
        counts_survive = np.zeros([self.n_gen,16])
        counts_extinct = np.zeros([self.n_gen,16])
        
        n_survive = 0
        for i in range(self.n_sims):
            
            if self.v2:
                counts_t = self.run_abm_v2()
            else:
                counts_t = self.run_abm()
                
            if any(counts_t[self.n_gen-1,:]>0.1*self.max_cells):
                n_survive+=1
                counts_survive += counts_t
            # else:
                # counts_extinct += counts_t
                
            counts+=counts_t
                                                                                                      
        counts = counts/self.n_sims
        counts_survive = counts_survive/n_survive
        # counts_extinct = counts_extinct/(self.n_sims-n_survive)
        
        self.counts = counts
        self.counts_survive = counts_survive
        self.counts_extinct = counts_extinct
        
        return counts, n_survive

    def plot_timecourse(self,counts_t=None):
        
        if (self.counts == 0).all() and counts_t==None:
            print('No data to plot!')
            return
        elif counts_t is None:
            counts = self.counts
        else:
            counts = counts_t # an input other than self overrides self
            
        fig, ax = plt.subplots(figsize = (6,4))
    #    plt.rcParams.update({'font.size': 22})
        counts_total = np.sum(counts,axis=0)
        
        sorted_index = counts_total.argsort()
        sorted_index_big = sorted_index[8:]
        
        colors = sns.color_palette('bright')
        colors = np.concatenate((colors[0:9],colors[0:7]),axis=0)
        # shuffle colors

        colors[[14,15]] = colors[[15,14]]
        
        cc = (cycler(color=colors) + 
              cycler(linestyle=['-', '-','-','-','-','-','-','-','-',
                                '--','--','--','--','--','--','--']))
        
        ax.set_prop_cycle(cc)
        
        ax2 = ax.twinx()

        color = [0.5,0.5,0.5]
        ax2.set_ylabel('Drug Concentration (uM)', color=color,fontsize=20) # we already handled the x-label with ax1
        
        if self.drug_log_scale:
            if all(self.drug_curve>0):
                drug_curve = np.log10(self.drug_curve)
            yticks = np.log10([10**-4,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3])    
            ax2.set_yticks(yticks)
            ax2.set_yticklabels(['0','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$',
                             '$10^1$','$10^2$','$10^3$'])
            ax2.set_ylim(-4,3)
        else:
            drug_curve = self.drug_curve
            ax2.set_ylim(0,1.1*max(drug_curve))
    
    #    ax2.plot(drug_curve, color=color, linewidth=3.0, linestyle = 'dashed')
        ax2.plot(drug_curve, color=color, linewidth=2.0)
        ax2.tick_params(axis='y', labelcolor=color)
            
        ax2.legend(['Drug Conc.'],loc=(1.25,0.85),frameon=False,fontsize=15)
        
        ax2.tick_params(labelsize=18)
    #    plt.yticks(fontsize=18)
        ax2.set_title(self.fig_title,fontsize=20)
        
        if self.normalize:
            counts = counts/np.max(counts)
            
        for allele in range(counts.shape[1]):
            if allele in sorted_index_big:
                ax.plot(counts[:,allele],linewidth=3.0,label=str(self.int_to_binary(allele)))
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
        
        if self.counts_log_scale:
            ax.set_yscale('log')
            ax.set_ylim(1,5*10**5)
        else:
    #        ax.set_yticks([0,20000,40000,60000,80000,100000])
    #        ax.set_yticklabels(['0','$2x10^{5}$','$4x10^{5}$','$6x10^{5}$',
    #                            '$8x10^{5}$','$10x10^{5}$'])
            ax.set_ylim(0,np.max(counts))
    
    #    fig.tight_layout()
        plt.show()
        return fig,ax
    
    # Calculate the shannon-gibbs entropy (normalized population size)
    def entropy(self,counts=None):
        if counts is None:
            counts = self.counts

        k = np.sum(counts,1)
        entropy = np.zeros(counts.shape[0])
        counts_t = np.zeros(counts.shape)
        for i in range(counts.shape[0]):
            counts_t[i,:] = np.divide(counts[i,:],k[i])
            # counts_log = np.log(counts_t[i,:]+1)
            # entropy[i] = np.dot(counts_t[i,:],counts_log)
            entropy[i] = sp.stats.entropy(counts_t[i,:])

        return entropy
    
###############################################################################
# Testing

# p1 = Population(v2=True)
# c = p1.simulate()
# p1.plot_timecourse()

# options = {'n_gen':1000,'max_dose':1,'n_sims':10}
# p1 = Population(**options)
# c = p1.simulate()
# p1.plot_timecourse()