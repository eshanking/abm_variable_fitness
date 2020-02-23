# Hybrid model of pathogen population evolution - strict agent-based and vectorized
# Agent-based controls at lower popualation size

import numpy as np
import pandas as pd

###############################################################################
# Helper functions

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
###############################################################################
# Hybrid model
# threshold based on population size
#def hybrid_evol(drugless_rates,
#                ic50,
#                n_gen=40,  # Number of simulated generations
#                mut_rate=0.001,  # probability of mutation per generation
#                mut_noise=0.05,
#                max_cells=10**5,  # Max number of cells
#                death_rate=0.3,  # Death rate
#                death_noise=0.1,
#                init_counts=None,
#                carrying_cap=True,
#                plot = True,
#                curve_type = 'linear',
#                const_dose = 0,
#                slope=100,
#                max_dose = 10,
#                min_dose=1,
#                h_step=100,
#                k_elim=0.01,
#                k_abs=0.1,
#                pharm_impulse_response = 0,
#                div_scale = 1,
#                lower_thresh = 1000, # transition to ABM
#                upper_thresh = 2000 # transition to vectorized model
#                ):
#    
#    n_allele = len(drugless_rates)
#    P = random_mutations( n_allele )
#    counts = np.zeros([n_gen,n_allele])
#    drug_curve = np.zeros(n_gen)
#    
#    if init_counts is None:
#        counts[0] = 10*np.ones(n_allele)
#    else:
#        counts[0] = init_counts
#        
#    use_abm = False
################################################################################
## Begin loop    
#    for mm  in range(n_gen-1):
#        if mm > 0:
#            if sum(counts[mm-1]) < lower_thresh:
#                use_abm = True
#            elif sum(counts[mm-1]) > upper_thresh:
#                use_abm = False
#                
##        print(str(sum(counts[mm-1])))
#        
#        if use_abm:
##            print('abm')
#            if curve_type == 'constant':
#                # dose is in uM
#                conc = const_dose
#            elif curve_type == 'impulse-response':
#                if mm>pharm_impulse_response.shape[0]-1:
#                    conc=0
#                else:
#                    conc = pharm_impulse_response[mm]
#            else:
#                conc = calc_conc(mm,curve_type,steepness=slope,max_dose=max_dose,
#                                 h_step=h_step,
#                                 min_dose=min_dose,
#                                 K_elim=k_elim,
#                                 K_abs=k_abs)
#            
#            drug_curve[mm] = conc
#            
#            fit_land = np.zeros(n_allele)
#            
#    
#            for kk in range(n_allele):
#                if kk == 3:
#                    fit_land[kk] = 0
#                elif kk < 3:
#                    fit_land[kk] = gen_fitness(kk,conc,drugless_rates,ic50)
#                elif kk > 3:
#                    fit_land[kk] = gen_fitness(kk,conc,drugless_rates,ic50)
#            
#            fit_land = fit_land*div_scale
#            n_cells = np.sum( counts[mm] )
#            n_cells = n_cells.astype(int)
#            # Scale division rates based on carrying capacity
#            if carrying_cap:
#                division_scale = 1 / (1+(2*np.sum(counts[mm])/max_cells)**4)
#            else:
#                division_scale = 1
#    
#            if counts[mm].sum()>max_cells:
#                division_scale = 0
#            
#            counts = np.ceil(counts)
#            counts = counts.astype(int)
#            div_rate = np.repeat( fit_land*division_scale, counts[mm] )
#            cell_types = np.repeat( np.arange(n_allele) , counts[mm] )
#            counts = counts.astype(float)
#            # Death of cells
#            death_rates = np.random.rand(n_cells)
#            surv_ind = death_rates > death_rate
#            div_rate = div_rate[surv_ind]
#            cell_types = cell_types[surv_ind]
#            n_cells = len(cell_types)
#    
#            counts[mm+1] = np.bincount(cell_types, minlength=n_allele)
#    
#            #Divide and mutate cells
#            div_ind = np.random.rand(n_cells) < div_rate
#    
#            # Mutate cells
#            # initial state of allele types
#            daughter_types = cell_types[div_ind].copy()
#    
#            # Generate random numbers to check for mutation
#            daughter_counts = np.bincount( daughter_types , minlength=n_allele)
#    
#            # Mutate cells of each allele type
#            for allele in np.random.permutation(np.arange(n_allele)):
#                n_mut = np.sum( np.random.rand( daughter_counts[allele] ) < mut_rate )
#    
#                # note that columns in P are normalized to probability densities (columns sum to 1)
#                mutations = np.random.choice(n_allele, size=n_mut, p=P[:,allele]).astype(np.uint8)
#    
#                #Add mutating cell to their final types
#                counts[mm+1] +=np.bincount( mutations , minlength=n_allele)
#                counts[:,3] =  0
#                #Substract mutating cells from that allele
#                daughter_counts[allele] -=n_mut
#    
#            counts[mm+1] += daughter_counts
#        
#        else:
##            print('here')
#            if curve_type == 'constant':
#                # dose is in uM
#                conc = const_dose
#            elif curve_type == 'impulse-response':
#                if mm>pharm_impulse_response.shape[0]-1:
#                    conc=0
#                else:
#                    conc = pharm_impulse_response[mm]
#            else:
#                conc = calc_conc(mm,curve_type,steepness=slope,max_dose=max_dose,
#                                 h_step=h_step,
#                                 min_dose=min_dose,
#                                 K_elim=k_elim,
#                                 K_abs=k_abs)
#            
#            drug_curve[mm] = conc
#            
#            fit_land = np.zeros(n_allele)
#            
#    
#            for kk in range(n_allele):
#                if kk == 3:
#                    fit_land[kk] = 0
#                elif kk < 3:
#                    fit_land[kk] = gen_fitness(kk,conc,drugless_rates,ic50)
#                elif kk > 3:
#                    fit_land[kk] = gen_fitness(kk,conc,drugless_rates,ic50)
#    
#            fit_land = fit_land*div_scale # scale division rate
#            # Death of cells
#    #        n_cells = np.sum(counts[mm])
#    
#            dead_cells = np.random.normal(death_rate, death_noise, n_allele)
#            dead_cells =  counts[mm]* dead_cells
#    
#            counts[mm] = counts[mm] - np.int_(dead_cells)
#            counts[mm, counts[mm] < 0] = 0
#    
#            # Divide and mutate
#            # Scale division rates based on carrying capacity
#            if carrying_cap:
#                division_scale = 1 / (1+(2*np.sum(counts[mm])/max_cells)**4)
#            else:
#                division_scale = 1
#    
#            if counts[mm].sum()>max_cells:
#                division_scale = 0
#    
#            dividing_cells = np.int_(counts[mm]*fit_land*division_scale)
#    
#    #         mutating_cells = dividing_cells*mut_rate
#    
#            mutating_cells = np.random.normal(mut_rate, mut_noise, n_allele)
#            mutating_cells =  dividing_cells* mutating_cells
#            mutating_cells = np.int_(mutating_cells)
#    
#            final_types = np.zeros(n_allele)
#    
#            # Mutate cells of each allele type
#            for allele in np.random.permutation(np.arange(n_allele)):
#                if mutating_cells[allele] > 0:
#                    mutations = np.random.choice(
#                        n_allele, size=mutating_cells[allele], p=P[allele])
#    
#                    final_types += np.bincount(mutations, minlength=n_allele)
#    
#            # Add final types to the cell counts
#            new_counts = counts[mm] + dividing_cells - mutating_cells + final_types
#    
#            counts[mm] = new_counts
#            counts[mm, counts[mm] < 0] = 0
#    
#            if mm < n_gen-1:
#                counts[mm+1] = counts[mm]
################################################################################
#            
#    return counts, drug_curve

# threshold based on allele cell count
def hybrid_evol_2(drugless_rates,
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
                curve_type = 'linear',
                const_dose = 0,
                slope=100,
                max_dose = 10,
                min_dose=1,
                h_step=100,
                k_elim=0.01,
                k_abs=0.1,
                pharm_impulse_response = 0,
                div_scale = 1,
#                lower_thresh = 1000, # transition to ABM
#                upper_thresh = 2000 # transition to vectorized model
                thresh = 1000
                ):
    n_allele = len(drugless_rates)
    P = random_mutations( n_allele )
    counts = np.zeros([n_gen,n_allele])
    drug_curve = np.zeros(n_gen)

#    counts_vect_t = counts
#    counts_abm_t = counts    
    
    if init_counts is None:
        counts[0] = 10*np.ones(n_allele)
    else:
        counts[0] = init_counts
        
#    use_abm = counts[0,:] > 0
    
    for mm  in range(n_gen-1):
        # filter population matrix        
#        switch_to_abm = counts[mm] < lower_thresh # 1000
#        switch_to_vectorized = counts[mm] > upper_thresh # 2000
#        
##        print(str(switch_to_abm))
##        print(str(switch_to_vectorized))
##        switch_to_ab = [not i for i in switch_to_vectorized]
#        
#        use_abm[switch_to_abm] = 1
#        use_abm[switch_to_vectorized] = 0
#        
#        print(str(use_abm))
#        
#        counts_abm = counts*use_abm
#        counts_vectorized = counts*(1-use_abm)
#        
#        counts_abm_t[mm] = switch_to_abm
##        counts_abm_t[mm] = switch_to_ab
#        counts_vect_t[mm] = switch_to_vectorized
        
#        abm_indx = counts[mm] < thresh
##        abm_indx = abm_indx*1
#        abm_indx = abm_indx.astype(int)
#        abm_indx = np.atleast_2d(abm_indx)
#        
#        ident_abm = np.identity(16)
#        ident_vect = np.identity(16)
#        
#        for nn in range(16):
#            if abm_indx[0,nn]==0:
#                ident_abm[nn,nn] = 0
#            else:
#                ident_vect[nn,nn] = 0
#        
#        counts_abm = np.matmul(counts,ident_abm)
#        counts_vectorized = np.matmul(counts,ident_vect)
#        indx_abm = counts[:,mm] < thresh
        
        ident_abm = np.identity(16)
        ident = np.identity(16)
        
        for nn in range(16):
            if nn%2==1:
                ident_abm[nn,nn] = 0
        
        counts_abm = np.matmul(counts,ident_abm)
        counts_vectorized = np.matmul(counts,ident-ident_abm)
        
#        print(str(counts_abm.shape))
#        counts_abm_t[mm] = counts_abm[mm]
#        counts_vect_t[mm] = counts_vectorized[mm]
#        counts_abm_t[mm] = counts_abm[mm]
#        counts_vect_t[mm] = counts_vectorized[mm]
#        n_abm = sum(use_abm)
#        print(str(n_abm))
        counts_abm_t = counts_abm
        counts_vect_t = counts_vectorized
###############################################################################
        # ab model        
#        print(str(sum(counts_abm)))
        if curve_type == 'constant':
            # dose is in uMnn
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
        n_cells = np.sum( counts_abm[mm] )
        n_cells = n_cells.astype(int)
        # Scale division rates based on carrying capacity
        if carrying_cap:
            division_scale = 1 / (1+(2*np.sum(counts_abm[mm])/max_cells)**4)
        else:
            division_scale = 1

        if counts_abm[mm].sum()>max_cells:
            division_scale = 0
        
        counts_abm = np.ceil(counts_abm)
        counts_abm = counts_abm.astype(int)
        div_rate = np.repeat( fit_land*division_scale, counts_abm[mm] )
        cell_types = np.repeat( np.arange(n_allele) , counts_abm[mm] )
        counts_abm = counts_abm.astype(float)
        # Death of cells
        death_rates = np.random.rand(n_cells)
        surv_ind = death_rates > death_rate
        div_rate = div_rate[surv_ind]
        cell_types = cell_types[surv_ind]
        n_cells = len(cell_types)

        counts_abm[mm+1] = np.bincount(cell_types, minlength=n_allele)

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
            counts_abm[mm+1] +=np.bincount( mutations , minlength=n_allele)
            counts_abm[:,3] =  0
            #Substract mutating cells from that allele
            daughter_counts[allele] -=n_mut

        counts_abm[mm+1] += daughter_counts
###############################################################################
# vectorized model

#            print('here')
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
        dead_cells =  counts_vectorized[mm]* dead_cells

        counts_vectorized[mm] = counts_vectorized[mm] - np.int_(dead_cells)
        counts_vectorized[mm, counts_vectorized[mm] < 0] = 0

        # Divide and mutate
        # Scale division rates based on carrying capacity
        if carrying_cap:
            division_scale = 1 / (1+(2*np.sum(counts_vectorized[mm])/max_cells)**4)
        else:
            division_scale = 1

        if counts_vectorized[mm].sum()>max_cells:
            division_scale = 0

        dividing_cells = np.int_(counts_vectorized[mm]*fit_land*division_scale)

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
        new_counts = counts_vectorized[mm] + dividing_cells - mutating_cells + final_types

        counts_vectorized[mm] = new_counts
        counts_vectorized[mm, counts_vectorized[mm] < 0] = 0

        if mm < n_gen-1:
            counts_vectorized[mm+1] = counts_vectorized[mm]
        
        counts[mm+1] = counts_abm[mm+1] + counts_vectorized[mm+1]
    return counts, drug_curve, counts_abm_t, counts_vect_t