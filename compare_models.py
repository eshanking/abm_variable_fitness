import numpy as np
import abm_variable_fitness as sim
import hybrid_evol_model as hybrid_sim
#import math
import matplotlib.pyplot as plt
import time
# Run n_sims number of simulations per model over a grid of parameters
###############################################################################
# Global parameters
drugless_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\ogbunugafor_drugless.csv"
ic50_cyc = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\cycloguanil_ic50.csv"
ic50_pyr = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\pyrimethamine_ic50.csv"
drugless_rates = sim.load_fitness(drugless_path)
ic50_pyr = sim.load_fitness(ic50_pyr)
###############################################################################
# Constant parameters

n_gen = 1000
max_cells = 10**6
init_counts = np.zeros(16)
#init_counts[0] = 10000
init_counts[0] = 10**4

mut_rate = 0.01
death_rate = 0.15
death_noise = 0.01
mut_noise = 0.005

carrying_cap = True
plot = False
drug_log_scale = False
counts_log_scale = False

#lower_thresh = 1000
#upper_thresh = 2000

#thresh = math.inf
thresh = 5000
#thresh = 10000

k_elim = .001
k_abs = .07
div_scale = 1
curve_type = 'constant'

n_sims = 10
###############################################################################
# Scanned parameters
# 1 parameter scanned -> bar chart, 2 parameters -> heat map
graph_type = 'barchart'
max_dose = np.array([100,300,500,700])
#max_dose = np.array([700])
#max_dose = np.array([1])
#max_dose = np.array([0])
labels = ['agent-based','vectorized','hybrid']
###############################################################################
# Generate dose curve
#u = sim.gen_impulses(1,n_gen)
#conv = sim.convolve_pharm(u,2000,k_elim=k_elim,k_abs=k_abs, max_dose=1)
###############################################################################
# Run agent-based model
start = time.time()
n_survive_ab = np.zeros(max_dose.shape)
dose_num=0
for dose in max_dose:
    counts = np.zeros((n_gen,ic50_pyr.shape[0]))
#    drug_curve_t = conv*dose
    for sim_num in range(n_sims):
        counts_t, drug_curve = sim.var_fit_automaton(drugless_rates,
                                        ic50_pyr,
                                        n_gen=n_gen,  # Number of simulated generations
                                        mut_rate=mut_rate,  # probability of mutation per generation
                                        max_cells=max_cells,  # Max number of cells
                                        death_rate=death_rate,  # Death rate
                                        init_counts=init_counts,
                                        carrying_cap=carrying_cap,
                                        plot=plot,
                                        const_dose=dose,
                                        curve_type=curve_type)
#                                        pharm_impulse_response=drug_curve_t)
        
        if any(counts_t[n_gen-1,:]>0.1*max_cells):
            n_survive_ab[dose_num]+=1
        counts += counts_t
    dose_num+=1
    counts=np.divide(counts,n_sims)
    title = 'agent-based model, dose = ' + str(dose) + ' uM'
    sim.plot_timecourse(counts,drug_curve,drug_log_scale=drug_log_scale,
                        counts_log_scale=counts_log_scale,fig_title=title)    
elapsed_abm = time.time() - start

# Run vectorized model
start = time.time()
n_survive_vect = np.zeros(max_dose.shape)
dose_num=0
for dose in max_dose:
    counts = np.zeros((n_gen,ic50_pyr.shape[0]))
#    drug_curve_t = conv*dose
    for sim_num in range(n_sims):
        counts_t, drug_curve= sim.vectorized_abm(drugless_rates,
                                        ic50_pyr,
                                        n_gen=n_gen,  # Number of simulated generations
                                        mut_rate=mut_rate,  # probability of mutation per generation
                                        max_cells=max_cells,  # Max number of cells
                                        death_rate=death_rate,  # Death rate
                                        init_counts=init_counts,
                                        carrying_cap=carrying_cap,
                                        plot=plot,
                                        const_dose=dose,
                                        curve_type=curve_type,
                                        death_noise = death_noise,
                                        mut_noise=mut_noise)
#                                        pharm_impulse_response=drug_curve_t)
        
        if any(counts_t[n_gen-1,:]>0.1*max_cells):
            n_survive_vect[dose_num]+=1
        counts += counts_t
    dose_num+=1
    counts=np.divide(counts,n_sims)
    title = 'vectorized model, dose = ' + str(dose) + ' uM'
    sim.plot_timecourse(counts,drug_curve,drug_log_scale=drug_log_scale,
                        counts_log_scale=counts_log_scale,fig_title=title)
    
elapsed_vectorized = time.time() - start

# Run hybrid model
start = time.time()
n_survive_hyb = np.zeros(max_dose.shape)
dose_num=0
for dose in max_dose:
    counts = np.zeros((n_gen,ic50_pyr.shape[0]))
#    drug_curve_t = conv*dose
    for sim_num in range(n_sims):
        counts_t, drug_curve = hybrid_sim.hybrid_evol(drugless_rates,
                                        ic50_pyr,
                                        n_gen=n_gen,  # Number of simulated generations
                                        mut_rate=mut_rate,  # probability of mutation per generation
                                        max_cells=max_cells,  # Max number of cells
                                        death_rate=death_rate,  # Death rate
                                        init_counts=init_counts,
                                        carrying_cap=carrying_cap,
                                        plot=plot,
                                        const_dose=dose,
                                        curve_type=curve_type,
                                        death_noise=death_noise,
                                        mut_noise=mut_noise,
                                        thresh=thresh)
#                                        pharm_impulse_response=drug_curve_t)
        
        if any(counts_t[n_gen-1,:]>0.1*max_cells):
            n_survive_hyb[dose_num]+=1
        counts += counts_t
    dose_num+=1
    counts=np.divide(counts,n_sims)
    title = 'hybrid model, dose = ' + str(dose) + ' uM'
    sim.plot_timecourse(counts,drug_curve,drug_log_scale=drug_log_scale,
                        counts_log_scale=counts_log_scale,fig_title=title)
    
elapsed_hybrid = time.time() - start
###############################################################################    
# abm bar chart
abm_bar_fig, abm_bar_ax = plt.subplots(figsize=(8,6))
#dose = ['1','100','500','1000']
dose = [str(x) for x in max_dose]
abm_bar_ax.bar(dose,n_survive_ab)
abm_bar_ax.set_ylabel('# survived',fontsize=20)
abm_bar_ax.set_xlabel('Dose (uM)',fontsize=20)
abm_bar_ax.set_title('Agent-based model fixation rate',fontsize=20)
abm_bar_ax.tick_params(labelsize=20)
abm_bar_ax.set_ylim(0,n_sims+0.1*n_sims)
# vectorized bar chart
vect_bar_fig, vect_bar_ax = plt.subplots(figsize=(8,6))
#dose = ['1','100','500','1000']
vect_bar_ax.bar(dose,n_survive_vect)
vect_bar_ax.set_ylabel('# survived',fontsize=20)
vect_bar_ax.set_xlabel('Dose (uM)',fontsize=20)
vect_bar_ax.set_title('Vectorized model fixation rate',fontsize=20)
vect_bar_ax.tick_params(labelsize=20)
vect_bar_ax.set_ylim(0,n_sims+0.1*n_sims)
# hybrid model bar chart
hyb_bar_fig, hyb_bar_ax = plt.subplots(figsize=(8,6))
#dose = ['1','100','500','1000']
hyb_bar_ax.bar(dose,n_survive_hyb)
hyb_bar_ax.set_ylabel('# survived',fontsize=20)
hyb_bar_ax.set_xlabel('Dose (uM)',fontsize=20)
hyb_bar_ax.set_title('Hybrid model fixation rate',fontsize=20)
hyb_bar_ax.tick_params(labelsize=20)
hyb_bar_ax.set_ylim(0,n_sims+0.1*n_sims)