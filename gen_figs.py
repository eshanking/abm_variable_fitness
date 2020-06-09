import numpy as np
#import matplotlib.pyplot as plp
import abm_variable_fitness as sim
import matplotlib.pyplot as plt
###############################################################################
# Global parameters
drugless_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\ogbunugafor_drugless.csv"
ic50_cyc = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\cycloguanil_ic50.csv"
ic50_pyr = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\pyrimethamine_ic50.csv"
###############################################################################
# Step 1: compare vectorized model with agent-based model - constant dose

# Vectorized model

n_gen=2000
curve_type = 'constant'
const_dose = np.array([0,10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3])
#const_dose = np.array([0])
mut_rate = 0.003
death_rate = 0.15
max_cells=10**6  # Max number of cells
death_noise = 0.01
mut_noise = 0.005

init_counts = np.zeros(16)
init_counts[0] = 10000

carrying_cap=True
plot = False

drug_log_scale = True
counts_log_scale = False

n_sims = 10
drugless_rates = sim.load_fitness(drugless_path)
ic50_pyr = sim.load_fitness(ic50_pyr)

#"""
#run simulations
#"""
#n_doses = const_dose.shape
#for dose in const_dose:
#    counts = np.zeros((n_gen,ic50_pyr.shape[0]))
#    for sim_num in range(n_sims):
#        counts_t, drug_curve = sim.vectorized_abm(drugless_rates,
#                                        ic50_pyr,
#                                        n_gen=n_gen,  # Number of simulated generations
#                                        mut_rate=mut_rate,  # probability of mutation per generation
#                                        max_cells=max_cells,  # Max number of cells
#                                        death_rate=death_rate,  # Death rate
#                                        init_counts=init_counts,
#                                        carrying_cap=carrying_cap,
#                                        plot=plot,
#                                        const_dose=dose,
#                                        death_noise = death_noise,
#                                        mut_noise=mut_noise,
#                                        curve_type=curve_type)
#        counts += counts_t
#    counts=np.divide(counts,n_sims)
#    sim.plot_timecourse(counts,drug_curve,drug_log_scale=drug_log_scale,
#                        counts_log_scale=counts_log_scale,fig_title='Vectorized')
#    
## Agent-based model
#for dose in const_dose:
#    counts = np.zeros((n_gen,ic50_pyr.shape[0]))
#    for sim_num in range(n_sims):
#        counts_t, drug_curve = sim.var_fit_automaton(drugless_rates,
#                                        ic50_pyr,
#                                        n_gen=n_gen,  # Number of simulated generations
#                                        mut_rate=mut_rate,  # probability of mutation per generation
#                                        max_cells=max_cells,  # Max number of cells
#                                        death_rate=death_rate,  # Death rate
#                                        init_counts=init_counts,
#                                        carrying_cap=carrying_cap,
#                                        plot=plot,
#                                        const_dose=dose,
#                                        curve_type=curve_type)
#        counts += counts_t
#    counts=np.divide(counts,n_sims)
#    sim.plot_timecourse(counts,drug_curve,drug_log_scale=drug_log_scale,
#                        counts_log_scale=counts_log_scale,fig_title='Agent-Based')
#"""
#end simulations
#"""
###############################################################################
# Step 2: compare the probability of death between constant, heaviside, ramped, and pharmacokinetic dose curves
#n_dead_const_dose = 0
drug_log_scale = False

const_dose = np.array([1,200,2000])
#max_dose = const_dose[0]

n_sims = 10

labels = ['Constant','Heaviside','Ramped','Pharm']

#constant_counts = []
#heaviside_counts = []
#ramped_counts = []
#pharm_counts = []

dose_counts = np.zeros((const_dose.shape[0],len(labels)))

# Don't want to recompute convolution for every dose...
u = sim.gen_impulses(1,2000)
conv_t = sim.convolve_pharm(u,2000,k_elim=.002,k_abs=0.01, max_dose=1)

dose_num = 0

for dose in const_dose:
    curve_type = 'constant'
    max_dose=dose
    counts = np.zeros((n_gen,ic50_pyr.shape[0]))
#    n_dead_const_dose = 0
    n_survive_const_dose = 0
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
#                                        death_noise = death_noise,
#                                        mut_noise=mut_noise,
                                        curve_type=curve_type)
        counts += counts_t
        if any(counts_t[1999,:]>0.1*max_cells):
#            n_dead_const_dose+=1
            n_survive_const_dose+=1
    counts=np.divide(counts,n_sims)
    sim.plot_timecourse(counts,drug_curve,drug_log_scale=drug_log_scale,
                        counts_log_scale=counts_log_scale)
#    constant_counts.append(n_dead_const_dose)
    dose_counts[dose_num,0] = n_survive_const_dose
    curve_type = 'heaviside'
    
    min_dose = 0
    h_step = 100
    
    n_survive_heaviside_dose = 0

    counts = np.zeros((n_gen,ic50_pyr.shape[0]))
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
#                                        death_noise = death_noise,
#                                        mut_noise=mut_noise,
                                        curve_type=curve_type,
                                        max_dose=max_dose,
                                        min_dose=min_dose,
                                        h_step=h_step)
        counts += counts_t
        if any(counts_t[1999,:]>0.1*max_cells):
            n_survive_heaviside_dose+=1
    counts=np.divide(counts,n_sims)
    sim.plot_timecourse(counts,drug_curve,drug_log_scale=drug_log_scale,
                        counts_log_scale=counts_log_scale)
    
#    heaviside_counts.append(n_dead_heaviside_dose)
    dose_counts[dose_num,1] = n_survive_heaviside_dose
    
    curve_type = 'linear'
    slope = 1000
    
    n_survive_ramped_dose = 0

    counts = np.zeros((n_gen,ic50_pyr.shape[0]))
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
#                                        death_noise = death_noise,
#                                        mut_noise=mut_noise,
                                        curve_type=curve_type,
                                        max_dose=max_dose,
                                        min_dose=min_dose,
                                        h_step=h_step,
                                        slope=slope)
        counts += counts_t
        if any(counts_t[1999,:]>0.1*max_cells):
            n_survive_ramped_dose+=1
    counts=np.divide(counts,n_sims)
    sim.plot_timecourse(counts,drug_curve,drug_log_scale=drug_log_scale,
                        counts_log_scale=counts_log_scale)
    
#    ramped_counts.append(n_dead_ramped_dose)
    dose_counts[dose_num,2] = n_survive_ramped_dose
    
    n_survive_pharm_dose = 0
#    u = sim.gen_impulses(2,2000)
#    conv = sim.convolve_pharm(u,2000,k_elim=.005,k_abs=0.05, max_dose=max_dose)
    conv = conv_t*max_dose
    curve_type = 'impulse-response'

    counts = np.zeros((n_gen,ic50_pyr.shape[0]))
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
#                                        death_noise = death_noise,
#                                        mut_noise=mut_noise,
                                        curve_type=curve_type,
                                        max_dose=max_dose,
                                        min_dose=min_dose,
                                        h_step=h_step,
                                        slope=slope,
                                        pharm_impulse_response=conv)
        counts += counts_t
        if any(counts_t[1999,:]>0.1*max_cells):
            n_survive_pharm_dose+=1
    counts=np.divide(counts,n_sims)
    sim.plot_timecourse(counts,drug_curve,drug_log_scale=drug_log_scale,
                        counts_log_scale=counts_log_scale)
    
#    pharm_counts.append(n_dead_pharm_dose)
    dose_counts[dose_num,3] = n_survive_pharm_dose
    
    dose_num+=1

dose_counts_perc = 100*dose_counts/n_sims
dose_counts_perc = dose_counts_perc+1

x = np.arange(len(labels))/2
width = 0.35

bar_fig, bar_ax = plt.subplots(figsize = (8,6))

rects1 = bar_ax.bar(x-width/3,dose_counts_perc[0,:],width/3,label=str(const_dose[0]))
rects2 = bar_ax.bar(x,dose_counts_perc[1,:],width/3,label=str(const_dose[1]))
rects3 = bar_ax.bar(x+width/3,dose_counts_perc[2,:],width/3,label=str(const_dose[2]))
#rects4 = bar_ax.bar(x+width/4,dose_counts_perc[3,:],width/4,label=str(const_dose[3]))

bar_ax.set_ylim(0,105)
bar_ax.set_ylabel('Percent Survival',fontsize=25)
bar_ax.set_xlabel('Dose Curve Type',fontsize=25)
bar_ax.tick_params(labelsize=25)
bar_ax.set_xticks([0,.5,1,1.5])
bar_ax.set_xticklabels(['Constant','Stepped','Ramped','Pharm'],fontsize=20)
bar_ax.legend(loc=(1,.5),frameon=False,fontsize=20)
#bar_ax.set_yticks(np.arange(1,11))
#bar_ax.set_yticklabels(['0','1','2','3','4','5','6','7','8','9','10'])