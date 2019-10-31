import numpy as np
#import matplotlib.pyplot as plp
import abm_variable_fitness as sim
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
max_cells=10**5  # Max number of cells
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

n_doses = const_dose.shape
for dose in const_dose:
    counts = np.zeros((n_gen,ic50_pyr.shape[0]))
    for sim_num in range(n_sims):
        counts_t, drug_curve = sim.vectorized_abm(drugless_rates,
                                        ic50_pyr,
                                        n_gen=n_gen,  # Number of simulated generations
                                        mut_rate=mut_rate,  # probability of mutation per generation
                                        max_cells=max_cells,  # Max number of cells
                                        death_rate=death_rate,  # Death rate
                                        init_counts=init_counts,
                                        carrying_cap=carrying_cap,
                                        plot=plot,
                                        const_dose=dose,
                                        death_noise = death_noise,
                                        mut_noise=mut_noise,
                                        curve_type=curve_type)
        counts += counts_t
    counts=np.divide(counts,n_sims)
    sim.plot_timecourse(counts,drug_curve,drug_log_scale=drug_log_scale,
                        counts_log_scale=counts_log_scale,fig_title='Vectorized')
    
# Agent-based model
#n_sims = 1
for dose in const_dose:
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
                                        curve_type=curve_type)
        counts += counts_t
    counts=np.divide(counts,n_sims)
    sim.plot_timecourse(counts,drug_curve,drug_log_scale=drug_log_scale,
                        counts_log_scale=counts_log_scale,fig_title='Agent-Based')