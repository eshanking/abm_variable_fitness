import abm_variable_fitness as sim 
import numpy as np
###############################################################################
n_gen=1000
mut_rate = 0.003
death_rate = 0.15
max_cells=10**5  # Max number of cells

init_counts = np.zeros(16)
init_counts[0] = 10000

carrying_cap=True
plot = False

drug_log_scale = False
counts_log_scale = False

n_sims = 1

u = sim.gen_impulses(3,1000)
conv = sim.convolve_pharm(u,2000,k_elim=.03,k_abs=0.05, max_dose=10)
curve_type = 'impulse-response'

drugless_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\ogbunugafor_drugless.csv"
ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\cycloguanil_ic50.csv"
#ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\pyrimethamine_ic50.csv"
###############################################################################

drugless_rates = sim.load_fitness(drugless_path)
ic50 = sim.load_fitness(ic50_path)

counts = np.zeros((n_gen+1,ic50.shape[0]))

for sim_num in range(n_sims):
    counts_t, drug_curve = sim.var_fit_automaton(drugless_rates,
                                    ic50,
                                    n_gen=n_gen,  # Number of simulated generations
                                    mut_rate=mut_rate,  # probability of mutation per generation
                                    max_cells=max_cells,  # Max number of cells
                                    death_rate=death_rate,  # Death rate
                                    init_counts=init_counts,
                                    carrying_cap=carrying_cap,
                                    plot=plot,
                                    curve_type=curve_type,
                                    pharm_impulse_response=conv)
    counts += counts_t
        

counts=np.divide(counts,n_sims)
sim.plot_timecourse(counts,drug_curve,drug_log_scale=drug_log_scale,
                    counts_log_scale=counts_log_scale)