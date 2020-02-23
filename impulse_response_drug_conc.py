#import abm_variable_fitness as sim 
import numpy as np
import hybrid_evol_model as hybrid_sim
import abm_variable_fitness as sim
###############################################################################
n_gen=2000

max_cells=10**5  # Max number of cells

init_counts = np.zeros(16)
init_counts[0] = 10000

carrying_cap=True
plot = False

drug_log_scale = False
counts_log_scale = False

lower_thresh = 0
upper_thresh = 2000

n_sims = 100

# "real-world" parameters
#k_elim = 0.007
#k_abs = 1.32
#max_dose = 4.83/4
k_elim = .0005
k_abs = .07
max_dose = 150
#div_scale = 0.66
#div_scale = 0.2 
div_scale = 1
#mut_rate = 0.003
#death_rate = 0.15
mut_rate = 0.01
death_rate = 0.15
death_noise = 0.01
mut_noise = 0.005

#u = sim.gen_impulses(1,n_gen)
#conv = sim.convolve_pharm(u,2000,k_elim=k_elim,k_abs=k_abs, max_dose=max_dose)
curve_type = 'constant'

drugless_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\ogbunugafor_drugless.csv"
#ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\cycloguanil_ic50.csv"
ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\pyrimethamine_ic50.csv"
###############################################################################

drugless_rates = sim.load_fitness(drugless_path)
ic50 = sim.load_fitness(ic50_path)

counts = np.zeros((n_gen,ic50.shape[0]))

for sim_num in range(n_sims):
#    counts_t, drug_curve = sim.vectorized_abm(drugless_rates,
    counts_t, drug_curve = hybrid_sim.hybrid_evol_2(drugless_rates,                                                 
                                    ic50,
                                    n_gen=n_gen,  # Number of simulated generations
                                    mut_rate=mut_rate,  # probability of mutation per generation
                                    max_cells=max_cells,  # Max number of cells
                                    death_rate=death_rate,  # Death rate
                                    init_counts=init_counts,
                                    carrying_cap=carrying_cap,
                                    plot=plot,
                                    curve_type=curve_type,
                                    pharm_impulse_response=conv,
                                    div_scale=div_scale,
                                    death_noise = death_noise,
                                    mut_noise = mut_noise,
                                    lower_thresh = lower_thresh,
                                    upper_thresh = upper_thresh,
                                    const_dose = max_dose)
#    sim.plot_timecourse(counts_t,drug_curve,drug_log_scale=drug_log_scale,
#                    counts_log_scale=counts_log_scale)
    counts += counts_t
        

counts=np.divide(counts,n_sims)
sim.plot_timecourse(counts,drug_curve,drug_log_scale=drug_log_scale,
                    counts_log_scale=counts_log_scale)