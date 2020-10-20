from experiment_class_raw import Experiment
import numpy as np
# import time

options = {'n_impulse':20,
           'k_abs':0.04,
           'k_elim':0.03,
           'max_dose':150,
           'n_gen':1000,
           'mut_rate':0.00005,
           'death_rate':0.3,
           'plot':False}

p = np.array([0,0.2,0.4,0.6,0.8])
n_sims = 100
experiment_type = 'drug-regimen'

e = Experiment(experiment_type=experiment_type,
               n_sims=n_sims,
               prob_drops=p,
               population_options = options,
               debug=False)

e.run_experiment()