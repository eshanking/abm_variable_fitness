from experiment_class import Experiment
import matplotlib.pyplot as plt

options = {'n_impulse':15,
           'k_abs':0.005,
           'k_elim':0.007,
           'max_dose':3000,
           'n_gen':2000}

e = Experiment(experiment_type='drug-regimen',
                n_sims=100 ,
                prob_drops=[0,.4,.7],
                population_options=options)

e.run_experiment()
e.plot_barchart()

# import matplotlib.pyplot as plt

# fig,ax = plt.subplots()
# ax.bar([0,1,2],[10,20,30])
# ax.set_xticks([0,1,2])
# ax.set_xticklabels(['a','b','c'])