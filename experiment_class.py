from population_class import Population
import numpy as np
import matplotlib.pyplot as plt
import time

class Experiment():
    # Initializer
    def __init__(self,
                 n_sims = 1,
                 curve_types = None,
                 max_doses = None,
                 population_options = {}):
        
        if curve_types is None:
            self.curve_types = ['constant']
        else:
            self.curve_types = curve_types
        
        if max_doses is None:
            self.max_doses = [1]
        else:
            self.max_doses = max_doses
            
        self.n_sims = n_sims
            
        self.population_options = population_options
        
        self.populations = []
        
        for curve_type in self.curve_types:
            for max_dose in self.max_doses:
                fig_title = 'Max dose = ' + str(max_dose) + ', curve type = ' + curve_type
                self.populations.append(Population(curve_type=curve_type,
                                                  n_sims = self.n_sims,
                                                  max_dose = max_dose,
                                                  fig_title = fig_title,
                                                  **self.population_options))
        self.n_survive = np.zeros([len(self.curve_types),len(self.max_doses)])
        self.perc_survive = np.zeros([len(self.curve_types),len(self.max_doses)])
###############################################################################
    # Methods for running experiments
    
    def run_experiment(self):
        # for pop in self.populations:
        #     pop.simulate()
        #     pop.plot_timecourse()
        #     n_condition+=1
        n_doses = len(self.max_doses)
        n_curves = len(self.curve_types)
        
        for curve_number in range(n_curves):
            for dose_number in range(n_doses):
                exp_num = curve_number*n_doses + dose_number
                pop = self.populations[exp_num]
                c,n_survive_t = pop.simulate()
                pop.plot_timecourse()
                self.n_survive[curve_number,dose_number] = n_survive_t
        self.perc_survive = 100*self.n_survive/self.n_sims
    
    def plot_barchart(self):
        # do stuff
        fig,ax = plt.subplots(1,1,figsize=(10,5))
        
        n_doses = len(self.max_doses)
        n_curves = len(self.curve_types)
        
        w = 1/(n_doses+1)
        
        for curve_number in range(n_curves):
            data = self.perc_survive[curve_number,:]
            # data[data==0] = 1
            N = len(data)
            ind = np.arange(N) + curve_number*w
            ax.bar(ind,data,w-.05,label = self.curve_types[curve_number])
            
        x_labels = [str(num) for num in self.max_doses]
        ax.set_xticks(np.arange(N)+w*n_curves/2-w/2)
        ax.set_xticklabels(x_labels)
        
        ax.legend(loc='best')
        ax.set_xlabel('Max dose (uM)',fontsize=15)
        ax.set_ylabel('Percent survival', fontsize=15)
        ax.tick_params(labelsize = 10)
        ax.set_ylim(0,100)

        return
###############################################################################
# Testing

# options = {'n_gen':1000,'v2':False}
# # e1 = Experiment(population_options=options,
# #                 curve_types=['constant','linear','pharm'],
# #                 max_doses = [1,10,100,500],
# #                 n_sims = 10)
# e1 = Experiment(population_options = options,n_sims=10)

# t0 = time.perf_counter()
# e1.run_experiment()
# t_elapsed = time.perf_counter() - t0
# print(str(t_elapsed))
# # e1.plot_barchart()


options = {'n_gen':1000,'v2':False}
e1 = Experiment(population_options = options, 
                   n_sims=100, 
                   curve_types = ['constant'],
                    # max_doses = [1,40,100]
                    max_doses = [1]
                    )
e1.run_experiment()
e1.plot_barchart()