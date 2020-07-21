from population_class import Population
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

class Experiment():
    
    # Initializer
    def __init__(self,
                 n_sims = 1,
                 curve_types = None,
                 max_doses = None,
                 inoculants = None,
                 experiment_type = None,
                 prob_drops = None,
                 n_impulse=1,
                 population_options = {}):
        
        allowed_types = ['linear', 'constant', 'heaviside', 'pharm', 'pulsed']
        allowed_experiments = ['inoculant-survival','dose-survival','drug-regimen']
        
        if curve_types is not None:
            if not all(elem in allowed_types for elem in curve_types):
                raise Exception('One or more curve types is not recognized.\nAllowable types are: linear, constant, heaviside, pharm, pulsed.')
                
        if experiment_type is not None:
            if experiment_type not in allowed_experiments:
                raise Exception('Experiment type not recognized.\nAllowable types are inoculant-survival and dose-survival.')
            
        # Curve type: linear, constant, heaviside, pharm, pulsed
        if curve_types is None:
            self.curve_types = ['constant']
        else:
            self.curve_types = curve_types
        
        if max_doses is None:
            self.max_doses = [1]
        else:
            self.max_doses = max_doses
            
        if inoculants is None:
            self.inoculants = [0]
        else:
            self.inoculants = inoculants
            
        self.n_sims = n_sims
            
        # Common options that are applied to each population
        self.population_options = population_options
        
        # initialize all populations
        self.populations = []
        
        if experiment_type is None:
            self.experiment_type = 'dose-survival'
            warnings.warn('No experiment type given - set to dose-survival by default.')
        else:
            self.experiment_type = experiment_type
        
        if experiment_type == 'dose-survival' and len(inoculants) > 1:
            raise Exception('The experiment type is set to dose-survival (default), but more than one inoculant is given.')
        elif experiment_type == 'inoculant-survival' and len(self.max_doses) > 1:
            # print('here')
            raise Exception('The experiment type is set to inoculant-survival, but more than one max dose is given.')
        
        if experiment_type == 'inoculant-survival' and inoculants is None:
            raise Exception('The experiment type is set to inoculant-survival, but no inoculants are given.')
        
        if self.experiment_type == 'dose-survival':
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
                    
        elif self.experiment_type == 'inoculant-survival':
            for curve_type in self.curve_types:
                for inoculant in self.inoculants:
                    fig_title = 'Inoculant = ' + str(inoculant) + ', curve type = ' + curve_type
                    
                    init_counts = np.zeros(16)
                    init_counts[0] = inoculant
                    
                    self.populations.append(Population(curve_type=curve_type,
                                                      n_sims = self.n_sims,
                                                      fig_title = fig_title,
                                                      init_counts=init_counts,
                                                      **self.population_options))
                    
            self.n_survive = np.zeros([len(self.curve_types),len(self.inoculants)])
            self.perc_survive = np.zeros([len(self.curve_types),len(self.inoculants)])
            
        elif self.experiment_type == 'drug-regimen':
            
            self.prob_drops = prob_drops
            
            for prob_drop in self.prob_drops:
                curve_type = 'pulsed'
                self.populations.append(Population(curve_type=curve_type,
                                                   prob_drop=prob_drop,
                                                   # n_impulse = 1,
                                                   n_sims = 1,
                                                   # fig_title = fig_title,
                                                   # init_counts=init_counts,
                                                   **self.population_options))
            self.n_survive = np.zeros([len(self.populations)])
            
        # self.n_survive = np.zeros([len(self.curve_types),len(self.max_doses)])
        # self.perc_survive = np.zeros([len(self.curve_types),len(self.max_doses)])
###############################################################################
    # Methods for running experiments
    
    def run_experiment(self):
        n_doses = len(self.max_doses)
        n_curves = len(self.curve_types)
        n_inoc = len(self.inoculants)
        
        # pbar = tqdm(total = n_curves*n_doses) # progress bar
        
        # Loop through each population, execute simulations, and store survival statistics
        
        if self.experiment_type == 'dose-survival':
            pbar = tqdm(total = n_curves*n_doses) # progress bar
            for curve_number in range(n_curves):
                for dose_number in range(n_doses):
                    
                    exp_num = curve_number*n_doses + dose_number
                    pop = self.populations[exp_num] # extract population in list of population
                    c,n_survive_t = pop.simulate()
                    pop.plot_timecourse()
                    self.n_survive[curve_number,dose_number] = n_survive_t
                    pbar.update()
                    
        elif self.experiment_type == 'inoculant-survival':
            pbar = tqdm(total = n_curves*n_inoc) # progress bar
            for curve_number in range(n_curves):
                for inoc_num in range(n_inoc):
                    
                    exp_num = curve_number*n_inoc + inoc_num
                    pop = self.populations[exp_num] # extract population in list of population
                    c,n_survive_t = pop.simulate()
                    pop.plot_timecourse()
                    self.n_survive[curve_number,inoc_num] = n_survive_t
                    pbar.update()           
        
        elif self.experiment_type == 'drug-regimen':
            pbar = tqdm(total=len(self.populations))
            kk=0
            for p in self.populations:
                for i in range(self.n_sims):
                    # initialize new drug curve
                    p.drug_curve = p.gen_curves()
                    c,n_survive = p.simulate()
                    p.plot_timecourse()
                    self.n_survive[kk] += n_survive
                kk+=1
                pbar.update()
                
        pbar.close() # close progress bar
        self.perc_survive = 100*self.n_survive/self.n_sims

    # Plot final results in a bar chart
    def plot_barchart(self):
        
        if self.experiment_type == 'dose-survival':
            fig,ax = plt.subplots(1,1,figsize=(10,5))
            n_doses = len(self.max_doses)
            n_curves = len(self.curve_types)
            
            w = 1/(n_doses+1)
            
            for curve_number in range(n_curves):
                data = self.perc_survive[curve_number,:]
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
            
        elif self.experiment_type == 'inoculant-survival':
            fig,ax = plt.subplots(1,1,figsize=(10,5))
            n_inoculants = len(self.inoculants)
            n_curves = len(self.curve_types)
            
            w = 1/(n_inoculants+1)
            
            for curve_number in range(n_curves):
                data = self.perc_survive[curve_number,:]
                N = len(data)
                ind = np.arange(N) + curve_number*w
                ax.bar(ind,data,w-.05,label = self.curve_types[curve_number])
                
            x_labels = [str(num) for num in self.inoculants]
            ax.set_xticks(np.arange(N)+w*n_curves/2-w/2)
            ax.set_xticklabels(x_labels)
            
            ax.legend(loc='best')
            ax.set_xlabel('Inoculant size (cells)',fontsize=15)
            ax.set_ylabel('Percent survival', fontsize=15)
            ax.tick_params(labelsize = 10)
            ax.set_ylim(0,100)
        
        elif self.experiment_type == 'drug-regimen':
            fig,ax = plt.subplots(1,1,figsize=(7,5))
            
            x = np.arange(len(self.populations))
            ax.bar(x,self.perc_survive)
            x_labels = [str(prob) for prob in self.prob_drops]            
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            
            ax.set_xlabel('Probability of dropping',fontsize=15)
            ax.set_ylabel('Percent survival', fontsize=15)
            ax.tick_params(labelsize = 15)
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

# options = {'n_gen':2000,'v2':True}
# e1 = Experiment(population_options = options, 
#                    n_sims=10, 
#                    # curve_types = ['constant','linear','pharm'],
#                     # max_doses = [1,40,100]
#                     curve_types = ['constant'],
#                     max_doses = [0]
#                     # max_doses = [1]
#                     )
# e1.run_experiment()
# e1.plot_barchart()

# e1 = Experiment()

# experiment_type = 'inoculant-survival'
# options = {'n_gen':1000,'v2':True,'max_dose':140,'max_cells':10**7}
# inoculants = [10**3,10**4,10**5,10**6]
# curve_types=['constant','linear','pharm']
# n_sims = 20

# e = Experiment(experiment_type = experiment_type,
#                inoculants=inoculants,
#                curve_types=curve_types,
#                n_sims=n_sims,
#                population_options=options)
# e.run_experiment()
# e.plot_barchart()