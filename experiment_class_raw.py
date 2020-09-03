# Non-interactive class for running experiments and saving raw data.
# Does not produce images by default.

from population_class import Population
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
import time

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
                 population_options = {},
                 slopes=None,
                 debug = True): # debug = True -> no save
        
        # list of allowed drug curve types
        allowed_types = ['linear',
                         'constant',
                         'heaviside',
                         'pharm',
                         'pulsed']
        
        # list of defined experiments
        allowed_experiments = ['inoculant-survival',
                               'dose-survival',
                               'drug-regimen',
                               'dose-entropy',
                               'rate-survival']
        
        if curve_types is not None:
            if not all(elem in allowed_types for elem in curve_types):
                raise Exception('One or more curve types is not recognized.\nAllowable types are: linear, constant, heaviside, pharm, pulsed.')
                
        if experiment_type is not None:
            if experiment_type not in allowed_experiments:
                raise Exception('Experiment type not recognized.\nAllowable types are inoculant-survival, dose-survival, drug-regimen, and dose-entropy.')
            
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
        
        # initialize a list of figures for saving
        self.figures = []
        
        if experiment_type is None:
            self.experiment_type = 'dose-survival'
            warnings.warn('No experiment type given - set to dose-survival by default.')
        else:
            self.experiment_type = experiment_type
        
        # if experiment_type == 'dose-survival' and len(inoculants) > 1:
        #     raise Exception('The experiment type is set to dose-survival (default), but more than one inoculant is given.')
        # elif experiment_type == 'inoculant-survival' and len(self.max_doses) > 1:
        #     # print('here')
        #     raise Exception('The experiment type is set to inoculant-survival, but more than one max dose is given.')
        
        # if experiment_type == 'inoculant-survival' and inoculants is None:
        #     raise Exception('The experiment type is set to inoculant-survival, but no inoculants are given.')
        
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
                                                   plot=False,
                                                   **self.population_options))
            self.n_survive = np.zeros([len(self.populations)])
            
        elif self.experiment_type == 'dose-entropy':
            for dose in self.max_doses:
                self.populations.append(Population(max_dose = dose,
                                                   curve_type=self.curve_types[0]))
            self.entropy_results = pd.DataFrame(columns=[]) # will become a dataframe later
            
        elif self.experiment_type == 'rate-survival':
            # if the curve type is 'pharm' then slope will be interpreted as k_abs
            self.slopes = slopes
            for slope in self.slopes:
                if curve_types[0] == 'pharm':
                    self.populations.append(Population(max_dose=self.max_doses[0],
                                                        k_abs=slope,
                                                        curve_type='pharm',
                                                        n_sims=1,
                                                        **self.population_options))
                else:
                    self.populations.append(Population(max_dose=self.max_doses[0],
                                                        slope=slope,
                                                        curve_type='linear',
                                                        n_sims=1,
                                                        **self.population_options))                        
                    
            # self.rate_survival_results = pd.DataFrame(columns=[])
            
        # generate new save folder        
        if not debug:
            num = 0
            num_str = str(num).zfill(4)
            
            date_str = time.strftime('%m%d%Y',time.localtime())
            
            save_folder = os.getcwd() + '\\results_' + date_str + '_' + num_str
            
            while(os.path.exists(save_folder)):
                num += 1
                num_str = str(num).zfill(4)
                save_folder = os.getcwd() + '\\results_' + date_str + '_' + num_str
            os.mkdir(save_folder) 
            
            self.results_path = save_folder
        
        # self.n_survive = np.zeros([len(self.curve_types),len(self.max_doses)])
        # self.perc_survive = np.zeros([len(self.curve_types),len(self.max_doses)])
###############################################################################
    # Methods for running experiments
    
    # run experiment and save results
    def run_experiment(self):
        n_doses = len(self.max_doses)
        n_curves = len(self.curve_types)
        n_inoc = len(self.inoculants)
        
        # pbar = tqdm(total = n_curves*n_doses) # progress bar
        
        # Loop through each population, execute simulations, and store survival statistics
        
        if self.experiment_type == 'dose-survival':
            # pbar = tqdm(total = n_curves*n_doses) # progress bar
            for curve_number in range(n_curves):
                for dose_number in range(n_doses):
                    
                    exp_num = curve_number*n_doses + dose_number
                    pop = self.populations[exp_num] # extract population in list of population
                    c,n_survive_t = pop.simulate()
                    pop.plot_timecourse()
                    self.n_survive[curve_number,dose_number] = n_survive_t
                    # pbar.update()
            self.perc_survive = 100*self.n_survive/self.n_sims   
                 
        elif self.experiment_type == 'inoculant-survival':
            # pbar = tqdm(total = n_curves*n_inoc) # progress bar
            for curve_number in range(n_curves):
                for inoc_num in range(n_inoc):
                    
                    exp_num = curve_number*n_inoc + inoc_num
                    pop = self.populations[exp_num] # extract population in list of population
                    c,n_survive_t = pop.simulate()
                    pop.plot_timecourse()
                    self.n_survive[curve_number,inoc_num] = n_survive_t
                    # pbar.update()           
            self.perc_survive = 100*self.n_survive/self.n_sims
            
        elif self.experiment_type == 'drug-regimen':
            # pbar = tqdm(total=len(self.populations))
            kk=0
            for p in self.populations:
                for i in range(self.n_sims):
                    # initialize new drug curve
                    p.drug_curve = p.gen_curves()
                    c,n_survive = p.simulate()
                    fig = p.plot_timecourse()
                    self.n_survive[kk] += n_survive
                    fig_savename = 'timecourse_p=' + str(p.prob_drop) + '_' + str(i)
                    fig_savename = fig_savename.replace('.','')
                    self.figures.append((fig_savename,fig))                    
                kk+=1
                # pbar.update()
                self.perc_survive = 100*self.n_survive/self.n_sims
            
        elif self.experiment_type == 'dose-entropy':
            # pbar = tqdm(total=len(self.populations)*self.n_sims)
            e_survived = []
            e_died = []
            for p in self.populations:
                
                for i in range(self.n_sims):
                    c,n_survive = p.simulate()
                    # e = max(p.entropy()) # compute max entropy
                    e_t = p.entropy()
                    e = max(e_t)
                    # e=1
                    # p.plot_timecourse()
                    
                    if n_survive == 1:
                        survive = 'survived' # survived
                        e_survived.append(e_t)
                    else:
                        survive = 'extinct' # died
                        e_died.append(e_t)      
                        
                    d = {'dose':[p.max_dose],
                         'survive condition':[survive],
                         'max entropy':[e]}
                    
                    entropy_results_t = pd.DataFrame(d)
                    self.entropy_results = self.entropy_results.append(entropy_results_t)
                    # pbar.update()
                    
                    
                    
                fig,ax=plt.subplots()
                # print('here')
                for i in range(len(e_survived)):
                    e_t = e_survived[i]
                    ax.plot(e_t[0:50],color ='red',label='survived')
                for i in range(len(e_died)):
                    e_t = e_died[i]
                    ax.plot(e_t[0:50],color ='blue',label='died')
                
                ax.set_xlabel('Time',fontsize=15)
                ax.set_ylabel('Entropy', fontsize=15)
                ax.tick_params(labelsize = 10)
                ax.set_xlim(0,50)
                
                ax.legend()
                handles, labels = ax.get_legend_handles_labels()
                labels, ids = np.unique(labels, return_index=True)
                handles = [handles[i] for i in ids]
                ax.legend(handles, labels, loc='best')
        
        elif self.experiment_type == 'rate-survival':
            # pbar = tqdm(total=len(self.populations))
            
            for p in self.populations:
                for n in range(self.n_sims):
                    counts,n_survive = p.simulate()
                    
                    drug = p.drug_curve
                    drug = np.array([drug])
                    drug = np.transpose(drug)
                    counts = np.concatenate((counts,drug),axis=1)
                    
                    if self.curve_types[0] == 'pharm':
                        save_folder = 'k_abs=' + str(p.k_abs)
                        save_folder.replace('.','pnt')
                    else:
                        save_folder = 'slope=' + str(p.slope)
                        save_folder.replace('.','pnt')
                    self.save_counts(counts,n,save_folder)
                    
                # fig_savename = 'slope = ' + str(p.slope)
                # self.figures = self.figures.append(fig)
                # pbar.update()
            # self.rate_survival_results.index = np.arange(len(self.rate_survival_results))
                
        # pbar.close() # close progress bar


    # Plot final results in a bar chart
    # def plot_barchart(self):
        
    #     if self.experiment_type == 'dose-survival':
    #         fig,ax = plt.subplots(1,1,figsize=(10,5))
    #         n_doses = len(self.max_doses)
    #         n_curves = len(self.curve_types)
            
    #         w = 1/(n_doses+1)
            
    #         for curve_number in range(n_curves):
    #             data = self.perc_survive[curve_number,:]
    #             N = len(data)
    #             ind = np.arange(N) + curve_number*w
    #             ax.bar(ind,data,w-.05,label = self.curve_types[curve_number])
                
    #         x_labels = [str(num) for num in self.max_doses]
    #         ax.set_xticks(np.arange(N)+w*n_curves/2-w/2)
    #         ax.set_xticklabels(x_labels)
            
    #         ax.legend(loc='best')
    #         ax.set_xlabel('Max dose (uM)',fontsize=15)
    #         ax.set_ylabel('Percent survival', fontsize=15)
    #         ax.tick_params(labelsize = 10)
    #         ax.set_ylim(0,100)
            
    #     elif self.experiment_type == 'inoculant-survival':
    #         fig,ax = plt.subplots(1,1,figsize=(10,5))
    #         n_inoculants = len(self.inoculants)
    #         n_curves = len(self.curve_types)
            
    #         w = 1/(n_inoculants+1)
            
    #         for curve_number in range(n_curves):
    #             data = self.perc_survive[curve_number,:]
    #             N = len(data)
    #             ind = np.arange(N) + curve_number*w
    #             ax.bar(ind,data,w-.05,label = self.curve_types[curve_number])
                
    #         x_labels = [str(num) for num in self.inoculants]
    #         ax.set_xticks(np.arange(N)+w*n_curves/2-w/2)
    #         ax.set_xticklabels(x_labels)
            
    #         ax.legend(loc='best')
    #         ax.set_xlabel('Inoculant size (cells)',fontsize=15)
    #         ax.set_ylabel('Percent survival', fontsize=15)
    #         ax.tick_params(labelsize = 10)
    #         ax.set_ylim(0,100)
        
    #     elif self.experiment_type == 'drug-regimen':
    #         fig,ax = plt.subplots(1,1,figsize=(7,5))
            
    #         x = np.arange(len(self.populations))
    #         ax.bar(x,self.perc_survive)
    #         x_labels = [str(prob) for prob in self.prob_drops]            
    #         ax.set_xticks(x)
    #         ax.set_xticklabels(x_labels)
            
    #         ax.set_xlabel('Probability of dropping',fontsize=15)
    #         ax.set_ylabel('Percent survival', fontsize=15)
    #         ax.tick_params(labelsize = 15)
    #         ax.set_ylim(0,100)
    #         self.figures.append(('barchart',fig))
            
    #     elif self.experiment_type == 'rate-survival':
    #         fig,ax = plt.subplots()
    #         data = self.rate_survival_results
    #         # data.reindex(index=data.index[::-1])
    #         # sns.swarmplot(x='dose',y='max entropy',data=e,ax=ax,hue='survive condition',dodge=True,color='black')
    #         # sns.boxplot(x='dose',y='max entropy',data=e,ax=ax,hue='survive condition',dodge=True,palette='Set2')
    #         if self.curve_types[0] == 'pharm':
    #             x_axis = 'k_abs'
    #         else:
    #             x_axis = 'slope'
                
    #         sns.barplot(x=x_axis,y='% survival',data = data,palette='Set2',ci='sd')
    #         if self.curve_types[0] == 'pharm':
    #             # x_label = 'k_{abs}'
    #             ax.set_xlabel(r'$k_{abs}$',fontsize=12)
    #         else:
    #             ax.set_xlabel('Slope (uM/time)',fontsize=12)
            
    #         # compute standard deviation
    #         p = data['% survival'].values
    #         n = self.n_sims
    #         p = p/100
    #         q = 1-p
            
    #         sd = 100*(p*q/n)**0.5 # variance of the estimator of the parameter of a bernoulli distribution
            
    #         y = data['% survival'].values
    #         x = data['k_abs'].values
    #         ax.errorbar(x=np.arange(len(data)), y=y, yerr=sd,linewidth=0,elinewidth=2,capsize=5)
    #         # ax.set_xlabel(x_label)
    #         l,r = ax.get_xlim()
    #         ax.set_xlim(r,l)
    #         # path = "C:\Users\Eshan\Documents\python scripts\theory division\abm_variable_fitness\figures\rate_survival_07312020"
    #         # path = os.path.normpath(path)
    #         # plt.savefig("C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\figures\\rate_survival_07312020\\barchart.svg")
    #     return
    
    # def save_images(self,save_folder=None):
    #     if save_folder is None:
    #         save_folder = self.experiment_type + '_figures'
    #     serial_ind = 0
    #     date_str = time.strftime('%m%d%Y',time.localtime())
    #     path = os.getcwd() + '\\' + save_folder + '_' + date_str + '_' + str(serial_ind)
        
    #     # Check if the path exists. If not, append a higher number
    #     while(os.path.exists(path)):
    #         serial_ind += 1
    #         path = path[:-1] + str(serial_ind)
    #     os.mkdir(path)    
        
    #     for figure in self.figures:
    #         fig_savename = figure[0]
    #         fig_savename = path + '\\' + fig_savename
    #         fig = figure[1]
    #         fig.savefig(fig_savename,bbox_inches="tight")
        
    #     return
    
    # save counts as a csv in the given subfolder with the label 'num'
    def save_counts(self,counts,num,save_folder,prefix='sim_'):
        
        # check if the desired save folder exists. If not, create it
        folder_path = self.results_path + '\\' + save_folder
        if os.path.exists(folder_path) != True:
            os.mkdir(folder_path)
            
        num = str(num).zfill(4)
        savename = self.results_path + '\\' + save_folder + '\\' + prefix + num + '.csv'
        np.savetxt(savename, counts, delimiter=",")
        return
        
###############################################################################
# Testing

# e = Experiment(experiment_type = 'drug-regimen',prob_drops=[0,.5,.7])
# e.run_experiment()
# e.save_images('figures')