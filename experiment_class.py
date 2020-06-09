from population_class import Population

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
            
        self.n_sims = 1
            
        self.population_options = population_options
        
        self.populations = []
        
        for curve_type in self.curve_types:
            for max_dose in self.max_doses:
                fig_title = 'Max dose = ' + str(max_dose) + ', curve type = ' + curve_type
                self.populations.append(Population(curve_type=curve_type,
                                                  max_dose = max_dose,
                                                  fig_title = fig_title,
                                                  **self.population_options))
###############################################################################
    # Methods for running experiments
    
    def run_experiment(self):
        for pop in self.populations:
            pop.simulate()
            pop.plot_timecourse()
            
###############################################################################
# Testing

options = {'n_gen':100}
e1 = Experiment(population_options=options)
e1.run_experiment()