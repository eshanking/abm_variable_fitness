from experiment_class import Experiment
import seaborn as sns
import matplotlib.pyplot as plt

# max_doses = [100,300,400,500]
max_doses = [350]
experiment_type = 'dose-entropy'
n_sims = 100

e1 = Experiment(max_doses=max_doses,experiment_type=experiment_type,n_sims=n_sims)
e1.run_experiment()

e = e1.entropy_results

# fig,ax = plt.subplots()
# sns.swarmplot(x='dose',y='max entropy',data=e,ax=ax,hue='survive condition',dodge=True,color='black')
# sns.boxplot(x='dose',y='max entropy',data=e,ax=ax,hue='survive condition',dodge=True,palette='Set2')

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[:2], labels[:2])