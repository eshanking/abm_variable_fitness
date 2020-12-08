import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
# import seaborn as sns

def int_to_binary(num, pad=4):
    return bin(num)[2:].zfill(pad)

def chi_square(n,y,p=None):
    k = len(y)
    if p is None:
        p = np.ones(k)*1/k # uniform distribution
    if len(p) == 1:
        p = np.ones(k)*p
    V = 0
    for i in range(k):
        V += ((y[i]-n*p[i])**2)/(n*p[i])

    return V

data_folder = 'results_10202020_0000'
max_cells = 10**6
###############################################################################
# generate figure and axes

results_dir = os.getcwd() + '//' + data_folder # folder containing all of the results
experiment_folders = os.listdir(path=results_dir) # each of these folders corresponds to a different k_abs

n_params = len(experiment_folders)

# Full page figure with room for caption
fig,ax = plt.subplots(2,1,figsize=(6.25,7.75),sharex=True)

# ax = fig.add_subplot()

##############################################################################
# data analysis and plotting

exp_num = 1
exp = experiment_folders[exp_num]

exp_path = results_dir + '\\' + exp
sim_files = os.listdir(path=exp_path)

n_sims = len(sim_files)

survived_regimen = np.zeros((1,19))
perished_regimen = np.zeros((1,19))
first_perished = True
first_survived = True

indx = 0

while indx < n_sims:
    
    sim = sim_files[indx]
    indx+=1
    sim_path = exp_path + '\\' + sim
    data_df = pd.read_csv(sim_path)
    data = data_df.to_numpy()
    counts = data[:,0:-2]
    regimen = data[0:19,-1]
    regimen = np.array([regimen])
    
    if any(counts[-1,:]>0.1*max_cells):
        if first_survived is True:
            survived_regimen[0,:] = regimen
            first_survived = False
        else:
            survived_regimen = np.concatenate((survived_regimen,regimen),axis=0)
    else:
        if first_perished is True:
            perished_regimen[0,:] = regimen
            first_perished = False
        else:
            perished_regimen = np.concatenate((perished_regimen,regimen),axis=0)   

cmap = mpl.colors.ListedColormap(['cornflowerblue','w'])
# total_regimen = np.concatenate((survived_regimen,perished_regimen))
ax[0].imshow(survived_regimen,cmap=cmap,aspect=0.3)
ax[1].imshow(perished_regimen,cmap=cmap,aspect=0.3)

ax0_pos = ax[0].get_position()
ax1_pos = ax[1].get_position()
ax[1].set_position([ax0_pos.x0,
                    ax1_pos.y0-ax1_pos.height*(ax0_pos.width/ax1_pos.width-1.3),
                    ax0_pos.width,
                    ax1_pos.height*ax0_pos.width/ax1_pos.width])  
xticks = np.arange(0,19,2)  
ax[1].set_xticks(xticks)
xlabels = [str(x) for x in np.arange(1,20,2)]
ax[1].set_xticklabels(xlabels)
ax[1].set_xlabel('Dose number',fontsize=15)
ax[1].set_ylabel('Perished',fontsize=15)
ax[0].set_ylabel('Survived',fontsize=15)        

legend_elements = [Patch(facecolor='cornflowerblue',label='Missed dose'),Patch(facecolor='w',edgecolor='black',label='Scheduled dose')]
ax[1].legend(handles=legend_elements,loc=(.15,-0.14),ncol=2,edgecolor='w')   
ax[0].set_title('$p_{forget}$ = 0.2',fontsize=15)     

# plt.savefig('figure_6.svg',bbox_inches="tight")
        
fig2, ax2 = plt.subplots(2,1,sharex=True,figsize=(3,4))

# for i in range(perished_regimen.shape[0]):
#     y = np.correlate(perished_regimen[i,:],perished_regimen[i,:],mode='full')
#     y = y[int(np.floor(y.size/2)):]
#     x = np.arange(y.size)*2
#     ax2.scatter(x,y,color='red')
    
# for i in range(survived_regimen.shape[0]):
#     y = np.correlate(survived_regimen[i,:],survived_regimen[i,:],mode='full')
#     y = y[int(np.floor(y.size/2)):]
#     x = np.arange(y.size)*2+0.5
#     ax2.scatter(x,y,color='blue')

# ax2.set_ylim(0.3,1)
# ax2.set_xlim(0,11)

n_survived = survived_regimen.shape[0]
n_perished = perished_regimen.shape[0]

survived_hist = np.sum(survived_regimen,axis=0)
survived_hist = survived_hist/n_survived
perished_hist = np.sum(perished_regimen,axis=0)
perished_hist = perished_hist/n_perished

dose_num = np.arange(len(survived_hist))
p = 0.8*np.ones(len(dose_num))

ax2[0].bar(dose_num,survived_hist,width=1,color='red',alpha=0.5,label='Survived')
ax2[0].plot(dose_num,p,'--',color='black',label='p = 0.8')
ax2[1].bar(dose_num,perished_hist,width=1,color='blue',alpha=0.5,label='Perished')
ax2[1].plot(dose_num,p,'--',color='black',label='p = 0.8')

ax2[0].legend(loc=(0.45,0.05))
ax2[1].legend(loc=(0.45,0.05))   

ax2[0].set_title('Histogram of Doses')
ax2[0].set_ylabel('Probability')
ax2[1].set_ylabel('Probability')    
ax2[1].set_xlabel('Dose number') 

plt.savefig('figure_7.svg',bbox_inches="tight")
p = p[0:10]
v_survived = chi_square(n_survived,np.sum(survived_regimen[:,0:10],axis=0),p)
v_perished = chi_square(n_perished,np.sum(perished_regimen[:,0:10],axis=0),p)