# Eshan's object-oriented figure cheat sheet
# Good resource for using axes: https://matplotlib.org/3.1.0/api/axes_api.html

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mpl

data = np.random.normal(50,20,1000)

fig,ax = plt.subplots(1,2,figsize=(10,5))

ax[0].hist(data,bins=10,rwidth=0.85)
#labels = ax[0].get_xticklabels()
#ind = np.arange(9)
ax[0].set_xticks([0,10,20,30,40,50,60,70,80,90,100])
ax[0].set_xticklabels(['0','10','20','30','40','50','60','70','80','90','100'])
ax[0].tick_params(labelsize = 10)

ax[1].plot([0,1,2,3,4,5])
ax[1].set_xlabel('x label',fontsize=15)
ax[1].legend('data1')