import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

mp.rcParams['xtick.minor.size'] = 0
mp.rcParams['xtick.minor.width'] = 0
mp.rcParams['xtick.major.size'] = 0
mp.rcParams['xtick.major.width'] = 0

def sigmoid(conc,
            ec50=1,
            n=1):
    e = 1/(1+(ec50/conc)**n)
    return e

conc = np.logspace(-3,3,num=20)
response = sigmoid(conc)

fig,ax = plt.subplots(figsize=(8,5))
ax.set_xscale('log')
ax.plot(conc,response,linewidth=3,color=np.array([37, 61, 138])/255)

ax.set_xticks([10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3])
ax.tick_params(labelsize=20)
ax.set_xlabel('Drug concentration ($\mathrm{\mu}$M)',fontsize=20)
ax.set_ylabel('Response',fontsize=20)
ax.set_frame_on(False)