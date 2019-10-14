import persistant_homologies as ph
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
###############################################################################
conc_step = 10
min_prominence = 0

drugless_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\ogbunugafor_drugless.csv"
#ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\cycloguanil_ic50.csv"
ic50_path = "C:\\Users\\Eshan\\Documents\\python scripts\\theory division\\abm_variable_fitness\\data\\pyrimethamine_ic50.csv"

###############################################################################
drugless_rate = ph.load_fitness(drugless_path)
ic50 = ph.load_fitness(ic50_path)

conc = np.logspace(-3,5,9)
conc = np.concatenate(([0],conc))
alleles = np.linspace(0,15,16)
alleles = alleles.astype(int)

binary_peaks = np.zeros((16,10))
#n_checks = 0

for conc_indx in range(0,10):
#    print(str(conc_indx))
    for allele_indx in range(0,16):
        check_local_max = ph.is_local_max(alleles[allele_indx],
                                          conc[conc_indx],
                                          ic50=ic50,
                                          drugless_rate=drugless_rate,
                                          min_prominence=min_prominence)
#        print(str(alleles[allele_indx]) + ' ' + str(conc[conc_indx]))
#        print(str(check_local_max[1]))
#        n_checks+=1
        if check_local_max[0]:
            binary_peaks[allele_indx,conc_indx] = 1

#x_ticks = ['0','10^-3','10^-2','10^-1','10^0','10^1','10^2','10^3','10^4','10^5']
y_ticks = []
for allele in range(16):
    y_ticks.append(str(ph.int_to_binary(allele)))
    
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(binary_peaks,cmap='magma',cbar=False,linewidth=0.5,square=False)
ax.set_xticklabels(['0','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$',
                         '$10^1$','$10^2$','$10^3$','$10^4$','$10^5$'])
ax.set_yticklabels(y_ticks)

ax.yaxis.set_tick_params(rotation=0)
ax.tick_params(axis='y',labelsize=15)
ax.tick_params(axis='x',labelsize=15)

plt.xlabel('Concentration (uM)',fontsize=20)
plt.ylabel('Allele',fontsize=20)
#ax.set_title('Cycloguanil',fontsize=20)