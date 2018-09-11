import numpy as np
import cPickle as pickle

kk=10 # This is the dimension of recombined individual.
k1s=[3,4,5] # The dimension of individual for recombination.
mut_k1=[8,9] # The remaining k1 used for additional mutations.
num_of_runs=10 # This makes sample size of all recombinations equal to len(k1s)*num_of_runs.

# Load the entire progress data.
all_file_name='all_recom_mutat.pickle'
with open(all_file_name,'wb') as all_file:
    [gradd,recom,mutat]=pickle.load(all_file)

# Analyze the data.
l_gd=max(map(len,gradd[kk]))
lengths=map(lambda x: map(len,x), mutat)
l_mu=max([item for sublist in lengths for item in sublist])
l=max(l_gd,l_mu)
# Control group.
gdkknp=np.zeros((len(k1s)*num_of_runs,l))
for i in range(len(k1s)*num_of_runs):
    gdkknp[i,:len(gradd[kk][i])]=np.asarray(gradd[kk][i])
    gdkknp[i,len(gradd[kk][i]):]=gradd[kk][i][-1]
med_control=np.median(gdkknp,axis=0).reshape((1,l))
# For mutations.
m=np.zeros((kk-min(k1s),num_of_runs,l))
for ki,k1 in enumerate(range(min(k1s),kk)):
    for r in range(num_of_runs):
        m[ki,r,:len(mutat[k1][r])]=np.asarray(mutat[k1][r])
        m[ki,r,len(mutat[k1][r]):]=mutat[k1][r][-1]
m=np.median(m,axis=1) 
csv_m=np.stack([m,med_control])
csv_m=csv_m.T
np.savetxt('mut_med.csv',csv_m,delimiter=',')
