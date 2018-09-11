import numpy as np
from os import listdir
#import cPickle as pickle
#from platypus import PlatypusError
from matrix_utilities import read_R
from local_search import GradientDescentOptimizerAdam, AdamLocalSearchProgress
from evolutionary_operators import JoinMatrices, add_columns
from problems import TriFactorization, ConstantKMatrixGenerator

kk=10 # This is the dimension of recombined individual.
k1s=[3,4,5] # The dimension of individual for recombination.
mut_k1=[8,9] # The remaining k1 used for additional mutations.
num_of_runs=10 # This makes sample size of all recombinations equal to len(k1s)*num_of_runs.

# Specifications of R matrices.
n=800
k=50
d=6
# Maximum number of steps for gradient descent.
max_gd_steps=5000
# The scale of components when column addition is performed.
almost_zero=1e-8

# Find directory for that specific n, k, d
directories=listdir('data')
n_string='n='+str(n)+'_'
directories=[s for s in directories if n_string in s]
k_string='k='+str(k)+'_'
directories=[s for s in directories if k_string in s]
d_string='density='+str(d)+'.'
directories=[s for s in directories if d_string in s]
# Raise error if none or multiple directories were found.
if len(directories) == 0:
    raise PlatypusError('Directory with those n, k, d not found!')
if len(directories) != 1:
    raise PlatypusError('Multiple directories with those n, k, d was found!')
directory='data/'+directories[0]+'/'
# Find the names of files that store the values of R matrices.
matrices=listdir(directory)
matrices=[s for s in matrices if 'R' in s]
# Raise error if no files 'R*.csv' were found
if len(matrices) == 0:
    raise PlatypusError('Directory with those n, k, d not found!')
R=read_R(directory, matrices, n)
gd=GradientDescentOptimizerAdam(R)
local_search=AdamLocalSearchProgress(gd,steps=max_gd_steps)
problem=TriFactorization(gd)
generator=ConstantKMatrixGenerator(len(matrices),n)
recombination=JoinMatrices()

def init_solution(k):
    return generator.generate(problem,k)

gradd=[[] for _ in range(kk+1)]
recom=[[] for _ in range(max(k1s)+1)]
mutat=[[] for _ in range(max(mut_k1)+1)]

for run in range(num_of_runs):
    print('run='+str(run))
    for k1 in k1s:
        print('k1='+str(k1))
        # Initial points.
        a=init_solution(k1)
        b=init_solution(kk-k1)
        # Do gradient descent on initial points.
        a,a_p=local_search.mutate(a)
        b,b_p=local_search.mutate(b)
        print('grad desc for k='+str(k1)+','+str(kk-k1)+' is done')
        # Perform recombination and perform gradient descent.
        c=recombination.evolve([a,b])
        c,c_p=local_search.mutate(c[0])
        print('recombination and grad desc is done')
        # Perform two mutations and perform gradient descent.
        d=add_columns(a,kk-k1,almost_zero)
        d,d_p=local_search.mutate(d)
        if k1 != kk/2:
            e=add_columns(b,k1,almost_zero)
            e,e_p=local_search.mutate(e)
        print('mutations and grad desc is done')
        # Do gradient descent from random point for dimension of recombined or mutated individuals.
        f=init_solution(kk)
        f,f_p=local_search.mutate(f)
        print('grad desc of control was done')
        # Save all progresses.
        gradd[k1].append(a_p)
        gradd[kk-k1].append(b_p)
        recom[k1].append(c_p)
        mutat[k1].append(d_p)
        if k1 != kk/2:
            mutat[kk-k1].append(e_p)
        gradd[kk].append(f_p)
    for k1 in mut_k1:
        print('k1='+str(k1))
        # Initial points.
        g=init_solution(k1)
        # Do gradient descent on initial points.
        g,g_p=local_search.mutate(g)
        print('additional mutation was done')
        # Perform mutation and perform gradient descent.
        h=add_columns(g,kk-k1,almost_zero)
        h,h_p=local_search.mutate(h)
        # Save all progresses.
        gradd[k1].append(g_p)
        mutat[k1].append(h_p)

# Save the entire progress data.
all_file_name='all_recom_mutat.pickle'
with open(all_file_name,'wb') as all_file:
    pickle.dump([gradd,recom,mutat],all_file)

# Analyze the data.
l_gd=max(map(len,gradd[kk]))
lengths=map(lambda x: map(len,x), recom)
l_re=max([item for sublist in lengths for item in sublist])
l=max(l_gd,l_re)
# Control group.
gdkknp=np.zeros((len(k1s)*num_of_runs,l))
for i in range(len(k1s)*num_of_runs):
    gdkknp[i,:len(gradd[kk][i])]=np.asarray(gradd[kk][i])
    gdkknp[i,len(gradd[kk][i]):]=gradd[kk][i][-1]
gdkknp=np.sort(gdkknp,axis=0)
# For specifically 30 runs.
up_kk=gdkknp[5]
mid_kk=0.5*(gdkknp[14]+gdkknp[15])
down_kk=gdkknp[24]
# For recombinations.
x=np.zeros((len(k1s),num_of_runs,l))
for ki,k1 in enumerate(k1s):
    for r in range(num_of_runs):
        x[ki,r,:len(recom[k1][r])]=np.asarray(recom[k1][r])
        x[ki,r,len(recom[k1][r]):]=recom[k1][r][-1]
x=x.reshape((-1,l))
x=np.sort(x,axis=0)
up_re=x[5]
mid_re=0.5*(x[14]+x[15])
down_re=x[24]
csv_x=np.stack([up_kk,mid_kk,down_kk,up_re,mid_re,down_re])
csv_x=csv_x.T
np.savetxt('recom.csv',csv_x,delimiter=',')
# Mutations.
csv_mut=[]
for k1 in range(min(k1s),kk):
    for instance in mutat[k1]:
        csv_mut.append([k1,min(instance)])
for instance in gradd[kk]:
    csv_mut.append([kk,min(instance)])
csv_mut=np.asarray(csv_mut)
np.savetxt('mut.csv',csv_mut,delimiter=',')
