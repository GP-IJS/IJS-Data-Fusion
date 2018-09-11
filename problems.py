from __future__ import print_function
import numpy as np
from platypus import Problem, Solution, Generator, PlatypusError
from matrix_utilities import read_R
from local_search import GradientDescentOptimizerAdam, AdamLocalSearch
from stochastic_algorithm import stochastic_algorithm, MaxEvaluationsAndTermination
from deterministic_algorithm import deterministic_algorithm, MaxEvaluationsAndTreshold
from functools import partial
from abc import ABCMeta
from os import listdir
import pickle as pickle
from hypervolume import HyperVolume

class HypervolumeDynamics:

    def __init__(self,k):
        self.hv=HyperVolume([1.,k])
        self.current_front=[]
        self.dynamics=[]

    def add_new(self,individual):
        candidate=individual.objectives
        i=0
        while i < len(self.current_front):
            # In case new one is dominated by someone, return without adding it.
            if candidate[0] > self.current_front[i][0] and candidate[1] >= self.current_front[i][1]:
                self.dynamics.append(self.dynamics[-1])
                return
            # In case new one dominates someone in the front, delete it.
            if candidate[0] < self.current_front[i][0] and candidate[1] <= self.current_front[i][1]:
                del self.current_front[i]
                continue
            i+=1
        # If we came this far, it is safe to add the candidate to the front.
        self.current_front.append(candidate)
        self.dynamics.append(self.hv.compute(self.current_front))

class Tensor(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Tensor, self).__init__()

    def rand(self):
        raise NotImplementedError("method not implemented")

    def encode(self, value):
        return value

    def decode(self, value):
        return value

class TriFactorization(Problem):
    def __init__(self, gd_optimizer, number_of_variables=2, number_of_objectives=2):

        # The objective function
        def obj_fun(cost_evaluator, x):
            G=x[0]
            S=x[1]
            k=S.shape[1]
            cost=cost_evaluator.calculate_cost(G,S)
            return [cost, k]

        # Partial application of argument cost_evaluator
        obj_fun_partial = partial(obj_fun, gd_optimizer)
        super(TriFactorization,self).__init__(nvars=number_of_variables,nobjs=number_of_objectives,function=obj_fun_partial)

        # Set type of variables
        self.types[:] = Tensor()
        self.gdOptimizer = gd_optimizer

class RandomKMatrixGenerator(Generator):
    def __init__(self,m,n,min_k,max_k,scale=0.01):
        super(RandomKMatrixGenerator, self).__init__()
        self.m=m
        self.n=n
        self.min_k=min_k
        self.max_k=max_k
        self.scale=scale

    def generate(self, problem):
        solution = Solution(problem)
        k=np.random.randint(self.min_k,self.max_k+1)
        solution.variables[0]=self.scale*np.random.rand(self.n,k)
        solution.variables[1]=self.scale*np.random.rand(self.m,k,k)
        return solution

class ConstantKMatrixGenerator(Generator):
    def __init__(self,m,n,scale=0.01):
        super(ConstantKMatrixGenerator, self).__init__()
        self.m=m
        self.n=n
        self.scale=scale

    def generate(self, problem, k):
        solution=Solution(problem)
        solution.variables[0]=self.scale*np.random.rand(self.n,k)
        solution.variables[1]=self.scale*np.random.rand(self.m,k,k)
        return solution

def prepare_stochastic(n,k,d,r,max_gd_steps,max_nfe,k_init_min=1,k_init_max=7):
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
    local_search=AdamLocalSearch(gd,steps=max_gd_steps)
    problem=TriFactorization(gd)
    generator=RandomKMatrixGenerator(len(matrices),n,k_init_min,k_init_max)
    hypervolume_dynamics=HypervolumeDynamics(k)
    print_file=open('results/prog_n'+str(n)+'k'+str(k)+'d'+str(d)+'r'+str(r)+'.txt','w')
    algorithm=stochastic_algorithm(problem=problem,local_search=local_search,generator=generator,
                                   hypervolume_dynamics=hypervolume_dynamics,print_file=print_file)
    termination_condition=MaxEvaluationsAndTermination(max_nfe)
    return algorithm,termination_condition

def prepare_deterministic(n,k,d,r,max_gd_steps,max_nfe):
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
    local_search=AdamLocalSearch(gd,steps=max_gd_steps)
    problem=TriFactorization(gd)
    generator=ConstantKMatrixGenerator(len(matrices),n)
    hypervolume_dynamics=HypervolumeDynamics(k)
    algorithm=deterministic_algorithm(problem=problem,local_search=local_search,generator=generator,
                                   hypervolume_dynamics=hypervolume_dynamics)
    termination_condition=MaxEvaluationsAndTreshold(max_nfe)
    return algorithm,termination_condition

def save_results(algorithm,n,k,d,r,elapsed_time,subcategory=''):
    # Gather all data from the population.
    obj=np.empty((len(algorithm.population),2))
    Gs=[]
    Ss=[]
    for i,sol in enumerate(algorithm.result):
        obj[i,1]=sol.objectives[0]
        obj[i,0]=sol.objectives[1]
        Gs.append(sol.variables[0])
        Ss.append(sol.variables[1])
    # Save values of objectives for the entire population
    obj_file_name='results/'+subcategory+'obj_n'+str(n)+'k'+str(k)+'d'+str(d)+'r'+str(r)+'.csv'
    np.savetxt(obj_file_name,obj,delimiter=',')
    # Save the best RSE values for each k gotten.
    best_file_name='results/'+subcategory+'best_n'+str(n)+'k'+str(k)+'d'+str(d)+'r'+str(r)+'.csv'
    ks=[]
    rses=[]
    k_max=-np.inf
    for kk in obj[:,0]:
        new_k=int(kk)
        if new_k > k_max:
            k_max=new_k
        if new_k not in ks:
            ks.append(new_k)
    for kk in ks:
        c_best=np.inf
        for i in range(obj.shape[0]):
            if int(obj[i,0]) == kk and obj[i,1] < c_best:
                c_best=obj[i,1]
        if not np.isinf(c_best):
            rses.append(c_best)
    best=[ks,rses] 
    best=np.asarray(best)
    best=best.T
    # Sort the best individuals with respect to k and save them.
    best=best[best[:,0].argsort()]    
    np.savetxt(best_file_name,best,delimiter=',')
    # Save Pareto front.
    front_file_name='results/'+subcategory+'front_n'+str(n)+'k'+str(k)+'d'+str(d)+'r'+str(r)+'.csv'
    front=np.asarray(algorithm.hypervolume_dynamics.current_front)
    front=np.fliplr(front)
    front=front[front[:,0].argsort()]
    np.savetxt(front_file_name,front,delimiter=',')
    # Save the dynamics of the hypervolume.
    hvd_file_name='results/'+subcategory+'hvd_'+str(n)+'k'+str(k)+'d'+str(d)+'r'+str(r)+'.csv'
    hvd=np.asarray(algorithm.hypervolume_dynamics.dynamics)
    hvd=hvd.T
    np.savetxt(hvd_file_name,hvd,delimiter=',')
    # Save the population
    pop_file_name='results/'+subcategory+'pop_n'+str(n)+'k'+str(k)+'d'+str(d)+'r'+str(r)+'.pickle'
    with open(pop_file_name,'wb') as pop_file:
        pickle.dump([Gs,Ss,obj],pop_file)
    # Save time needed to perform this run.
    with open('results/'+subcategory+'time_n'+str(n)+'k'+str(k)+'.csv','a') as time_file:
        print(elapsed_time,file=time_file)
    # Save number of steps needed to perform this run.
    with open('results/'+subcategory+'nfe_n'+str(n)+'k'+str(k)+'.csv','a') as nfe_file:
        print(algorithm.nfe,file=nfe_file)


