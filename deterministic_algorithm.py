# This file contains the components for deterministic optimization algorithm for solving NMTF

import numpy as np
from platypus import AbstractGeneticAlgorithm, TerminationCondition

class MaxEvaluationsAndTreshold(TerminationCondition):
    
    def __init__(self, nfe):
        super(MaxEvaluationsAndTreshold, self).__init__()
        self.nfe = nfe
        self.starting_nfe = 0
        
    def initialize(self, algorithm):
        self.starting_nfe = algorithm.nfe
        
    def shouldTerminate(self, algorithm):
        # Termination when sufficient treshold is reached
        other_termination=algorithm.c_best < algorithm.c_treshold
        # Termination when the number of function evaluations was exceeded.
        nfe_termination=algorithm.nfe - self.starting_nfe >= self.nfe
        return nfe_termination or other_termination

def update_best(individual,c_best):
    c_new=individual.objectives[0]
    if c_new < c_best:
        return c_new
    else:
        return c_best

class deterministic_algorithm(AbstractGeneticAlgorithm):

    def __init__(self,problem,local_search,generator,hypervolume_dynamics,
                 c_treshold=0.01,
                 desired_density=1,
                 **kwargs):
        super(deterministic_algorithm,self).__init__(problem,desired_density,generator,**kwargs)
        self.c_best=np.inf
        self.hypervolume_dynamics=hypervolume_dynamics
        self.local_search=local_search
        self.c_treshold=c_treshold
        self.density=desired_density
        self.k=1

    def initialize(self):
        self.population=[self.generator.generate(self.problem,self.k) for _ in range(self.density)]
        new_population=[]
        for individual in self.population:
            new_individual,dnfe=self.local_search.mutate(individual)
            print(str(new_individual.objectives[1])+', steps='+str(dnfe)+', c='+str(new_individual.objectives[0]))
            self.c_best=update_best(new_individual,self.c_best)
            new_population.append(new_individual)
            self.nfe+=dnfe
        self.population=new_population
        # Add population to hypervolume_dynamics object and calculate initial hypervolume.
        for individual in self.population:
            self.hypervolume_dynamics.add_new(individual)
        self.hypervolume_dynamics.dynamics=[self.hypervolume_dynamics.dynamics[-1]]

    def iterate(self):
        self.k+=1
        new_individuals=[self.generator.generate(self.problem,self.k) for _ in range(self.density)]
        for individual in new_individuals:
            new_individual,dnfe=self.local_search.mutate(individual)
            print(str(new_individual.objectives[1])+', steps='+str(dnfe)+', c='+str(new_individual.objectives[0]))
            self.population.append(new_individual)
            self.nfe+=dnfe
            self.c_best=update_best(new_individual,self.c_best)
            self.hypervolume_dynamics.add_new(new_individual)
