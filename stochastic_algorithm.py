# This file contains the components for stochastic optimization algorithm for solving NMTF

from __future__ import print_function
import numpy as np
import itertools
from platypus import AbstractGeneticAlgorithm, AttributeDominance, objective_key, TerminationCondition
from evolutionary_operators import AddDeleteColumns, JoinMatrices
from abc import ABCMeta, abstractmethod

class MaxEvaluationsAndTermination(TerminationCondition):
    
    def __init__(self, nfe):
        super(MaxEvaluationsAndTermination, self).__init__()
        self.nfe = nfe
        self.starting_nfe = 0
        
    def initialize(self, algorithm):
        self.starting_nfe = algorithm.nfe
        
    def shouldTerminate(self, algorithm):
        # Termination when sufficient density is reached.
        other_termination=False
        if algorithm.c_best < algorithm.c_treshold:
            # Count individuals that have smaller k than k_best.
            smaller_than_best=0
            for individual in algorithm.population:
                if individual.objectives[1] < algorithm.k_best:
                    smaller_than_best+=1
            # If approximation of Pareto front for k < k_best has sufficient density, terminate.
            if smaller_than_best > algorithm.density*algorithm.k_best:
                other_termination=True
                print('termination',file=algorithm.print_file)
                print('termination')
                algorithm.print_file.close()
        # Termination when the number of function evaluations was exceeded.
        nfe_termination=algorithm.nfe - self.starting_nfe >= self.nfe
        if nfe_termination:
            print('nfe exceeded',file=algorithm.print_file)
            print('nfe exceeded')
            algorithm.print_file.close()
        return nfe_termination or other_termination

class Selector(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        super(Selector, self).__init__()
        
    def select(self, n, population, tournament_size):
        return list(map(lambda x: self.select_one(x,tournament_size), itertools.repeat(population, n)))
        
    @abstractmethod
    def select_one(self, population):
        raise NotImplementedError("method not implemented")

class TournamentSelector(Selector):
    
    def __init__(self, dominance):
        super(TournamentSelector, self).__init__()
        self.dominance = dominance
    
    def select_one(self, population, tournament_size):
        winner = np.random.randint(len(population))
        for _ in range(tournament_size-1):
            candidate = np.random.randint(len(population))
            flag = self.dominance.compare(population[winner], population[candidate])
            if flag > 0:
                winner = candidate
        return winner

def update_best(individual,c_best,k_best):
    c_new,k_new=individual.objectives
    if c_new < c_best:
        return c_new,k_new
    else:
        return c_best,k_best

class stochastic_algorithm(AbstractGeneticAlgorithm):

    def __init__(self,problem,local_search,generator,hypervolume_dynamics,print_file,
                 init_size=4,
                 c_treshold=0.1,
                 desired_density=1.,
                 number_of_recombinations=1,
                 number_of_mutations=2,
                 max_gd_descents=3,
                 selector=TournamentSelector(AttributeDominance(objective_key,False)),
                 variator=JoinMatrices(),
                 mutator=AddDeleteColumns(),
                 **kwargs):
        super(stochastic_algorithm,self).__init__(problem,init_size,generator,**kwargs)
        self.c_best=np.inf
        self.k_best=0
        self.gd_history=dict()
        self.init_size=init_size
        self.hypervolume_dynamics=hypervolume_dynamics
        self.variator=variator
        self.mutator=mutator
        self.local_search=local_search
        self.selector=selector
        self.c_treshold=c_treshold
        self.number_of_recombinations=number_of_recombinations
        self.number_of_mutations=number_of_mutations
        self.max_gd_descents=max_gd_descents
        self.tournament_size=1
        self.density=desired_density
        self.print_file=print_file

    def initialize(self):
        new_population=[]
        while len(new_population) < self.population_size:
            individual=self.generator.generate(self.problem)
            new_k=individual.variables[1].shape[1]
            if self.gd_history.setdefault(new_k,0) >= self.max_gd_descents:
                continue
            self.gd_history[new_k]+=1
            new_individual,dnfe=self.local_search.mutate(individual)
            print(str(new_individual.objectives[1])+', steps='+str(dnfe)+', c='+str(new_individual.objectives[0]),file=self.print_file)
            print(str(new_individual.objectives[1])+', steps='+str(dnfe)+', c='+str(new_individual.objectives[0]))
            self.c_best,self.k_best=update_best(new_individual,self.c_best,self.k_best)
            new_population.append(new_individual)
            self.nfe+=dnfe
        self.population=new_population
        # Add population to hypervolume_dynamics object and calculate initial hypervolume.
        for individual in self.population:
            self.hypervolume_dynamics.add_new(individual)
        self.hypervolume_dynamics.dynamics=[self.hypervolume_dynamics.dynamics[-1]]
    
    def perform_recombinations(self):
        recombinations_performed=0
        while recombinations_performed < self.number_of_recombinations:
            # Select parents.
            parents_indices=self.selector.select(self.variator.arity,self.population,self.tournament_size)
            # If one individual is to be recombined with itself, select again.
            if parents_indices[0] == parents_indices[1]:
                continue
            parents=[self.population[parents_indices[0]],self.population[parents_indices[1]]]
            # Test whether the recombined individual has been constructed many times before
            new_k=parents[0].objectives[1]+parents[1].objectives[1]
            if self.gd_history.setdefault(new_k,0) >= self.max_gd_descents:
                continue
            self.gd_history[new_k]+=1
            # Construct an offspring.
            offspring=self.variator.evolve(parents)[0]
            # Improve the offspring via local search
            offspring,dnfe=self.local_search.mutate(offspring)
            print('recombination: '+str(parents[0].objectives[1])+' + '+str(parents[1].objectives[1])+' = '+ str(new_k)+', steps='+str(dnfe)+', c='+str(offspring.objectives[0]),file=self.print_file)
            print('recombination: '+str(parents[0].objectives[1])+' + '+str(parents[1].objectives[1])+' = '+ str(new_k)+', steps='+str(dnfe)+', c='+str(offspring.objectives[0]))
            self.nfe+=dnfe
            # Add the offspring to the population
            self.population.append(offspring)
            # Update the best individual if necessary.
            self.c_best,self.k_best=update_best(offspring,self.c_best,self.k_best)
            # Add individual to Pareto front from which hypervolume dynamics is calculated.
            self.hypervolume_dynamics.add_new(offspring)
            recombinations_performed+=1

    def perform_mutations(self):
        mutations_performed=0
        while mutations_performed < self.number_of_mutations:
            # Select individual to be mutated.
            individual_index=self.selector.select_one(self.population,self.tournament_size)
            individual=self.population[individual_index]
            # Choose the number of columns to be deleted or added.
            k=individual.objectives[1]
            dk=np.random.exponential(self.mutator.dk_mean)
            dk=np.random.choice([-1,1])*(int(dk)+1)
            # Test whether the mutated individual has been constructed many times before
            new_k=k+dk
            if new_k < 1 or self.gd_history.setdefault(new_k,0) >= self.max_gd_descents:
                continue
            self.gd_history[new_k]+=1
            # Execute the mutation.
            mutated=self.mutator.mutate(individual,dk)
            # Improve the offspring via local search
            mutated,dnfe=self.local_search.mutate(mutated)
            print('mutation: '+str(k)+' -> '+str(new_k)+', steps='+str(dnfe)+', c='+str(mutated.objectives[0]),file=self.print_file)
            print('mutation: '+str(k)+' -> '+str(new_k)+', steps='+str(dnfe)+', c='+str(mutated.objectives[0]))
            self.nfe+=dnfe
            # Add the offspring to the population
            self.population.append(mutated)
            mutations_performed+=1
            # Update the best individual if necessary.
            self.c_best,self.k_best=update_best(mutated,self.c_best,self.k_best)
            # Add individual to Pareto front from which hypervolume dynamics is calculated.
            self.hypervolume_dynamics.add_new(mutated)

    def iterate(self):
        # Adjust the tournament size.
        self.tournament_size=int(len(self.population)/self.init_size)
        # Perform recombination, gradient descent and add new subject to the population.
        if self.c_best > self.c_treshold:
            self.perform_recombinations()
        # Perform mutation, gradient descent and add new subjects to the population.
        self.perform_mutations()
        
