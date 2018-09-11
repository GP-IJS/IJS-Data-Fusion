# This script tests emirically which mutations and recombination are better

from abc import ABCMeta
import numpy as np
from platypus import Generator, Solution, Problem

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

class RandomKMatrixGenerator(Generator):
    def __init__(self,m,n,max_k,scale=0.01):
        super(RandomKMatrixGenerator, self).__init__()
        self.m=m
        self.n=n
        self.max_k=max_k
        self.scale=scale

    def generate(self, problem):
        solution = Solution(problem)
        i=np.random.randint(self.max_k)+1
        solution.variables[0]=self.scale*np.random.rand(self.n,i)
        solution.variables[1]=self.scale*np.random.rand(self.m,i,i)
        return solution

class ConstantKMatrixGenerator(Generator):
    def __init__(self,m,n,k,scale=0.01):
        super(ConstantKMatrixGenerator, self).__init__()
        self.m=m
        self.n=n
        self.k=k
        self.scale=scale

    def generate(self, problem):
        solution = Solution(problem)
        solution.variables[0]=self.scale*np.random.rand(self.n, self.k)
        solution.variables[1]=self.scale*np.random.rand(self.m, self.k, self.k)
        return solution

class Tri_Factorization(Problem):
    def __init__(self, gd_optimizer, number_of_variables=2, number_of_objectives=2):

        # The objective function
        def obj_fun(cost_evaluator, x):
            G = x[0]
            S = x[1]
            k = S.shape[1]
            cost = cost_evaluator.calculate_cost(G, S)
            return [cost, k]

        # Partial application of argument cost_evaluator
        obj_fun_partial = partial(obj_fun, gd_optimizer)
        super().__init__(number_of_variables,number_of_objectives,function=obj_fun_partial)

        # Set type of variables
        self.types[:] = Tensor()
        self.gdOptimizer = gd_optimizer
