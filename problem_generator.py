import warnings

import numpy as np
import tensorflow as tf

#np.random.seed(42)
from local_search import GradientDescentAdam
from matrix_utilities import construct_problem_orthogonal

dims=[900,900,900]
ks=[100,23,202]
num_matrices=[5,2,0,1,6,3]
S_spar = 0.2


# Construct optimization problem
R,G,S=construct_problem_orthogonal(dims,ks,num_matrices, S_sparsity=S_spar, G_row_sparsity=0.2)


alpha = 0.01
#regs = [reg1]#, reg3]
p1=GradientDescentAdam(R, lr=alpha)
#p2=GradientDescentAdam(R, regularizations=regs, lr=alpha)

c,_,_,_,G1,S1=p1.optimize(50,ks=ks)
print("Cost1", c, p1.calculate_cost(G1,S1))
c,_,_,_,G2,S2=p1.optimize(50,ks=ks)
print("Cost2", c, p1.calculate_cost(G2,S2))

#cr = ArithmeticCrossover(number_of_parents=2)
#G, S = cr.cross([(G1, S1),(G2, S2)])#, (G3, S3), (G4, S4)])
#print("Cost3", p1.calculate_cost(G, S))


