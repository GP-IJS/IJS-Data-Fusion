from crossovers import ArithmeticCrossover
from local_search import GradientDescentAdam, GradientDescent
from local_search_PNMTF_poly import S2NormRegularization
from matrix_utilities import construct_problem, get_NMTF_problem
import numpy as np
import tensorflow as tf

#np.random.seed(42)

#dims=[6,5,6]
#ks=[2,3,3]
#num_matrices=[2,1,0,1,3,2]
#G_spar = 0.8
#S_spar = 0.7

# Construct optimization problem
#R1,G1,S1=construct_problem(dims,ks,num_matrices, G_sparsity=G_spar ,S_sparsity=S_spar)
#R2,G2,S2=construct_problem(dims,ks,num_matrices, G_sparsity=G_spar ,S_sparsity=S_spar)
#R3,G3,S3=construct_problem(dims,ks,num_matrices, G_sparsity=G_spar ,S_sparsity=S_spar)
#R4,G4,S4=construct_problem(dims,ks,num_matrices, G_sparsity=G_spar ,S_sparsity=S_spar)

#print([x[0].shape for x in zip(G1, G2, G3)])
#print([len(x) for x in S1])
from regularizations import OrthonormalColumnsRegularization

n=500
ks=[20]
d=6
R=get_NMTF_problem(n,ks[0],d)

reg1 = OrthonormalColumnsRegularization(0.001)

alpha = 0.01
#regs = [reg1]#, reg3]
p1=GradientDescentAdam(R, lr=alpha)
#p2=GradientDescentAdam(R, regularizations=regs, lr=alpha)

c,_,_,_,G1,S1=p1.optimize(50,ks=ks)
print("Cost1", c, p1.calculate_cost(G1,S1))
c,_,_,_,G2,S2=p1.optimize(50,ks=ks)
print("Cost2", c, p1.calculate_cost(G2,S2))

cr = ArithmeticCrossover(number_of_parents=2)
G, S = cr.cross([(G1, S1),(G2, S2)])#, (G3, S3), (G4, S4)])
print("Cost3", p1.calculate_cost(G, S))
