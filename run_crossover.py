from crossovers import ArithmeticCrossover
from local_search import GradientDescentAdam
from matrix_utilities import get_NMTF_problem

steps = 50
n=500
ks=[20]
d=6
R=get_NMTF_problem(n,ks[0],d)
p = GradientDescentAdam(R, lr=0.01)
_, _, _, _, G1, S1 = p.optimize(steps,ks=ks)
_, _, _, _, G2, S2 = p.optimize(steps,ks=ks)

cr = ArithmeticCrossover()
cr.cross([(G1, S1),(G2, S2)])



