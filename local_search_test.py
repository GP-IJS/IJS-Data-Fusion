# This script tests methods from local_search.py on a real problem
import matplotlib.pyplot as plt
import numpy as np

#from matrix_utilities import construct_problem, construct_starting_point
#from local_search import GradientDescentAdam
#
#dims=[70,90,100]
#ks=[12,33,41]
#steps=100
#
## Construct optimization problem
#R,G,S=construct_problem(dims,ks)
## Initial point for gradient descent
#G,S=construct_starting_point(dims,ks)
## Construct gradient descent computational graph and run it
#p=GradientDescentAdam(R)
## Do gradient descent
#p.adam(G,S,steps,5.,150,3.)
## This script tests methods from local_search.py on a real problem
#
#from matrix_utilities import construct_problem, construct_starting_point, get_NMTF_problem
#from local_search import GradientDescent, GradientDescentAdam, GradientDescentPoly, S2NormRegularization, OrthonormalColumnsRegularization
#
#dims=[1813,1697,2230]
#ks=[42,33,51]
#num_matrices=[2,1,0,1,3,2]
#
##steps_poly=1000
from regularizations import OrthonormalColumnsRegularization, SAbsNormRegularization, S2NormRegularization

steps_adam=2000
#
## Construct optimization problem
#R,G,S,W=construct_problem(dims,ks,num_matrices,W=True)

# Take a saved syntetic problem.
from local_search import GradientDescentAdam, GradientDescentPoly
from matrix_utilities import get_NMTF_problem

n=500
ks=[20]
d=6
R=get_NMTF_problem(n,ks[0],d)

# Construct gradient descent computational graph and run it
#regs=[OrthonormalColumnsRegularization(0.1)]
#regs=[S2NormRegularization(1e-5)]
reg1 = OrthonormalColumnsRegularization(0.001)
reg3 = S2NormRegularization(0.05)
#reg2 = SAbsNormRegularization(0.04)

#reg2 = S2NormRegularization(0)
#p=GradientDescentPoly(R,W,regularizations=regs)
#p=GradientDescentPoly(R,W)
#p=GradientDescentPoly(R)
#p=GradientDescentAdam(R,W)

r = dict()
alpha = 0.001
for _ in range(1):

    regs = [reg1, reg3]
    p=GradientDescentPoly(R, regularizations=regs)
    #p = GradientDescentAdam(R, lr=alpha)
    p_best,p_progress,G_best,S_best=p.optimize(steps_adam,ks=ks)
    r[alpha] = p_progress
    #p_progress=[]

    #plt.imshow(G_best[0])
    #plt.show()

    #cum = np.cumsum(r_progress, axis=0)
    #print(cum)
    ##plt.plot(p_progress, label="cost")
    ##plt.plot(pr_progress, label="reg. cost")
    #for i, reg in enumerate(regs):
    #    plt.plot(cum[i,:]+p_progress)
    plt.plot(p_progress)

    #print(G_best)
    #print()
    #print(S_best[0][0])
    alpha /= 2


plt.legend()
plt.show()



# End session for poly.
#p.close()

#regs=[OrthonormalColumnsRegularization(0.5), S2NormRegularization(1e-5)]
#regs=[S2NormRegularization(1e-5)]
#regs=[OrthonormalColumnsRegularization(0.1)]

#r=GradientDescentPoly(R,W,regularizations=regs)
#r_progress,Go,So=r.optimize(steps_poly,G=G_poly,S=S_poly,restart=False)
#r=GradientDescentAdam(R,regularizations=regs)
#r_best,r_progress,G,S=r.optimize(steps_adam,G=G_poly,S=S_poly)

# Launch adam where we left of.
#r=GradientDescentAdam(R,W)

#r_best,r_progress,G_adam2,S_adam2=r.optimize(steps_adam,G_adam,S_adam)
