# This file contains recombination and mutations used for NMTF problem

import numpy as np
from platypus import Variator, Mutation, copy, PlatypusError

def delete_random_column(parent):
    child = copy.deepcopy(parent)
    k=child.objectives[1]
    G=child.variables[0]
    S=child.variables[1]
    i=np.random.randint(k)
    G=np.delete(G,i,1)
    S=np.delete(S,i,1)
    S=np.delete(S,i,2)
    child.variables=[G,S]
    child.evaluated=False
    return child

def delete_lest_significant_column(parent):
    G=parent.variables[0]
    S=parent.variables[1]
    k=parent.objectives[1]
    s=np.inf
    least_significant_i=-1
    for i in range(k):
        s_new=np.sum(S[:,i,:])+np.sum(S[:,:,i])
        if s_new < s:
            s=s_new
            least_significant_i=i
    if least_significant_i < 0:
        raise PlatypusError("S matrices have inf values.")
    G=np.delete(G,least_significant_i,1)
    S=np.delete(S,least_significant_i,1)
    S=np.delete(S,least_significant_i,2)
    parent.variables=[G,S]
    parent.objectives[1]=k-1
    parent.evaluated=False
    return parent
    
def add_columns(parent,dk,almost_zero):
    child = copy.deepcopy(parent)
    G=child.variables[0]
    S=child.variables[1]
    m=S.shape[0]
    n=G.shape[0]
    k=child.objectives[1]
    # create larger arrays and store G,S in them
    newG=np.zeros((n,k+dk))
    newS=np.zeros((m,k+dk,k+dk))
    newG[:,:k]=G
    newS[:,:k,:k]=S
    # add random columns to newG
    newG[:,k:]=almost_zero*np.random.rand(n,dk)
    # create side and corner block of S in symmetric way ...
    S_side=almost_zero*np.random.rand(m,k,dk)
    S_corner=0.5*almost_zero*np.random.rand(m,dk,dk)
    S_corner=S_corner+np.transpose(S_corner,(0,2,1))
    # ... and add those blocks to newS
    newS[:,:k,k:]=S_side
    newS[:,k:,:k]=np.transpose(S_side,(0,2,1))
    newS[:,k:,k:]=S_corner
    child.variables=[newG,newS]
    child.evaluated=False
    return child

class AddDeleteColumns(Mutation):
    def __init__(self,dk_mean=3.,almost_zero=1e-8):
        super(AddDeleteColumns,self).__init__()
        self.almost_zero=almost_zero
        self.dk_mean=dk_mean

    def mutate(self,parent,dk):
        if dk > 0:
            return add_columns(parent,dk,self.almost_zero)
        elif dk < 0:
            child = copy.deepcopy(parent)
            for i in range(-dk):
                child=delete_lest_significant_column(child)
            return child
        else:
            raise PlatypusError('Algorithm wanted to delete 0 columns!')

class JoinMatrices(Variator):
    def __init__(self,almost_zero=1e-8):
        super(JoinMatrices,self).__init__(2)
        self.almost_zero=almost_zero
        self.factorG=0.5**0.25
        self.factorS=0.5**0.5

    def evolve(self,parents):
        result=copy.deepcopy(parents[0])
        # Join G matrices side by side
        G1=parents[0].variables[0]
        G2=parents[1].variables[0]
        newG=self.factorG*np.concatenate((G1,G2),axis=1)
        # Join S tensors as a direct sum
        S1=parents[0].variables[1]
        S2=parents[1].variables[1]
        k1=parents[0].objectives[1]
        k2=parents[1].objectives[1]
        m=S1.shape[0]
        newS=np.empty((m,k1+k2,k1+k2))
        newS[:,:k1,:k1]=self.factorS*S1
        newS[:,k1:,k1:]=self.factorS*S2
        almost_zero=self.almost_zero*np.random.rand(m,k1,k2)
        newS[:,:k1,k1:]=almost_zero
        newS[:,k1:,:k1]=np.transpose(almost_zero,(0,2,1))
        # Save to result
        result.variables=[newG,newS]
        result.evaluated=False
        return [result]
