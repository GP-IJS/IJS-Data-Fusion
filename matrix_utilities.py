"""This file contains supplementary methods for matrix manipulations
needed for other algorithms. All matrices are persumed to be nonnegative."""
import warnings

import numpy as np
import heapq
import itertools
from os import listdir
from itertools import product

from local_search import GradientDescentAdam

p3map = map
map = lambda func, *iterable: list(p3map(func, *iterable))

def norm(G,S):
    """Changes G and S matrices so that G_i.S_ij.G_j^T stay
    the same and so that the columns of G are normed."""
    # Norm the columns and save the norms for S matrices transformation.
    g=[np.sqrt(np.sum(np.square(Gi),axis=0,keepdims=True)) for Gi in G]
    G=[Gi/gi for Gi,gi in zip(G,g)]
    # Transform matrices S.
    r=0
    for i,gi in enumerate(g):
        for j in range(i,len(g)):
            gj=g[j]
            S[r]=map(lambda Si: Si*np.matmul(gi.T,gj), S[r])
            r+=1
    return G,S

def read_sparse_matrices(directory, matrices_names):
    """Gets file names of saved sparse matrices and returns them as a list of numpy arrays."""
    R=[]
    matrices_names = [directory + mat for mat in matrices_names]
    for file_name in matrices_names:
        sparseR=np.loadtxt(file_name, delimiter=',', skiprows=1)
        ni=int(np.round(np.max(sparseR[:,0:1])))
        Ri=np.zeros((ni,ni))
        for row in range(ni):
            i=int(np.round(sparseR[row,0]))-1
            j=int(np.round(sparseR[row,1]))-1
            Ri[i,j]=sparseR[row,2]
        R.append(Ri)
    return R

def get_NMTF_problem(n,k,d):
    """Imports matrices saved in sparse form in data directory
    and returns a NMTF problem encoded in PNMTF problem form."""
    # Find directory for that specific n, k, d
    directories=listdir('../data')
    n_string='n='+str(n)+'_'
    directories=[s for s in directories if n_string in s]
    k_string='k='+str(k)+'_'
    directories=[s for s in directories if k_string in s]
    d_string='density='+str(d)+'.'
    directories=[s for s in directories if d_string in s]
    # Raise error if none or multiple directories were found.
    if len(directories) == 0:
        raise RuntimeError('Directory with such n, k, d not found!')
    if len(directories) != 1:
        raise RuntimeError('Multiple directories with such n, k, d was found!')
    directory='../data/'+directories[0]+'/'
    # Find the names of files that store the values of R matrices.
    matrices=listdir(directory)
    matrices=[s for s in matrices if 'R' in s]
    # Raise error if no files 'R*.csv' were found
    if len(matrices) == 0:
        raise RuntimeError('Directory with those n, k, d not found!')
    # Read matrices and save them to list of numpy arrays
    R=read_sparse_matrices(directory, matrices)
    # Return data in the form of PNMTF problem
    return [R]

def pop_heap_degeneracy(heap, degeneracy, number, num_unique_rows):
    """Supplementary recursive function used by closest_orthogonal_masks function."""
    # When the heap is to small do not pop.
    if len(heap) < number:
        return degeneracy, num_unique_rows
    # When we have exactly the right number of unique rows and the
    # worst candidate is the only one in the row we also do not pop.
    worst_element=heap[0]
    i_worst=worst_element[1][0][0]
    if degeneracy[i_worst] == 1:
        if num_unique_rows == number:
            return degeneracy, num_unique_rows
    # In any other case we pop an element and call the function again.
    heapq.heappop(heap)
    degeneracy[i_worst]-=1
    if degeneracy[i_worst] == 0:
        num_unique_rows-=1
    return pop_heap_degeneracy(heap, degeneracy, number, num_unique_rows)

def closest_orthogonal_masks(M, number=1, fast_mode=True, norm_function=lambda x: x*x):
    """Finds a given number of closest orthogonal matrices and returns their masks."""
    # Find maximum value indices.
    arg_max=np.argmax(M,1)
    # Put ones on those indices and zeros otherwise to get the closest mask.
    n=M.shape[0]
    m=M.shape[1]
    best_mask=np.zeros(M.shape)
    best_mask[range(n),arg_max]=1.
    # If we only need the closest mask, return it.
    if number <= 1:
        return [best_mask]
    # Get the difference matrix.
    dM=norm_function(M)-norm_function(np.reshape(M[range(n),arg_max],(-1,1)))
    # Find the smallest values and insert them to a heap data structure.
    heap=[]
    degeneracy=np.zeros(n+1,dtype=np.int16)
    num_unique_rows=0
    # Push a dummy element that ensures that heap gets filled
    # by any element at the start when len(heap) < number.
    heapq.heappush(heap,(-np.inf,[[n,m]]))
    degeneracy[n]=1
    for j in range(m):
        for i in range(n):
            # Ignore the best mask components.
            if j == arg_max[i]:
                continue
            # Add element to the heap if it is large enough.
            if heap[0][0] < dM[i,j]:
                heapq.heappush(heap,(dM[i,j],[[i,j]]))
                # Add to degeneracy array.
                degeneracy[i]+=1
                # Increment the number of unique rows if this is the case.
                if degeneracy[i] == 1:
                    num_unique_rows+=1
                # Pop elements if needed.
                degeneracy,num_unique_rows=pop_heap_degeneracy(heap,degeneracy,number,num_unique_rows)
    # Sort the heap and save to a list.
    sorted_list=heapq.nlargest(len(heap),heap,key=lambda x: x[0])
    # Loop through the power set of all elements of the heap in
    # an inteligent way and get the best composited elements.
    if not fast_mode:
        for i in range(1,len(sorted_list)):
            s=sorted_list[:i]
            power_set=itertools.chain.from_iterable(itertools.combinations(s,r) for r in range(1,len(s)+1))
            end_the_loop=True
            # Loop through the power set and determine if constructed elements are worthy to
            # include them in the heap and break the outher loop if entire power set is unworthy.
            for composite in power_set:
                element=list(sorted_list[i])
                colisions=set(element[1][0][:1])
                colision_flag=False
                # Construct an element that is encoded as the ones in the heap.
                for pertubation in composite:
                    # Mind that there might be colisions and we note that with colision_flag.
                    if pertubation[1][0][0] in colisions:
                        colision_flag=True
                    colisions.add(pertubation[1][0][0])
                    element[0]=element[0]+pertubation[0]
                    element[1]=element[1]+pertubation[1]
                # Find if such an element is worthy to be in the heap.
                if element[0] > heap[0][0]:
                    end_the_loop=False
                    # If there were no colisions we push the element to the heap.
                    if not colision_flag:
                        heapq.heappush(heap,tuple(element))
            if end_the_loop:
                break
    # Sort the heap again and truncate it.
    sorted_list=heapq.nlargest(number-1,heap,key=lambda x: x[0])
    # Construct the masks from pertubations and return them in a list.
    masks=[best_mask]
    for element in sorted_list:
        new_mask=np.copy(best_mask)
        for i,j in element[1]:
            new_mask[i,arg_max[i]]=0.
            new_mask[i,j]=1.
        masks.append(new_mask)
    return masks

def construct_problem(dims,ks,num_matrices,G_sparsity=0.2,S_sparsity=0.3,W=False,W_sparsity=0.7):
    """Constructs a syntetic PNMTF problem."""
    nd=len(dims)
    # Randomly choose matrices G.
    G=[]
    for d1,d2 in zip(dims,ks):
        G.append(np.random.choice([1.0,0.0],(d1,d2),p=[G_sparsity,1-G_sparsity])*np.random.rand(d1,d2))
    # Randomly choose matrices S and calculate matrices R.
    S=[]
    R=[]
    r=0
    for i in range(nd):
        for j in range(i,nd):
            S.append([])
            R.append([])
            for _ in range(num_matrices[r]):
                S[r].append(np.random.choice([1.0,0.0],(ks[i],ks[j]),p=[S_sparsity,1-S_sparsity])*np.random.rand(ks[i],ks[j]))
                R_new=np.dot(np.dot(G[i],S[r][-1]),G[j].T)
                R_new/=np.max(R_new)
                R[r].append(R_new)
            r+=1
    # In case missing values should also be present, generate masks W.
    if W:
        W=map(lambda R_list: map(lambda Ri: np.random.choice([1.0,0.0],Ri.shape,p=[W_sparsity,1-W_sparsity]), R_list), R)
    if W:
        return R,G,S,W
    else:
        return R,G,S

def construct_problem_orthogonal(dims,ks,num_matrices,S_sparsity=0.3,W=False,W_sparsity=0.7, G_row_sparsity=0.0):
    """Constructs a syntetic PNMTF problem with orthogonal G."""
    nd=len(dims)
    # Randomly choose matrices G.
    G=[]
    G_density = 0
    for d1,d2 in zip(dims,ks):
        g = np.zeros((d1,d2))
        g[np.array(range(d1)), np.random.randint(0, high=d2, size=d1)] = np.maximum(0, np.random.rand(d1)-G_row_sparsity)
        G.append(g)
        G_density +=np.count_nonzero(g)/(d1*d2*len(ks))

    # Randomly choose matrices S and calculate matrices R.
    S=[]
    R=[]
    r=0
    for i in range(nd):
        for j in range(i,nd):
            S.append([])
            R.append([])
            for _ in range(num_matrices[r]):
                S[r].append(np.random.choice([1.0,0.0],(ks[i],ks[j]),p=[S_sparsity,1-S_sparsity])*np.random.rand(ks[i],ks[j]))
                R_new=np.dot(np.dot(G[i],S[r][-1]),G[j].T)
                R_new/=np.max(R_new)+0.00001
                R[r].append(R_new)
            r+=1
    # In case missing values should also be present, generate masks W.
    if W:
        W=map(lambda R_list: map(lambda Ri: np.random.choice([1.0,0.0],Ri.shape,p=[W_sparsity,1-W_sparsity]), R_list), R)
    if W:
        return R,G,S,W
    else:
        return R,G,S

def construct_starting_point(dims,ks,num_matrices,scale=0.01):
    """Constructs a random starting point based on the specifics of the problem
    and is used by optimisation algorithms to generate initial populations."""
    nd=len(dims)
    G=[]
    S=[]
    r=0
    for i in range(nd):
        G.append(scale*np.random.rand(dims[i],ks[i]))
        for j in range(i,nd):
            S.append([])
            for _ in range(num_matrices[r]):
                S[r].append(scale*np.random.rand(ks[i],ks[j]))
            r+=1
    return G,S

def two_step_orthogonal_optimization(R, ks, iterations_1=500, iterations_2=200, number_of_masks=2,
                                     adam1_params=None, adam2_params=None, reg1=None, reg2=None):
    """Returns a list of orthogonal solutions of length len(ks)^number_of_masks.
    A regular solution is first found with gradient descent. This solution is then used
    to construct masks used for further optimization of solutions with orthogonal matrices. """
    if adam1_params is None:
        adam1_params = {"lr": 0.01, "beta_1": 0.9, "beta_2": 0.99, "eps": 1e-8}
    if adam2_params is None:
        adam2_params = {"lr": 0.01, "beta_1": 0.9, "beta_2": 0.99, "eps": 1e-8}
    if reg1 is None:
        reg1 = []
    if reg2 is None:
        reg2 = []
    optimizer1 = GradientDescentAdam(R, regularizations=reg1, **adam1_params)
    c, _, _, _, G, S = optimizer1.optimize(iterations_1, ks=ks)
    optimizer1.close()
    G_masks = [closest_orthogonal_masks(g, number=number_of_masks) for g in G]
    results = []
    optimizer2 = GradientDescentAdam(R, orthogonality_constraint=True, regularizations=reg2, **adam2_params)
    for i, x in enumerate(product(*G_masks)):
        returned = optimizer2.optimize(iterations_2, G=G, S=S, M=x)
        results.append(returned)
    optimizer2.close()
    return results
