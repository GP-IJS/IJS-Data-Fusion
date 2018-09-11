# This file contains methods for local search that is needed for the main algorithms

import tensorflow as tf
import numpy as np
from matrix_utilities import norm
from platypus import Mutation, copy

def construct_poly(poly,dR,G1,S,G2,G1S,dG1,dS,dG2,order,W=None):
    poly_array=tf.Variable(0.,shape=[order+1,G1.shape[0],G2.shape[0]])
    poly_array[order,:,:]=dR
    dG1S=tf.matmult(dG1,S)
    G1dS=tf.matmult(G1,dS)
    poly_array[order-1,:,:]=tf.matmult(dG1S,G2)+tf.matmult(G1dS,G2)+tf.matmult(G1S,dG2)
    if order > 1:
        dG1dS=tf.matmult(dG1,dS)
        poly_array[order-2,:,:]=tf.matmult(dG1dS,G2)+tf.matmult(G1dS,dG2)+tf.matmult(dG1S,G2)
    if ordery > 2:
        poly_array[order-3,:,:]=tf.matmult(dG1dS,dG2)
    if order > 3:
        raise PlatypusError('Polynomial order too large in construct_poly!')
    if W is not None:
        poly_array*=W
    new_order=2*order
    conv_poly_array=tf.Variable(0.,shape=[new_order,G1.shape[0],G2.shape[0]])
    conv_poly_array[new_order-1,:,:]=2.*poly_array[order,:,:]*poly_array[order-1,:,:]
    conv_poly_array[new_order-2,:,:]=2*poly_array[order,:,:]*poly_array[order-2,:,:]+poly_array[order-1,:,:]*poly_array[order-1,:,:]
    if order > 1:
        conv_poly_array[new_order-3,:,:]=2*poly_array[order,:,:]*poly_array[order-3,:,:]+2.*poly_array[order-1,:,:]*poly_array[order-2,:,:]
        conv_poly_array[new_order-4,:,:]=2*poly_array[order-3,:,:]*poly_array[order-1,:,:]+poly_array[order-2,:,:]*poly_array[order-2,:,:]
    if order > 2:
        conv_poly_array[new_order-5,:,:]=2*poly_array[order-2,:,:]*poly_array[order-3,:,:]
        conv_poly_array[new_order-6,:,:]=poly_array[order-3,:,:]*poly_array[order-3,:,:]
    poly+=tf.reduce_sum(conv_poly_array,axis=[1,2])
    return poly
################################################################################
#                    WHAT ABOUT REGULARIZATION ?
################################################################################

def find_stepsize(poly):
    diff_poly=poly*np.arange(poly.shape[0],0,-1)
    poly_diff_rts=np.roots(diff_poly)
    poly_eval=np.append(poly,0)
    best=np.inf
    best_i=-1
    for rt_i,rt in enumerate(poly_diff_rts):
        if np.isreal(rt):
            new=np.polyval(poly_eval,rt)
            if new < best:
                best=new
                best_i=rt_i
    print('predicted dc='+str(best))
    if best_i < 0:
        raise PlatypusError("No good roots were found in find_stepsize.")
    return poly_diff_rts[best_i]

class AngleRegulatrization:
    def __init__(self,_lambda):
        self._lambda=_lambda

    def add_regularization(self,G,S,dtype):
        cost=tf.Variable(initial_value=0.,dtype=dtype)
        for Gi in G:
            # Dot products of pairs.
            Ci=tf.matmul(Gi,tf.transpose(Gi))
            Li=tf.sqrt(tf.diag_part(Ci))
            # Lengths.
            LTLi=tf.multiply(tf.transpose(Li), Li)
            # Angles between columns.
            Di=tf.divide(Ci,LTLi)
            # Mean on angles.
            cost+=tf.reduce_mean(tf.matrix_band_part(D,0,-1))
        return self._lambda*cost

    def construct_poly(poly,G,S,dG,dS,order):
        raise PlatipusError("Method construct_poly in AngleRegulatrization is not defined!")

class GNormalColumnsRegularization:
    def __init__(self,_lambda):
        self._lambda=_lambda

    def add_regularization(self,G,S,dtype):
        for Gi in G:
            norm_columns=tf.reduce_sum(Gi*Gi,axis=1)-1.
            cost=tf.reduce_sum(norm_columns*norm_columns)
        return self._lambda*cost

    def construct_poly(poly,G,S,dG,dS,order):
        for Gi,dGi in zip(G,dG):
            pass

class S2NormRegularization:
    def __init__(self,_lambda):
        self._lambda=_lambda

    def add_regularization(self,G,S,dtype):
        cost=tf.Variable(initial_value=0.,dtype=dtype)
        for Si in S:
            cost+=tf.norm(Si,ord=self.order)
        return self._lambda*cost

    def construct_poly(poly,G,S,dG,dS,order):
        for Si,dSi in zip(S,dS):
            poly[-2]+=tf.reduce_sum(dSi*dSi)
            poly[-1]+=2.*tf.reduce_sum(Si*dSi)
        return poly

class GradientDescentOptimizer:
    def __init__(self, R_data, lr=0.001, beta_1=0.9, beta_2=0.99, eps=1e-8, poly_order=0, W=None, mask=False, regularizations=[], dtype=tf.float64):
        # To get ordinary gradient descent: beta_1=0, beta_2=1, eps=1
        # Save the parameters of Adam gradient descent
        self.lr, self.beta_1, self.beta_2, self.eps = lr, beta_1, beta_2, eps
        self.R_data = R_data

        # The number of matrices R and S
        l=len(self.R_data)
        # The number of matrices G
        m=(np.sqrt(8*l+1)-1)/2.
        if np.abs(m-int(m)) > 1e-8:
            raise PlatipusError("Length of R_data is not of the form n(n+1)/2 in GradientDescentOptimizer!")
        m=int(m)

        # Sum of square of Frobenius norm of matrices R
        fnR=[]
        for R in self.R_data:
            fnR.append(np.sum(np.square(R)))
        
        # Tensor definitions
        self.R=[]
        self.S=[]
        self.Sm=[]
        self.Sv=[]
        self.Sp=[]
        if W is not None:
            self.W=[]
        for i in range(l):
            self.R.append(tf.constant(self.R_data[i],name='R',dtype=dtype))
            self.S.append(tf.Variable(initial_value=[],validate_shape=False,name='S',dtype=dtype))
            self.Sm.append(tf.Variable(initial_value=[],validate_shape=False,name='Sm',dtype=dtype))
            self.Sv.append(tf.Variable(initial_value=[],validate_shape=False,name='Sv',dtype=dtype))
            self.Sp.append(tf.placeholder(dtype))
            if W is not None:
                self.W.append(tf.placeholder(dtype))
        self.G=[]
        self.Gm=[]
        self.Gv=[]
        self.Gp=[]
        for _ in range(m):
            self.G.append(tf.Variable(initial_value=[],validate_shape=False,name='G',dtype=dtype))
            self.Gm.append(tf.Variable(initial_value=[],validate_shape=False,name='Gm',dtype=dtype))
            self.Gv.append(tf.Variable(initial_value=[],validate_shape=False,name='Gv',dtype=dtype))
            self.Gp.append(tf.placeholder(dtype))
        self.t=tf.Variable(initial_value=0.0,name='t',dtype=dtype)
        if mask:
            self.mask=[]
            for _ in range(m):
                self.mask.append(tf.placeholder(dtype))
        if poly_order > 0:
            self.poly=tf.Variable(initial_value=2*poly_order*[0.],name='poly',dtype=dtype)
        self.cost=tf.Variable(initial_value=0.,name='cost',dtype=dtype)
        self.step_size=tf.placeholder(dtype)

        # Initialization of tensors
        new_cost_list=[tf.assign(cost,0.)]
        new_step_list=[]
        for Si,Spi,Smi,Svi in zip(self.S,self.Sp,self.Sm,self.Sv):
            new_cost_list.append(tf.assign(Si,Spi,validate_shape=False))
            new_step_list.append(tf.assign(Smi,tf.zeros_like(Spi),validate_shape=False))
            new_step_list.append(tf.assign(Svi,tf.zeros_like(Spi),validate_shape=False))
        for i,(Gi,Gpi,Gmi,Gvi) in enumerate(zip(self.G,self.Gp,self.Gm,self.Gv)):
            if mask:
                new_cost_list.append(tf.assign(Gi,Gpi*self.mask[i],validate_shape=False))
            else:
                new_cost_list.append(tf.assign(Gi,Gpi,validate_shape=False))
            new_step_list.append(tf.assign(Gmi,tf.zeros_like(Gpi),validate_shape=False))
            new_step_list.append(tf.assign(Gvi,tf.zeros_like(Gpi),validate_shape=False))
        self.assign_t=tf.assign(self.t,0.0)
        self.new_cost=tf.group(*new_cost_list)
        self.new_descent=tf.group(self.assign_t,*(new_cost_list+new_step_list))

        # Cost calculation
        self.G_abs=[]
        self.S_abs=[]
        for Gi in self.G:
            self.G_abs.append(tf.abs(Gi))
        for Si in self.S:
            self.S_abs.append(tf.abs(Si))
        self.GS=[]
        self.dR=[]
        r=0
        for i in range(m):
            for j in range(i,m):
                self.GS.append(tf.matmul(self.G_abs[i], self.S_abs[r]))
                self.dR.append(tf.matmul(self.GS[r],tf.transpose(self.G[j]))-self.R[r])
                self.cost+=tf.norm(self.dR[r])/fnR[r]
                r+=1

        self.regularized_cost=self.cost
        for regularization in regularizations:
            self.regularized_cost+=regularization.add_regularization(self.G_abs,self.S_abs,dtype)

        # Gradient calculation.
        g=tf.gradients(self.regularized_cost,self.G+self.S)
        self.g_G=g[:m]
        self.g_S=g[m:]
        if mask:
            for i,maski in enumerate(self.mask):
                self.g_G[i]=self.g_G[i]*maski

        # Step of gradient descent using ideal step method
        if poly_order > 0:
            r=0
            for i in range(m):
                for j in range(i,m):
                    if W is not None:
                        Wi=self.W[r]
                    else:
                        Wi=None
                    self.poly+=construct_poly(self.poly,self.dR[r],self.G[i],self.S[r],self.G[j],self.GS[r],self.g_G[i],self.g_S[r],self.g_G[j],W=Wi,order=poly_order)
        gradient_descent=[]
        for Gi,g_Gi in zip(self.G,self.g_G):
            gradient_descent.append(tf.assign(Gi,Gi+self.step_size*g_Gi))
        for Si,g_Si in zip(self.S,self.g_S):
            gradient_descent.append(tf.assign(Si,Si+self.step_size*g_Si))
        self.new_step_poly=tf.group(*gradient_descent)

        # Step of gradient descent using Adam
        newt=tf.assign(self.t,self.t+1)
        with tf.control_dependencies([newt]):
            self.alpha_t=self.lr*tf.sqrt(1.-self.beta_2**self.t)/(1.-self.beta_1**self.t)
            new_mv=[]
            for Gmi,Gvi,g_Gi in zip(self.Gm,self.Gv,self.g_G):
                new_mv.append(tf.assign(Gmi,self.beta_1*Gmi+(1.-self.beta_1)*g_Gi))
                new_mv.append(tf.assign(Gvi,self.beta_2*Gvi+(1.-self.beta_2)*g_Gi**2))
            for Smi,Svi,g_Si in zip(self.Sm,self.Sv,self.g_S):
                new_mv.append(tf.assign(Smi,self.beta_1*Smi+(1.-self.beta_1)*g_Si))
                new_mv.append(tf.assign(Svi,self.beta_2*Svi+(1.-self.beta_2)*g_Si**2))
            with tf.control_dependencies(new_mv):
                new_GS=[]
                for Gi,Gmi,Gvi in zip(self.G,self.Gm,self.Gv):
                    new_GS.append(tf.assign(Gi,Gi-self.alpha_t*Gmi/(tf.sqrt(Gvi)+self.eps)))
                for Si,Smi,Svi in zip(self.S,self.Sm,self.Sv):
                    new_GS.append(tf.assign(Si,Si-self.alpha_t*Smi/(tf.sqrt(Svi)+self.eps)))
        self.new_step_adam=tf.group(newt,*(new_mv+new_GS))
        
        # Start tensorflow session
        self.sess=tf.Session()

    def grad_desc_poly(self,npG,npS,steps,mask=None):
        input_dict=dict()
        for Gpi,npGi in zip(self.Gp,npG):
            input_dict[Gpi]=npGi
        for Spi,npSi in zip(self.Sp,npS):
            input_dict[Spi]=npSi
        if mask is not None:
            for maskpi,masknpi in zip(self.mask,mask):
                input_dict[maskpi]=masknpi
        self.sess.run(self.new_cost,feed_dict=input_dict)
        for i in range(steps):
            poly=self.sess.run(self.poly)
            step_size=find_stepsize(poly)
            self.sess.run(self.new_step_poly,feed_dict={self.step_size: step_size})
            c=self.sess.run(self.cost)
        
    # Function that does gradient descent using Adam algorithm
    def adam(self,npG,npS,steps,worsening_factor,convergence_steps,convergence_factor,mask=None):
        input_dict=dict()
        for Gpi,npGi in zip(self.Gp,npG):
            input_dict[Gpi]=npGi
        for Spi,npSi in zip(self.Sp,npS):
            input_dict[Spi]=npSi
        if mask is not None:
            for maskpi,masknpi in zip(self.mask,mask):
                input_dict[maskpi]=masknpi
        self.sess.run(self.new_descent,feed_dict=input_dict)
        c_best=np.inf
        c_old=1.
        dc_list=[]
        dc_new=0.
        dc_worst=-np.inf
        dc_worst_ok=np.inf
        for i in range(steps):
            # perform new step
            self.sess.run(self.new_step_adam)
            # calculate the cost at new point
            c_new=self.sess.run(self.cost)
            print(c_new)
            # save best point seen so far
            if c_new < c_best:
                c_best=c_new
                npG,npS=self.sess.run([self.G_abs,self.S_abs])
            # monitor the progress of descent
            dc_list.append(c_new/c_old-1.)
            if i >= convergence_steps:
                # stop descent if the result worsens
                if c_new > c_best*worsening_factor:
                    break
                # delete old point
                del dc_list[0]
                # calculate median of the last convergence_steps points
                dc_new=np.median(np.asarray(dc_list))
                # save new median to dc_worst if it is the worst seen yet
                if dc_new > dc_worst:
                    dc_worst=dc_new
                # if dc_worst was a local minimum, save it to dc_worst_ok
                if dc_new < dc_worst:
                    dc_worst_ok=dc_worst
                # convergence happens if current median is low compared to dc_worst_ok
                # or if new median becomes positive
                if dc_new*convergence_factor > dc_worst_ok or dc_new > 0.:
                    break
            c_old=c_new
        # norm the columns of G and adjust S accordingly
        G_best,S_best=norm(npG,npS)
        return c_best,G_best,S_best,i+1

    # Function that calculates the cost
    def calculate_cost(self,npG,npS,mask=None):
        input_dict=dict()
        for Gpi,npGi in zip(self.Gp,npG):
            input_dict[Gpi]=npGi
        for Spi,npSi in zip(self.Sp,npS):
            input_dict[Spi]=npSi
        if mask is not None:
            for maskpi,masknpi in zip(self.mask,mask):
                input_dict[maskpi]=masknpi
        self.sess.run(self.new_cost,feed_dict=input_dict)
        c=self.sess.run(self.cost)
        return c

    # Method that closes tensorflow session
    def close(self):
        self.sess.close()

class AdamLocalSearch(Mutation):
    def __init__(self, gd, steps):
        super(AdamLocalSearch, self).__init__()
        self.gd=gd
        self.steps=steps

    def mutate(self,parent,worsening_factor=5.,convergence_steps=150,convergence_factor=3.):
        child = copy.deepcopy(parent)
        G=child.variables[0]
        S=child.variables[1]
        k=S.shape[1]
        c,newG,newS,dnfe=self.gd.adam(G,S,self.steps,worsening_factor,convergence_steps,convergence_factor)
        child.variables=[newG,newS]
        child.objectives=[c,k]
        child.evaluated=True
        return child,dnfe
