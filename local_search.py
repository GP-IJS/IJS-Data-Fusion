"""This module contains methods for local search techniques and
incorporates gradient descent using two different algorithms."""

# RSE_flag=True
#
#        sum_ij sum_by_matrix_components (G_i.S_ij.G_j - R_ij)^2
# RSE = ---------------------------------------------------------
#               sum_ij sum_by_matrix_components (R_ij)^2
#
# RSE_flag=False
#
#                sum_by_matrix_components (G_i.S_ij.G_j - R_ij)^2
# RSE = mean_ij --------------------------------------------------
#                     sum_by_matrix_components (R_ij)^2

import tensorflow as tf
import numpy as np
from matrix_utilities import norm

# Redefine map so it always produces an object of type list
p3map = map
map = lambda func, *iterable: list(p3map(func, *iterable))

class CostAndGradients(object):
    """Includes methods for cost calculation and is used by classes that perform gradient descent."""
    def __init__(self, data, missing_values_mask, orthogonality_constraint, use_projection, regularizations, RSE_flag,
                 dtype):
        """Prepares the needed tensorflow tensors and constructs the
        computational graph for cost (and its gradient) calculation."""

        # The number of interactions among different types of data.
        num_interactions = len(data)
        # The number of matrices for each interaction.
        self.num_matrices = map(lambda x: len(x), data)
        # Regularization used in optimization
        self.regularizations = regularizations
        self.orthogonality_constraint = orthogonality_constraint
        # The number of data types, i.e. number of matrices G.
        num_data_types_float = (np.sqrt(8 * num_interactions + 1) - 1) / 2.
        self.num_data_types = int(np.rint(num_data_types_float))
        if np.abs(num_data_types_float - self.num_data_types) > 1e-8:
            raise RuntimeError("Length of data is not of the form n(n+1)/2 in GradientDescentOptimizer!")
        # Find out whether to use W matrices or not.
        if missing_values_mask is None:
            self.use_W = False
        else:
            self.use_W = True
        # The number of data instances for each data type.
        self.num_instances = [0] * self.num_data_types
        r = 0
        for i in range(self.num_data_types):
            for j in range(i, self.num_data_types):
                for Ri in data[r]:
                    self.num_instances[i] = Ri.shape[0]
                r += 1
        if any(elem == 0 for elem in self.num_instances):
            raise RuntimeError("At least one data type is redundant in CostAndGradients' init!")

        # Sum of square of Frobenius norm of matrices R (stored in the same way as data - nested list).
        if self.use_W:
            self.fnR = map(lambda R_list, W_list: map(lambda Ri, Wi: np.sum(np.square(Ri * Wi)), R_list, W_list), data,
                           missing_values_mask)
        else:
            self.fnR = map(lambda R_list: map(lambda Ri: np.sum(np.square(Ri)), R_list), data)
        if RSE_flag:
            total_fnR = sum(map(sum, self.fnR))
            self.fnR = map(lambda R_list: map(lambda Ri: total_fnR, R_list), data)

        # Tensor definitions that are used for cost calculation.
        self.R = map(lambda R_list: map(lambda Ri: tf.constant(Ri, name='R', dtype=dtype), R_list), data)
        self.S = map(
            lambda S_list: map(lambda Si: tf.Variable(initial_value=[], name='S', validate_shape=False, dtype=dtype),
                               S_list), self.fnR)
        self.Sp = map(lambda Sp_list: map(lambda Spi: tf.placeholder(dtype), Sp_list), self.fnR)
        self.G = [tf.Variable(initial_value=[], name='G', validate_shape=False, dtype=dtype) for _ in
                  range(self.num_data_types)]
        self.Gp = [tf.placeholder(dtype, name="Gp") for _ in range(self.num_data_types)]
        if self.use_W:
            self.W = map(lambda W_list: map(lambda Wi: tf.constant(Wi, name='W', dtype=dtype), W_list),
                         missing_values_mask)
        if orthogonality_constraint:
            self.M = [tf.Variable(initial_value=[], name='M', validate_shape=False, dtype=dtype) for _ in range(self.num_data_types)]
            self.Mp = [tf.placeholder(dtype, name="Mp") for _ in range(self.num_data_types)]

        # Initialization of tensors before cost calculation.
        self.S_assign_list = []
        # Assign placeholders of matrices S to variables S which will change during gradient descent while placeholders can't.
        self.S_assign_list.extend(
            [tf.assign(Si, Spi, validate_shape=False) for S_list, Sp_list in zip(self.S, self.Sp) for Si, Spi in
             zip(S_list, Sp_list)])
        # Same for matrices G.

        self.M_assign_list = []
        self.G_assign_list = []
        if orthogonality_constraint:
            self.M_assign_list.extend([tf.assign(Mi, Mpi, validate_shape=False) for Mi, Mpi in zip(self.M, self.Mp)])

            with tf.control_dependencies(self.M_assign_list):
                # Just in case Gs are projected to the feasible region if orthogonality constraint is in effect.
                self.G_assign_list.extend(
                    [tf.assign(Gi, Gpi * Mi, validate_shape=False) for Gi, Gpi, Mi in zip(self.G, self.Gp, self.M)])#TODO MP TO M
        else:
            self.G_assign_list.extend([tf.assign(Gi, Gpi, validate_shape=False) for Gi, Gpi in zip(self.G, self.Gp)])
        # Group all assignments to directive new_cost.
        self.new_cost = tf.group(*(self.M_assign_list+self.S_assign_list+self.G_assign_list))

        # Cost calculation.
        # Force matrices to be non-negative before calculating the cost when use_projection is not chosen.
        if use_projection:
            make_nonnegative = lambda x: x  # {G,S}_nneg is equivalent to {G,S}
        else:
            make_nonnegative = tf.abs  # {G,S}_nneg is equivalent to {|G|,|S|}
        self.G_nneg = [make_nonnegative(Gi) for Gi in self.G]
        self.S_nneg = map(lambda S_list: map(lambda Si: make_nonnegative(Si), S_list), self.S)
        self.GS = []
        self.dR = []
        r = 0
        # Go through all datatype combinations and ...
        for i in range(self.num_data_types):
            for j in range(i, self.num_data_types):
                # ... multiply each S matrix instance with appropriate G to get GS and ...
                self.GS.append(map(lambda Si: tf.matmul(self.G_nneg[i], Si), self.S_nneg[r]))
                # ... multiply each GS matrix instance with appropriate G to get GSG^T and subtract matrix R to get the difference ...
                if self.use_W:
                    # ... and possibly multiply with appropriate mask W to get dR ...

                    #TODO
                    self.dR.append(
                        map(lambda GSi, Ri, Wi: Wi * (tf.matmul(GSi, tf.transpose(self.G_nneg[j])) - Ri), self.GS[r],
                            self.R[r], self.W[r]))
                else:
                    self.dR.append(
                        map(lambda GSi, Ri: tf.matmul(GSi, tf.transpose(self.G_nneg[j])) - Ri, self.GS[r], self.R[r]))
                for k, fnRi in enumerate(self.fnR[r]):
                    # ... from which Frobenius norm is calculated and division with fnR is made to get RSE cost.
                    if r == 0 and k == 0:
                        self.cost = tf.reduce_sum(tf.square(self.dR[r][k])) / fnRi
                    else:
                        self.cost += tf.reduce_sum(tf.square(self.dR[r][k])) / fnRi
                r += 1

        # To make the gradient descent code simpler we define flattened version of S (as is g_S).
        self.Sf = [Si for S_list in self.S for Si in S_list]
        # Divide the cost with the number of R matrices so that the trivial solution has RSE=1 as intended.
        if not RSE_flag:
            self.cost /= len(self.Sf)

        # Add regularization cost that include penalties on matrices G, S.
        self.regularized_cost = self.cost


        for regularization in self.regularizations:
            self.regularized_cost += regularization.add_regularization(self.G_nneg, self.S_nneg, self.Sf, self.GS)



        # Gradient calculation.
        self.g = tf.gradients(self.regularized_cost, self.G + self.Sf)
        self.g_G = self.g[:self.num_data_types]
        self.g_S = self.g[self.num_data_types:]  # Unlike S, this is a flattened list.
        # If orthogonality is being forced, multiply with appropriate mask.
        if orthogonality_constraint:
            self.g_G = [g_Gi * Mi for g_Gi, Mi in zip(self.g_G, self.M)]

        # Start tensorflow session.
        self.sess = tf.Session()

    def generate_input_dict(self, G, S, M):
        """Generates a dictionary mapping numpy arrays given
        as arguments to placeholders defined in init."""
        input_dict = dict()
        for Gpi, Gi in zip(self.Gp, G):
            input_dict[Gpi] = Gi
        for Sp_list, S_list in zip(self.Sp, S):
            for Spi, Si in zip(Sp_list, S_list):
                input_dict[Spi] = Si
        if M is not None:
            for Mpi, Mi in zip(self.Mp, M):
                input_dict[Mpi] = Mi
        return input_dict

    def calculate_cost(self, G, S, M=None):
        """Calculates the cost when given the numpy matrices G_i,S_ij."""


        if self.orthogonality_constraint:
            self.sess.run(self.S_assign_list, feed_dict=self.generate_input_dict(G, S, M))
            self.sess.run(self.M_assign_list, feed_dict=self.generate_input_dict(G, S, M))
            self.sess.run(self.G_assign_list, feed_dict=self.generate_input_dict(G, S, M))

        else:
            self.sess.run(self.new_cost, feed_dict=self.generate_input_dict(G, S, M))
        return self.sess.run(self.cost)

    def construct_starting_point(self, ks, scale):
        """Constructs a random starting point based on the specifics of the problem
        and can be used by optimization algorithms to generate initial populations."""
        G = []
        S = []
        r = 0
        for i in range(self.num_data_types):
            G.append(scale * np.random.rand(self.num_instances[i], ks[i]))
            for j in range(i, self.num_data_types):
                S.append([])
                for _ in range(self.num_matrices[r]):
                    S[r].append(scale * np.random.rand(ks[i], ks[j]))
                r += 1
        return G, S

    def get_starting_point(self, G, S, ks, scale):
        """Prepares the initial point if necessary and calcualtes ks if necessary."""
        if G is None or S is None:
            if ks is None:
                raise RuntimeError("Both {G,S} and ks are not set in get_starting_point!")
            else:
                G, S = self.construct_starting_point(ks, scale)
        else:
            if ks is None:
                ks = map(lambda Gi: Gi.shape[1], G)
        return G, S, ks

    def close(self):
        """Closes the tensorflow session and deletes the graph."""
        self.sess.close()
        tf.reset_default_graph()


class GradientDescent(CostAndGradients):
    """Class that is able to perform ordinary gradient descent."""

    def __init__(self, data, missing_values_mask=None, orthogonality_constraint=False, use_projection=False,
                 regularizations=[], lr=10, RSE_flag=True, dtype=tf.float64):
        """Constructs tensorflow computational graph for ordinary gradient descent."""

        # Inherit CostAndGradients' variables and methods.
        super(GradientDescent, self).__init__(data, missing_values_mask, orthogonality_constraint, use_projection,
                                              regularizations, RSE_flag, dtype)

        # Step of gradient descent.
        new_GS = []
        for Gi, g_Gi in zip(self.G, self.g_G):
            new_GS.append(tf.assign(Gi, Gi - lr * g_Gi))
        for Si, g_Si in zip(self.Sf, self.g_S):
            new_GS.append(tf.assign(Si, Si - lr * g_Si))
        # Use projection to force non-negativity if use_projection flag was chosen.
        force_nneg = []
        if use_projection:
            with tf.control_dependencies(new_GS):
                for Gi in self.G:
                    force_nneg.append(tf.assign(Gi, tf.nn.relu(Gi)))
                for Si in self.Sf:
                    force_nneg.append(tf.assign(Si, tf.nn.relu(Si)))
        self.new_step = tf.group(*(new_GS + force_nneg))

    def optimize(self, steps, G=None, S=None, M=None, ks=None, scale=0.01):
        """Performs gradient descent from the initial point for
        a selected number of steps using ordinary gradient descent."""
        # First calculate the initial cost and prepare everything for the new descent.
        G, S, ks = self.get_starting_point(G, S, ks, scale)
        c_best = self.calculate_cost(G, S, M)
        c_progress = np.zeros(steps + 1)
        c_progress[0] = c_best

        cr_progress = np.zeros(steps + 1)
        cr_progress[0] = self.sess.run(self.regularized_cost)

        r_progress = np.zeros((len(self.regularizations), steps+1))
        r_progress[:, 0] = np.array(self.sess.run([reg.regularization_cost for reg in self.regularizations]))

        print('0, ' + str(self.sess.run(self.regularized_cost)))
        # Perform gradient descent for desired number of steps.
        for i in range(1, steps + 1):
            # Perform a new step.
            self.sess.run(self.new_step)
            # Calculate the cost at the new point and save it.
            c = self.sess.run(self.cost)
            cr = self.sess.run(self.regularized_cost)
            print(str(i) + ', ' + str(c) + ', ' + str(cr))
            c_progress[i] = c
            cr_progress[i] = cr
            r_progress[:, i] = np.array(self.sess.run([reg.regularization_cost for reg in self.regularizations]))
            # Save best point seen so far.
            if c < c_best:
                c_best = c
                G = self.sess.run(self.G_nneg)
                S = map(lambda S_list: self.sess.run(S_list), self.S_nneg)
        # Norm the columns of G and adjust S accordingly.
        #G, S = norm(G, S)
        return c_best, c_progress, cr_progress, r_progress, G, S


class GradientDescentAdam(CostAndGradients):
    """Class that is able to perform gradient descent using the Adam algorithm."""

    def __init__(self, data, missing_values_mask=None, orthogonality_constraint=False, use_projection=False,
                 regularizations=[], lr=0.001, beta_1=0.9, beta_2=0.99, eps=1e-8, RSE_flag=True, dtype=tf.float64):
        """Constructs tensorflow computational graph for gradient descent using Adam algorithm."""

        # Inherit CostAndGradients' variables and methods.
        super(GradientDescentAdam, self).__init__(data, missing_values_mask, orthogonality_constraint, use_projection,
                                                  regularizations, RSE_flag, dtype)

        # Tensor definitions needed for the Adam algorithm.
        self.Sm = map(
            lambda Sm_list: map(lambda Smi: tf.Variable(initial_value=[], validate_shape=False, name='Sm', dtype=dtype),
                                Sm_list), self.fnR)
        self.Sv = map(
            lambda Sv_list: map(lambda Svi: tf.Variable(initial_value=[], validate_shape=False, name='Sv', dtype=dtype),
                                Sv_list), self.fnR)
        self.Gm = [tf.Variable(initial_value=[], name='Gm', validate_shape=False, dtype=dtype) for _ in
                   range(self.num_data_types)]
        self.Gv = [tf.Variable(initial_value=[], name='Gv', validate_shape=False, dtype=dtype) for _ in
                   range(self.num_data_types)]
        self.t = tf.Variable(initial_value=0.0, name='t', dtype=dtype)

        # Initialization of tensors before Adam algorithm starts the gradient descent.
        new_descent_list = [tf.assign(self.t, 0.0)]
        new_descent_list.extend(
            [tf.assign(Smi, tf.zeros_like(Si), validate_shape=False) for Sm_list, S_list in zip(self.Sm, self.S) for
             Smi, Si in zip(Sm_list, S_list)])
        new_descent_list.extend(
            [tf.assign(Svi, tf.zeros_like(Si), validate_shape=False) for Sv_list, S_list in zip(self.Sv, self.S) for
             Svi, Si in zip(Sv_list, S_list)])
        new_descent_list.extend(
            [tf.assign(Gmi, tf.zeros_like(Gi), validate_shape=False) for Gmi, Gi in zip(self.Gm, self.G)])
        new_descent_list.extend(
            [tf.assign(Gvi, tf.zeros_like(Gi), validate_shape=False) for Gvi, Gi in zip(self.Gv, self.G)])
        self.new_descent = tf.group(*new_descent_list)

        # To make the following block of code simpler we define flattened versions of Sm and Sv.
        self.Smf = [Smi for Sm_list in self.Sm for Smi in Sm_list]
        self.Svf = [Svi for Sv_list in self.Sv for Svi in Sv_list]

        # Step of gradient descent using the Adam algorithm.
        newt = [tf.assign(self.t, self.t + 1.)]
        with tf.control_dependencies(newt):
            # Update step size with respect to the time variable.
            self.alpha_t = lr * tf.sqrt(1. - beta_2 ** self.t) / (1. - beta_1 ** self.t)
            # Update momentum and second moments.
            new_mv = []
            for Gmi, Gvi, g_Gi in zip(self.Gm, self.Gv, self.g_G):
                new_mv.append(tf.assign(Gmi, beta_1 * Gmi + (1. - beta_1) * g_Gi))
                new_mv.append(tf.assign(Gvi, beta_2 * Gvi + (1. - beta_2) * tf.square(g_Gi)))
            for Smi, Svi, g_Si in zip(self.Smf, self.Svf, self.g_S):
                new_mv.append(tf.assign(Smi, beta_1 * Smi + (1. - beta_1) * g_Si))
                new_mv.append(tf.assign(Svi, beta_2 * Svi + (1. - beta_2) * tf.square(g_Si)))
            with tf.control_dependencies(new_mv):
                # Update the actual variables.
                new_GS = []
                for Gi, Gmi, Gvi in zip(self.G, self.Gm, self.Gv):
                    new_GS.append(tf.assign(Gi, Gi - self.alpha_t * Gmi / (tf.sqrt(Gvi) + eps)))
                for Si, Smi, Svi in zip(self.Sf, self.Smf, self.Svf):
                    new_GS.append(tf.assign(Si, Si - self.alpha_t * Smi / (tf.sqrt(Svi) + eps)))
                # Use projection to force non-negativity if use_projection flag was chosen.
                force_nneg = []
                if use_projection:
                    with tf.control_dependencies(new_GS):
                        for Gi in self.G:
                            force_nneg.append(tf.assign(Gi, tf.nn.relu(Gi)))
                        for Si in self.Sf:
                            force_nneg.append(tf.assign(Si, tf.nn.relu(Si)))
        self.new_step = tf.group(*(newt + new_mv + new_GS + force_nneg))

    def optimize(self, steps, G=None, S=None, M=None, ks=None, scale=0.01):
        """Performs gradient descent from the initial point for
        a selected number of steps using Adam algorithm"""
        # First calculate the initial cost and prepare everything for the new descent.
        G, S, ks = self.get_starting_point(G, S, ks, scale)
        c_best = self.calculate_cost(G, S, M)
        self.sess.run(self.new_descent)
        c_progress = np.zeros(steps + 1)
        c_progress[0] = c_best

        cr_progress = np.zeros(steps + 1)
        cr_progress[0] = self.sess.run(self.regularized_cost)

        r_progress = np.zeros((len(self.regularizations), steps+1))
        r_progress[:, 0] = np.array(self.sess.run([reg.regularization_cost for reg in self.regularizations]))

        print('0, ' + str(self.sess.run(self.regularized_cost)))
        # Perform gradient descent for desired number of steps.
        for i in range(1, steps + 1):
            # Perform a new step.
            self.sess.run(self.new_step)
            # Calculate the cost at the new point and save it.
            c = self.sess.run(self.cost)
            cr = self.sess.run(self.regularized_cost)

            print(str(i) + ', ' + str(c) + ', ' + str(cr))
            c_progress[i] = c
            cr_progress[i] = cr
            r_progress[:, i] = np.array(self.sess.run([reg.regularization_cost for reg in self.regularizations]))
            # Save best point seen so far.
            if c < c_best:
                c_best = c
                G = self.sess.run(self.G_nneg)
                S = list(map(lambda S_list: self.sess.run(S_list), self.S_nneg))
        # Norm the columns of G and adjust S accordingly.
        #G, S = norm(G, S)
        return c_best, c_progress, cr_progress, r_progress, G, S

def arg_min_poly(poly):
    """Returns the position of the global minimum of a
    polynomial with constant term equal to zero."""
    # Differentiate the polynomial and find stationary points.
    diff_poly = poly * np.arange(poly.shape[0], 0, -1)
    diff_poly_rts = np.roots(diff_poly)
    # Put poly to appropriate numpy form, i.e. add constant term.
    poly_eval = np.append(poly, 0)
    # Find stationary point that has the lowest value.
    best = np.inf
    best_i = -1
    for rt_i, rt in enumerate(diff_poly_rts):
        if np.isreal(rt):
            new = np.polyval(poly_eval, rt)
            if new < best:
                best = new
                best_i = rt_i
    if best_i < 0:
        raise RuntimeError("No good roots were found in arg_min_poly.")
    return diff_poly_rts[best_i].real


class GradientDescentPoly(CostAndGradients):
    """Constructs tensorflow computational graph for gradient descent where optimal step
    size is calculated by finding a minimum of a polynomial function of one variable."""

    def __init__(self, data, missing_values_mask=None, orthogonality_constraint=False, regularizations=[], order=3,
                 RSE_flag=True, dtype=tf.float64):

        # Inherit CostAndGradients' variables and methods.
        super(GradientDescentPoly, self).__init__(data, missing_values_mask, orthogonality_constraint, True,
                                                  regularizations, RSE_flag, dtype)

        # Prepare tensors that are needed for this type of gradient descent.
        self.order = order
        self.step_size = tf.placeholder(dtype)

        # Definition of an adjusted gradients whose components that will end up in unfeasible region after gradient descent step are set to zero.
        self.g_G_adjusted = []
        self.g_S_adjusted = []
        self.G_zero = [tf.zeros_like(Gi) for Gi in self.G]
        self.S_zero = [tf.zeros_like(Si) for Si in self.Sf]
        for Gi, g_Gi, G_zeroi in zip(self.G, self.g_G, self.G_zero):
            self.g_G_adjusted.append(tf.where(tf.logical_and(tf.equal(Gi, 0.0), tf.less(g_Gi, 0.0)), G_zeroi, g_Gi))
        for Si, g_Si, S_zeroi in zip(self.Sf, self.g_S, self.S_zero):
            self.g_S_adjusted.append(tf.where(tf.logical_and(tf.equal(Si, 0.0), tf.less(g_Si, 0.0)), S_zeroi, g_Si))

        # Calculation of polynomial that is used to find an optimal step size for gradient descent.
        if self.use_W:
            W = missing_values_mask
        else:
            W = map(lambda W_list: map(lambda W: None, W_list), self.fnR)  # This is easier for looping.
        r = 0  # Index that goes through sublists of S, dR and GS.
        s = 0  # Index that goes through (a flattened) list dS.
        for i in range(self.num_data_types):
            for j in range(i, self.num_data_types):
                for Si, dRi, GSi, fnRi, Wi in zip(self.S[r], self.dR[r], self.GS[r], self.fnR[r], W[r]):
                    poly_current = self.poly_calculation(self.G[i], Si, self.G[j], self.g_G_adjusted[i],
                                                         self.g_S_adjusted[s], self.g_G_adjusted[j], dRi, GSi,
                                                         Wi) / fnRi
                    if s:
                        self.poly += poly_current
                    else:
                        self.poly = poly_current
                    s += 1
                r += 1
        self.poly /= len(self.Sf)
        # Add the contribution of regularizations to the polynomial.
        for regularization in regularizations:
            self.poly += regularization.construct_poly(self.G, self.S, self.Sf, self.GS, self.g_G, self.g_S, self.order)

        # Step of gradient descent using the step size from a placeholder followed by a projection to the feasible region.
        new_step_list = []
        for Gi, g_Gi in zip(self.G, self.g_G_adjusted):
            new_step_list.append(tf.assign(Gi, tf.nn.relu(Gi + self.step_size * g_Gi)))
        for Si, g_Si in zip(self.Sf, self.g_S_adjusted):
            new_step_list.append(tf.assign(Si, tf.nn.relu(Si + self.step_size * g_Si)))
        self.new_step = tf.group(*new_step_list)

    def poly_calculation(self, G1, S, G2, dG1, dS, dG2, dR, GS, W):
        """Returns a polynomial that tells how the cost varies in direction of the
        gradient with respect to the step size for specific matrix instance."""
        dGS = tf.matmul(dG1, S)
        GdS = tf.matmul(G1, dS)
        dGdS = tf.matmul(dG1, dS)
        G2T = tf.transpose(G2)
        dG2T = tf.transpose(dG2)
        # Coeficients of polynomial of the third degree - before Frobenius.
        A = dR
        B = tf.matmul(dGS, G2T) + tf.matmul(GdS, G2T) + tf.matmul(GS, dG2T)
        C = tf.matmul(dGS, dG2T) + tf.matmul(GdS, dG2T) + tf.matmul(dGdS, G2T)
        D = tf.matmul(dGdS, dG2T)
        # Missing value mask if we have one.
        if W is not None:
            B *= W
            C *= W
            D *= W
        # Coeficients of polynomial of the sixth degree.
        u1 = 2.0 * tf.reduce_sum(A * B)
        u2 = tf.reduce_sum(tf.square(B))
        if self.order == 1:
            u = tf.stack([u2, u1])
        else:
            u2 += 2.0 * tf.reduce_sum(A * C)
            u3 = tf.reduce_sum(B * C)
            u4 = tf.reduce_sum(tf.square(C))
            if self.order == 2:
                u = tf.stack([u4, u3, u2, u1])
            else:
                u3 += 2.0 * (tf.reduce_sum(A * D))
                u4 += 2.0 * tf.reduce_sum(B * D)
                u5 = 2.0 * tf.reduce_sum(C * D)
                u6 = tf.reduce_sum(tf.square(D))
                if self.order == 3:
                    u = tf.stack([u6, u5, u4, u3, u2, u1])
                else:
                    raise RuntimeError('Polynomial order too large in poly_calculation!')
        # The polynomial is returned.
        return u

    def optimize(self, steps, restart=True, G=None, S=None, M=None, ks=None, scale=0.01):
        """Performs gradient descent for desired number of steps using ideal
        step method and in case restart flag is chosen it restarts the descent
        if the cost degrades above the cost of trivial solution."""
        # Calculate the cost for initial point.
        G, S, ks = self.get_starting_point(G, S, ks, scale)
        c_progress = np.zeros(steps + 1)
        c_progress[0] = self.calculate_cost(G, S, M)

        cr_progress = np.zeros(steps + 1)
        cr_progress[0] = self.sess.run(self.regularized_cost)

        r_progress = np.zeros((len(self.regularizations), steps+1))
        r_progress[:, 0] = np.array(self.sess.run([reg.regularization_cost for reg in self.regularizations]))

        print('0, ' + str(c_progress[0]))
        # Perform gradient descent for specified number of steps.
        for i in range(1, steps + 1):
            poly = self.sess.run(self.poly)
            step_size = arg_min_poly(poly)
            self.sess.run(self.new_step, feed_dict={self.step_size: step_size})
            #c_progress[i] = self.sess.run(self.cost)
            #print(str(i) + ', ' + str(c_progress[i]))


            # Calculate the cost at the new point and save it.
            c = self.sess.run(self.cost)
            cr = self.sess.run(self.regularized_cost)

            print(str(i) + ', ' + str(c) + ', ' + str(cr))
            c_progress[i] = c
            cr_progress[i] = cr
            r_progress[:, i] = np.array(self.sess.run([reg.regularization_cost for reg in self.regularizations]))

            # If the step was bad, restart.
            if restart and c_progress[i] > 1.0:
                print('Restarting the descent.')
                return self.optimize(steps, True, None, None, None, ks, scale)
        # Extract G,S and norm the columns of G and adjust S accordingly.
        G = self.sess.run(self.G)
        S = map(lambda S_list: self.sess.run(S_list), self.S)
        #G, S = norm(G, S)
        return c_progress[0], c_progress, cr_progress, r_progress, G, S
        #return c_progress[0], c_progress, G, S

