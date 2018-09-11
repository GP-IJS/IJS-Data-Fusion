from abc import abstractmethod, ABC
import tensorflow as tf
import numpy as np
from matrix_utilities import norm


class AbstractRegularization(ABC):
    """Abstract class used as a blueprint for regularization methods."""

    def __init__(self, regularization_constant):
        self.regularization_constant = regularization_constant
        self.regularization_cost = tf.constant(0.0, dtype=tf.float64)

    @abstractmethod
    def add_regularization(self, G, S, Sf, GS):
        """Adds regularization penalty to the current cost."""
        pass

    @abstractmethod
    def construct_poly(self, G, S, Sf, GS, dG, dS, order):
        """Adds regularization penalty to the current cost in case of polynomial step optimization"""
        pass


class OrthonormalColumnsRegularization(AbstractRegularization):
    """A regularization for sum((G^T.G - I)^2)."""

    def __init__(self, regular_const):
        super(OrthonormalColumnsRegularization, self).__init__(regular_const)
        self._scalar_products = []
        self._scalar_products_I = []

    def add_regularization(self, G, S, Sf, GS):
        for i, Gi in enumerate(G):
            self._scalar_products.append(tf.matmul(tf.transpose(Gi), Gi))
            self._scalar_products_I.append(self._scalar_products[i] - tf.eye(tf.shape(Gi)[1], dtype=tf.float64))
            self.regularization_cost += tf.reduce_sum(tf.square(self._scalar_products_I[i]))
        self.regularization_cost *= self.regularization_constant
        return self.regularization_cost

    def construct_poly_for_one(self, i, Gi, dGi, order):
        A = self._scalar_products_I[i]
        mixed = tf.matmul(tf.transpose(Gi), dGi)
        B = mixed + tf.transpose(mixed)
        C = tf.matmul(tf.transpose(dGi), dGi)
        u1 = 2.0 * tf.reduce_sum(A * B)
        u2 = (2.0 * tf.reduce_sum(A * C) + tf.reduce_sum(tf.square(B)))
        if order > 1:
            u3 = 2.0 * tf.reduce_sum(B * C)
            u4 = tf.reduce_sum(tf.square(C))
        if order == 1:
            return tf.stack([u2, u1])
        else:
            return tf.stack([u4, u3, u2, u1])

    def construct_poly(self, G, S, Sf, GS, dG, dS, order):
        for i, (Gi, dGi) in enumerate(zip(G, dG)):
            if i:
                poly += self.construct_poly_for_one(i, Gi, dGi, order)
            else:
                poly = self.construct_poly_for_one(i, Gi, dGi, order)
        paddings = tf.constant([[2 * order - 4, 0]])
        return self.regularization_constant * tf.pad(poly, paddings)


class S2NormRegularization(AbstractRegularization):
    """A regularization for sum(S^2)."""

    def __init__(self, regular_const):
        super(S2NormRegularization, self).__init__(regular_const)

    def add_regularization(self, G, S, Sf, GS):
        for Si in Sf:
            self.regularization_cost += tf.reduce_mean(tf.square(Si))
        self.regularization_cost *= self.regularization_constant
        return self.regularization_cost

    def construct_poly(self, G, S, Sf, GS, dG, dS, order):
        for i, (Si, dSi) in enumerate(zip(Sf, dS)):
            if i:
                u1 += tf.reduce_sum(Si * dSi)
                u2 += tf.reduce_sum(dSi * dSi)
            else:
                u1 = tf.reduce_sum(Si * dSi)
                u2 = tf.reduce_sum(dSi * dSi)
        poly = tf.stack([u2, 2.0 * u1])
        paddings = tf.constant([[2 * order - 2, 0]])
        return self.regularization_constant * tf.pad(poly, paddings)

class SAbsNormRegularization(AbstractRegularization):
    """A regularization for sum(abs(S))."""

    def __init__(self, regular_const):
        super(SAbsNormRegularization, self).__init__(regular_const)

    def add_regularization(self, G, S, Sf, GS):
        for Si in Sf:
            self.regularization_cost += tf.reduce_mean(tf.abs(Si))
        self.regularization_cost *= self.regularization_constant
        return self.regularization_cost

    def construct_poly(self, G, S, Sf, GS, dG, dS, order):
        raise NotImplementedError


class PearsonCorrelationRegularization(AbstractRegularization):
    """Pearson regularization between all columns of matrices G.
    Only positive correlations are summed to get a final constant"""

    def __init__(self, regular_const):
        super(PearsonCorrelationRegularization, self).__init__(regular_const)

    def add_regularization(self, G, S, Sf, GS):
        for i, Gi in enumerate(G):
            #X = tf.Variable(Gi)
            X_norm = tf.subtract(Gi, tf.reduce_mean(Gi, axis=0))
            cov = tf.matmul(tf.transpose(X_norm), X_norm)
            std = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(X_norm), 0)), 1)
            std_pairs = tf.matmul(std, tf.transpose(std))
            pearson_correlation = tf.divide(cov, std_pairs)

            #self._pearson_matrices.append(pearson_correlation)
            dim = tf.cast(tf.shape(pearson_correlation)[0], tf.float64)#TODO change float

            # Ignore negative values and sum
            self.regularization_cost += (tf.reduce_sum(tf.nn.relu(pearson_correlation))-dim)/(dim*dim-dim)
        self.regularization_cost *= self.regularization_constant
        return self.regularization_cost

    def construct_poly(self, G, S, Sf, GS, dG, dS, order):
        raise NotImplementedError