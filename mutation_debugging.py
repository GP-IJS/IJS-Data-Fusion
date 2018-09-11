import os
from os import listdir

import matplotlib.pyplot as plt
import numpy as np

from evolutionary_operators import add_columns
from local_search import GradientDescentOptimizerAdam, AdamLocalSearchProgress
from matrix_utilities import read_R
from problems import TriFactorization, ConstantKMatrixGenerator


def init_solution(k):
    return generator.generate(problem, k)


def save_matrix(M, file_name):
    plt.clf()
    plt.imshow(M, aspect=0.4)
    plt.savefig(file_name)


def save_matrix_with_text(M, file_name):
    plt.clf()
    plt.imshow(M)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j, i, "{0:.2f}".format(M[i, j]),
                     ha="center", va="center", color="w")
    plt.savefig(file_name)


def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)


def resolve_correlation_conflict(cor):
    # Resolve conflict where one column contains multiple max elements
    # Elements are deleted until there is no more conflicts left and order is returned
    max_cor_column = np.argmax(cor, axis=1)
    if len(np.unique(max_cor_column)) == len(max_cor_column):
        return max_cor_column
    for i, x in enumerate(max_cor_column):
        for j, y in enumerate(max_cor_column):
            if i != j and x == y:
                if cor[i, x] < cor[j, y]:
                    cor[i, x] = 0
                else:
                    cor[j, y] = 0
                return resolve_correlation_conflict(cor)


def shuffle_G(G_original, G_to_shuffle):
    cor = np.dot(G_original.T, G_to_shuffle)
    max_cor_column = resolve_correlation_conflict(cor.copy())
    return cor, G_to_shuffle[:, max_cor_column], max_cor_column


def shuffle_S(S, order):
    return S[:, order][order]


def plot_error(results, file_name):
    plt.clf()
    for k1, a_p in results.items():
        plt.plot(a_p, label=k1)
    plt.legend()
    plt.savefig(file_name)


def save_G_S(G, S, file):
    save_matrix(G, file)
    for ig, s in enumerate(S):
        save_matrix(s, file + "_g=" + str(ig))

kk = 7  # This is the dimension of recombined individual.
# np.random.seed(52)
# Specifications of R matrices.
n = 100
k = 20
d = 4
# Maximum number of steps for gradient descent
# in first (before mutation) and second part (after mutation) of optimization
gd_steps_part1 = 6000
gd_steps_part_2 = 6000
# The scale of components when column addition is performed.
almost_zero = 1e-7
# k's to visualize
ks = [3,5]
# Directory for generated images
dir_for_pictures = "viz"

# Find directory for that specific n, k, d
dir = '../data'
directories = listdir(dir)
n_string = 'n=' + str(n) + '_'
directories = [s for s in directories if n_string in s]
k_string = 'k=' + str(k) + '_'
directories = [s for s in directories if k_string in s]
d_string = 'density=' + str(d) + '.'
directories = [s for s in directories if d_string in s]

directory = dir + '/' + directories[0] + '/'
# Find the names of files that store the values of R matrices.
matrices = listdir(directory)
matrices = [s for s in matrices if 'R' in s]
R = read_R(directory, matrices, n)

gd = GradientDescentOptimizerAdam(R)
local_search = AdamLocalSearchProgress(gd, steps=gd_steps_part1)
local_search1 = AdamLocalSearchProgress(gd, steps=gd_steps_part_2)

generator = ConstantKMatrixGenerator(len(matrices), n, scale=0.1)
problem = TriFactorization(gd)

# Saved errors for diffrerent k's
gd_step_1_results = dict()
gd_step_2_results = dict()

create_folder(dir_for_pictures)

# def optimization for random init
a = init_solution(kk)
a, a_p = local_search1.mutate(a, convergence_steps=gd_steps_part_2)
G_random = a.variables[0]
S_random = a.variables[1]
name = dir_for_pictures + "/random_k=" + str(kk) + "_er=" + str(a_p[-1]).replace(".", ",")
save_matrix(G_random, name)

correlation_matrix, _, _ = shuffle_G(G_random, G_random)
save_matrix_with_text(correlation_matrix, dir_for_pictures + "/cor_GrGr_k=" + str(kk))

for ig, s in enumerate(S_random):
    save_matrix(s, name + "_g=" + str(ig))

gd_step_2_results["random"] = a_p
gd_step_1_results["random"] = a_p

for k1 in ks:
    print("k="+str(k1))

    # Local search before adding rows
    a = init_solution(k1)
    a, a_p = local_search.mutate(a)
    gd_step_1_results[k1] = a_p # Save quality of the solution
    G = a.variables[0]
    S = a.variables[1]

    sub_dir = dir_for_pictures + "/k=" + str(k1)
    create_folder(sub_dir)

    name = sub_dir + "/init_k=" + str(k1)
    save_G_S(G, S, name)

    # Padding matriec with additional columns
    d = add_columns(a, kk - k1, almost_zero)

    name = sub_dir + "/padded_k=" + str(k1)
    G = d.variables[0]
    S = d.variables[1]
    save_G_S(G, S, name)

    # Second step of the local search
    a, a_p = local_search1.mutate(d, convergence_steps=gd_steps_part_2)
    gd_step_2_results[k1] = a_p # Store results
    G = a.variables[0]
    S = a.variables[1]

    name = sub_dir + "/padded_gd_k=" + str(k1)
    save_G_S(G, S, name)
    #save_matrix(G, name)

    correlation_matrix, newG, swaps = shuffle_G(G_random, G)
    save_matrix_with_text(correlation_matrix, sub_dir + "/correlation_randomG_G_k=" + str(k1))

    correlation_matrix, _, _ = shuffle_G(G, G)
    save_matrix_with_text(correlation_matrix, sub_dir + "/correlation_with_itself_G_k=" + str(k1))

    name = sub_dir + "/shuffled_gd_k=" + str(k1)
    save_G_S(newG, [shuffle_S(s, swaps) for s in S], name)
    #save_matrix(newG, name)
    #for ig, s in enumerate(S):
    #    save_matrix(shuffle_S(s.copy(), swaps), name + "_g=" + str(ig))



plot_error(gd_step_1_results, dir_for_pictures + "/GD_1")
plot_error(gd_step_2_results, dir_for_pictures + "/GD_2")
