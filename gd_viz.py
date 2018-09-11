import random
from functools import partial

from crossovers import roll, unroll
from local_search import GradientDescentAdam
from matrix_utilities import get_NMTF_problem
import numpy as np
import matplotlib.pyplot as plt
from platypus import NSGAII, Problem, Real
from platypus.operators import TournamentSelector, RandomGenerator
from platypus.operators import SBX, UNDX, SPX, DifferentialEvolution

n = 500
ks = [20]
d = 6
R = get_NMTF_problem(n, ks[0], d)

alpha = 0.01


def fit(cost_calc, vector_unrolling_meta, x):
    G, S = unroll(np.array(x), vector_unrolling_meta)
    return [cost_calc.calculate_cost(G, S)]


def generate_initial_population(pop_size=10, number_of_gd_steps=50):
    p1 = GradientDescentAdam(R, lr=alpha)
    initial_pop = []
    meta_g = None
    vec_len = None

    for x in range(pop_size):
        _, _, _, _, G, S = p1.optimize(number_of_gd_steps, ks=ks)
        v, meta = roll(G, S)
        meta_g = meta
        vec_len = len(v)
        initial_pop.append(v.tolist())

    problem = Problem(vec_len, 1)
    problem.types[:] = Real(0, 1000)
    problem.function = partial(fit, p1, meta_g)

    generator = RandomGenerator()
    population = []

    for i in range(pop_size):
        p = generator.generate(problem)
        p.variables = initial_pop[i]
        problem.evaluate(p)
        population.append(p)

    return problem, population, meta_g, p1


problem, population, meta, optimizer = generate_initial_population(2, 250)
G1, S1 = unroll(np.array(population[0].variables), meta)
G2, S2 = unroll(np.array(population[1].variables), meta)

print("G1,S1", optimizer.calculate_cost(G1,S1))
print("G2,S2", optimizer.calculate_cost(G2,S2))

print("G1,S2", optimizer.calculate_cost(G1,S2))
print("G2,S1", optimizer.calculate_cost(G2,S1))

G_mean, S_mean = unroll(0.5*np.array(population[0].variables) + 0.5*np.array(population[1].variables), meta)

print("G_mean,S_mean", optimizer.calculate_cost(G_mean,S_mean))



plt.imshow(G1[0][0:50,:])
plt.savefig('G1.pdf')
plt.clf()
plt.imshow(G2[0][0:50,:])
plt.savefig('G2.pdf')
plt.clf()
plt.imshow(G_mean[0][0:50,:])
plt.savefig('G_mean.pdf')
plt.clf()

v = np.linspace(0, 1, 100)
errors = []
for x in v:
    G_mean, S_mean = unroll(x * np.array(population[0].variables) + (1-x) * np.array(population[1].variables), meta)
    errors.append(optimizer.calculate_cost(G_mean, S_mean))

plt.plot(v, errors)
plt.savefig('lin_comb.pdf')
plt.clf()

def calc_dif(G1, G2):
    v = np.sqrt(np.sum(np.square(G1-G2), axis=0))
    print(v)

calc_dif(G1[0], G2[0])





