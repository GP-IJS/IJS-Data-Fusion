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

init_pop_size = 2
number_of_gd_steps = 150

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

    return problem, population


problem, population = generate_initial_population(init_pop_size, number_of_gd_steps)

obj_base = [p.objectives[0] for p in population]
print(obj_base)


spx = SPX()
for x in range(10):
    random.shuffle(population)
    p = spx.evolve(population)[0]
    problem.evaluate(p)
    print(p.objectives)


