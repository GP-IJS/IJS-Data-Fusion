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


def generate_initial_population(number_of_gd_steps=50):
    p1 = GradientDescentAdam(R, lr=alpha)
    initial_pop = []
    scale = 0.01

    noise_scale=0.0000001

    Gv, Sv = p1.construct_starting_point(ks, scale=scale)

    _, _, _, _, G, S = p1.optimize(G=Gv.copy(), S=Sv.copy(), steps=number_of_gd_steps, ks=ks)
    v, meta = roll(G, S)
    initial_pop.append(v.tolist())

    GvNoise = []
    for x in Gv:
        GvNoise.append(x+np.random.randn(*x.shape)*noise_scale)

    SvNoise = []
    for x in Sv:
        n=[]
        for y in x:
            n.append(y+np.random.randn(*y.shape)*noise_scale)
        SvNoise.append(n)

    _, _, _, _, G, S = p1.optimize(G=GvNoise.copy(), S=SvNoise.copy(), steps=number_of_gd_steps, ks=ks)
    v, meta = roll(G, S)
    initial_pop.append(v.tolist())

    np.random.rand()
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

    for i in range(3):
        p = generator.generate(problem)
        p.variables = initial_pop[i]
        problem.evaluate(p)
        population.append(p)

    return problem, population, meta_g, p1


problem, population, meta, optimizer = generate_initial_population(250)
G1, S1 = unroll(np.array(population[0].variables), meta)
G2, S2 = unroll(np.array(population[1].variables), meta)
G3, S3 = unroll(np.array(population[2].variables), meta)

plt.imshow(G1[0][0:100,:])
plt.show()
plt.imshow(G2[0][0:100,:])
plt.show()
plt.imshow(G3[0][0:100,:])
plt.show()

def calc_dif(G1, G2):
    v = np.sqrt(np.sum(np.square(G1-G2), axis=0))
    return v

print(calc_dif(G1[0], G2[0]))
print(calc_dif(G1[0], G3[0]))







