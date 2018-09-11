import random
from functools import partial

from crossovers import roll, unroll, RandomColumns
from local_search import GradientDescentAdam
from matrix_utilities import get_NMTF_problem
import numpy as np
import matplotlib.pyplot as plt
from platypus import NSGAII, Problem, Real
from platypus.operators import TournamentSelector, RandomGenerator
from platypus.operators import SBX, UNDX, SPX, DifferentialEvolution

init_pop_size = 25
number_of_gd_steps = 150

generated_pop_size=300

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

    return problem, population, meta_g, generator


problem, population, meta, generator = generate_initial_population(init_pop_size, number_of_gd_steps)

obj_base = [p.objectives[0] for p in population]

#pcx = PCX(5, 1)
#pcx_pop = []
#while len(pcx_pop) < init_pop_size:
#    random.shuffle(population)
#    pcx_pop += pcx.evolve(population)
#obj_pcx = [p.objectives[0] for p in pcx_pop]

sbx = SBX()
sbx_pop = []
while len(sbx_pop) < generated_pop_size:
    print("SBX", len(sbx_pop))
    random.shuffle(population)
    p = sbx.evolve(population)[0]
    problem.evaluate(p)
    sbx_pop.append(p)
    #sbx_pop += sbx.evolve(population)
#map(lambda x:problem.evaluate(x), sbx_pop)
obj_sbx = [p.objectives[0] for p in sbx_pop]

rc = RandomColumns(meta, generator, problem)
rc_pop = []
while len(rc_pop) < generated_pop_size:
    print("RC", len(rc_pop))
    random.shuffle(population)
    p = rc.evolve(population)[0]
    problem.evaluate(p)
    rc_pop.append(p)
obj_rc = [p.objectives[0] for p in rc_pop]

spx = SPX()
spx_pop = []
while len(spx_pop) < generated_pop_size:
    print("SPX", len(spx_pop))
    random.shuffle(population)
    p = spx.evolve(population)[0]
    problem.evaluate(p)
    spx_pop.append(p)
#map(lambda x:problem.evaluate(x), spx_pop)
obj_spx = [p.objectives[0] for p in spx_pop]

de = DifferentialEvolution()
de_pop = []
while len(de_pop) < generated_pop_size:
    print("DE", len(de_pop))
    random.shuffle(population)
    #de_pop += de.evolve(population)
    p = de.evolve(population)[0]
    problem.evaluate(p)
    de_pop.append(p)
#map(lambda x:problem.evaluate(x), de_pop)
obj_de = [p.objectives[0] for p in de_pop]

data = np.vstack((obj_sbx, obj_spx, obj_de))

plt.boxplot([obj_base, obj_sbx, obj_spx, obj_de, obj_rc])
#plt.boxplot(data.T, positions=[2,3,4])

plt.xticks([1, 2, 3, 4, 5], ['Base', 'SBX', 'SPX', 'DE', 'RC'])
plt.show()

