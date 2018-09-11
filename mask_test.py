from matrix_utilities import closest_orthogonal_masks
from itertools import product
import numpy as np

# Set dimensions of the test matrix.
n = 10
m = 3

# Number of closest masks.
number = 20

# Generate a random test matrix.
M = np.random.rand(n, m)
M = 1.0 / M
print(M)

# Find actual closest orthogonal matrices.
lst = [0.0 for _ in range(m ** n)]
r = 0
for idxs in product(range(m), repeat=n):
    new_M = np.copy(M)
    new_M[range(n), idxs] = 0.0
    d = np.sum(np.square(new_M))
    lst[r] = d
    r += 1
lst = sorted(lst)
best1 = lst[0:number]
print("Done with greedy method.")

# Use the algorithm.
masks = closest_orthogonal_masks(M, number=number, fast_mode=True)
lst = [0.0 for _ in range(number)]
for r, mask in enumerate(masks):
    new_M = M * mask
    d = np.sum(np.square(new_M - M))
    lst[r] = d
best2 = sorted(lst)

# Print the result.
print(best1)
print(best2)
