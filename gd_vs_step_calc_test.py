import warnings

import numpy as np
import tensorflow as tf

#np.random.seed(42)
from local_search import GradientDescentAdam

dims=[900,900,900]
ks=[100,23,202]
num_matrices=[5,2,0,1,6,3]
S_spar = 0.2