# This script runs the stochastic algorithm to test it.

import numpy as np
from problems import prepare_deterministic, save_results
import time

for n in [800]:
    for k in [50]:
        for d in [6]:
            for r in range(12):
                # Construct an instance of algorithm class.
                algorithm,termination_condition=prepare_deterministic(n,k,d,r,max_gd_steps=5000,max_nfe=300000)
                # Set a random seed for repeatability.
                np.random.seed(r)
                # Run the optimization.
                start_time=time.time()
                algorithm.run(termination_condition)
                elapsed_time=time.time()-start_time
                algorithm.problem.gdOptimizer.close()
                # Save the results.
                save_results(algorithm,n,k,d,r,elapsed_time,'deterministic/')
