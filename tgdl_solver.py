#!/usr/bin/env python3
"""Solver for the TGDL in 2D with a one dimensional external potential.

Parameters must be choosen in order to obey the stability condition here:
https://en.wikipedia.org/wiki/Von_Neumann_stability_analysis
"""
import numpy as np
import tgdl_functions as tf
import multiprocessing as mp
import time
import os

# Define number of processes
n_proc = 10
# Define number of runs
number_runs = 100

# Data location
data_loc = '/home/james/Ising_Model_Codes/TGDL_Solutions/'
if not os.path.exists(data_loc):
    os.makedirs(data_loc)

# Define square grid length (float or int)
grid_length = 100.
# Define number of grid points (int)
n_points = 500
# Define Point spacing
dx = np.float32(grid_length / n_points)
# Set time step
delta_time = np.float32(0.01 * dx ** 2.)

neighbours = np.zeros((2, n_points, 4), dtype=np.int64)
tf.get_all_neighbours(neighbours, n_points)

sample_times = np.linspace(0, 200, 201, dtype=np.float32)

# Start the clock to estimate the total time taken for this lattice size
tic = time.time()
# Launch the quenches as independent parallel processes
pool = mp.Pool(processes=n_proc)
results = [pool.apply_async(tf.solve_tgdl,
           args=(n_points, dx, neighbours, sample_times, data_loc, delta_time))
           for j in range(number_runs)]
output = [p.get() for p in results]
# Stop the clock for time estimation for simualtion on this lattice size
toc = time.time()
