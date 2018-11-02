#!/usr/bin/env python3
"""Solver for the TGDL in 2D with a one dimensional external potential."""
import numpy as np
import tgdl_functions as tf
import multiprocessing as mp
import time

# Define number of processes
n_proc = 4
# Define number of runs
number_runs = 4

# Data location
data_loc = '/home/james/Ising_Model_Codes/TGDL_Solutions/'

# Define square grid length (float or int)
grid_length = 50.
# Define number of grid points (int)
n_points = 50
# Define Point spacing
dx = grid_length / n_points

neighbours = np.zeros((2, n_points, 4), dtype=np.int64)
tf.get_all_neighbours(neighbours, n_points)

sample_times = np.linspace(0, 10, 100)


# Start the clock to estimate the total time taken for this lattice size
tic = time.time()
# Launch the quenches as independent parallel processes
pool = mp.Pool(processes=n_proc)
results = [pool.apply_async(tf.solve_tgdl,
           args=(n_points, dx, neighbours, sample_times, data_loc))
           for j in range(number_runs)]
output = [p.get() for p in results]
# Stop the clock for time estimation for simualtion on this lattice size
toc = time.time()
