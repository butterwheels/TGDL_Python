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
n_proc = 15
# Define number of runs
number_runs = 100
# Where to save the numerical solutions
data_loc = '/home/james/Ising_Model_Codes/TGDL_Solutions/'
if not os.path.exists(data_loc):
    os.makedirs(data_loc)
# Define square grid length (float or int)
grid_length = 512
# Define number of grid points (int)
n_points = 512
# Define Point spacing
dx = np.float32(grid_length / n_points)
# Set time step
delta_time = np.float32(0.01 * dx ** 2)
print("The time step is %f" % delta_time)
# Check stability condition
VN = (delta_time / dx ** 2.)
if VN <= 0.5:
    # If stability criterion has been met, print that we are all good
    print("Von Neumann stability criterion met, %0.2f <= 0.5" % VN)
else:
    # Print warning and stop the program
    print("Unstable time step to spacing ratio, aborting")
    exit()
# Array to hold the nearst neighbours of each grid point
neighbours = np.zeros((2, n_points, 4), dtype=np.int64)
# Get the nearest neighbours of each grid point
tf.get_all_neighbours(neighbours, n_points)
# Specify the number of snapshots to save
n_snaps = 501
# Array tol hold points in time at which to store current state of the grid
sample_times = np.zeros(n_snaps, dtype=np.float32)
# Set log spaced points in time
sample_times[1:] = np.logspace(-2, 4, n_snaps - 1, dtype=np.float32)
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

print("Total wall clock time = %f" % (toc - tic))
