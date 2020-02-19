#!/usr/bin/env python3
"""Solver for the TGDL in 2D with a one dimensional external potential.

Parameters must be choosen in order to obey the stability condition here:
https://en.wikipedia.org/wiki/Von_Neumann_stability_analysis
"""
import os
import time
import multiprocessing as mp
import numpy as np
import tgdl_functions as tf


def main():
    """Solve the TGDL Equation."""
    # Define number of processes
    n_proc = 15
    # Define number of runs
    number_runs = 1
    # Where to save the numerical solutions
    data_loc = '/home/james/Ising_Model_Codes/TGDL_Solutions/'
    if not os.path.exists(data_loc):
        os.makedirs(data_loc)
    # Define square grid length (int)
    grid_length = 100
    # Define number of grid points (int)
    n_points = 20 * grid_length
    # Define Point spacing
    delta_x = np.float32(grid_length / n_points)
    # Set time step
    delta_time = np.float32(0.01 * delta_x ** 2)
    print("The time step is %f" % delta_time)
    # Check stability condition
    von_newmann = (delta_time / delta_x ** 2.)
    if von_newmann <= 0.5:
        # If stability criterion has been met, print that we are all good
        print("Von Neumann stability criterion met, %0.2f <= 0.5" %
              von_newmann)
    else:
        # Print warning and stop the program
        print("Unstable time step to spacing ratio %0.2f, aborting" %
              von_newmann)
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
    sample_times[1:] = np.logspace(-2, 3, n_snaps - 1, dtype=np.float32)
    # Start the clock to estimate the total time taken for this lattice size
    times = np.zeros(2, dtype=np.float64)
    times[0] = time.time()
    # Launch the quenches as independent parallel processes
    pool = mp.Pool(processes=n_proc)
    results = [pool.apply_async(tf.solve_tgdl,
                                args=(n_points, delta_x, neighbours,
                                      sample_times, data_loc, delta_time))
               for j in range(number_runs)]
    output = [p.get() for p in results]
    print("%i Completed" % len(output))
    # Stop the clock for time estimation for simualtion on this lattice size
    times[1] = time.time()

    print("Total wall clock time = %f" % (times[1] - times[0]))

main()
