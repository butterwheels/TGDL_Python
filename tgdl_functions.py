#!/usr/bin/env python3
"""Solver for the TGDL in 2D with a one dimensional external potential."""
import numpy as np
from numba import jit, float64, int64
import uuid
import matplotlib.pyplot as plt


@jit(int64(float64[:, :], float64[:, :]), nopython=True)
def potential_derivative(input_grid, output_grid):
    """Evaluate the derivative of the potential at each point on the grid."""
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):

            output_grid[i, j] = (2 * input_grid[i, j] *
                                 (1 - input_grid[i, j] ** 2.))
    return(1)


@jit(int64(float64[:, :], float64[:, :], int64[:, :, :], float64),
     nopython=True)
def compute_laplacian(input_grid, output_grid, neighbours, dx):
    """Compute the laplacian."""
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):

            output_grid[i, j] = 0.
            for k in range(neighbours.shape[2]):

                output_grid[i, j] += input_grid[neighbours[0, i, k],
                                                neighbours[1, j, k]]
                output_grid[i, j] -= input_grid[i, j]
    output_grid /= dx ** 2.
    return(1)


@jit(int64(int64[:, :, :], int64), nopython=True)
def get_all_neighbours(neighbours, n_points):
    """Get array to store the neighbours of all sites in the system."""
    for l in range(n_points):
        neighbours[0, l, 0] = l
        neighbours[0, l, 1] = l
        neighbours[0, l, 2] = l + 1
        neighbours[0, l, 3] = l - 1

        neighbours[1, l, 0] = l + 1
        neighbours[1, l, 1] = l - 1
        neighbours[1, l, 2] = l
        neighbours[1, l, 3] = l
    neighbours %= n_points
    return(1)


@jit(int64(float64[:], float64, float64[:, :], int64[:, :, :], float64,
     float64[:, :, :]), nopython=True)
def get_snapshots(sample_times, delta_time, grid_sol, neighbours, dx,
                  snapshots):
    """Get snapshots of coarsening evolution."""
    current_time = np.float64(0)
    counter = np.int32(0)
    output_grid = np.zeros((grid_sol.shape[0], grid_sol.shape[1]),
                           dtype=np.float64)

    while counter < len(sample_times):

        current_time += delta_time
        compute_laplacian(grid_sol, output_grid, neighbours, dx)
        grid_sol = grid_sol + (delta_time * output_grid)
        potential_derivative(grid_sol, output_grid)
        grid_sol = grid_sol + (delta_time * output_grid)

        if current_time >= sample_times[counter]:
            snapshots[:, :, counter] = grid_sol.copy()
            counter += 1
    return(1)


def solve_tgdl(n_points, dx, neighbours, sample_times, data_loc, delta_time):
    """Launch simulations to solve the TGDL."""
    np.random.seed()
    grid_sol = 2 * np.random.rand(n_points, n_points) - 1.0

    snapshots = np.zeros((n_points, n_points, len(sample_times)),
                         dtype=np.float64)
    get_snapshots(sample_times, delta_time, grid_sol, neighbours, dx,
                  snapshots)
    unique_string = str(uuid.uuid4())
    np.save(data_loc + unique_string, snapshots)
    np.save(data_loc + 'times', sample_times)

    fig, ax = plt.subplots(1, figsize=(1, 1))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    ax.imshow(snapshots[:, :, -1], vmin=-1, vmax=1, cmap='RdGy')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(data_loc + unique_string + '.png', dpi=1000, format='png')

    return()
