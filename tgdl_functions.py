#!/usr/bin/env python3
"""Solver for the TGDL in 2D with a one dimensional external potential."""
import numpy as np
from numba import jit, float64, int64


@jit(float64[:, :](float64[:, :]), nopython=True)
def potential_derivative(input_grid):
    """Evaluate the derivative of the potential at each point on the grid."""
    output_grid = 2. * input_grid * (1 - input_grid ** 2.)
    return(output_grid)


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


@jit(float64[:, :](float64[:, :], float64[:, :], float64, int64[:, :, :]),
     nopython=True)
def compute_laplacian(grid_sol, delta_grid, dx, neighbours):
    """Compute the laplacian."""
    for i in range(delta_grid.shape[0]):
        for j in range(delta_grid.shape[1]):
            nbr_sum = 0
            for k in range(neighbours.shape[2]):

                nbr_sum += grid_sol[neighbours[0, i, k], neighbours[1, j, k]]
            delta_grid[i, j] = nbr_sum
    delta_grid -= 4 * grid_sol
    delta_grid /= (dx ** 2.)
    return(delta_grid)


@jit(float64[:, :, :](float64, float64[:, :], float64[:, :], float64,
     float64, int64[:, :, :], float64[:], float64[:, :, :]), nopython=True)
def evolve_till_time(current_time, delta_grid, grid_sol,
                     delta_time, dx, neighbours, sample_times, snapshots):
    """Solve the pde untill sepcified time."""
    counter = 1
    while current_time <= np.max(sample_times):

        delta_grid[:, :] = 0
        grid_sol += delta_time * (potential_derivative(grid_sol) +
                                  compute_laplacian(grid_sol, delta_grid, dx,
                                                    neighbours))
        current_time += delta_time

        if current_time >= sample_times[counter]:
            snapshots[:, :, counter] = grid_sol.copy()
            counter += 1
            print(current_time)
    return(snapshots)


def solve_tgdl(n_points, dx, neighbours, sample_times, data_loc, delta_time):
    """Launch simulations to solve the TGDL."""
    import os
    import uuid
    np.random.seed()
    # Initial grid
    grid_sol = 2 * np.random.rand(n_points, n_points) - 1
    # Setup array to store the change in grid over each time stip
    delta_grid = np.zeros((n_points, n_points), dtype=np.float64)
    # Set up current time
    current_time = 0.

    snapshots = np.zeros((n_points, n_points, len(sample_times)))

    evolve_till_time(current_time, delta_grid, grid_sol, delta_time, dx,
                     neighbours, sample_times, snapshots)

    # Get a unique string for the name of this quench
    data_name = str(uuid.uuid4())
    if not os.path.exists(data_loc):
        os.makedirs(data_loc)
    np.save(data_loc + data_name, snapshots)
    np.save(data_loc + 'times', sample_times)
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    fig, ax = plt.subplots(1, figsize=(1, 1))
    ax.imshow(snapshots[:, :, -1], cmap='RdGy')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(data_loc + data_name + '.png', dpi=1000, format='png')
    return()
