#!/usr/bin/env python3
"""Solver for the TGDL in 2D with a one dimensional external potential."""
import numpy as np
from numba import jit, float32, int64
import uuid
import matplotlib.pyplot as plt
# plt.switch_backend("agg")


@jit(int64(float32[:, :], float32[:, :]), nopython=True)
def potential_derivative(input_grid, output_grid):
    """Evaluate the derivative of the potential at each point on the grid."""
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):

            output_grid[i, j] = (2 * input_grid[i, j] *
                                 (1 - input_grid[i, j] ** 2.))
    return(1)


@jit(int64(float32[:, :], float32[:, :], int64[:, :, :], float32),
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


@jit(int64(float32[:], float32, float32[:, :], int64[:, :, :], float32,
           float32[:, :, :], float32[:]), nopython=True)
def get_snapshots(sample_times, delta_time, grid_sol, neighbours, dx,
                  snapshots, measured_times):
    """Get snapshots of coarsening evolution."""
    current_time = np.float32(0)
    counter = np.int32(1)
    output_grid = np.zeros((grid_sol.shape[0], grid_sol.shape[1]),
                           dtype=np.float32)
    snapshots[:, :, 0] = grid_sol.copy()

    while counter < len(sample_times):

        current_time += delta_time
        compute_laplacian(grid_sol, output_grid, neighbours, dx)
        grid_sol += (delta_time * output_grid)
        potential_derivative(grid_sol, output_grid)
        grid_sol += (delta_time * output_grid)

        if current_time >= sample_times[counter]:
            snapshots[:, :, counter] = grid_sol.copy()
            measured_times[counter] = current_time
            counter += 1
    return(1)


@jit(nopython=True)
def random_shuffle_grid(input_grid):
    """Random shuffle the input grid."""
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):

            swap_i = np.random.randint(input_grid.shape[0])
            swap_j = np.random.randint(input_grid.shape[1])

            site_val = input_grid[i, j]
            swap_val = input_grid[swap_i, swap_j]

            input_grid[i, j] = swap_val
            input_grid[swap_i, swap_i] = site_val
    return()


def solve_tgdl(n_points, dx, neighbours, sample_times, data_loc, delta_time):
    """Launch simulations to solve the TGDL."""
    np.random.seed()
    # Initialize array to hold field at each grid point
    grid_sol = np.zeros(n_points ** 2, dtype=np.float32)
    # Set half the initial condition to m = 1
    # Actual times
    measured_times = np.zeros(len(sample_times), dtype=np.float32)
    # Initial condition
    grid_sol[0::2] = 1
    grid_sol[1::2] = -1
    # Set half the initial condition to m = -1
    grid_sol = np.reshape(grid_sol, (n_points, n_points))
    # Random order the initial condition
    random_shuffle_grid(grid_sol)
    # Set up array to hold snapshots of the solution
    snapshots = np.zeros((n_points, n_points, len(sample_times)),
                         dtype=np.float32)
    # Get the snapshots
    get_snapshots(sample_times, delta_time, grid_sol, neighbours, dx,
                  snapshots, measured_times)
    unique_string = str(uuid.uuid4())
    np.save(data_loc + unique_string, snapshots)
    np.save(data_loc + 'aimed_times', sample_times)
    np.save(data_loc + 'measured_times', measured_times)

    print(np.min(grid_sol))
    print(np.max(grid_sol))

    fig, ax = plt.subplots(1, figsize=(1, 1))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    ax.imshow(snapshots[:, :, -1], vmin=-1, vmax=1, cmap='RdBu')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(data_loc + unique_string + '.png', dpi=1000, format='png')

    return()
