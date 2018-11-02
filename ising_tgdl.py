#!/usr/bin/env python3
"""Attempt to solve the ginzburg landau fomrualtion for 3 state potts."""
import numpy as np
import matplotlib.pyplot as plt


def laplacian(u, dx):
    """Compute the laplacian of the grid u."""
    out = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
           np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u) / (dx ** 2.)
    return(out)


# f = lambda u: (u - u ** 3.)
f = lambda u: (2. * u * (1 - u ** 2.))


grid_length = 50.
number_points = 300

dx = grid_length / number_points

current_time = 0.
total_time = 1000.

dt = dx ** 2. / 100

u = 2. * np.random.rand(number_points, number_points) - 1.
print(np.sum(u))

plt.imshow(u, cmap='BuPu', vmin=-1, vmax=1)
plt.title("%0.3f" % current_time)
plt.colorbar()
plt.show()


check = 0.5


while current_time < total_time:
    current_time += dt
    u = u + dt * (f(u) + laplacian(u, dx))

    if current_time > check:
        plt.imshow(u, cmap='RdGy', vmin=-1, vmax=1)
        plt.title("%0.3f" % current_time)
        plt.colorbar()
        plt.show()
        check += 0.5
