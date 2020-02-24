#!/usr/bin/env python3
"""Code ot plot thew TGDL solutions."""
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Path to the location you want to make a movie of
load_path = '/home/james/Ising_Model_Codes/TGDL_Solutions/'
file_names = ["ground_1.npy", "stripe_1.npy", "diagonal_1.npy"]

the_times = [1.5, 2.5, 20, 300]

width = 3.375
height = 3 * width / (len(the_times)) + 0.125

fig, ax = plt.subplots(3, len(the_times), figsize=(width, height))
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.06, top=0.99,
                    wspace=0.05, hspace=0.05)

for file_name in enumerate(file_names):

    snapshots = np.load(load_path + file_name[1])
    sample_times = np.load(load_path + 'measured_times.npy')

    for i in range(len(the_times)):

        index = np.argmin(np.abs(sample_times - the_times[i]))

        frame_num = index
        ax[file_name[0], i].imshow(snapshots[:, :, frame_num],
                                   vmin=-1, vmax=1, cmap="RdGy")

        ax[file_name[0], i].set_xticks([])
        ax[file_name[0], i].set_yticks([])

        ax[-1, i].set_xlabel(r"$t = %.2f$" %
                             np.round(sample_times[frame_num], 2),
                             fontsize=10, labelpad=2)

x_pos, y_pos = 0.7, 0.86
ax[0, -1].text(x_pos, y_pos, r"(a)", color="white", fontsize=10,
               transform=ax[0, -1].transAxes)
ax[1, -1].text(x_pos, y_pos, r"(b)", color="white", fontsize=10,
               transform=ax[1, -1].transAxes)
ax[2, -1].text(x_pos, y_pos, r"(c)", color="white", fontsize=10,
               transform=ax[2, -1].transAxes)


if not os.path.exists("images/"):
    os.makedirs("images/")

fig.savefig("images/tgdl_evo.pdf", format="pdf", dpi=1000)
