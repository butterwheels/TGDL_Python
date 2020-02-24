#!/usr/bin/env python3
"""Code ot plot thew TGDL solutions."""
import numpy as np
import matplotlib.pyplot as plt

# Path to the location you want to make a movie of
load_path = '/home/james/Ising_Model_Codes/TGDL_Solutions/'
file_name = 'diagonal.npy'

snapshots = np.load(load_path + file_name)
sample_times = np.load(load_path + 'measured_times.npy')

frame_nums = [200, 250, 400, 400]

width = 6
height = 2

fig, ax = plt.subplots(1, len(frame_nums), figsize=(width, height))

import os
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Path to the location you want to make a movie of
load_path = '/home/james/Ising_Model_Codes/TGDL_Solutions/'
file_names = ["ground.npy", "on_axis.npy", "diagonal.npy"]

frame_nums = [225, 250, 400, 500]

width = 3.375
height = 2.75

fig, ax = plt.subplots(3, len(frame_nums), figsize=(width, height))
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.075, top=0.99,
                    wspace=0.05, hspace=0.1)

for file_name in enumerate(file_names):

    snapshots = np.load(load_path + file_name[1])
    sample_times = np.load(load_path + 'measured_times.npy')

    for i in range(len(frame_nums)):

        frame_num = frame_nums[i]

        ax[file_name[0], i].imshow(snapshots[:, :, frame_num],
                                   vmin=-1, vmax=1, cmap="YlGnBu")

        ax[file_name[0], i].set_xticks([])
        ax[file_name[0], i].set_yticks([])

        ax[-1, i].set_xlabel(r"$t = %0.4f$" % sample_times[i])

if not os.path.exists("images/"):
    os.makedirs("images/")

fig.savefig("images/tgdl_evo.eps", format="eps", dpi=1000)
