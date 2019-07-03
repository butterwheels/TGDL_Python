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

for i in range(len(frame_nums)):

    frame_num = frame_nums[i]
    print(sample_times[i])

    ax[i].imshow(snapshots[:, :, frame_num], vmin=-1, vmax=1, cmap="RdBu")

    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.show()
