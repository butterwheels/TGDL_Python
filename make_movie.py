#!/usr/bin/env python3
"""Attempt to solve the ginzburg landau fomrualtion for 3 state potts."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Path to the location you want to make a movie of
load_path = '/home/james/Ising_Model_Codes/TGDL_Solutions/'
file_name = 'test.npy'

snapshots = np.load(load_path + file_name)
sample_times = np.load(load_path + 'times.npy')

# Where to save the movie
file_name = file_name[:-4]
save_path = ('/home/james/Ising_Model_Codes/TGDL_movies/')
if not os.path.exists(save_path):
    os.makedirs(save_path)

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='TGDL mess around',
                artist='James Denholm', comment='Movie support!')
writer = FFMpegWriter(fps=25, metadata=metadata)

cmap = 'RdPu'

fig, ax = plt.subplots(1, figsize=(1.2, 1.1))
fig.subplots_adjust(left=0.05, right=0.8, bottom=0.05)

im = ax.imshow(snapshots[:, :, 0], cmap=cmap, vmin=-1, vmax=1)
ax.set_title('t = %0.2f' % sample_times[0], fontsize=8)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_ticks([1, 0, -1])
cbar.ax.tick_params(labelsize=6)


ax.set_xticks([])
ax.set_yticks([])

counter = 0

with writer.saving(fig, save_path + file_name + '.mp4', dpi=1000):

    writer.grab_frame()

    for i in range(1, snapshots.shape[2]):

        im = ax.imshow(snapshots[:, :, i], cmap=cmap, vmin=-1, vmax=1)
        ax.set_title('t = %0.3f' % sample_times[i], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        writer.grab_frame()
        ax.clear()
