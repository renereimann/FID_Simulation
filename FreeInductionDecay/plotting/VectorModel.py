# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class AnimateVectorModelHistory(object):
    def __init__(self):
        self.history = []

def AnimateVectorModel(data, index_range=[0,-1], N_steps=100, interval=0.1, savepath=None):
    if index_range[-1] == -1:
        index_range[-1] = len(data.history)-1
    dN = (index_range[-1] - index_range[0])/N_steps

    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.view_init(30, 110)

    ax1.set_xlim(-1,1)
    ax1.set_ylim(-1,1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2.set_xlim(-1,1)
    ax2.set_ylim(-1,1)
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")

    ax3.set_xlim(-1,1)
    ax3.set_ylim(-1,1)
    ax3.set_xlabel("z")
    ax3.set_ylabel("y")

    line = ax.scatter(data.history[0][2], data.history[0][0], data.history[0][1], color="b")

    line1 = ax1.scatter(data.history[0][0], data.history[0][1], color="b")
    line2 = ax2.scatter(data.history[0][0], data.history[0][2], color="b")
    line3 = ax3.scatter(data.history[0][2], data.history[0][1], color="b")

    def update(num, ani, line):
        idx = int(num*dN)
        line._offsets3d = (ani[idx][2], ani[idx][0], ani[idx][1])
        line1.set_offsets(np.array((ani[idx][0], ani[idx][1])).T)
        line2.set_offsets(np.array((ani[idx][0], ani[idx][2])).T)
        line3.set_offsets(np.array((ani[idx][1], ani[idx][2])).T)

    # interval: Delay between frames in milliseconds

    ani = animation.FuncAnimation(fig, update, range(N_steps), fargs=(data.history, line), interval=interval)
    if savepath is not None:
        ani.save(savepath, writer='imagemagick')
    plt.show()
