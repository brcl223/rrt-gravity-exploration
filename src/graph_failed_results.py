#!/usr/bin/env python3
import sys

import numpy as np
from matplotlib import pyplot as plt

i = 15
T = "successful"
EXPLORER = "rrt"
OUTPUT_FILE = f"./data/current/panda-5dof-{EXPLORER}-qpos_{T}_graph-{i}.png"


def load_data():
    qpos_filename = f"./data/current/panda-5dof-rrt-qpos_{T}-{i}.txt"
    taus_filename = f"./data/current/panda-5dof-rrt-taus_{T}-{i}.txt"
    return np.loadtxt(qpos_filename), np.loadtxt(taus_filename)


def main():
    qpos, taus = load_data()

    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].plot(qpos[:,0], label='Joint 1')
    axes[0].plot(qpos[:,1], label='Joint 2')
    axes[0].plot(qpos[:,2], label='Joint 3')
    axes[0].plot(qpos[:,3], label='Joint 4')
    axes[0].plot(qpos[:,4], label='Joint 5')

    axes[1].plot(taus[:,0], label='Joint 1')
    axes[1].plot(taus[:,1], label='Joint 2')
    axes[1].plot(taus[:,2], label='Joint 3')
    axes[1].plot(taus[:,3], label='Joint 4')
    axes[1].plot(taus[:,4], label='Joint 5')

    axes[0].set_title("QPos vs Time")
    axes[0].set(ylabel="QPos", xlabel="Time")
    axes[0].legend(loc='best')

    axes[1].set_title("Taus vs Time")
    axes[1].set(ylabel="Taus", xlabel="Time")
    axes[1].legend(loc='best')
    plt.savefig(OUTPUT_FILE)


if __name__ == '__main__':
    main()
