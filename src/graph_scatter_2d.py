#!/user/bin/env python3
import sys

import numpy as np
from matplotlib import pyplot as plt

MODEL = "panda-2dof"
EXPLORERS = ["random", "rrt", "rrt_star"]
OUTPUT_FILE = f"./data/current/graphs/{MODEL}-manual-test.png"


def load_data_files():
    data = {}
    for explorer in EXPLORERS:
        # We want 2000-20000 inclusive
        data[explorer] = {}
        for i in range(2000, 20001, 2000):
            filename = f"./data/current/{MODEL}-{explorer}-{i}_nodes-cleaned_qpos.txt"
            filedata = np.loadtxt(filename)
            data[explorer][i] = filedata
    return data


def main():
    data = load_data_files()

    for explorer in EXPLORERS:
        fig, axs = plt.subplots(4, 3, figsize=(30, 30))
        for idx, i in enumerate(range(2000, 20001, 2000)):
            iy = idx // 4
            ix = idx % 4
            xs = data[explorer][i][:,0]
            ys = data[explorer][i][:,1]
            axs[ix,iy].scatter(xs, ys)
            axs[ix,iy].set_xlim(-2.8973,2.8973)
            axs[ix,iy].set_ylim(-1.7628,1.7628)
        fig.savefig(f"./data/current/graphs/panda-2dof-{explorer}-scatter-plot.png")


main()
