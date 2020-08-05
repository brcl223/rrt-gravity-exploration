import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

MODELS = [f"panda-{i}dof" for i in range(2,8)]
AVG_OUTPUT_FILENAME = "./data/current/graphs/avg_collision_data.png"
MAX_OUTPUT_FILENAME = "./data/current/graphs/max_collision_data.png"


def gen_filename(model, explorer, nodes):
    return f"./data/current/metadata/{model}-{explorer}-{nodes}_nodes-metadata.pickle"


def load_data(model, explorer):
    avg_nodes_rem = []
    max_nodes_rem = []
    for nodes in range(2000, 20001, 2000):
        filename = gen_filename(model, explorer, nodes)
        with open(filename, "rb") as f:
            d = pickle.load(f)
            avg_nodes_rem.append(d['total_nodes_removed_during_collision'] / d["collision_count"])
            max_nodes_rem.append(d['max_nodes_removed_during_collision'])

    return avg_nodes_rem, max_nodes_rem


def main():
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    ls = {
        5: '-',
        6: '--',
        7: ':',
    }
    for i in range(5,8):
        model = f"panda-{i}dof"
       
        rrt_data_avg, rrt_data_max = load_data(model, "rrt")
        rrt_star_data_avg, rrt_star_data_max = load_data(model, "rrt_star")
        x_range = range(2000, 20001, 2000)

        ax1.plot(x_range, rrt_data_avg, label=f'RRT ({i}DOF)', color='b', linestyle=ls[i])
        ax1.plot(x_range, rrt_star_data_avg, label='RRT* ({i}DOF)', color='g', linestyle=ls[i])

    ax1.ylabel("Average Nodes Removed per Collision")
    ax1.xlabel("Collected Samples")
    ax1.legend(loc="best")
    fig.savefig(AVG_OUTPUT_FILENAME)


if __name__ == '__main__':
    main()
