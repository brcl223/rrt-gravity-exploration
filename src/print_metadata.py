import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt

MODELS = [f"panda-{i}dof" for i in range(5,8)]
EXPLORERS = ["rrt", "rrt_star", "random"]
OUTPUT_FILE = "./data/current/graphs/metadataplot.png"


def filename(model, explorer, i=0):
    return f"./data/current/metadata/{model}-{explorer}-20000_nodes-metadata-{i}.pickle"


def load_data(explorer):
    out = {
        "avg_nodes_lost": [],
        "max_nodes_lost": [],
        "total_nodes_lost": [],
        "collisions": [],
    }

    for model in MODELS:
        max_n = []
        total_n = []
        avg_n = []
        collisions = []
        for i in range(4):
            fname = filename(model, explorer, i)
            data = None
            with open(fname, "rb") as f:
                data = pickle.load(f)
            max_n.append(data['max_nodes_removed_during_collision'])
            total_n.append(data['total_nodes_removed_during_collision'])
            collisions.append(data['collision_count'])
            if data['collision_count'] > 0:
                avg_n.append(data['total_nodes_removed_during_collision'] / data['collision_count'])
            else:
                avg_n.append(0)
        out['avg_nodes_lost'].append(np.average(avg_n))
        out['max_nodes_lost'].append(np.average(max_n))
        out['total_nodes_lost'].append(np.average(total_n))
        out['collisions'].append(np.average(collisions))

    return out


def main():
    rrt_data = load_data("rrt")
    rrt_star_data = load_data("rrt_star")
    xs = range(5,8)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    ax1.plot(xs, rrt_data['avg_nodes_lost'], label="RRT")
    ax1.plot(xs, rrt_star_data['avg_nodes_lost'], label="RRT*")
    ax1.set(xlabel="DOF", ylabel="Nodes / Collision", title="Average Nodes Pruned per Collision")

    ax2.plot(xs, rrt_data['max_nodes_lost'])
    ax2.plot(xs, rrt_star_data['max_nodes_lost'])
    ax2.set(xlabel="DOF", ylabel="Nodes", title="Max Nodes Pruned")

    ax3.plot(xs, rrt_data['total_nodes_lost'])
    ax3.plot(xs, rrt_star_data['total_nodes_lost'])
    ax3.set(xlabel="DOF", ylabel="Nodes", title="Total Nodes Pruned")

    ax4.plot(xs, rrt_data['collisions'])
    ax4.plot(xs, rrt_star_data['collisions'])
    ax4.set(xlabel="DOF", ylabel="Collisions", title="Total Collisions")

    fig.legend(loc='best')
    fig.suptitle("Metadata Collision Metrics Averaged over 5 Runs")
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    fig.savefig(OUTPUT_FILE)


if __name__ == '__main__':
    main()
