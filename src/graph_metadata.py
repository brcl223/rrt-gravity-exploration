import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt


def gen_filename(i, explorer, nodes, ext):
    return f"./data/current/panda-{i}dof-{explorer}-{nodes}_nodes-metadata.{ext}"


def load_data(explorer):
    collisions = []
    for i in range(2,8):
        filename = gen_filename(i, explorer, 20000, "pickle")
        with open(filename, "rb") as f:
            d = pickle.load(f)
            collisions.append(d["collision_count"])

    return collisions


def main():
    rrt_data = load_data("rrt")
    rrt_star_data = load_data("rrt_star")
    random_data = load_data("random")
    x_range = range(2,8)

    plt.plot(x_range, rrt_data, label='RRT')
    plt.plot(x_range, rrt_star_data, label='RRT*')
    plt.plot(x_range, random_data, label='Random')

    plt.title("Collision counts per DOF")
    plt.ylabel("Number of collisions")
    plt.xlabel("Degrees of Freedom")
    plt.legend(loc="best")
    plt.savefig(f"./data/current/graphs/metadata_plot.png")


if __name__ == '__main__':
    main()
