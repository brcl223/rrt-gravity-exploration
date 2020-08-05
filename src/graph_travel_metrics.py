import sys

import numpy as np
from matplotlib import pyplot as plt

MODELS = [f"panda-{i}dof" for i in range(2,8)]
OUTPUT_FILENAME_AVG_DIST = f"./data/current/graphs/distance-metric.png"
OUTPUT_FILENAME_AVG_STEPS = f"./data/current/graphs/steps-metric.png"
OUTPUT_FILENAME_AVG_ENERGY = f"./data/current/graphs/energy-metric.png"
RANDOM_GRAPH_COLOR = 'r'
RRT_GRAPH_COLOR = 'b'
RRT_STAR_GRAPH_COLOR = 'g'


def gen_filename(model, explorer, nodes):
    return f"./data/current/metadata/{model}-{explorer}-{nodes}_nodes-simdist.txt"


def gen_filename_steps(model, explorer, nodes):
    return f"./data/current/metadata/{model}-{explorer}-{nodes}_nodes-simsteps.txt"


def gen_filename_energy(model, explorer, nodes):
    return f"./data/current/metadata/{model}-{explorer}-{nodes}_nodes-simenergy.txt"


def load_data(explorer):
    dists = []
    steps = []
    energy = []
    for model in MODELS:
        fname_dists = gen_filename(model, explorer, 20000)
        fname_steps = gen_filename_steps(model, explorer, 20000)
        fname_energy = gen_filename_energy(model, explorer, 10000)

        data_dists = np.loadtxt(fname_dists)
        data_steps = np.loadtxt(fname_steps)
        data_energy = np.loadtxt(fname_energy)

        dists.append([np.sum(data_dists), np.average(data_dists), np.std(data_dists)])
        steps.append([np.sum(data_steps), np.average(data_steps), np.std(data_steps)])
        energy.append([np.sum(data_energy), np.average(data_energy), np.std(data_energy)])

    return np.array(dists), np.array(steps), np.array(energy)


def main():
    rrt_dists, rrt_steps, rrt_energy = load_data("rrt")
    rrt_star_dists, rrt_star_steps, rrt_star_energy = load_data("rrt_star")
    random_dists, random_steps, random_energy = load_data("random")
    x_range = range(2,8)

    fig1, ax1 = plt.subplots()

    ax1.plot(x_range, random_dists[:, 1], label='Random', color=RANDOM_GRAPH_COLOR)
    ax1.plot(x_range, rrt_dists[:, 1], label='RRT', color=RRT_GRAPH_COLOR)
    ax1.plot(x_range, rrt_star_dists[:, 1], label='RRT*', color=RRT_STAR_GRAPH_COLOR)
    ax1.set(xlabel='Degree of Freedom', ylabel='Distance (Rad)')
    ax1.legend(loc='best')
    _, top = ax1.get_ylim()
    ax1.set_ylim(0, top)
    ax1.grid(b=True, which='major', color=(0.7, 0.7, 0.7, 0.6), linestyle='-')

    fig2, ax2 = plt.subplots()
    ax2.plot(x_range, random_steps[:, 1], label='Random', color=RANDOM_GRAPH_COLOR)
    ax2.plot(x_range, rrt_steps[:, 1], label='RRT', color=RRT_GRAPH_COLOR)
    ax2.plot(x_range, rrt_star_steps[:, 1], label='RRT*', color=RRT_STAR_GRAPH_COLOR)
    ax2.set(xlabel='Degree of Freedom', ylabel='Simulation Steps')
    _, top = ax2.get_ylim()
    ax2.set_ylim(0, top)
    ax2.legend(loc='best')
    ax2.grid(b=True, which='major', color=(0.7, 0.7, 0.7, 0.6), linestyle='-')

    fig3, ax3 = plt.subplots()
    ax3.plot(x_range, random_energy[:, 1], label='Random', color=RANDOM_GRAPH_COLOR)
    ax3.plot(x_range, rrt_energy[:, 1], label='RRT', color=RRT_GRAPH_COLOR)
    ax3.plot(x_range, rrt_star_energy[:, 1], label='RRT*', color=RRT_STAR_GRAPH_COLOR)
    ax3.set(xlabel='Degree of Freedom', ylabel='Energy (J)')
    _, top = ax3.get_ylim()
    ax3.set_ylim(0, top)
    ax3.legend(loc='best')
    ax3.grid(b=True, which='major', color=(0.7, 0.7, 0.7, 0.6), linestyle='-')

    fig1.savefig(OUTPUT_FILENAME_AVG_DIST)
    fig2.savefig(OUTPUT_FILENAME_AVG_STEPS)
    fig3.savefig(OUTPUT_FILENAME_AVG_ENERGY)


if __name__ == '__main__':
    main()
