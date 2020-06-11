import sys

import numpy as np
from matplotlib import pyplot as plt

def gen_filename(i, explorer, nodes):
    return f"./data/current/panda-{i}dof-{explorer}-{nodes}_nodes-simdist.txt"


def gen_filename_steps(i, explorer, nodes):
    return f"./data/current/panda-{i}dof-{explorer}-{nodes}_nodes-simsteps.txt"


def gen_filename_loss(i, explorer, nodes):
    return f"./data/current/panda-{i}dof-{explorer}-{nodes}-manual-test.txt"


def load_data(explorer):
    dists = []
    steps = []
    for i in range(2,8):
        fname_dists = gen_filename(i, explorer, 20000)
        fname_steps = gen_filename_steps(i, explorer, 20000)

        data_dists = np.loadtxt(fname_dists)
        data_steps = np.loadtxt(fname_steps)

        dists.append([np.sum(data_dists), np.average(data_dists), np.std(data_dists)])
        steps.append([np.sum(data_steps), np.average(data_steps), np.std(data_steps)])

    return np.array(dists), np.array(steps)


def load_loss_data(explorer):
    loss = []
    for i in range(2,8):
        current_loss = []
        for nodes in range(2000, 22000, 2000):
            data_fname = gen_filename_loss(i, explorer, nodes)

            data = np.loadtxt(data_fname)

            current_loss.append(np.average(data))

        loss.append(np.array(current_loss))

    return np.array(loss)


def main():
    # rrt_dists, rrt_steps = load_data("rrt")
    # rrt_star_dists, rrt_star_steps = load_data("rrt_star")
    # random_dists, random_steps = load_data("random")
    # x_range = range(2,8)

    # fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle(f"Exploration Travel Metrics")

    # ax1.plot(x_range, rrt_dists[:,1], label='RRT')
    # ax1.plot(x_range, rrt_star_dists[:,1], label='RRT*')
    # ax1.plot(x_range, random_dists[:,1], label='Random')
    # ax1.set(xlabel='DOF', ylabel='Average Travel Distance (Rad)')

    # ax2.plot(x_range, rrt_steps[:,1], label='RRT')
    # ax2.plot(x_range, rrt_star_steps[:,1], label='RRT*')
    # ax2.plot(x_range, random_steps[:,1], label='Random')
    # ax2.set(xlabel='DOF', ylabel='Average Simulation Steps')

    # plt.legend(loc="best")
    # plt.savefig(f"./data/current/graphs/travel_metrics_plot.png")

    rrt_loss = load_loss_data("rrt")
    rrt_star_loss = load_loss_data("rrt_star")
    random_loss = load_loss_data("random")

    x_range = range(2000, 22000, 2000)

    plt.figure(0)
    for i in range(2,8):
        plt.plot(x_range, rrt_loss[i,:], label=f"RRT (DOF={i})")
        plt.plot(x_range, rrt_star_loss[i,:], label=f"RRT* (DOF={i})")
        plt.plot(x_range, random_loss[i,:], label=f"Random (DOF={i})")

    plt.legend(loc="best")
    plt.savefig(f"./data/current/graphs/loss_metrics_plot.png")


if __name__ == '__main__':
    main()
