#!/user/bin/env python3
import sys

import numpy as np
from matplotlib import pyplot as plt

MODELS = [f"panda-{i}dof" for i in range(2, 8)]
EXPLORERS = ["random", "rrt", "rrt_star"]
EXPLORER_NAMES = ["Random", "RRT", "RRT*"]
OUTPUT_FILE = f"./data/current/graphs/accuracy_test_results.png"
NUM_NN_MODELS = 5
MAX_TORQUES = [87., 87., 87., 87., 12., 12., 12.]


def max_torque_norm(dof):
    return np.linalg.norm(MAX_TORQUES[:dof])


def is_below_norm_error_threshold(avg, dof, err_threshold=0.015):
    err = err_threshold * max_torque_norm(dof)
    print(f"DOF: {dof}\nAvg: {avg}\nError: {err}\n")
    return avg < err


def input_filename(model, explorer, node_count, weight_model):
    return f"./data/current/results/{model}-{explorer}-{node_count}-model_{weight_model}-manual_test_results.txt"


def output_filename(model):
    return f"./data/current/graphs/{model}-accuracy_test_results.png"


def load_samples_for_error(explorer):
    samples = []
    for i, model in enumerate(MODELS):
        dof = i + 2
        for nodes in range(2000, 20001, 2000):
            cur_avg = []
            for weight in range(NUM_NN_MODELS):
                filename = input_filename(model, explorer, nodes, weight)
                filedata = np.loadtxt(filename)
                cur_avg.append(np.average(filedata))
            avg = np.average(cur_avg)
            print(f"Explorer: {explorer}\nNodes: {nodes}")
            if is_below_norm_error_threshold(avg, dof):
                samples.append(nodes)
                break
    assert len(samples) == 6
    return samples


def load_data(model, explorer):
    data = []
    for nodes in range(2000, 40001, 2000):
        cur_data = []
        for weight in range(NUM_NN_MODELS):
            filename = input_filename(model, explorer, nodes, weight)
            filedata = np.loadtxt(filename)
            cur_data.append(np.average(filedata))
        data.append(np.average(cur_data))
    return data


def main():
    random_data = load_data("panda-7dof", "random")
    rrt_data = load_data("panda-7dof", "rrt")
    rrt_star_data = load_data("panda-7dof", "rrt_star")
    # random_data6 = load_data("panda-6dof", "random")
    # rrt_data6 = load_data("panda-6dof", "rrt")
    # rrt_star_data6 = load_data("panda-6dof", "rrt_star")
    random_data5 = load_data("panda-5dof", "random")
    rrt_data5 = load_data("panda-5dof", "rrt")
    rrt_star_data5 = load_data("panda-5dof", "rrt_star")
    xs = range(2000, 40001, 2000)

    plt.plot(xs, random_data, label='Random (7DOF)', color='r')
    plt.plot(xs, rrt_data, label='RRT (7DOF)', color='b')
    plt.plot(xs, rrt_star_data, label='RRT* (7DOF)', color='g')
    # plt.plot(xs, random_data6, label='Random (6DOF)', color='r', linestyle='--')
    # plt.plot(xs, rrt_data6, label='RRT (6DOF)', color='b', linestyle='--')
    # plt.plot(xs, rrt_star_data6, label='RRT* (6DOF)', color='g', linestyle='--')
    plt.plot(xs, random_data5, label='Random (5DOF)', color='r', linestyle=':')
    plt.plot(xs, rrt_data5, label='RRT (5DOF)', color='b', linestyle=':')
    plt.plot(xs, rrt_star_data5, label='RRT* (5DOF)', color='g', linestyle=':')
    plt.legend(loc='best')
    plt.xlabel("Collected Samples")
    plt.ylabel("NN Prediction Error")
    plt.xlim(2000, 40000)
    plt.grid(which='major', color=(0.7, 0.7, 0.7, 0.6))

    plt.savefig(output_filename("panda-567dof"))


main()
