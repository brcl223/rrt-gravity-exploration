#!/user/bin/env python3
import sys

import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) != 2:
    print(f"Expected arguments: <model name>")
    print("Exiting...")
    sys.exit(1)

MODEL = sys.argv[1]
EXPLORERS = ["random", "rrt", "rrt_star"]
OUTPUT_FILE = f"./data/current/graphs/{MODEL}-test_results.png"
NUM_NN_MODELS = 5


def input_filename(explorer, node_count, model):
    return f"./data/current/results/{MODEL}-{explorer}-{node_count}-model_{model}-manual_test_results.txt"


def load_data_files():
    out = {}
    for model in range(NUM_NN_MODELS):
        data = {}
        for explorer in EXPLORERS:
            # We want 2000-20000 inclusive
            avgs = []
            stddevs = []
            xs = []
            for node_count in range(2000, 20001, 2000):
                filename = input_filename(explorer, node_count, model)
                filedata = np.loadtxt(filename)
                avgs.append(np.average(filedata))
                stddevs.append(np.std(filedata))
                xs.append(node_count)
            data[explorer] = {}
            data[explorer]["avgs"] = avgs
            data[explorer]["stddevs"] = stddevs
            data[explorer]["xs"] = xs
        out[model] = data
    return out


def main():
    ylabel = f"Average Prediction MSE ({MODEL})"
    xlabel = "Training Samples"

    all_data = load_data_files()

    fig, axs = plt.subplots(nrows=NUM_NN_MODELS, figsize=(10,10))
    for model in range(NUM_NN_MODELS):
        data = all_data[model]
        cur_ax = axs[model]
        for explorer in EXPLORERS:
            avgs = data[explorer]["avgs"]
            stddevs = data[explorer]["stddevs"]
            xs = data[explorer]["xs"]

            cur_ax.errorbar(xs, avgs, stddevs, fmt='-o', label=explorer)

        cur_ax.set_title(f"{MODEL} NN Performance per Sample Count (NN Model {model})")
        # cur_ax.set_ylabel(ylabel)
        # cur_ax.set_xlabel(xlabel)

    fig.legend(loc='best')
    fig.subplots_adjust(hspace = 0.5)
    fig.savefig(OUTPUT_FILE)

main()
