#!/usr/bin/env python3
import sys

import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) != 5:
    print(f"Expected arguments: <model name>, <explorer>, <metric>, and <num nodes>")
    print("Exiting...")
    sys.exit(1)

MODEL = sys.argv[1]
EXPLORER = sys.argv[2]
# Metric should be one of: avg_loss, avg_var, max_loss, max_var
METRIC = sys.argv[3]
DESIRED_NODES = sys.argv[4]
DATA_FILE = f"./data/current/{MODEL}-{EXPLORER}-{DESIRED_NODES}_nodes-{METRIC}.txt"
OUTPUT_FILE = f"./data/current/graphs/{MODEL}-{EXPLORER}-{DESIRED_NODES}_nodes-{METRIC}_graph.png"


def main():
    ylabel = "Loss" if "loss" in METRIC else "Variance"

    data = np.loadtxt(DATA_FILE)
    train = data[:, 0]
    val = data[:, 1]
    test = data[:, 2]
    assert len(train) == len(val) == len(test)

    plt.plot(train, label='Train')
    plt.plot(val, label='Validation')
    plt.plot(test, label='Test')

    plt.title(f"{MODEL.title()} {EXPLORER.title()} NN {METRIC.title().replace('_', ' ')}")
    plt.ylabel(f"MSE {ylabel}")
    plt.xlabel("Training Batch")
    plt.ylim(0, 50)
    plt.legend(loc='best')
    plt.savefig(OUTPUT_FILE)


if __name__ == '__main__':
    main()
