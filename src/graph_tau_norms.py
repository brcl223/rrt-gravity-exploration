#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np

tau_data = np.loadtxt("./data/current/tau_norm_avg.txt")
tau_pred_data = np.loadtxt("./data/current/tau_pred_norm_avg.txt")

plt.plot(tau_data, label='Taus')
plt.plot(tau_pred_data, label='Tau Predicted')

plt.title("Avg Norm Tau vs Tau Predicted")
plt.ylabel("Norm")
plt.xlabel("Different samples")
plt.legend(loc="best")
plt.savefig(f"./data/current/graphs/tau_plot.png")
