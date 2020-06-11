import math
import os.path as path
import random
import sys

from mujoco_py import load_model_from_xml, MjSim, MjViewer
import numpy as np
from numpy.linalg import norm

# Check system arguments for model name
if len(sys.argv) != 2:
    print("Wrong number of arguments supplied. Requires <model name>.")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
MODEL_XML_PATH = f"./src/models/{MODEL_NAME}.xml"
DESIRED_NODE_COUNT = 10000
MAX_SIM_LENGTH = 100000
RANDOM_QPOS_FILENAME = f"./data/current/{MODEL_NAME}-random-qpos.txt"
RANDOM_TAUS_FILENAME = f"./data/current/{MODEL_NAME}-random-taus.txt"
RANDOM_TIME_FILENAME = f"./data/current/{MODEL_NAME}-random-simsteps.txt"
QPOS_DATA_PATH = f"./data/current/{MODEL_NAME}-cleaned_qpos.txt"
MAX_VALUE = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
MIN_VALUE = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]

def random_save_failed(a, t, i):
    a = np.array(a)
    filename = f"./data/current/{MODEL_NAME}-random-{t}-{i}.txt"
    np.savetxt(filename, a)


class PDController:
    def __init__(self, kp, kd, dim):
        # self.KP = np.diag([150,75,40,20,10,3,0.1]) * kp
        # self.KD = np.diag([150,75,40,20,10,3,0.1]) * kd
        self.KP = np.identity(dim) * kp
        self.KD = np.identity(dim) * kd

    def __call__(self, qd, q, qdot):
        return (self.KP @ (qd - q)) - (self.KD @ qdot)


def load_data_if_available(fail_if_no_file=True):
    qpos = []

    if path.exists(QPOS_DATA_PATH):
        qpos = list(np.loadtxt(QPOS_DATA_PATH))
    elif fail_if_no_file:
        raise Exception("[Error]: Unable to load previous data files.")
    else:
        print(f"[WARNING]: Unable to load previous data files.")

    return qpos


def generate_random_samples(dim, sample_size=DESIRED_NODE_COUNT):
    points = []
    for _ in range(sample_size):
        points.append(np.random.uniform(low=MIN_VALUE[0:dim], high=MAX_VALUE[0:dim]))
    return points


def adjust_model(model_attr, values):
    for i, v in enumerate(values):
        model_attr[i] = v


def reset_sim(sim, qpos=None, taus=None):
    nq = sim.data.qpos.shape[0]
    nt = sim.data.ctrl.shape[0]
    zeros = np.zeros(nq)
    qpos = qpos if qpos is not None else zeros
    taus = taus if taus is not None else np.zeros(nt)
    adjust_model(sim.data.qpos, qpos)
    adjust_model(sim.data.qvel, zeros)
    adjust_model(sim.data.qacc, zeros)
    adjust_model(sim.data.ctrl, taus)


def check_bad_sim(sim):
    if sim.data.ncon > 1:
        return True
    return False


def check_end_of_sim(sim):
    for i, v in enumerate(sim.data.qvel):
        # print(f"v[{i}] = {v}")
        if abs(v) >= 1e-5:
            return False
    for i, v in enumerate(sim.data.qacc):
        # print(f"a[{i}] = {v}")
        if abs(v) >= 1e-5:
            return False
    return True


def dist(node_a, node_b):
    return norm(node_a - node_b)


# Simulation exception for collision
class CollisionException(Exception):
    pass


# Simulation exception for settling time
class SettlingTimeException(Exception):
    pass


i = 0
def find_q_and_g_at_point(sim, q_goal, pd, tau_prev=None, viewer=None):
    # failed_qpos = []
    # failed_taus = []
    # global i
    q_start = sim.data.qpos
    sim_dist = 0
    qc = None
    tau_pd = None
    q_prev = q_start
    for _ in range(MAX_SIM_LENGTH):
        qc = sim.data.qpos.copy()
        qv = sim.data.qvel
        tau_pd = pd(q_goal, qc, qv)
        if tau_prev is not None:
            tau_pd += tau_prev
        adjust_model(sim.data.ctrl, tau_pd)
        sim.step()
        # failed_qpos.append(qc.copy())
        # failed_taus.append(tau_pd.copy())
        if viewer is not None:
            viewer.render()
        sim_dist += norm(qc - q_prev)
        q_prev = qc

        if check_bad_sim(sim) or check_end_of_sim(sim):
            break

    if check_bad_sim(sim):
        raise CollisionException("Collision detected!")
    if not check_end_of_sim(sim):
        print(f"Start point: {q_start}\nEnd Point: {q_goal}")
        raise SettlingTimeException("Simulation did not terminate in timestep limit")

    # Random chance to save results to explore
    # if random.random() > 0.99:
    #     random_save_failed(failed_qpos, "qpos_successful", i)
    #     random_save_failed(failed_taus, "taus_successful", i)
    #     i += 1

    # Return the actual q value we stopped at and the tau value
    # needed to support that point under gravity
    # Also return number of  sim steps to settle
    return qc, tau_pd, sim_dist


def find_closest_g(coord, nodes):
    min_dist = float('inf')
    min_tau = float('inf')
    for node in nodes:
        d = dist(node[0], coord)
        if d < min_dist:
            min_dist = d
            min_tau = node[1]
    return min_tau


def print_info(node):
    print("\n---------------------------------------------")
    print(f"Explored current node: {node.coord}")
    print(f"Tau value discovered: {node.tau}")
    print("---------------------------------------------\n")


def print_stats(sim_steps, it=0):
    print(f"\n-----------------Run {it}---------------------")
    print(f"Average travel dist: {np.average(sim_steps)}")
    print(f"Std Dev travel dist: {np.std(sim_steps)}")
    print(f"Total travel dist: {np.sum(sim_steps)}")
    print("---------------------------------------------\n")


def main():
    xml = None
    with open(MODEL_XML_PATH, "r") as xml_file:
        xml = xml_file.read()

    model = load_model_from_xml(xml)
    sim = MjSim(model)
    # viewer = MjViewer(sim)
    viewer = None

    nq = sim.data.qpos.shape[0]
    pd = PDController(165, 35, nq)

    # initial_data = load_data_if_available()
    initial_data = generate_random_samples(nq)
    start_point = np.zeros(nq)
    qstart, tau_start, sim_dist = find_q_and_g_at_point(sim, start_point, pd)
    collected_data = [(qstart, tau_start)]
    sim_dists = [sim_dist]

    # Sequentially go through random list of points and record information
    for i, qpos in enumerate(initial_data):
        tau_prev = find_closest_g(qpos, collected_data)
        try:
            q, t, sim_dist = find_q_and_g_at_point(sim, qpos, pd, tau_prev)
        except CollisionException as e:
            print("Collision Occurred!")
            reset_sim(sim)
            continue
        except SettlingTimeException as e:
            print("Settling time error!")
            continue
        collected_data.append((q,t))
        sim_dists.append(sim_dist)

        if i % 100 == 0:
            print_stats(sim_dists, i)

        if len(collected_data) >= DESIRED_NODE_COUNT:
            break

    # Once finished, convert our array of tuples into matrix and save values
    collected_data = np.array(collected_data)
    sim_dists = np.array(sim_dists)

    assert len(collected_data) == len(sim_dists)

    np.savetxt(RANDOM_QPOS_FILENAME, collected_data[:,0])
    np.savetxt(RANDOM_TAUS_FILENAME, collected_data[:,1])
    np.savetxt(RANDOM_TIME_FILENAME, sim_dists)


if __name__ == '__main__':
    main()
