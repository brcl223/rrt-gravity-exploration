import math
import os.path as path
import random
import time
import sys

from mujoco_py import load_model_from_xml, MjSim, MjViewer
import numpy as np
import torch as T
import torch.nn as nn

# Check system arguments
if len(sys.argv) != 2:
    print("Wrong number of arguments supplied. Requires <model name>.")
    sys.exit(1)

# First get model name
MODEL_NAME = sys.argv[1]

SAMPLES_TO_COLLECT = 50000
MAX_EPISODE_LENGTH = 10000
EPISODE_WATCHING_COUNT = 20
DEVICE = 'cuda' if T.cuda.is_available() else 'cpu'
BATCH_SIZE = 300
INITIAL_QPOS_PATH = f"./data/current/{MODEL_NAME}-initial-qpos.txt"
NN_MODEL_PATH = f"./data/current/{MODEL_NAME}.pt"
TAU_DATA_PATH = f"./data/current/{MODEL_NAME}-tau_actuals.txt"
QPOS_DATA_PATH = f"./data/current/{MODEL_NAME}-qpos.txt"
MODEL_XML_PATH = f"./src/models/{MODEL_NAME}.xml"
LOSS_PATH = f"./data/current/{MODEL_NAME}-loss.txt"


def load_data_if_available():
    tau_actuals = []
    qpos = []

    if path.exists(TAU_DATA_PATH) and path.exists(QPOS_DATA_PATH):
        tau_actuals = list(np.loadtxt(TAU_DATA_PATH))
        qpos = list(np.loadtxt(QPOS_DATA_PATH))
    else:
        print(f"[WARNING]: Unable to load previous data files.")

    return tau_actuals, qpos


class PDController:
    def __init__(self, kp, kd, dim):
        # self.KP = np.diag([150,75,40,20,10,3,0.1]) * kp
        # self.KD = np.diag([150,75,40,20,10,3,0.1]) * kd
        self.KP = np.identity(dim) * kp
        self.KD = np.identity(dim) * kd

    def __call__(self, qd, q, qdot):
        return (self.KP @ (qd - q)) - (self.KD @ qdot)


def adjust_model(model_attr, values):
    # try:
    for i, v in enumerate(values):
        model_attr[i] = v
    # except TypeError:
        # If we get here we have a 1-dof system
        # Numpy is dumb and defaults zeros(1) to be a
        # scalar instead of a 1x1 vector, so this is
        # where we ended up
        # model_attr[0] = values


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


# TODO: Get rid of reliance on qacc here
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


def check_bad_sim(sim):
    if sim.data.ncon > 1:
        return True
    return False


def explore_new_point(sim, viewer=None, random_start=False):
    good_pos = False

    while not good_pos:
        taus = None
        qs = None
        if random_start:
            nq = sim.data.qpos.shape[0]
            qs = np.zeros(nq)
            # TODO: Don't hardcode these values
            for i in range(nq):
                if i in [0, 2, 4, 6]:
                    # 0.925 = MAX_ANGLE / math.pi for joints according
                    # to panda manufacturer
                    qs[i] = random.uniform(-math.pi, math.pi) * 0.925
                if i == 3:
                    qs[i] = random.uniform(-math.pi,0) / 8
                if i == 5:
                    # This is the annoying one at the end that is very
                    # different than all the other joints
                    qs[i] = random.uniform(-0.0175, math.pi + 0.4)
            nt = sim.data.ctrl.shape[0]
            taus = [random.uniform(-1, 1) * (nt - i) ** 1.2 for i in range(nt)]

        reset_sim(sim, qpos=qs, taus=taus)
        random_turns_to_fall = random.randrange(10, 500)

        for _ in range(random_turns_to_fall):
            if viewer:
                viewer.render()
            sim.step()

        if not check_bad_sim(sim):
            good_pos = True

    return sim.data.qpos.copy()


def explore_the_world(sim, viewer=None):
    starting_qpos = []
    for i in range(SAMPLES_TO_COLLECT):
        vx = None
        if i % 100 == 0:
            print(f"Currently on iteration {i}...")
        if i % EPISODE_WATCHING_COUNT == 0:
            vx = viewer
        q = explore_new_point(sim, viewer=vx, random_start=True)
        starting_qpos.append(q)
        if viewer and i % EPISODE_WATCHING_COUNT == 0:
            input("Enter for next iteration...")

    np.savetxt(INITIAL_QPOS_PATH, np.array(starting_qpos))


def get_qpos_desired(qpos_previous, qpos, i, new_sample_count=10):
    if i % new_sample_count == 0:
        return qpos[i // new_sample_count]

    nq = qpos[0].shape[0]
    return qpos_previous + \
        np.array([random.uniform(-math.pi, math.pi) / 16 for _ in range(nq)])


def get_next_tau(previous_tau, nt, i, new_sample_count=10):
    if i % new_sample_count == 0:
        return np.zeros(nt)
    return previous_tau


def sample_initial_qpos(model, sim, viewer=None):
    qpos = np.loadtxt(INITIAL_QPOS_PATH)

    # When numpy loads this data with loadtxt, it returns a
    # (n,) array in the case when our data is an nx1 matrix.
    # This throws errors later in the code, since these qpos
    # values are assumed to be iterable later on. So reshape
    # it to avoid headache
    if len(qpos.shape) == 1:
        qpos.shape = (-1, 1)

    nq = sim.data.qpos.shape[0]
    nt = sim.data.ctrl.shape[0]

    # Seemingly stable gains for the Panda arm
    pd = PDController(140, 45, nq)

    final_qpos = []
    final_taus = []

    new_sample_count = 10
    num_samples = len(qpos) * new_sample_count

    t_previous = np.zeros(nt)
    qpos_previous = np.zeros(nq)

    num_bad_samples = 0
    failed_sims = 0
    bad_sim = False

    start_time = time.time()
    for i in range(num_samples):
        if i % 100 == 0:
            end_time = time.time()
            print(f"Last iteration took {end_time - start_time} seconds")
            print(f"Currently at iteration {i}...")
            print(f"Bad Samples: {num_bad_samples}\t\tFailed Sims: {failed_sims}")
            start_time = end_time

        q_desired = get_qpos_desired(qpos_previous, qpos, i, new_sample_count=new_sample_count)
        t_previous = get_next_tau(t_previous, nt, i, new_sample_count=new_sample_count)

        reset_sim(sim, qpos=q_desired, taus=t_previous)

        for j in range(MAX_EPISODE_LENGTH):
            if viewer and i % EPISODE_WATCHING_COUNT == 0:
                viewer.render()

            q_current = sim.data.qpos.copy()
            qdot_current = sim.data.qvel.copy()

            tau_pd = pd(q_desired, q_current, qdot_current)
            tau_desired = tau_pd + t_previous
            adjust_model(sim.data.ctrl, tau_desired)

            try:
                sim.step()
            except:
                bad_sim = True
                break

            if check_end_of_sim(sim):
                break

        if viewer and i % EPISODE_WATCHING_COUNT == 0:
            input("Enter for next episode...")

        if check_bad_sim(sim) or not check_end_of_sim(sim) or bad_sim:
            if bad_sim:
                sim = MjSim(model)
                failed_sims += 1
                bad_sim = False
            num_bad_samples += 1
            continue

        qpos_previous = q_current.copy()
        t_previous = tau_desired.copy()
        final_qpos.append(q_current)
        final_taus.append(tau_desired)

        if i % 1000 == 0:
            np.savetxt(QPOS_DATA_PATH, np.array(final_qpos))
            np.savetxt(TAU_DATA_PATH, np.array(final_taus))


def main(render=True):
    xml = None
    with open(MODEL_XML_PATH, "r") as f:
        xml = f.read()

    model = load_model_from_xml(xml)
    sim = MjSim(model)
    viewer = None
    if render:
        viewer = MjViewer(sim)

    explore_the_world(sim, viewer=viewer)
    sample_initial_qpos(model, sim, viewer=viewer)


if __name__ == '__main__':
    print(f"------------RUNNING GRAVITY EXPLORER FOR MODEL {MODEL_NAME}----------\n\n")
    main(render=False)
    print(f"\n\n-----------END OF GRAVITY SIMULATION FOR MODEL {MODEL_NAME}----------\n\n")
