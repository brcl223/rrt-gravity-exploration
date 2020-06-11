import random
import os.path as path

from mujoco_py import load_model_from_xml, MjSim, MjViewer
import numpy as np
import torch as T

DEVICE = 'cuda' if T.cuda.is_available() else 'cpu'
NN_MODEL_PATH = "./data/7dof-panda-robot-noee.pt"
MODEL_XML_PATH = "./src/panda.xml"
TAU_DATA_PATH = "./data/cleaned_tau_actuals.txt"
QPOS_DATA_PATH = "./data/cleaned_qpos.txt"
TEST_LENGTH = 25000


def load_data():
    tau_actuals = []
    qpos = []

    if path.exists(TAU_DATA_PATH) and path.exists(QPOS_DATA_PATH):
        tau_actuals = list(np.loadtxt(TAU_DATA_PATH))
        qpos = list(np.loadtxt(QPOS_DATA_PATH))
    else:
        raise Exception(f"[ERROR]: Unable to load previous data files.")

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


def load_nn():
    if path.exists(NN_MODEL_PATH):
        return T.load(NN_MODEL_PATH)
    else:
        raise Exception(f"[ERROR]: Unable to load NN model from path {NN_MODEL_PATH}")


def main():
    xml = None
    with open(MODEL_XML_PATH, "r") as f:
        xml = f.read()

    model = load_model_from_xml(xml)
    sim = MjSim(model)
    # viewer = MjViewer(sim)
    viewer = None

    nq = sim.data.qpos.shape[0]
    pd = PDController(65, 65, nq)

    print("Loading nn...")
    net = load_nn()
    print("Loading QPos/Tau data...")
    _, qpos = load_data()
    print("Done loading data")
    errors = []

    num_bad_sims = 0

    for i in range(len(qpos)):
        if i % 100 == 0:
            print(f"Starting iteration {i}\nNum Bad Sims so far: {num_bad_sims}")

        if viewer:
            input("Enter to start episode...")

        q_start = random.choice(qpos)
        reset_sim(sim, qpos=q_start)
        qc = T.from_numpy(q_start).to(DEVICE)
        tau_nn = net(qc).detach().cpu().numpy()
        adjust_model(sim.data.ctrl, tau_nn)
        for j in range(TEST_LENGTH):
            if viewer:
                viewer.render()
            qc = sim.data.qpos.copy()
            qv = sim.data.qvel.copy()
            tau_pd = pd(q_start, qc, qv)
            adjust_model(sim.data.ctrl, tau_nn + tau_pd)

            sim.step()

            if check_bad_sim(sim) or check_end_of_sim(sim):
                # print(f"Stopping after {j} iterations...")
                break

        if check_bad_sim(sim) or not check_end_of_sim(sim):
            # print(f"Bad simulation")
            num_bad_sims += 1
            continue

        qprime_final = T.from_numpy(qc).to(DEVICE)
        tau_nn_qprime = net(qprime_final).detach().cpu().numpy()
        errors.append(np.linalg.norm(tau_nn + tau_pd - tau_nn_qprime))

        if i % 1000 == 0:
            np.savetxt("./data/test-errors.txt", np.array(errors))



if __name__ == '__main__':
    main()
