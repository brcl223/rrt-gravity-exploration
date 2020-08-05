import os.path as path
import sys

from mujoco_py import load_model_from_xml, MjSim
import numpy as np

# Check system arguments for model name
if len(sys.argv) != 4:
    print("Wrong number of arguments supplied. Requires <model name> <explorer> <num nodes>")
    sys.exit(1)

XML_MODEL_NAME = sys.argv[1]
MODEL_NAME = f"{XML_MODEL_NAME}-{sys.argv[2]}-{sys.argv[3]}_nodes"

DIRTY_TAU_DATA_PATH = f"./data/current/data-initial/{MODEL_NAME}-taus.txt"
DIRTY_QPOS_DATA_PATH = f"./data/current/data-initial/{MODEL_NAME}-qpos.txt"
TAU_DATA_PATH = f"./data/current/data-cleaned/{MODEL_NAME}-cleaned_taus.txt"
QPOS_DATA_PATH = f"./data/current/data-cleaned/{MODEL_NAME}-cleaned_qpos.txt"
MODEL_XML_PATH = f"./src/models/{XML_MODEL_NAME}.xml"
VEL_EPS = 1e-5
ACC_EPS = 1e-5
SIM_STEPS_TO_TEST = 10

MAX_TORQUES = np.array([87., 87., 87., 87., 12., 12., 12.])


def load_data_if_available(fail_if_no_file=True):
    tau_actuals = []
    qpos = []

    if path.exists(DIRTY_TAU_DATA_PATH) and path.exists(DIRTY_QPOS_DATA_PATH):
        tau_actuals = list(np.loadtxt(DIRTY_TAU_DATA_PATH))
        qpos = list(np.loadtxt(DIRTY_QPOS_DATA_PATH))
    elif fail_if_no_file:
        raise Exception("[Error]: Unable to load previous data files.")
    else:
        print(f"[WARNING]: Unable to load previous data files.")

    return tau_actuals, qpos


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
        if abs(v) >= VEL_EPS:
            return False
    for i, v in enumerate(sim.data.qacc):
        # print(f"a[{i}] = {v}")
        if abs(v) >= ACC_EPS:
            return False
    return True


def check_collision(sim):
    for contact in sim.data.contact:
        if contact.geom1 == 0 or contact.geom2 == 0:
            break
        # This appears to be a false positive due to one
        # of the joints colliding incorrectly
        if contact.geom1 == 19 and contact.geom2 == 23:
            continue
        # print(f"CG1: {contact.geom1}, CG2: {contact.geom2}, Dist: {contact.dist}")
        # input("Enter to continue")
        reset_collisions(sim)
        return True
    return False


def reset_collisions(sim):
    for contact in sim.data.contact:
        if contact.geom1 == 0 or contact.geom2 == 0:
            break
        contact.geom1 = 0
        contact.geom2 = 0


def check_bad_gravity_point(sim):
    dim = sim.data.qpos.shape[0]
    return check_collision(sim) or (abs(sim.data.ctrl) > MAX_TORQUES[0:dim]).any()


def main():
    xml = None
    with open(MODEL_XML_PATH, "r") as xml_file:
        xml = xml_file.read()

    model = load_model_from_xml(xml)
    sim = MjSim(model)
    # viewer = MjViewer(sim)

    # Basically numpy is dumb when loading 1D data
    # To avoid enumerate(.) issues later on in code,
    # make sure that we turn these into 2D arrays with
    # a single column (or row, I don't know)
    tau_finals, qpos_finals = load_data_if_available()
    assert len(tau_finals) == len(qpos_finals)
    tau_finals = np.array(tau_finals)
    qpos_finals = np.array(qpos_finals)
    if len(qpos_finals.shape) == 1:
        qpos_finals.shape = (-1,1)
    if len(tau_finals.shape) == 1:
        tau_finals.shape = (-1,1)

    num_samples = len(tau_finals)
    good_q_samples = []
    good_t_samples = []

    for i in range(num_samples):
        if i % 1000 == 0:
            print(f"Currently running step {i}")

        qpos = qpos_finals[i]
        tau = tau_finals[i]
        reset_sim(sim, qpos=qpos, taus=tau)

        for _ in range(SIM_STEPS_TO_TEST):
            sim.step()

            if not check_end_of_sim(sim) or check_collision(sim):
                break

        if not check_end_of_sim(sim) or check_bad_gravity_point(sim):
            continue  # We have a bad sample

        good_q_samples.append(qpos)
        good_t_samples.append(tau)

    print(f"Number of samples: {num_samples}")
    print(f"Number of good samples: {len(good_q_samples)}")

    assert len(good_q_samples) == len(good_t_samples)

    np.savetxt(QPOS_DATA_PATH, np.array(good_q_samples))
    np.savetxt(TAU_DATA_PATH, np.array(good_t_samples))


if __name__ == '__main__':
    main()
