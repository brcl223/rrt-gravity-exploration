import sys

from mujoco_py import load_model_from_xml, MjSim
import numpy as np
from numpy.linalg import norm
import torch as T

from shared import gen_and_load_nn, get_logger

if len(sys.argv) != 4:
    print("Wrong number of arguments supplied. Requires <model name>, <explorer>, and <num nodes>")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
EXPLORER = sys.argv[2]
NODE_COUNT = sys.argv[3]
DEVICE = 'cuda' if T.cuda.is_available() else 'cpu'
NN_MODEL_CHECKPOINT_PATH = f"./data/current/weights-trained/{MODEL_NAME}-{EXPLORER}-{NODE_COUNT}_nodes-model_IDX.chkpnt"
MODEL_XML_PATH = f"./src/models/{MODEL_NAME}.xml"
DESIRED_NODE_COUNT = 10000
MAX_SIM_LENGTH = 100000
MAX_VALUE = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
MIN_VALUE = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
AVG_LOSS_FILENAME = f"./data/current/results/{MODEL_NAME}-{EXPLORER}-{NODE_COUNT}-model_IDX-manual_test_results.txt"
MAX_TORQUES = np.array([87., 87., 87., 87., 12., 12., 12.])


######################################################################
# Logger Details
######################################################################
LOGGER_NAME = f"{MODEL_NAME}-{EXPLORER}-{NODE_COUNT}-automated-testing"
LOGGER = get_logger(LOGGER_NAME, log_level="info")


class PDController:
    def __init__(self, kp, kd, dim):
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
    # print(f"Abs sim data: {abs(sim.data.ctrl)}")
    # print(f"Max Torques: {MAX_TORQUES[0:dim]}")
    # print(f"ASD > Torques: {(abs(sim.data.ctrl) > MAX_TORQUES[0:dim]).any()}")
    return check_collision(sim) or (abs(sim.data.ctrl) > MAX_TORQUES[0:dim]).any()


def dist(node_a, node_b):
    return norm(node_a - node_b)


def find_q_and_g_at_point(sim, q_goal, pd, tau_prev=None, viewer=None):
    q_start = sim.data.qpos
    sim_steps = 0
    qc = None
    tau_pd = None
    for _ in range(MAX_SIM_LENGTH):
        qc = sim.data.qpos.copy()
        qv = sim.data.qvel
        tau_pd = pd(q_goal, qc, qv)
        if tau_prev is not None:
            tau_pd += tau_prev
        adjust_model(sim.data.ctrl, tau_pd)
        sim.step()
        if viewer is not None:
            viewer.render()
        sim_steps += 1

        if check_collision(sim) or check_end_of_sim(sim):
            break

    if check_bad_gravity_point(sim):
        raise Exception("Collision detected!")
    if not check_end_of_sim(sim):
        LOGGER.debug(f"FAILED SIM: Start point: {q_start}\nEnd Point: {q_goal}")
        raise Exception("Simulation did not terminate in timestep limit")

    # Return the actual q value we stopped at and the tau value
    # needed to support that point under gravity
    # Also return number of  sim steps to settle
    return qc, tau_pd, sim_steps


def find_closest_g(coord, nodes):
    min_dist = float('inf')
    min_tau = float('inf')
    for node in nodes:
        d = dist(node[0], coord)
        if d < min_dist:
            min_dist = d
            min_tau = node[1]
    return min_tau


def test_network(sim, pd, net, it):
    loss_values = []
    nq = sim.data.qpos.shape[0]

    # Sequentially go through random list of points and record information
    for i in range(DESIRED_NODE_COUNT):
        try:
            q_rand = np.random.uniform(low=MIN_VALUE[0:nq], high=MAX_VALUE[0:nq])
            q, t, sim_step = find_q_and_g_at_point(sim, q_rand, pd)
            q_tensor = T.Tensor(q).double().to(DEVICE)
            tau_predicted = net(q_tensor).to('cpu').detach().numpy()
            n = norm(tau_predicted - t)
            loss_values.append(n)

            print("\n---------------RESULTS------------------")
            print(f"Tau from PD: {t}")
            print(f"Tau predicted from NN: {tau_predicted}")
            print(f"Norm difference in estimation: {n}")
            print("----------------------------------------\n")
            input("Enter to continue...")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            LOGGER.critical(f"Exception: {str(e)}")
            LOGGER.critical("Simulation Error. Restarting...")
            reset_sim(sim)

    np.savetxt(AVG_LOSS_FILENAME.replace("IDX", str(it)), np.array(loss_values))


def main():
    LOGGER.info(f"Beginning automated tests\nModel Name: {MODEL_NAME}\nExplorer: {EXPLORER}\nNode Count: {NODE_COUNT}")

    xml = None
    with open(MODEL_XML_PATH, "r") as xml_file:
        xml = xml_file.read()

    model = load_model_from_xml(xml)
    sim = MjSim(model)
    nq = sim.data.qpos.shape[0]
    nt = sim.data.ctrl.shape[0]
    pd = PDController(165, 35, nq)

    # TODO: Fix this value
    for i in range(5):
        LOGGER.info(f"Beginning test iteration {i}")
        model_path = f"./data/current/weights-trained/{MODEL_NAME}-{EXPLORER}-{NODE_COUNT}_nodes-model_{i}.chkpnt"
        net, _, _ = gen_and_load_nn(model_path, nq, nt)
        test_network(sim, pd, net, i)
    LOGGER.info("Automated testing completed")


if __name__ == '__main__':
    main()
