import math
import pickle
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

# HYPERPARAMETERS (eps, rewire range):
# 3DOF: (0.5, 1.0)
# 4DOF: (0.5, 1.0)
# 5DOF: (0.5, 1.0)

MODEL_NAME = sys.argv[1]
MODEL_XML_PATH = f"./src/models/{MODEL_NAME}.xml"
RRT_NODE_COUNT = 5000
MAX_VALUE = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
MIN_VALUE = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
RRT_STEP_EPS = 0.5
RRT_REWIRE_RANGE = 1.0
MAX_SIM_LENGTH = 100000
RRT_QPOS_FILENAME = f"./data/current/{MODEL_NAME}-rrt-qpos.txt"
RRT_TAUS_FILENAME = f"./data/current/{MODEL_NAME}-rrt-taus.txt"
RRT_TIME_FILENAME = f"./data/current/{MODEL_NAME}-rrt-simsteps.txt"
RRT_TREE_DATA_FILENAME = f"./data/current/{MODEL_NAME}-rrt_tree-{RRT_NODE_COUNT}-{RRT_STEP_EPS}-{RRT_REWIRE_RANGE}.pickle"
RRT_AVOID_CACHE = False


def rrt_save_failed(a, t, i):
    a = np.array(a)
    filename = f"./data/current/{MODEL_NAME}-rrt-{t}-{i}.txt"
    np.savetxt(filename, a)


########################################################
# Start RRT related code
########################################################
class Node:
    def __init__(self,
                 dim=None,
                 parent=None,
                 coord=None,
                 tau=None,
                 cost=0):
        self.parent = parent
        self.children = []
        if coord is None and dim is None:
            raise ValueError("Coord and Dim cannot both be None!")
        if coord is None:
            self.coord = np.zeros(dim)
        else:
            self.coord = coord
        self.tau = tau
        self.cost = cost

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None


def dist(node_a, node_b):
    return norm(node_a - node_b)


def find_closest_node(cur_q, nodes):
    min_val = float('inf')
    min_idx = -1
    for i, v in enumerate(nodes):
        d = dist(cur_q, v.coord)
        if d < min_val:
            min_val = d
            min_idx = i
    return min_idx, min_val


def steer(q_rand, q_near, val, eps):
    q_next = None
    if val > eps:
        q_next = q_near + (q_rand - q_near) * eps / val
    else:
        q_next = q_rand
    return q_next


def save_tree(nodes):
    with open(RRT_TREE_DATA_FILENAME, "wb") as f:
        pickle.dump(nodes,f)


def load_tree_if_available():
    try:
        with open(RRT_TREE_DATA_FILENAME, "rb") as f:
            return pickle.load(f)
    except IOError:
        return None


def build_tree(dim, eps=RRT_STEP_EPS, node_count=RRT_NODE_COUNT):
    # Grab cached results if those are available
    old_data = load_tree_if_available() if not RRT_AVOID_CACHE else None
    if old_data is not None:
        print("[INFO]: Cached tree results discovered. Using these instead of"
        "generating new tree...")
        return old_data

    start_node = Node(dim=dim)
    nodes = [start_node]

    for i in range(node_count):
        if i % 1000 == 0:
            print(f"Current node count: {i}")

        # First generate random node within limits
        q_rand = np.random.uniform(low=MIN_VALUE[0:dim], high=MAX_VALUE[0:dim])

        # Next find closest existing node
        idx, val = find_closest_node(q_rand, nodes)
        q_near = nodes[idx]

        q_new_coord = steer(q_rand, q_near.coord, val, eps)

        # We don't have to worry about collision here, unlike real RRT*
        q_new_cost = dist(q_new_coord, q_near.coord) + q_near.cost

        q_min = q_near
        c_min = q_new_cost
        # TODO: For now this will be vanilla RRT due to the O(n^2) nature
        # of this current algorithm. However, in the future we could use
        # collision detection or other algorithms to begin rewiring in an
        # optimal manner.
        #
        # Unlike matlab impl, just combine the two loops to avoid having
        # to double loop for no reason each time
        for node in nodes:
            if dist(node.coord, q_new_coord) <= RRT_REWIRE_RANGE:
                cost = node.cost + dist(node.coord, q_new_coord)
                if cost < c_min:
                    q_min = node
                    c_min = cost

        q_new = Node(parent=q_min,
                     cost=c_min,
                     coord=q_new_coord)
        q_min.children.append(q_new)
        nodes.append(q_new)

    # Cache our tree results if this is the first time running with these
    # conditions
    save_tree(nodes)
    return nodes


########################################################
# Non-RRT related code
########################################################
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


# Simulation exception for collision
class CollisionException(Exception):
    pass


# Simulation exception for settling time
class SettlingTimeException(Exception):
    pass


i = 0
def find_q_and_g_at_node(sim, node, pd, tau_prev=None, viewer=None):
    # failed_qpos = []
    # failed_taus = []
    # global i
    # sim_steps = 0
    sim_dist = 0
    q_goal = node.coord
    q_start = sim.data.qpos.copy()
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
        # failed_qpos.append(qc.copy())
        # failed_taus.append(tau_pd.copy())
        sim.step()
        if viewer is not None:
            viewer.render()
        sim_dist += norm(qc - q_prev)

        q_prev = qc

        if check_bad_sim(sim) or check_end_of_sim(sim):
            break

    if check_bad_sim(sim):
        raise CollisionException("Bad simulation - Contact occurred")
    elif not check_end_of_sim(sim):
        # rrt_save_failed(failed_qpos, "qpos_failed", i)
        # rrt_save_failed(failed_taus, "taus_failed", i)
        # i += 1
        # failed_qpos = []
        # failed_taus = []
        print(f"Start node: {q_start}\nEnd node: {q_goal}\nDist: {dist(q_start, q_goal)}")
        raise SettlingTimeException("Bad simulation - Settling time exceeded max")

    # Random chance to save results to explore
    # if sim_steps > 6000:
    #     rrt_save_failed(failed_qpos, "qpos_successful", i)
    #     rrt_save_failed(failed_taus, "taus_successful", i)
    #     i += 1
    #     failed_qpos = []
    #     failed_taus = []

    # Return the actual q value we stopped at and the tau value
    # needed to support that point under gravity
    # Also return number of  sim steps to settle
    return qc, tau_pd, sim_dist


def find_closest_g(coord, nodes):
    min_dist = float('inf')
    min_tau = float('inf')
    # print("\n------------------------------------------------")
    # print(f"Searching for closest point to {coord}")
    for node in nodes:
        d = dist(node[0], coord)
        if d < min_dist:
            # print("\nFound new point!")
            # print(f"[INFO]\nq_org: {coord}\nq_new: {node[0]}\ndist: {d}")
            # input("Enter to continue...")
            min_dist = d
            min_tau = node[1]
    # print("------------------------------------------------\n")
    return min_tau


def print_info(node):
    print("\n---------------------------------------------")
    print(f"Explored current node: {node.coord}")
    print(f"Tau value discovered: {node.tau}")
    print("---------------------------------------------\n")


def print_stats(sim_steps, node_num=0):
    print(f"\n-----------------Run {node_num}---------------------")
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

    # Build tree of points
    print(f"Building tree with {RRT_NODE_COUNT} nodes...")
    nodes = build_tree(nq)
    print("Finished building tree.")
    current_node = nodes[0] # Start with root node - always at 0
    # Go ahead and find the gravity of our first node
    print("Discovering info on initial node...")
    q, tau, dist = find_q_and_g_at_node(sim, current_node, pd)
    current_node.coord = q
    current_node.tau = tau

    collected_data = [[current_node.coord, current_node.tau]]
    sim_dist = [dist]
    print_info(current_node)
    print("Beginning exploration through entire tree...")

    while True:
        # First check if we need the tau at the current node
        # If so, find it
        if current_node.tau is None:
            tau_prev = find_closest_g(current_node.coord, collected_data)
            try:
                q, t, dist = find_q_and_g_at_node(sim, current_node, pd, tau_prev, viewer=viewer)
            except CollisionException as e:
                print("Collision Occurred!")
                # TODO: May cause infinite loop here? Maybe need to pop child
                # Maybe fixed it here?
                current_node = current_node.parent
                # If the node we just had wasn't a leaf, we can get stuck in a loop
                # So remove the offending child if it hasn't been already
                if len(current_node.children) > 0 and not current_node.children[-1].is_leaf():
                    current_node.children.pop()
                reset_sim(sim, qpos=current_node.coord)
                continue
            except SettlingTimeException as e:
                print("Settling time error!")
                current_node.tau = 0 # Make it not None, but don't give it a real value
                continue
            current_node.tau = t
            collected_data.append([q,t])
            sim_dist.append(dist)

            num_samples_collected = len(collected_data)
            if num_samples_collected % 100 == 0:
                print_info(current_node)
                print_stats(sim_dist, num_samples_collected)

        # Now check whether or not we're a leaf node. If we're not, keep
        # exploring up the tree
        if len(current_node.children) > 0:
            if current_node.children[-1].is_leaf():
                current_node = current_node.children.pop()
            else:
                current_node = current_node.children[-1]
        else:
            # If we are at a leaf node, explore and then move back up
            # Iterate back to root node (Iterative depth first search)
            if current_node.is_root():
                break
            # print("Moving back to root node...")
            while not current_node.is_root():
                current_node = current_node.parent
            # Reset simulation to 0v
            # reset_sim(sim)

    # Once finished, convert our array of tuples into matrix and save values
    collected_data = np.array(collected_data)
    sim_dist = np.array(sim_dist)

    assert len(collected_data) == len(sim_dist)

    np.savetxt(RRT_QPOS_FILENAME, collected_data[:,0])
    np.savetxt(RRT_TAUS_FILENAME, collected_data[:,1])
    np.savetxt(RRT_TIME_FILENAME, sim_dist)


if __name__ == '__main__':
    main()
