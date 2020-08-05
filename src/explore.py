import pickle
import sys
import time

from mujoco_py import load_model_from_xml, MjSim, MjViewer
import numpy as np
from numpy.linalg import norm

from shared import get_logger

# Check system arguments for model name
if len(sys.argv) != 4:
    print("Wrong number of arguments supplied. Requires:")
    print("Model Name: panda-*dof")
    print("Explorer: (rrt | rrt_star | random)")
    print("Desired node count: (positive integer value)")
    sys.exit(1)


######################################################################
# Generic Params
######################################################################
CURRENT_TIME = time.time()
MODEL_NAME = sys.argv[1]
EXPLORER = sys.argv[2]
MODEL_XML_PATH = f"./src/models/{MODEL_NAME}.xml"
MAX_VALUE = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
MIN_VALUE = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
MAX_SIM_LENGTH = 40000
VEL_EPS = 1e-5
ACC_EPS = 1e-5
USE_TAU_PREV = True
NODE_COUNT = int(sys.argv[3])
QPOS_FILENAME = f"./data/current/data-initial/{MODEL_NAME}-{EXPLORER}-{NODE_COUNT}_nodes-qpos.txt"
TAUS_FILENAME = f"./data/current/data-initial/{MODEL_NAME}-{EXPLORER}-{NODE_COUNT}_nodes-taus.txt"
DIST_FILENAME = f"./data/current/metadata/{MODEL_NAME}-{EXPLORER}-{NODE_COUNT}_nodes-simdist.txt"
STEPS_FILENAME = f"./data/current/metadata/{MODEL_NAME}-{EXPLORER}-{NODE_COUNT}_nodes-simsteps.txt"
ENERGY_FILENAME = f"./data/current/metadata/{MODEL_NAME}-{EXPLORER}-{NODE_COUNT}_nodes-simenergy.txt"
METADATA_FILENAME = f"./data/current/metadata/{MODEL_NAME}-{EXPLORER}-{NODE_COUNT}_nodes-metadata.pickle"


######################################################################
# Logger Details
######################################################################
LOGGER_NAME = f"{MODEL_NAME}-{EXPLORER}-{NODE_COUNT}-exploration"
LOGGER = get_logger(LOGGER_NAME, log_level="info")


######################################################################
# RRT Params
######################################################################
RRT_USE_CACHE = False
RRT_STAR = True if EXPLORER == "rrt_star" else False
RRT_IDFS = False
RRT_STEP_EPS = 0.5
RRT_REWIRE_RANGE = 1.0


########################################################
# Start RRT related code
########################################################
def rrt_select_params(dims):
    # Values selected from bound:
    # max(d_steer, d_rewire) < 1/6 * n ^ (1/2)
    # where n is the dimension
    limit = 1/6 * dims ** (0.5) - 0.01
    return (limit, limit)


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

    def branch_size(self):
        # Add one to account for the node itself when pruned
        return child_count(self) + 1


def child_count(node):
    num_children = len(node.children)
    for child in node.children:
        num_children += child_count(child)
    return num_children


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


def rrt_tree_filename(eps, rewire):
    return f"./data/current/cache/{MODEL_NAME}-{EXPLORER}_tree-{NODE_COUNT}-{eps}-{rewire}.pickle"


def save_tree(nodes, eps, rewire):
    filename = rrt_tree_filename(eps, rewire)
    with open(filename, "wb") as f:
        pickle.dump(nodes, f)


def load_tree_if_available(eps, rewire):
    filename = rrt_tree_filename(eps, rewire)
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except IOError:
        return None


def build_tree(dim, node_count=NODE_COUNT, rrt_star=True):
    eps, rewire = rrt_select_params(dim)
    # Grab cached results if those are available
    old_data = load_tree_if_available(eps, rewire) if RRT_USE_CACHE else None
    if old_data is not None:
        print("[INFO]: Cached tree results discovered. Using these instead of"
              " generating new tree...")
        return old_data

    start_node = Node(dim=dim)
    nodes = [start_node]

    print("Building tree with params:")
    print(f"Step Epsilon: {eps}")
    print(f"Rewire Range: {rewire}")
    print(f"Node Count: {NODE_COUNT}")

    for i in range(node_count):
        if i % 200 == 0:
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
        if rrt_star:
            for node in nodes:
                if dist(node.coord, q_new_coord) <= rewire:
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
    save_tree(nodes, eps, rewire)
    return nodes


########################################################
# Random Exploration related code
########################################################
def generate_random_samples(dim, sample_size=NODE_COUNT):
    points = []
    for _ in range(sample_size):
        points.append(np.random.uniform(low=MIN_VALUE[0:dim], high=MAX_VALUE[0:dim]))
    return points


########################################################
# General Code
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


def check_bad_sim(sim, reset=False):
    if check_collision(sim, reset=reset):
        return True
    return False


def check_collision(sim, reset=False):
    colliding = False
    for contact in sim.data.contact:
        if contact.geom1 == 0 or contact.geom2 == 0:
            break
        # This appears to be a false positive due to one
        # of the joints colliding incorrectly
        if contact.geom1 == 19 and contact.geom2 == 23:
            continue
        # print(f"CG1: {contact.geom1}, CG2: {contact.geom2}, Dist: {contact.dist}")
        # input("Enter to continue")
        if reset:
            reset_collisions(sim)
        return True
    return False


def reset_collisions(sim):
    for contact in sim.data.contact:
        if contact.geom1 == 0 or contact.geom2 == 0:
            break
        contact.geom1 = 0
        contact.geom2 = 0


# TODO: Base these off of norms instead of individual values?
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


# Simulation exception for collision
class CollisionException(Exception):
    pass


# Simulation exception for settling time
class SettlingTimeException(Exception):
    pass


def find_q_and_g_at_node(sim, q_goal, pd, tau_prev=None, viewer=None):
    sim_dist = 0
    sim_steps = 0
    energy = 0
    qc = None
    tau_pd = None
    q_prev = sim.data.qpos.copy()
    for _ in range(MAX_SIM_LENGTH):
        qv = sim.data.qvel
        tau_pd = pd(q_goal, q_prev, qv)
        if USE_TAU_PREV and tau_prev is not None:
            tau_pd += tau_prev
        adjust_model(sim.data.ctrl, tau_pd)
        sim.step()
        if viewer is not None:
            viewer.render()

        # Calculate distance moved in current timestep
        # and then update previous configuration
        qc = sim.data.qpos.copy()
        sim_dist += norm(qc - q_prev)
        sim_steps += 1
        q_prev = qc
        energy += np.dot(tau_pd, qv)

        if check_bad_sim(sim) or check_end_of_sim(sim):
            break

    if check_bad_sim(sim, reset=True):
        raise CollisionException("Bad simulation - Contact occurred")
    elif not check_end_of_sim(sim):
        raise SettlingTimeException("Bad simulation - Settling time exceeded max")

    # Return the actual q value we stopped at and the tau value
    # needed to support that point under gravity
    # Also return total distance traveled
    return qc, tau_pd, sim_dist, sim_steps, energy


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


def print_stats(sim_steps, node_num=0):
    print(f"\n-----------------Run {node_num}---------------------")
    print(f"Average travel dist: {np.average(sim_steps)}")
    print(f"Std Dev travel dist: {np.std(sim_steps)}")
    print(f"Total travel dist: {np.sum(sim_steps)}")
    print("---------------------------------------------\n")


def rrt_explore(sim, pd, rrt_star=False, viewer=None):
    nq = sim.data.qpos.shape[0]
    metadata = {
        'collision_count': 0,
        'max_nodes_removed_during_collision': 0,
        'total_nodes_removed_during_collision': 0,
        'timeouts': 0,
    }

    # Build tree of points
    print(f"Building tree with {NODE_COUNT} nodes...")
    nodes = build_tree(nq, rrt_star=rrt_star)
    print("Finished building tree.")
    current_node = nodes[0] # Start with root node - always at 0
    # Go ahead and find the gravity of our first node
    print("Discovering info on initial node...")
    q, tau, dist, steps, energy = find_q_and_g_at_node(sim, current_node.coord, pd)
    current_node.coord = q
    current_node.tau = tau

    collected_data = [[current_node.coord, current_node.tau]]
    sim_dist = [dist]
    sim_steps = [steps]
    sim_energy = [energy]
    print("Beginning exploration through entire tree...")

    while True:
        # First check if we need the tau at the current node
        # If so, find it
        if current_node.tau is None:
            tau_prev = find_closest_g(current_node.coord, collected_data)
            try:
                q, t, dist, steps, energy = find_q_and_g_at_node(sim, current_node.coord, pd, tau_prev, viewer=viewer)
            except CollisionException as e:
                print("Collision Occurred!")
                branch_size_pruned = current_node.branch_size()
                metadata['collision_count'] += 1
                metadata['max_nodes_removed_during_collision'] = max(metadata['max_nodes_removed_during_collision'], branch_size_pruned)
                metadata['total_nodes_removed_during_collision'] += branch_size_pruned
                current_node = current_node.parent
                # If the node we just had wasn't a leaf, we can get stuck in a loop
                # So remove the offending child if it hasn't been already
                if len(current_node.children) > 0 and not current_node.children[-1].is_leaf():
                    current_node.children.pop()
                reset_sim(sim, qpos=current_node.coord)
                continue
            except SettlingTimeException as e:
                print("Settling time error!")
                metadata['timeouts'] += 1
                current_node.tau = 0 # Make it not None, but don't give it a real value
                continue
            current_node.tau = t
            collected_data.append([q,t])
            sim_dist.append(dist)
            sim_steps.append(steps)
            sim_energy.append(energy)

            num_samples_collected = len(collected_data)
            if num_samples_collected % 100 == 0:
                print_info(current_node)
                print_stats(sim_dist, num_samples_collected)

        # Now check whether or not we're a leaf node. If we're not, keep
        # exploring up the tree
        if len(current_node.children) > 0:
            # Add some randomness to which branch we select. This way we
            # explore all throughout the tree
            # Was formerly -1 for last child
            # next_child = randint(0, len(current_node.children) - 1)
            next_child = -1
            if current_node.children[next_child].is_leaf():
                current_node = current_node.children.pop(next_child)
            else:
                current_node = current_node.children[next_child]
        else:
            # If we are at a leaf node, explore and then move back up
            # Iterate back to root node (Iterative depth first search)
            if current_node.is_root():
                break

            # Here we have two options.
            # 1) Depth first search (RRT_IDFS = False)
            # 2) Iterative Depth first search(RRT_IDFS = True)
            if RRT_IDFS:
                while not current_node.is_root():
                    current_node = current_node.parent
            else:
                current_node = current_node.parent
            # Reset simulation to 0v
            # reset_sim(sim)

    # Once finished, convert our array of tuples into matrix and save values
    collected_data = np.array(collected_data)
    sim_dist = np.array(sim_dist)
    sim_steps = np.array(sim_steps)

    assert len(collected_data) == len(sim_dist) == len(sim_steps) == len(sim_energy)

    return collected_data, sim_dist, sim_steps, sim_energy, metadata


def sequential_explore(sim, pd, viewer=None):
    nq = sim.data.qpos.shape[0]
    metadata = {
        'collision_count': 0,
        'timeouts': 0,
    }

    # initial_data = load_data_if_available()
    initial_data = generate_random_samples(nq)

    start_point = np.zeros(nq)
    qstart, tau_start, dist, steps, energy = find_q_and_g_at_node(sim, start_point, pd)
    collected_data = [(qstart, tau_start)]
    sim_dists = [dist]
    sim_steps = [steps]
    sim_energy = [energy]

    # Sequentially go through random list of points and record information
    for i, qpos in enumerate(initial_data):
        tau_prev = find_closest_g(qpos, collected_data)
        try:
            q, t, dist, steps, energy = find_q_and_g_at_node(sim, qpos, pd, tau_prev)
        except CollisionException as e:
            print("Collision Occurred!")
            metadata['collision_count'] += 1
            reset_sim(sim)
            continue
        except SettlingTimeException as e:
            print("Settling time error!")
            metadata['timeouts'] += 1
            continue
        collected_data.append((q,t))
        sim_dists.append(dist)
        sim_steps.append(steps)
        sim_energy.append(energy)

        if i != 0 and (i + 1) % 100 == 0:
            print_stats(sim_dists, i + 1)

        if len(collected_data) >= NODE_COUNT:
            break

    # Once finished, convert our array of tuples into matrix and save values
    collected_data = np.array(collected_data)
    sim_dists = np.array(sim_dists)
    sim_steps = np.array(sim_steps)

    assert len(collected_data) == len(sim_dists) == len(sim_steps) == len(sim_energy)

    return collected_data, sim_dists, sim_steps, sim_energy, metadata


def main():
    xml = None
    with open(MODEL_XML_PATH, "r") as xml_file:
        xml = xml_file.read()

    model = load_model_from_xml(xml)
    sim = MjSim(model)

    nq = sim.data.qpos.shape[0]
    pd = PDController(165, 35, nq)

    if EXPLORER == "rrt" or EXPLORER == "rrt_star":
        rrt_star = True if EXPLORER == "rrt_star" else False
        collected_data, sim_dist, sim_steps, sim_energy, md = rrt_explore(sim, pd, rrt_star=rrt_star)
    elif EXPLORER == "random":
        collected_data, sim_dist, sim_steps, sim_energy, md = sequential_explore(sim, pd)
    else:
        print(f"Unknown explorer {EXPLORER}. Must be one of (rrt | rrt_star | random)")
        sys.exit(1)
   
    np.savetxt(QPOS_FILENAME, collected_data[:,0])
    np.savetxt(TAUS_FILENAME, collected_data[:,1])
    np.savetxt(DIST_FILENAME, sim_dist)
    np.savetxt(STEPS_FILENAME, sim_steps)
    np.savetxt(ENERGY_FILENAME, sim_energy)
    with open(METADATA_FILENAME, "wb") as f:
        pickle.dump(md,f)


if __name__ == '__main__':
    main()
