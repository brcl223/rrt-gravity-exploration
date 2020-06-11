#!/usr/bin/env python3

import math
import pickle
import os.path as path
import random
import sys

from mujoco_py import load_model_from_xml, MjSim, MjViewer
import numpy as np
from numpy.linalg import norm

from random import randint

# Check system arguments for model name
if len(sys.argv) != 3:
    print("Wrong number of arguments supplied. Requires:")
    print("Model Name: panda-*dof")
    print("Desired node count: (positive integer value)")
    sys.exit(1)

######################################################################
# Generic Params
######################################################################
MODEL_NAME = sys.argv[1]
MODEL_XML_PATH = f"./src/models/{MODEL_NAME}.xml"
MAX_VALUE = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
MIN_VALUE = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
MAX_SIM_LENGTH = 100000
VEL_EPS = 1e-5
ACC_EPS = 1e-5
USE_TAU_PREV = True
DESIRED_NODE_COUNT = int(sys.argv[2])
COLLISION_FILE_PATH = "./data/current/collision_data.txt"


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



class PDController:
    def __init__(self, kp, kd, dim):
        # self.KP = np.diag([150,75,40,20,10,3,0.1]) * kp
        # self.KD = np.diag([150,75,40,20,10,3,0.1]) * kd
        self.KP = np.identity(dim) * kp
        self.KD = np.identity(dim) * kd

    def __call__(self, qd, q, qdot):
        return (self.KP @ (qd - q)) - (self.KD @ qdot)

def check_collision(sim):
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
        colliding = True
        break
    reset_collisions(sim)
    return colliding


def reset_collisions(sim):
    for contact in sim.data.contact:
        if contact.geom1 == 0 or contact.geom2 == 0:
            break
        contact.geom1 = 0
        contact.geom2 = 0


def generate_random_samples(dim, sample_size=DESIRED_NODE_COUNT):
    points = []
    zero_idx = [2,4,5,6]
    for _ in range(sample_size):
        rand_samp = np.random.uniform(low=MIN_VALUE[0:dim], high=MAX_VALUE[0:dim])
        rand_samp[zero_idx] = 0
        points.append(rand_samp)
    return points


collision_points = []

xml = None
with open(MODEL_XML_PATH, "r") as xml_file:
    xml = xml_file.read()

model = load_model_from_xml(xml)
sim = MjSim(model)

reset_sim(sim)
for sample in generate_random_samples(7):
    reset_sim(sim, qpos=sample)
    sim.step()
    if check_collision(sim):
        # print("Found collision!")
        collision_points.append(sample.copy())

print(f"Number of collisions: {len(collision_points)}")
np.savetxt(COLLISION_FILE_PATH, np.array(collision_points))
