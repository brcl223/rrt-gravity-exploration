#!/usr/bin/env python3
"""
Example of how bodies interact with each other. For a body to be able to
move it needs to have joints. In this example, the "robot" is a red ball
with X and Y slide joints (and a Z slide joint that isn't controlled).
On the floor, there's a cylinder with X and Y slide joints, so it can
be pushed around with the robot. There's also a box without joints. Since
the box doesn't have joints, it's fixed and can't be pushed around.
"""
import math
import os
import random

from mujoco_py import load_model_from_xml, MjSim, MjViewer

# MODEL_XML_PATH =
# "./robosuite/robosuite/models/assets/robots/sawyer/robot_torque.xml"
# MODEL_XML_PATH = "./panda.xml"
MODEL_XML_PATH = "./src/three-dof.xml"
xml = None
model = None
with open(MODEL_XML_PATH, "r") as f:
    xml = f.read()

model = load_model_from_xml(xml)
print(f"Type model: {type(model)}")

sim = MjSim(model)
viewer = MjViewer(sim)
t = 0

nq = sim.data.qpos.shape[0]


while True:
    if t % 10000 == 0:
        for i in range(nq):
            sim.data.qpos[i] = 0 ##random.random() * 2 * math.pi
            sim.data.qvel[i] = 0
            sim.data.qacc[i] = 0
            sim.data.ctrl[i] = 1000
    t += 1
    sim.step()
    viewer.render()
    print(f"Number of contacts: {sim.data.ncon}")
    if t > 100 and os.getenv('TESTING') is not None:
        break
