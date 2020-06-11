#!/usr/bin/env python3

import numpy as np
import sys

if len(sys.argv) != 3:
    print("Expected 2 arguments: <model name> and <num nodes>")
    print("Exiting...")
    sys.exit(1)


MODEL_NAME = sys.argv[1]
MODEL_XML_PATH = f"./src/models/{MODEL_NAME}.xml"
DESIRED_NODE_COUNT = 100
MAX_VALUE = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
MIN_VALUE = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
TSP_IMPROV_THRESHOLD = 0.001
RANDOM_QPOS_FILENAME = f"./data/current/{MODEL_NAME}-random-qpos.txt"


def generate_random_samples(dim, sample_size=DESIRED_NODE_COUNT):
    return np.random.uniform(low=MIN_VALUE[0:dim], high=MAX_VALUE[0:dim], size=(sample_size, dim))


# Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
path_distance = lambda r,c: np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])
# Reverse the order of all elements from element i to element k in array r.
two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))

# 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
def two_opt(cities,improvement_threshold):
    # Make an array of row numbers corresponding to cities.
    route = np.arange(cities.shape[0])
    # Initialize the improvement factor.
    improvement_factor = 1
    # Calculate the distance of the initial path.
    best_distance = path_distance(route,cities)
    # If the route is still improving, keep going!
    while improvement_factor > improvement_threshold:
        # Record the distance at the beginning of the loop.
        distance_to_beat = best_distance
        # From each city except the first and last,
        for swap_first in range(1,len(route)-2):
            # to each of the cities following,
            for swap_last in range(swap_first+1,len(route)):
                # try reversing the order of these cities
                new_route = two_opt_swap(route,swap_first,swap_last)
                # and check the total distance with this modification.
                new_distance = path_distance(new_route,cities)
                # If the path distance is an improvement,
                if new_distance < best_distance:
                    # make this the accepted best route
                    route = new_route
                    # and update the distance corresponding to this route.
                    best_distance = new_distance
                    # Calculate how much the route has improved.
        improvement_factor = 1 - best_distance/distance_to_beat
    # When the route is no longer improving substantially, stop searching and return the route.
    return route

samples = generate_random_samples(3)
original_route = np.arange(samples.shape[0])
print(f"Original distance: {path_distance(original_route, samples)}")

route = two_opt(samples, TSP_IMPROV_THRESHOLD)

pdist = path_distance(route, samples)
print(f"New distance: {pdist}")
print(f"Values:\n{samples[route]}")
