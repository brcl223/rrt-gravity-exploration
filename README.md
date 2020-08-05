# *A Self-Supervised Learning Approach to Gravity Compensation Approximation on a Robotic Manipulator* Codebase

## Dependencies
All of the following code was exclusively tested on Ubuntu 18.04.
However, with the correct packages installed, it should run on any *nix system.

A valid MuJoCo 2.0 license file and installation is required to run the code in this codebase.
While mujoco-py is included, it is simply a wrapper for the actual MuJoCo physics engine, and cannot run without it.

In order to run the code, the following Python 3 libraries are required:

* [Poetry](https://github.com/python-poetry/poetry) - Package Management
* [PyTorch](https://github.com/pytorch/pytorch) - Deep Neural Network Library
* [Numpy](https://github.com/numpy/numpy) - Scientific Computation Library
* [mujoco-py](https://github.com/openai/mujoco-py) - MuJoCo Python 3 Wrapper Library
* [Matplotlib](https://github.com/matplotlib/matplotlib) - Python Graphing Library

It is recommended to use Poetry to install the packages.
A pyproject.toml file is included in the root directory, which should make package installation simple with Poetry.
However, manually installing the libraries may work as well.
All instructions following will assume Poetry has been used to install and manage the project dependencies.

## Initial Setup
After pulling down the code but before running, complete the following steps:

* `cd data`
* `./backup_data`

This will ensure that the proper data directories are present when running the scripts.

## File Descriptions and Instructions
The following files are listed in order in which they should be run.
Each file may depend on results produced from a previous script.
Please run these in order as listed.

### src/explore.py
* Description: Main file used for collecting joint configuration/torque data. Three variants are available to run: Random, RRT, RRT\*.
* Running Instructions: `poetry run python src/explore.py <model> <explorer> <iterations>`

### src/grav_point_cleaner.py (optional)
* Description: Verifies collected data points are valid (non-self-intersecting, joint velocities/accelerations valid, etc.). 
Not strictly necessary, but may help produce better results, as sometimes MuJoCo does not accurately report collisions.
* Running Instructions: `poetry run python src/grav_point_cleaner.py <model> <explorer> <iterations>`

### src/grav_explore_trainer.py
* Description: Trains deep neural network based on previously collected data samples.
* Running Instructions: `poetry run python src/grav_explore_trainer.py <model> <explorer> <iterations>`

### src/test_nn_manual.py (optional)
* Description: Simple helper script to display neural network accuracy/error.
* Running Instructions: `poetry run python src/test_nn_manual <model> <explorer> <iterations>`

### Command Line Arguments
In order to run the scripts, the following command line arguments are required:
* `<model>` - (panda-2dof, panda-3dof, panda-4dof, panda-5dof, panda-6dof, panda-7dof)
* `<explorer>` - (rrt, rrt_star, random)
* `<iterations>` - Any positive integer value

## Acknowledgements

The Panda Robot XML file used in the `src/models` folder was modified from the following source: https://github.com/StanfordVL/robosuite.
