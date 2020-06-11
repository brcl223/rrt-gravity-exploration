import logging
import os.path as path

import torch as T
import torch.nn as nn


def gen_nn(inputs, outputs, hidden=100, lr=4e-3, device='cuda'):
    net = nn.Sequential(nn.Linear(inputs, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, outputs)).double().to(device)
    optimizer = T.optim.Adam(net.parameters(), lr=lr)
    loss_fn = T.nn.MSELoss(reduction='none')
    return net, optimizer, loss_fn


def load_nn(model_path, net, optimizer, fail_if_unavailable=False):
    failure_message = f"Unable to load NN checkpoint from path {path}"
    if path.exists(model_path):
        checkpoint = T.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif fail_if_unavailable:
        logging.critical(failure_message)
        raise Exception("Failed to load neural network")
    else:
        logging.warning(failure_message)


def gen_and_load_nn(model_path, inputs, outputs):
    net, optimizer, loss_fn = gen_nn(inputs, outputs)
    load_nn(model_path, net, optimizer, fail_if_unavailable=True)
    return net, optimizer, loss_fn


def save_nn(model_path, net, optimizer):
    T.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)


def get_logger(logname, log_level=""):
    log_level = str_to_log_level(log_level)
    log_filename = f"./data/current/logs/{logname}.log"
    log_format = "[%(asctime)s] <%(levelname)s>: %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=log_filename, level=log_level, format=log_format, datefmt=log_datefmt)
    return logging


def str_to_log_level(level):
    levels = {
        "": logging.WARNING, # Default level
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "critical": logging.CRITICAL,
        "fatal": logging.FATAL,
    }

    return levels[level.lower()]
