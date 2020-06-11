import os.path as path

from mujoco_py import load_model_from_xml, MjSim
import numpy as np
import torch as T
import torch.nn as nn

BATCH_SIZE = 100
DEVICE = 'cuda' if T.cuda.is_available() else 'cpu'
NN_MODEL_PATH = "./data/7dof-panda-robot-noee.pt"
NN_MODEL_CHECKPOINT_PATH = "./data/7dof-panda-robot-noee.chkpnt"
TAU_DATA_PATH = "./data/cleaned_tau_actuals.txt"
QPOS_DATA_PATH = "./data/cleaned_qpos.txt"
MODEL_XML_PATH = "./src/panda.xml"
MAX_LOSS_PATH = "./data/max-loss.txt"
AVG_LOSS_PATH = "./data/avg-loss.txt"
MAX_VAR_PATH = "./data/max-var.txt"


def gen_nn(inputs, outputs, hidden=100):
    net = nn.Sequential(nn.Linear(inputs, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, outputs)).double().to(DEVICE)
    optimizer = T.optim.Adam(net.parameters(), lr=4e-3)
    loss_fn = T.nn.MSELoss(reduction='none')
    return net, optimizer, loss_fn


def load_nn_if_avaiable(net, optimizer):
    if path.exists(NN_MODEL_PATH):
        checkpoint = T.load(NN_MODEL_PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print(f"[WARNING]: Unable to load NN checkpoint from path {NN_MODEL_PATH}")


def load_data():
    tau_actuals = None
    qpos = None

    if path.exists(TAU_DATA_PATH) and path.exists(QPOS_DATA_PATH):
        tau_actuals = np.loadtxt(TAU_DATA_PATH)
        qpos = np.loadtxt(QPOS_DATA_PATH)
    else:
        raise Exception(f"[ERROR]: Unable to load previous data files.")

    return tau_actuals, qpos


def main():
    xml = None
    with open(MODEL_XML_PATH, "r") as f:
        xml = f.read()

    model = load_model_from_xml(xml)
    sim = MjSim(model)

    nq = sim.data.qpos.shape[0]
    nt = sim.data.ctrl.shape[0]

    net, optimizer, loss_fn = gen_nn(nq, nt)
    # load_nn_if_avaiable(net, optimizer)

    tau_finals, qpos_finals = load_data()
    iters = len(qpos_finals) // BATCH_SIZE

    max_loss = []
    max_var = []
    avg_loss = []

    for i in range(iters):
        qpos_final_tensor = T.Tensor(qpos_finals[:(i+1)*BATCH_SIZE]).double().to(DEVICE)
        tau_finals_tensor = T.Tensor(tau_finals[:(i+1)*BATCH_SIZE]).double().to(DEVICE)
        tau_predicted = net(qpos_final_tensor).to(DEVICE)

        optimizer.zero_grad()
        loss = loss_fn(tau_predicted, tau_finals_tensor)

        # Take loss statistics for mean and variance of current batch
        max_loss.append(loss.max().item())
        max_var.append(loss.var(dim=1).max().item())

        # Calculate actual loss and backprop
        loss = loss.mean()
        avg_loss.append(loss.item())
        print(f"Loss after {(i+1) * BATCH_SIZE} samples: {loss}")
        loss.backward()
        optimizer.step()

        if i > 0 and i % BATCH_SIZE == 0:
            print(f"Saving checkpoint and model info at iteration {i}")
            np.savetxt(AVG_LOSS_PATH, np.array(avg_loss))
            np.savetxt(MAX_LOSS_PATH, np.array(max_loss))
            np.savetxt(MAX_VAR_PATH, np.array(max_var))

            T.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, NN_MODEL_CHECKPOINT_PATH)

            T.save(net, NN_MODEL_PATH)


if __name__ == '__main__':
    main()
