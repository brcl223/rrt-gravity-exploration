import os.path as path
import sys

from mujoco_py import load_model_from_xml, MjSim
import numpy as np
import torch as T

from shared import gen_nn, get_logger, load_nn, save_nn

if len(sys.argv) != 4:
    print("Wrong number of arguments supplied. Requires <model name>, <explorer> and <num nodes>")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
EXPLORER_NAME = sys.argv[2]
NODE_COUNT = sys.argv[3]
EPOCHS = 15
BATCH_SIZE = 500
K_FOLDS = 5
DEVICE = 'cuda' if T.cuda.is_available() else 'cpu'
NN_MODEL_CHECKPOINT_PATH = f"./data/current/weights-trained/{MODEL_NAME}-{EXPLORER_NAME}-{NODE_COUNT}_nodes-model_IDX.chkpnt"
NN_LEARNING_RATE = 4e-3
TAU_DATA_PATH = f"./data/current/data-cleaned/{MODEL_NAME}-{EXPLORER_NAME}-{NODE_COUNT}_nodes-cleaned_taus.txt"
QPOS_DATA_PATH = f"./data/current/data-cleaned/{MODEL_NAME}-{EXPLORER_NAME}-{NODE_COUNT}_nodes-cleaned_qpos.txt"
MODEL_XML_PATH = f"./src/models/{MODEL_NAME}.xml"
AVG_LOSS_PATH = f"./data/current/data-nn/{MODEL_NAME}-{EXPLORER_NAME}-{NODE_COUNT}_nodes-avg_loss.txt"
AVG_VAR_PATH = f"./data/current/data-nn/{MODEL_NAME}-{EXPLORER_NAME}-{NODE_COUNT}_nodes-avg_var.txt"
MAX_LOSS_PATH = f"./data/current/data-nn/{MODEL_NAME}-{EXPLORER_NAME}-{NODE_COUNT}_nodes-max_loss.txt"
MAX_VAR_PATH = f"./data/current/data-nn/{MODEL_NAME}-{EXPLORER_NAME}-{NODE_COUNT}_nodes-max_var.txt"
NUM_SHUFFLES = 100

###########################################
# Logging related utilities
###########################################
LOGGER_NAME = f"{MODEL_NAME}-{EXPLORER_NAME}-{NODE_COUNT}-nn-training"
LOGGER = get_logger(LOGGER_NAME, log_level="info")


def model_save_path(i):
    return NN_MODEL_CHECKPOINT_PATH.replace("IDX", str(i))


def load_data():
    if path.exists(TAU_DATA_PATH) and path.exists(QPOS_DATA_PATH):
        taus = np.loadtxt(TAU_DATA_PATH)
        qpos = np.loadtxt(QPOS_DATA_PATH)
        return taus, qpos
    else:
        LOGGER.fatal("Unable to load previous data files.")
        raise Exception(f"[ERROR]: Failed to load previous data.")


def load_initial_nns(inputs, outputs, num_weights=5):
    nets = []
    optims = []
    loss_fn = None

    for i in range(num_weights):
        net, optim, loss = gen_nn(inputs, outputs, lr=NN_LEARNING_RATE, device=DEVICE)
        nets.append(net)
        optims.append(optim)
        loss_fn = loss_fn if loss_fn else loss

    base_path = f"./data/current/weights-initial/{MODEL_NAME}_IDX.chkpnt"
    initial_weights_found = True
    for i in range(num_weights):
        cur_path = base_path.replace("IDX", str(i))
        if not path.exists(cur_path):
            initial_weights_found = False
            break

    if initial_weights_found:
        LOGGER.info("Found initial weights files.")
        for i in range(num_weights):
            cur_path = base_path.replace("IDX", str(i))
            load_nn(cur_path, nets[i], optims[i])
    else:
        LOGGER.warning(f"No initial weights found for {MODEL_NAME}. Creating initial weights.")
        for i in range(num_weights):
            cur_path = base_path.replace("IDX", str(i))
            save_nn(cur_path, nets[i], optims[i])

    return nets, optims, loss_fn


def split_80_20(dataset, kfold=None):
    dataset = np.array(dataset)
    if kfold is True:
        return np.array_split(dataset, 5)
    else:
        split = [int(.8*len(dataset))]
        return np.split(dataset, split)


# Since shuffling via permutations, we need to make sure we have enough shuffles to be "random"
# See the following resource: https://www.dartmouth.edu/~chance/teaching_aids/Mann.pdf
def shuffle_unison(a, b, shuffles=NUM_SHUFFLES):
    assert len(a) == len(b)
    for i in range(shuffles):
        p = np.random.permutation(len(a))
        a, b = a[p], b[p]
    return a, b


def predict(net, optimizer, loss_fn, qpos, taus, backprop=False):
    qpos_tensor = T.Tensor(qpos).double().to(DEVICE)
    taus_tensor = T.Tensor(taus).double().to(DEVICE)

    tau_predicted = net(qpos_tensor).to(DEVICE)

    loss = loss_fn(tau_predicted, taus_tensor)

    max_loss = loss.max().item()
    max_var = loss.var(dim=1).max().item()
    avg_var = loss.var(dim=1).mean().item()
    avg_loss = loss.mean().item()

    # Backprop results from training
    if backprop:
        optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        optimizer.step()

    return avg_loss, avg_var, max_loss, max_var


def train_network(net, optimizer, loss_fn, taus, qpos):
    # Shuffle data as RRT/RRT* exploration won't be randomly collected
    tf_shuf, qpos_shuf = shuffle_unison(taus, qpos)

    # First create 80% train / 20% test split
    train_qpos, test_qpos = split_80_20(qpos_shuf)
    train_taus, test_taus = split_80_20(tf_shuf)

    assert train_qpos.shape == train_taus.shape
    assert test_qpos.shape == test_taus.shape

    max_loss = []
    max_var = []
    avg_loss = []
    avg_var = []

    batches_per_epoch = len(train_qpos) // BATCH_SIZE
    epoch_iters = batches_per_epoch * EPOCHS

    LOGGER.debug("[INFO] Run Stats:")
    LOGGER.debug(f"Training data set shape: {train_qpos.shape}")
    LOGGER.debug(f"Testing data set shape: {test_qpos.shape}")
    LOGGER.debug(f"Batches/Epoch: {batches_per_epoch}")
    LOGGER.debug(f"Number of Epochs: {EPOCHS}")
    LOGGER.debug(f"Epoch Iterations: {epoch_iters}")

    for i in range(epoch_iters):
        start_idx = (i % batches_per_epoch) * BATCH_SIZE
        # print(f"Current Range: {start_idx} to {start_idx + BATCH_SIZE - 1}")
        # Get current batch
        train_qpos_split = train_qpos[start_idx:start_idx + BATCH_SIZE]
        train_taus_split = train_taus[start_idx:start_idx + BATCH_SIZE]
        # Split batch into training and validation sets
        train_qpos_split = np.array(split_80_20(train_qpos_split, kfold=True))
        train_taus_split = np.array(split_80_20(train_taus_split, kfold=True))
        for j in range(K_FOLDS):
            # Begin K-Fold cross validation setup
            train_qpos_cur = np.concatenate(train_qpos_split[[y for y in range(K_FOLDS) if y != j]])
            train_taus_cur = np.concatenate(train_taus_split[[y for y in range(K_FOLDS) if y != j]])
            val_qpos = train_qpos_split[j]
            val_taus = train_taus_split[j]

            assert train_qpos_cur.shape == train_taus_cur.shape
            assert val_qpos.shape == val_taus.shape

            # First train on training batch
            train_stats = predict(net, optimizer, loss_fn, train_qpos_cur, train_taus_cur, backprop=True)

            # Now run through validation set and testing data
            val_stats = predict(net, optimizer, loss_fn, val_qpos, val_taus, backprop=False)
            test_stats = predict(net, optimizer, loss_fn, test_qpos, test_taus, backprop=False)

            avg_loss.append([train_stats[0], val_stats[0], test_stats[0]])
            avg_var.append([train_stats[1], val_stats[1], test_stats[1]])
            max_loss.append([train_stats[2], val_stats[2], test_stats[2]])
            max_var.append([train_stats[3], val_stats[3], test_stats[3]])

        if i > 0 and i % 25 == 0:
            avg_loss_stats = np.array(avg_loss)
            LOGGER.debug("----------------Stats---------------")
            LOGGER.debug(f"Avg Loss Train: {np.average(avg_loss_stats[-5:-1,0])}")
            LOGGER.debug(f"Avg Loss Val: {np.average(avg_loss_stats[-5:-1,1])}")
            LOGGER.debug(f"Avg Loss Test: {np.average(avg_loss_stats[-5:-1,2])}")
            LOGGER.debug("------------------------------------\n")

        if ((i + 1) % batches_per_epoch == 0):
            LOGGER.debug(f"Saving checkpoint of data at epoch {(i + 1) // batches_per_epoch}")
            np.savetxt(AVG_LOSS_PATH, np.array(avg_loss))
            np.savetxt(AVG_VAR_PATH, np.array(avg_var))
            np.savetxt(MAX_LOSS_PATH, np.array(max_loss))
            np.savetxt(MAX_VAR_PATH, np.array(max_var))


def main():
    log_msg = f"Beginning training phase:\nModel Name: {MODEL_NAME}\nExplorer: {EXPLORER_NAME}\nNode Count: {NODE_COUNT}"
    LOGGER.info(log_msg)
   
    xml = None
    with open(MODEL_XML_PATH, "r") as f:
        xml = f.read()

    model = load_model_from_xml(xml)
    sim = MjSim(model)

    nq = sim.data.qpos.shape[0]
    nt = sim.data.ctrl.shape[0]

    nets, optimizers, loss_fn = load_initial_nns(nq, nt)

    taus, qpos = load_data()
    assert taus.shape == qpos.shape

    for i, (net, optim) in enumerate(zip(nets, optimizers)):
        LOGGER.info(f"Beginning training for network {model_save_path(i)}")
        train_network(net, optim, loss_fn, taus, qpos)
        save_nn(model_save_path(i), net, optim)
        LOGGER.info(f"Model {model_save_path(i)} saved")


if __name__ == '__main__':
    main()
