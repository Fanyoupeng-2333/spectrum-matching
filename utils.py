import torch
import shutil
import numpy as np
from numpy.random import permutation
import scipy as sp
import scipy.stats
import logging
import datetime
import os

def save_checkpoint(state,seed,  is_best, filename, best_filename=None):
    """ Save checkpoints

    Args:
        state (dict): state to save
        is_best (bool): whether the state is the best so far. If True, will be
                        saved as best as well.
        filename (str): where to save the current checkpoint
        best_filename (str): where to save the current best checkpoint

    Returns:
        Return nothing

    """
    # torch.save(state, filename)
    # if is_best:
    #     if best_filename == None:
    #         best_filename = "best_"+filename
    #     shutil.copyfile(filename, best_filename)
    torch.save(state, filename)
    if is_best:
        if best_filename == None:
            best_filename = "best_"+str(seed)+'_'+filename
            print(seed,best_filename)
        shutil.copyfile(filename, best_filename)

def split_dataset(samples, labels, n_training=-1, n_test=-1, rnd_index=True):
    """Split data into training, test and validate sets.
    """
    # Sanity check. Ignore for now
    assert n_training != -1 or n_test != -1
    swap = (n_training == -1)
    if swap:
        n_subset = n_test
    else:
        n_subset = n_training
    X, y = samples, labels
    sub_indexes = []
    for i in range(np.max(y).astype(int) + 1):
        idx = np.where(y == i)
        if swap and len(idx[0]) <= 1:#If only one sample available
            continue

        if rnd_index:
            ridx = np.random.permutation(range(len(idx[0])))
        else:
            #ridx = range(len(idx[0]))[::-1]
            ridx = range(len(idx[0]))

        if 0 < n_subset and n_subset < 1:
            ridx = ridx[0:int(round(n_subset*len(idx[0])))]
        else:
            ridx = ridx[0:n_subset]

        sub_indexes.append(idx[0][ridx].tolist())

    sub_indexes = sum(sub_indexes, [])
    rest_indexes = [x for x in range(samples.shape[0])
                        if x not in sub_indexes]

    X_train = X[sub_indexes, :]
    y_train = y[sub_indexes]
    X_test = X[rest_indexes, :]
    y_test = y[rest_indexes]
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

    if not swap:
        return X_train, y_train, X_test, y_test
    else:
        return X_test, y_test, X_train, y_train

def get_logger(log_file_name):
    now = datetime.datetime.now()
    #log_file = r'.\Logs\log_' + now.strftime("%b-%d-%Y-%H%M%S") + '.txt'
    #logging.basicConfig(filename=log_file, level=logging.INFO)
    idx_log=0
    while True:
        filename = 'Logs/main{}.log'.format(idx_log)
        if not os.path.exists(filename):
            break
        else:
            idx_log += 1

    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    if not os.path.exists('./Logs'):
        os.makedirs('./Logs')

    file_handler = logging.FileHandler(filename=filename)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def torch_tensor(inputs, targets, device):
    """ Convert np.array data to tensors to fit the network
        we assume the shape of arr is 
        inputs:  [n_samples, signal_dim]
        targets: [n_samples,]
    """
    inputs = inputs.reshape(inputs.shape + (1,))
    inputs = np.transpose(inputs, axes=[0, 2, 1]).astype('float32')
    tensor_inputs = torch.from_numpy(inputs).float().to(device)
    targets = np.squeeze(targets)
    tensor_targets = torch.from_numpy(targets).long().to(device)
    return tensor_inputs, tensor_targets
