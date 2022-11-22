import numpy as np
import torch
import torch.utils.data as data
from utils import save_checkpoint
from utils import split_dataset
from utils import torch_tensor
from collections import Counter
from pair_generator import create_random_pairs
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from models.baseline import compute_accuracy 
from models.baseline import PairwiseDistances
import os
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from models.lenet import LeNet
from models.siamese import SiameseNetwork
from torch.optim import lr_scheduler

# default_logger = get_logger(log_file_name="default_log.log")

# torch.device object used throughout this script
_g_device = None

# This settings will affect all the functions below 
_g_settings = None
# logger = None

_g_lr_min_thresh = 1e-6


def set_config(settings):
    global _g_settings
    _g_settings = settings
    # global logger
    # logger = get_logger(log_file_name=_g_settings['log_dir'])
    global _g_device
    _g_device = torch.device("cuda:1" if _g_settings['cuda'] else "cpu")

def split_training_test_classes(dataset, sorting_labels=False, 
        one_shot_test_only=True):
    """Create pairs for siamese network training
    """
    X_train, y_train, X_test, y_test = \
        dataset.get_train_test_sets(splitting_scheme='leave_one_out',  #8020 leave_one_out
            n_fixed_set=-1, data_augmentation=False, class_balancing=False, 
            fixed_split_for_debugging=True, show_sample_status=False,
            n_pow_per_sample=10, n_mix_per_sample=10, n_shift_per_sample=10,
            max_wave_shift=5, n_mag_per_sample=0, max_mag_noise=3)

    n_train = X_train.shape[0]
    n_features = X_train.shape[1]
    n_test = X_test.shape[0]
    n_classes = dataset.n_classes
    
    # For pure one shot learning, we split classes along with all samples 
    # belonging to them. While for multi-class classification, we do 
    # leave-one-out first to produce test samples, then split classes into 
    # training, validation and test for one shot learning.
    if one_shot_test_only:
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)    
    else:
        X, y = X_train, y_train 

    # Some classes for training, some for testing
    n_classes = dataset.n_classes
    class_labels = np.random.permutation(n_classes)
    print(class_labels[0:30])
    print(class_labels[120:140])

    if sorting_labels:
        num_spectra_per_class = []
        for i in range(n_classes):
            xi = dataset.get_spectra(i)
            num_spectra_per_class.append(xi.shape[0])

        sorted_labels = np.argsort(num_spectra_per_class)
        class_labels = sorted_labels

    train_rate, valid_rate = 0.5, 0.1
    n_train_class = int(np.round(train_rate*n_classes))
    n_valid_class = int(np.round(valid_rate*n_classes))
    n_test_class = n_classes - n_train_class - n_valid_class
    # n_test_class = 50 #30 #0
    # n_valid_class = 12  #30
    # n_train_class = n_classes - n_valid_class - n_test_class

    train_labels = class_labels[0:n_train_class]
    valid_labels = class_labels[n_train_class:n_train_class+n_valid_class]
    # valid_labels = class_labels[0:n_valid_class]
    # train_labels = class_labels[n_valid_class:n_train_class+n_valid_class]
    test_labels = class_labels[n_train_class+n_valid_class:
        n_train_class+n_test_class+n_valid_class]

    X_tr = np.array([]).reshape(0, X.shape[1])
    y_tr = np.array([]).reshape(0, 1)
    for l in train_labels:
        idx = np.where(y == l)[0]
        X_tr = np.vstack([X_tr, X[idx]])
        y_tr = np.vstack([y_tr, y[idx]])

    X_va = np.array([]).reshape(0, X.shape[1])
    y_va = np.array([]).reshape(0, 1)
    for l in valid_labels:
        idx = np.where(y == l)[0]
        X_va = np.vstack([X_va, X[idx]])
        y_va = np.vstack([y_va, y[idx]])

    X_te = np.array([]).reshape(0, X.shape[1])
    y_te = np.array([]).reshape(0, 1)
    for l in test_labels:
        idx = np.where(y == l)[0]
        if len(idx) >= 1:
            X_te = np.vstack([X_te, X[idx]])
            y_te = np.vstack([y_te, y[idx]])
    print(y_tr.shape,y_va.shape,y_te.shape)
    splitted_data = {
        'train_input':  X_tr, 
        'train_target': y_tr,
        'valid_input':  X_va, 
        'valid_target': y_va,
        'test_input':   X_te, 
        'test_target':  y_te,
        'loo_input':    X_test, 
        'loo_target':   y_test 
    }
    return splitted_data

def train(splitted_data):
    # net, criterion, optimizer etc
    net = _g_settings['model']    
    models_dir = _g_settings['models_dir']
    logger = _g_settings['logger']
    use_cuda = _g_settings['cuda']
    epochs = _g_settings['epochs']
    dataset = _g_settings['dataset']
    batch_size = _g_settings['batch_size']
    data_augmentation = _g_settings['data_augmentation']
    num_random_pairs_per_sample = _g_settings['num_random_pairs_per_sample']
    num_ways = _g_settings['num_ways'] 
    num_shots = _g_settings['num_shots'] 
    num_episodes = _g_settings['num_episodes'] 
    num_queries = _g_settings['num_queries']
    num_episodes_halve_lr = _g_settings['num_episodes_halve_lr']
    num_episodes_per_epoch = _g_settings['num_episodes_per_epoch']
    verbose = _g_settings['verbose']    

    net = net.to(_g_device).float()    
    criterion = net.loss
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #6

    # splitted_data = load_create_sample_pairs(dataset)
    X_tr, y_tr = splitted_data['train_input'], splitted_data['train_target']
    X_va, y_va = splitted_data['valid_input'], splitted_data['valid_target']
    X_te, y_te = splitted_data['test_input'], splitted_data['test_target']
    

    # Augmentation of the training set
    if data_augmentation:
        X_tr_aug, y_tr_aug = dataset.augument_data(X=X_tr, y=y_tr,
            n_pow_per_sample=10, n_mix_per_sample=10, n_shift_per_sample=10,
            max_wave_shift=5, n_mag_per_sample=10, max_mag_noise=3, 
            class_balancing=False)
        X_tr = np.concatenate((X_tr, X_tr_aug), axis=0)
        y_tr = np.concatenate((y_tr, y_tr_aug), axis=0)

        # X_va_aug, y_va_aug = dataset.augument_data(X=X_va, y=y_va,
        #     n_pow_per_sample=5, n_mix_per_sample=0, n_shift_per_sample=5,
        #     max_wave_shift=5, n_mag_per_sample=5, max_mag_noise=3, 
        #     class_balancing=False)
        # X_va = np.concatenate((X_va, X_va_aug), axis=0)
        # y_va = np.concatenate((y_va, y_va_aug), axis=0)

    # Convert input data into 3D tensors
    X_tr, y_tr = torch_tensor(X_tr, y_tr, _g_device)
    X_va, y_va = torch_tensor(X_va, y_va, _g_device)
    X_te, y_te = torch_tensor(X_te, y_te, _g_device)
    print('#Training_Samples: {0}, #Valid_Samples: {1}, #Test_Samples: {2}'.
        format(y_tr.size()[0], y_va.size()[0], y_te.size()[0]))

    # Main loop 
    best_valid_loss = float('inf')
    train_loss_epochs = []
    valid_loss_epochs = []  
    net.train()
    for epoch in range(epochs):     
        # Update the learning rate. 
        # Reduce lr by half every 10 epochs
        scheduler.step()

        # Training 
        train_loss = _run_net_one_epoch(net, optimizer, criterion, [X_tr, y_tr], 
            batch_size, trainable=True, shuffle=True, num_random_pairs=
            num_random_pairs_per_sample)
        train_loss_epochs += [np.mean(train_loss)]
        print(epoch,epoch,epoch)
        # Validation
        with torch.no_grad():
            net.eval()
            valid_loss = _run_net_one_epoch(net, optimizer, criterion, 
                [X_va, y_va], batch_size, trainable=False, shuffle=True,
                num_random_pairs=num_random_pairs_per_sample)
            valid_loss_epochs += [np.mean(valid_loss)]
            net.train()

            is_better = np.mean(valid_loss) < best_valid_loss
            if is_better:
                best_valid_loss = np.mean(valid_loss)
            # save_checkpoint({
            #     'epoch': iter,
            #     'net': net,
            #     'best_valid_loss': np.mean(valid_loss),
            #     'optimizer' : optimizer.state_dict(),
            # }, is_better, filename=os.path.join(models_dir, 'net.tar'),
            # best_filename=os.path.join(models_dir, 'best_net.tar'))
            save_checkpoint({
                'epoch': iter,
                'net': net,
                'best_valid_loss': np.mean(valid_loss),
                'optimizer' : optimizer.state_dict(),
            }, _g_settings['seed'], is_better, filename=os.path.join(models_dir, 'net.tar'),
            best_filename=os.path.join(models_dir, 'best_net_'+str(_g_settings['seed'])+'.tar'))

        # Logging 
        if verbose:
            num_seen_pairs = X_tr.shape[0]*2*num_random_pairs_per_sample*(epoch+1)
            logger.info('Epoch: {}, Train_Loss: {:.3e}, Valid_Loss: {:.3e} '
                'LR: {:.3e} #Seen_Episodes: {}'.format(epoch, np.mean(train_loss), 
                np.mean(valid_loss),optimizer.param_groups[0]['lr'], num_seen_pairs))
        
        # torch.save(net, os.path.join(models_dir, 'net_' + str(epoch) + '.tar'))
        np.save(os.path.join(models_dir,'val_loss'),np.array(valid_loss_epochs))
        np.save(os.path.join(models_dir,'train_loss'),np.array(train_loss_epochs))

        # Stopping 
        if optimizer.param_groups[0]['lr'] < _g_lr_min_thresh:
            break

    # Save the trained network
    #torch.save(net, 'best_checkpoint_siamese.tar')


def _run_net_one_epoch(net, optimizer, criterion, episodes, batch_size, 
                           trainable, shuffle, num_random_pairs):
    """ Train the network for one epoch
    """
    dataset = data.TensorDataset(episodes[0], episodes[1])
    loader = data.DataLoader(dataset=dataset, shuffle=shuffle, batch_size=
        batch_size)
    losses = []
    # print(batch_size,batch_size,batch_size)
    for step, (x_batch, y_batch) in enumerate(loader):
        # print(step)
        #Create negative and positive pairs for training
        num_pos, num_neg = num_random_pairs, num_random_pairs
        pairs_idx, y = create_random_pairs(y_batch.cpu().numpy(), num_pos, num_neg)  
        if pairs_idx == []:
            continue
        x0 = (x_batch[pairs_idx[:, 0], :]).float().requires_grad_()
        x1 = (x_batch[pairs_idx[:, 1], :]).float().requires_grad_()
        y = (torch.from_numpy(y)[:, None]).float()
        # print(step,step,step)
        x0, x1, y = x0.to(_g_device), x1.to(_g_device), y.to(_g_device)
        # print(step,step,step,x0.size(),x1.size(),y.size())
        dist = net(x0, x1)
        # print(step,step,step,step)
        # print(dist.size(),y.size())
        loss = criterion(dist, y)
        losses.append(loss.item())

        # Network back-prop
        if trainable:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses

   






