import numpy as np
import random
from numpy.random import permutation

def create_random_pairs(input_labels, num_pos_pairs, num_neg_pairs):
    pairs = []
    labels = []
    cls_indices = []
    for i in range(int(np.max(input_labels) + 1)):
        idx = np.where(input_labels == i)[0]
        if idx.size != 0:
            cls_indices.append(idx)
    num_classes = len(cls_indices)

    if num_classes > 1:
        # Create positive/similar pairs
        for i in range(num_classes):
            n = len(cls_indices[i])
            if n < 2: continue
            for j in range(num_pos_pairs):
                perm_n = np.random.permutation(n)
                z1, z2 = cls_indices[i][perm_n[0]], cls_indices[i][perm_n[1]]
                pairs.append([z1, z2])
                labels += [0.]

        # Create negative/desimilar pairs
        for i in range(num_classes):
            for j in range(num_neg_pairs):
                inc = random.randrange(1, num_classes)
                iin = (i + inc) % num_classes
                di = random.randrange(0, len(cls_indices[i]))
                din = random.randrange(0, len(cls_indices[iin]))
                z1, z2 = cls_indices[i][di], cls_indices[iin][din]
                pairs.append([z1, z2])
                labels += [1.]
        pairs, labels = np.array(pairs), np.array(labels)
    return pairs, labels


def create_random_pairs_auto(y):
    pairs = []
    labels = []
    cls_indices = []
    for i in range(int(np.max(y) + 1)):
        idx = np.where(y == i)[0]
        if idx.size != 0:
            cls_indices.append(idx)
    num_class = len(cls_indices)

    # Create positive/similar pairs
    for i in range(num_class):
        n = len(cls_indices[i])
        s = 0.1
        num_pairs = int(n*(n-1)*0.5*s)
        if n < 2: continue
        for j in range(num_pairs):
            perm_n = np.random.permutation(n)
            z1, z2 = cls_indices[i][perm_n[0]], cls_indices[i][perm_n[1]]
            pairs.append([z1, z2])
            labels += [0.]

    # Create negative/desimilar pairs
    for i in range(num_class):
        n = len(cls_indices[i])
        s = 0.1
        num_pairs = int(n*(n-1)*0.5*s)
        for j in range(num_pairs):
            inc = random.randrange(1, num_class)
            iin = (i + inc) % num_class
            di = random.randrange(0, len(cls_indices[i]))
            din = random.randrange(0, len(cls_indices[iin]))
            z1, z2 = cls_indices[i][di], cls_indices[iin][din]
            pairs.append([z1, z2])
            labels += [1.]

    return np.array(pairs), np.array(labels)


def create_pairs_wrt_neighbors(X, y, k=10):
    uni_y = np.unique(y)
    data_dict = {}
    for uy in uni_y:
        idx = np.where(y == uy)[0]
        data_dict[uy] = X[idx,:]

    pos_pairs_x1 = []
    pos_pairs_x2 = []
    for key, val in data_dict.items():
        num_val = val.shape[0]
        dist_dict = {}  
        for i in range(num_val):
            for j in range(i+1, num_val):
                dist = np.sum(val[i,:,:]*val[j,:,:])
                dist_dict[dist] = [i,j]

        sorted_keys = np.sort(list(dist_dict.keys()))
        topn_keys = sorted_keys[-k:]
        for tpk in topn_keys:
            i, j = dist_dict[tpk]
            pos_pairs_x1 += [val[i,:,:]]
            pos_pairs_x2 += [val[j,:,:]]

    pos_pairs_x1 = np.array(pos_pairs_x1)
    pos_pairs_x2 = np.array(pos_pairs_x2)
    print(pos_pairs_x1.shape, pos_pairs_x2.shape)

    neg_pairs_x1 = []
    neg_pairs_x2 = []
    dist_dict = {}
    keys = list(data_dict.keys())
    vals = list(data_dict.values())
    for i in range(len(keys)):
        # Class i 
        for vi in vals[i]:
            for j in range(len(keys)):
                if i == j:
                    continue
                for vj in vals[j]:
                    dist = np.sum(vi*vj)
                    dist_dict[dist] = [vi, vj]

    print(dist_dict)
    exit()


def create_nn_pairs(X, y, k=10):
    uni_y = np.unique(y)
    num_classes = len(uni_y)
    data_dict = {}
    for uy in uni_y:
        idx = np.where(y == uy)[0]
        data_dict[uy] = X[idx,:,:]

    # Num_classes * k
    X_pos_1 = []
    X_pos_2 = []
    for key, val in data_dict.items():
        num_val = val.shape[0]
        dist_dict = {}  
        for i in range(num_val):
            for j in range(i+1, num_val):
                dist = np.sum(val[i,:,:]*val[j,:,:])
                dist_dict[dist] = [i,j]

        sorted_keys = np.sort(list(dist_dict.keys()))
        k = int(len(sorted_keys)*0.05)
        perm = np.random.permutation(sorted_keys)
        #topn_keys = sorted_keys[perm[0:k]]
        topn_keys = perm[0:k]
        
        for tpk in topn_keys:
            i, j = dist_dict[tpk]
            X_pos_1 += [val[i,:,:]]
            X_pos_2 += [val[j,:,:]]

    X_pos_1 = np.array(X_pos_1)
    X_pos_2 = np.array(X_pos_2)


    X_neg_1 = []
    X_neg_2 = []
    reps = int(num_classes)
    for rep in range(reps):
        sample_pool = []
        for key, val in data_dict.items():
            num_val = val.shape[0]
            # Select one sample from this class
            perm = np.random.permutation(num_val)
            sample_pool += [val[perm[0],:,:]]

        dist_dict = {}
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                dist = np.sum(sample_pool[i]*sample_pool[j])
                dist_dict[dist] = [i,j]

        sorted_keys = np.sort(list(dist_dict.keys()))
        k = int(0.0025*len(sorted_keys))
        #topn_keys = sorted_keys[0:k]
        perm = np.random.permutation(sorted_keys)
        topn_keys = perm[0:k]
        for tpk in topn_keys:
            i, j = dist_dict[tpk]
            X_neg_1 += [sample_pool[i]]
            X_neg_2 += [sample_pool[j]]

    X_neg_1 = np.array(X_neg_1)
    X_neg_2 = np.array(X_neg_2)

    #print(X_neg_1.shape, X_neg_2.shape, X_pos_1.shape,X_pos_2.shape)

    X1 = X_neg_1
    X2 = X_neg_2
    if X_pos_1.shape[0] > 0:
        X1 = np.concatenate((X1, X_pos_1), axis=0)
        X2 = np.concatenate((X2, X_pos_2), axis=0)
    y = np.array([0]*X_pos_1.shape[0] + [1]*X_neg_1.shape[0])

    return X1, X2, y
      

def create_pairs_wrt_means(X, y):
    pairs = []
    labels = []
    num_classes = int(np.max(y) + 1)
    cls_indices = []
    X_mean = []
    for i in range(num_classes):
        idx = np.where(y == i)[0]
        if len(idx) > 1:
            cls_indices.append(idx)
            xm = np.mean(X[idx,:], axis=0)
            X_mean += [xm]
    num_classes = len(cls_indices)
    X_mean = np.array(X_mean)

    X_pos_1 = []
    X_pos_2 = None
    count = 0
    for i in cls_indices:
        idx = np.where(y == i)[0]
        if X_pos_2 is None:
            X_pos_2 = X[idx,:]
        else:
            X_pos_2 = np.concatenate((X_pos_2, X[idx,:]), axis=0)
        X_pos_1 += [X_mean[count]]*len(idx)
        count += 1
    X_pos_1 = np.array(X_pos_1)

    # Positive pairs
    pos_pairs = [X_pos_1, X_pos_2]    

    # Negative pairs
    X_mean_neg = []
    for i in range(int(np.max(y) + 1)):
        idx = np.where(y == i)[0]
        if len(idx) > 0:
            xm = np.mean(X[idx,:], axis=0)
            X_mean_neg += [xm]
    X_mean_neg = np.array(X_mean_neg)

    perm = np.random.permutation(X_mean_neg.shape[0])
    X_neg_1 = np.copy(X_mean_neg)
    X_neg_2 = X_neg_1[perm,:]
    reps = X_mean_neg.shape[0]*2
    for rep in range(reps):
        perm = np.random.permutation(X_mean_neg.shape[0])
        # Check if same
        X_neg_1 = np.concatenate((X_neg_1, np.copy(X_mean_neg)), axis=0)
        X_neg_2 = np.concatenate((X_neg_2, X_neg_1[perm,:]), axis=0)

    neg_pairs = [X_neg_1, X_neg_2]

    print(X_pos_1.shape, X_neg_1.shape)
    
    X1 = X_neg_1
    X2 = X_neg_2
    if X_pos_1.shape[0] > 0:
        X1 = np.concatenate((X1, X_pos_1), axis=0)
        X2 = np.concatenate((X2, X_pos_2), axis=0)
    y = np.array([0]*X_pos_1.shape[0] + [1]*X_neg_1.shape[0])

    return X1, X2, y


def create_full_pairs(labels, batch_size=0):
    '''Create all the positive and negative pairs of given samples

    Parameters :
    -----------
        labels : labels of samples, of shape [n_samples, 1]

    Return :
    --------
        Return an index matrix containing all the pairs or
        a list of pair matrix if batch_size is greater than 1
    '''
    n_samples = len(labels.ravel())
    pair_lbs = []
    pair_idx = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            pair_idx.append([i, j])
            pair_lbs.append(float(labels[i] != labels[j]))

    pair_idx, pair_lbs = np.array(pair_idx), np.array(pair_lbs)
    if batch_size <= 1:
        return pair_idx, pair_lbs
    else:
        batch_pair_idx = []
        batch_pair_lbs = []
        n_pairs = pair_idx.shape[0]
        n_batches = int(np.floor(n_pairs / batch_size))
        for i in range(n_batches):
            batch_pair_idx.append(pair_idx[batch_size * i: batch_size * (i + 1), :])
            batch_pair_lbs.append(pair_lbs[batch_size * i: batch_size * (i + 1)])

        if batch_size * n_batches < n_pairs:
            batch_pair_idx.append(pair_idx[batch_size * n_batches: n_pairs, :])
            batch_pair_lbs.append(pair_lbs[batch_size * n_batches: n_pairs])

        return batch_pair_idx, batch_pair_lbs
