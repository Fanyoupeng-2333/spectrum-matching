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




