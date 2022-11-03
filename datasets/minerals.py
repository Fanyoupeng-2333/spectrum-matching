from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import numbers
import h5py
from numpy.random import permutation
import torch

# np.random.seed(1)
      
class Mineral(object):
    """Mineral dataset 
    """
    def __init__(self, preprocessing=False, verbose=False):
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.status = None
        print(self.preprocessing, self.verbose, self.status)

    def _load_from_disk(self):
        """Load preprocessed data from disk. Must be overrided."""
        self.spectra = None
        self.mineral_names = None
        self.n_classes = None
        self.n_spectra = None
        self.n_spec_dim = None

    def names(self, label=None):
        """Find the corresponding name given a label
        """
        if label is None:
            return self.mineral_names
        return self.mineral_names[label]

    def plot_spectra(self, label_or_name):
        """Plot spectra of a given mineral
        if isinstance(label_or_name, numbers.Number)   \
            and label_or_name < self.n_spectra and label_or_name >= 0:
            label = label_or_name
        elif isinstance(label_or_name, str) and label_or_name.lower() \
            in self.mineral_names:
            print label_or_name.lower()
            label = self.mineral_names.index(label_or_name.lower())
            print label
        else:
            print 'not valid argument'
            return

        indexes_label = np.where(self.labels == label)[0]
        """
        spec = self.get_spectra(label_or_name)
        plt.figure()
        for i in range(spec.shape[0]):
            plt.plot(spec[i,:], linestyle='-', c=np.random.rand(3,1))
        plt.ylabel('error')
        plt.xlabel('Frequency')
        #plt.title(self.names(label))
        plt.grid('on')
        plt.show()

    def get_spectra(self, label_or_name):
        if isinstance(label_or_name, numbers.Number)   \
            and label_or_name < self.n_spectra and label_or_name >= 0:
            label = label_or_name
        elif isinstance(label_or_name, str) and \
                (label_or_name.lower() in self.mineral_names):
            label = self.mineral_names.index(label_or_name.lower())
        else:
            print('Not valid argument')
            return
        indexes_label = np.where(self.labels == label)[0]
        return self.spectra[indexes_label, :]

    def get_data(self, copy=False):
        """Get data with label
        """
        if copy:
            return np.copy(self.spectra), np.copy(self.labels)
        else:
            return self.spectra, self.labels

    def augument_data_fixed_num(self, X, y,
                                total_num_per_sample,
                                max_wave_shift=5):

        # Compute some important info dict[label] = n_samples
        n_samples_per_label = []
        for label in y:
            idx = np.where(label == y)[0]
            n_samples_per_label.append(len(idx))
        n_samples_per_label = np.array(n_samples_per_label)
        u_n_smaples = np.unique(n_samples_per_label)
        X_aug = np.array([]).reshape(0, X.shape[1])
        y_aug = np.array([]).reshape(0, y.shape[1])
        for u in u_n_smaples:
            n_aug = np.ceil((total_num_per_sample-u)/u * 0.5).astype(int)
            idx = np.where(n_samples_per_label == u)[0]
            X_tmp = X[idx, :]
            y_tmp = y[idx]
            for i in range(n_aug):
                p = 5.0 * (i + 1.0) / (n_aug+1)
                X_aug = np.vstack([X_aug, np.power(np.abs(X_tmp), p)])
                y_aug = np.vstack([y_aug, y_tmp])
            y_aug = np.array(y_aug).flatten()
            y_aug = np.reshape(y_aug, (y_aug.shape[0], 1))

            # Shift
            for i in range(n_aug):
                X_aug_shift = np.copy(X_tmp)
                y_aug_shift = np.copy(y_tmp)
                n_right = np.random.randint(-max_wave_shift, max_wave_shift)
                X_aug_shift = np.roll(X_aug_shift, n_right, axis=1)
                for i in range(n_right):
                    X_aug_shift[:,i] = X_aug_shift[:,n_right+1]
                # Merge
                X_aug = np.concatenate((X_aug, X_aug_shift), axis=0)
                y_aug = np.concatenate((y_aug, y_aug_shift), axis=0)

        return X_aug, y_aug

    def augument_data(self, X, y,
                      n_pow_per_sample=10,
                      n_mix_per_sample=0,
                      n_shift_per_sample=10,
                      max_wave_shift=5,
                      n_mag_per_sample=0,
                      max_mag_noise=3,
                      class_balancing=False):
        """Data augumentation

        Parameters:
        -----------
        X : array. data to be augmented 

        y : array. label 

        n_pow_per_sample : integer. 
            number of samples augmented by applying power functions

        n_mix_per_sample : integer. 
            number of samples augmented by linearly combining spectra that 
            belong to the same class.
            
        n_shift_per_sample : integer. 
            number of samples augmented by shifting spectra left or right 
            a few wave numbers.

        class_balancing: Boolean. Default False
            If True, n_shift_per_sample will be ignored and we generate 
            sufficient samples so that each class has the same amount 
            of spectra.
        """

        # Compute some important info dict[label] = n_samples
        unique_labels = np.unique(y)
        label_nsamples_dict = dict()
        label_idx_dict = dict()
        for label in unique_labels:
            idx = np.where(label==y)[0]
            label_idx_dict[label] = idx
            label_nsamples_dict[label] = len(y[idx])
        max_nsamples = np.max(list(label_nsamples_dict.values()))

        # Balancing data
        if class_balancing:
            X_bala_data = np.copy(X)
            y_bala_data = np.copy(y)
            for label in unique_labels:
                idx = label_idx_dict[label]
                n_idx = label_nsamples_dict[label]
                diff = max_nsamples-n_idx
                if diff == 0: 
                    continue
                X_idx_copy = np.tile(X[idx,:],(int(diff/n_idx),1))
                y_idx_copy = np.tile(y[idx,:],(int(diff/n_idx),1))
                if diff%n_idx > 0:
                    X_idx_copy = np.concatenate((X_idx_copy, 
                        X[idx[0:diff%n_idx],:]), axis=0)
                    y_idx_copy = np.concatenate((y_idx_copy, 
                        y[idx[0:diff%n_idx],:]), axis=0)
                X_bala_data = np.concatenate((X_bala_data, X_idx_copy), axis=0)
                y_bala_data = np.concatenate((y_bala_data, y_idx_copy), axis=0)

            X = X_bala_data
            y = y_bala_data

        # Apply power functions to X
        X_aug_power = np.array([]).reshape(0, X.shape[1])
        y_aug_power = []
        for i in range(n_pow_per_sample - 1):
            # p = np.random.random() * 2
            # X_aug_power.append(np.power(np.abs(X_train), p))
            p = 5.0 * (i + 1.0) / n_pow_per_sample
            X_aug_power = np.vstack([X_aug_power, np.power(np.abs(X), p)])
            y_aug_power.append(y)
        if y_aug_power:
            X_aug_power = np.array(X_aug_power)
            y_aug_power = np.array(y_aug_power).flatten()
            y_aug_power = np.reshape(y_aug_power, (y_aug_power.shape[0], 1))
        X_aug = X_aug_power
        y_aug = y_aug_power

        # Mix spectra to generate linear combinations
        X_aug_mix = []
        y_aug_mix = []
        for i in range(np.max(y).astype(int)):
            idx = np.where(y == i)
            if len(idx[0]) > 1:
                for j in range(n_mix_per_sample):
                    coeffs = np.random.random((1, len(idx[0])))
                    coeffs = coeffs / np.sum(coeffs)
                    x_new = np.dot(coeffs, X[idx[0], :]).flatten()
                    X_aug_mix.append(x_new)
                    y_aug_mix.append(i)
        X_aug_mix = np.array(X_aug_mix)
        y_aug_mix = np.array(y_aug_mix)
        y_aug_mix = np.reshape(y_aug_mix, (y_aug_mix.shape[0], 1))
        
        # Merge
        if n_mix_per_sample > 0:
            X_aug = np.concatenate((X_aug, X_aug_mix), axis=0)
            y_aug = np.concatenate((y_aug, y_aug_mix), axis=0)

        # Shift
        for i in range(n_shift_per_sample):
            X_aug_shift = np.copy(X)
            y_aug_shift = np.copy(y)
            n_right = np.random.randint(-max_wave_shift, max_wave_shift)
            X_aug_shift = np.roll(X_aug_shift, n_right, axis=1)
            for i in range(n_right):
                X_aug_shift[:,i] = X_aug_shift[:,n_right+1]
            # Merge
            X_aug = np.concatenate((X_aug, X_aug_shift), axis=0)
            y_aug = np.concatenate((y_aug, y_aug_shift), axis=0)

        # Add random noises along y axis
        for i in range(n_mag_per_sample):
            X_aug_mag = np.copy(X)
            y_aug_mag = np.copy(y)
            mag_noise = np.random.rand(X.shape[0], 
                                        X.shape[1]) * max_mag_noise
            X_aug_mag = np.multiply(X_aug_mag, mag_noise)
            # Merge
            X_aug = np.concatenate((X_aug, X_aug_mag), axis=0)
            y_aug = np.concatenate((y_aug, y_aug_mag), axis=0)

        return X_aug, y_aug

    def get_train_test_sets(self,
                            splitting_scheme ='8020',
                            n_fixed_set = -1,  
                            fixed_split_for_debugging=False,
                            show_sample_status=False,
                            data_augmentation = False,
                            n_pow_per_sample=10,
                            n_mix_per_sample=0,
                            n_shift_per_sample=10,
                            max_wave_shift=5,
                            n_mag_per_sample=0,
                            max_mag_noise=3,
                            class_balancing=False):
        """ Can be Override. Split the dataset into the training and test sets
            and return them. 
        """
        self.supported_split_schemes = ['leave_one_out',
                                        'fixed_training_size',
                                        'fixed_test_size',
                                        '8020']
        if splitting_scheme.lower() in self.supported_split_schemes:
            self.splitting_scheme = splitting_scheme
        else:
            print('split_scheme is not valid.')

        self.class_balancing = class_balancing
        self.fixed_split_for_debugging = fixed_split_for_debugging
        if self.splitting_scheme.lower() == 'leave_one_out':
            X_train, y_train, X_test, y_test, test_status = \
                self._split_data(n_training=-1, n_test=1)

        elif self.splitting_scheme == 'fixed_training_size':
            X_train, y_train, X_test, y_test, test_status = \
                self._split_data(n_training=n_fixed_set, n_test=1)

        elif self.splitting_scheme == 'fixed_test_size':
            X_train, y_train, X_test, y_test, test_status = \
                self._split_data(n_training=-1, n_test=n_fixed_set)

        elif self.splitting_scheme == '8020':
            X_train, y_train, X_test, y_test, test_status = \
                self._split_data(n_training=0.8, n_test=0.2)

        if data_augmentation:
            X_aug, y_aug = \
                self.augument_data(X=X_train, y=y_train,
                                   n_pow_per_sample=n_pow_per_sample,
                                   n_mix_per_sample=n_mix_per_sample,
                                   n_shift_per_sample=n_shift_per_sample,
                                   max_wave_shift=max_wave_shift,
                                   n_mag_per_sample=n_mag_per_sample,
                                   max_mag_noise=max_mag_noise,
                                   class_balancing=class_balancing)
            X_train = np.concatenate((X_train, X_aug), axis=0)
            y_train = np.concatenate((y_train, y_aug), axis=0)

        if show_sample_status:
            return X_train, y_train, X_test, y_test, test_status
        else:
            return X_train, y_train, X_test, y_test

    def _split_data(self, 
                    n_training=-1, 
                    n_test=-1, 
                    n_valid=-1, 
                    rnd_index=True):
            """Split data into training, test and validate sets.
            """
            # Sanity check. Ignore for now
            assert n_training != -1 or n_test != -1
            swap = (n_training == -1)
            if swap:
                n_subset = n_test
            else:
                n_subset = n_training
            X, y = self.spectra, self.labels
            sub_indexes = []
            for i in range(np.max(y).astype(int) + 1):
                idx = np.where(y == i)
                if swap and len(idx[0]) <= 1:#If only one sample available
                    continue

                if self.fixed_split_for_debugging:
                    #ridx = range(len(idx[0]))[::-1]
                    ridx = range(len(idx[0]))
                else:
                    ridx = permutation(range(len(idx[0])))

                if 0 < n_subset and n_subset < 1:
                    ridx = ridx[0:int(round(n_subset*len(idx[0])))]
                else:
                    # ridx = ridx[0:n_subset]
                    ridx = ridx[-1:-1]
                sub_indexes.append(idx[0][ridx].tolist())

            sub_indexes = sum(sub_indexes, [])
            print(sub_indexes)
            rest_indexes = [x for x in range(self.n_spectra) 
                                if x not in sub_indexes]
            print(rest_indexes)

            X_train = X[sub_indexes, :]
            y_train = y[sub_indexes]
            X_test = X[rest_indexes, :]
            y_test = y[rest_indexes]
            y_train = np.reshape(y_train, (y_train.shape[0], 1))
            y_test = np.reshape(y_test, (y_test.shape[0], 1))

            if self.status is not None:
                train_status = self.status[sub_indexes]
                test_status = self.status[rest_indexes]
            else:
                train_status = None
                test_status = None

            if not swap:
                return X_train, y_train, X_test, y_test, test_status
            else:
                if self.status is not None:
                    test_status = self.status[sub_indexes]
                return X_test, y_test, X_train, y_train, train_status


class MINERAL_107(Mineral):
    """A small mineral dataset which contains 107 kinds of minerals and 163
    spectra in total.
    """
    def __init__(self, min_num_spec=1, preprocessing=None):
        self.min_num_spec = min_num_spec
        self.preprocessing = preprocessing
        self._load_from_disk()
        super(MINERAL_107, self).__init__()

    def _load_from_disk(self):
        """Load preprocessed data from disk"""
        """Load preprocessed data from disk"""
        if self.preprocessing is not None:
            self.spectra = np.loadtxt(
                'datasets/UniprAug/cleaned/' + self.preprocessing + \
                '_spec.txt')
        else:
            self.spectra = np.loadtxt(
                'datasets/UniprAug/cleaned/raw_spec.txt')

        self.labels = np.loadtxt(
            'Datasets/UniprAug/cleaned/labels.txt')
        with open('datasets/UniprAug/cleaned/names.txt', 'r') as f:
            self.mineral_names = [x.strip() for x in f.readlines()]
        self.mineral_names = [x.lower() for x in self.mineral_names]

        if self.min_num_spec > 1:
            X, y = self.spectra, self.labels
            X_sub_list = []
            y_sub_list = []
            count = 0
            for i in range(int(np.max(y) + 1)):
                idx = np.where(y == i)[0]
                if len(idx) >= self.min_num_spec:
                    X_sub_list.append(X[idx, :])
                    y_sub_list.append(np.array([count] * len(idx)))
                    count += 1

            X_sub = X_sub_list[0]
            y_sub = y_sub_list[0]
            for i in range(1, len(X_sub_list)):
                X_sub = np.concatenate((X_sub, X_sub_list[i]), axis=0)
                y_sub = np.concatenate((y_sub, y_sub_list[i]), axis=0)

            self.spectra, self.labels = X_sub, y_sub

        self.n_classes = len(np.unique(self.labels))
        self.n_spectra, self.n_spec_dim = self.spectra.shape


class RRUFF_ExUo(Mineral):
    """A large mineral dataset which contains 1696 kinds of minerals and 5127
    spectra in total.

    Parameters:
    -----------
        min_num_spec : default 1. 
            Only samples which have more than min_num_spec will be included. 
            Thus we can generate a subset which has more spectra per class.
    """
    def __init__(self, min_num_spec=1):
        self.min_num_spec = min_num_spec
        self._load_from_disk()
        return super(RRUFF_ExUo, self).__init__()

    def _load_from_disk(self):
        """Load preprocessed data from disk"""
        self.spectra = np.loadtxt(
            'datasets/RRUFF_ExUo/cleaned/' \
            'excellent_unoriented_spec_l0_u1500.txt')
        self.labels = np.loadtxt(
            'datasets/RRUFF_ExUo/cleaned/' \
            'excellent_unoriented_labels_l0_u1500.txt')
        self.status = np.loadtxt(
            'datasets/RRUFF_ExUo/cleaned/' \
            'status_labels.txt')
        with open('datasets/RRUFF_ExUo/cleaned/distinc_mineral_names.txt', 
                  'r') as f:
            self.mineral_names = [x.strip() for x in f.readlines()]
        self.mineral_names = [x.lower() for x in self.mineral_names]

        if self.min_num_spec > 1:
            X, y = self.spectra, self.labels
            X_sub_list = []
            y_sub_list = []
            count = 0
            for i in range(int(np.max(y)+1)):
                idx = np.where(y==i)[0]
                if len(idx) >= self.min_num_spec:
                    X_sub_list.append(X[idx, :])
                    y_sub_list.append(np.array([count]*len(idx)))
                    count += 1

            X_sub = X_sub_list[0]
            y_sub = y_sub_list[0]
            for i in range(1, len(X_sub_list)):
                X_sub = np.concatenate((X_sub, X_sub_list[i]), axis=0)
                y_sub = np.concatenate((y_sub, y_sub_list[i]), axis=0)

            self.spectra, self.labels = X_sub, y_sub

        self.n_classes = len(np.unique(self.labels))
        print(self.n_classes)
        self.n_spectra, self.n_spec_dim = self.spectra.shape


class RRUFF_LR(Mineral):
    """A large mineral dataset which contains 1696 kinds of minerals and 5127
    spectra in total.

    Parameters:
    -----------
        min_num_spec : default 1. 
            Only samples which have more than min_num_spec will be included. 
            Thus we can generate a subset which has more spectra per class.
    """
    def __init__(self, min_num_spec=1):
        self.min_num_spec = min_num_spec
        self._load_from_disk()
        return super(RRUFF_LR, self).__init__()

    def _load_from_disk(self):
        """Load preprocessed data from disk"""
        self.spectra = np.loadtxt(
            'datasets/RRUFF_LR/cleaned/low_raw_spec.txt')
        self.labels = np.loadtxt(
            'datasets/RRUFF_LR/cleaned/low_raw_labels.txt')
        with open('datasets/RRUFF_LR/cleaned/mineral_names.txt', 'r') as f:
            self.mineral_names = [x.strip() for x in f.readlines()]
        self.mineral_names = [x.lower() for x in self.mineral_names]

        if self.min_num_spec > 1:
            X, y = self.spectra, self.labels
            X_sub_list = []
            y_sub_list = []
            count = 0
            for i in range(int(np.max(y)+1)):
                idx = np.where(y==i)[0]
                if len(idx) >= self.min_num_spec:
                    X_sub_list.append(X[idx, :])
                    y_sub_list.append(np.array([count]*len(idx)))
                    count += 1

            X_sub = X_sub_list[0]
            y_sub = y_sub_list[0]
            for i in range(1, len(X_sub_list)):
                X_sub = np.concatenate((X_sub, X_sub_list[i]), axis=0)
                y_sub = np.concatenate((y_sub, y_sub_list[i]), axis=0)

            self.spectra, self.labels = X_sub, y_sub

        self.n_classes = len(np.unique(self.labels))
        self.n_spectra, self.n_spec_dim = self.spectra.shape


class RRUFF_UrUo(Mineral):
    """A large mineral dataset which contains 1696 kinds of minerals and 5127
    spectra in total.

    Parameters:
    -----------
        min_num_spec : default 1. 
            Only samples which have more than min_num_spec will be included. 
            Thus we can generate a subset which has more spectra per class.
    """
    def __init__(self, min_num_spec=1, preprocessing=None):
        self.min_num_spec = min_num_spec
        self.preprocessing = preprocessing
        self._load_from_disk()
        super(RRUFF_UrUo, self).__init__()

    def _load_from_disk(self):
        """Load preprocessed data from disk"""
        if self.preprocessing is not None:
            self.spectra = np.loadtxt(
                'datasets/RRUFF_UrUo/cleaned/'+ self.preprocessing + \
                '_spec.txt')
        else:
            self.spectra = np.loadtxt(
                'datasets/RRUFF_UrUo/cleaned/raw_spec.txt')

        self.labels = np.loadtxt(
            'Datasets/RRUFF_UrUo/cleaned/labels.txt')
        with open('datasets/RRUFF_UrUo/cleaned/names.txt', 'r') as f:
            self.mineral_names = [x.strip() for x in f.readlines()]
        self.mineral_names = [x.lower() for x in self.mineral_names]

        if self.min_num_spec > 1:
            X, y = self.spectra, self.labels
            X_sub_list = []
            y_sub_list = []
            count = 0
            for i in range(int(np.max(y)+1)):
                idx = np.where(y==i)[0]
                if len(idx) >= self.min_num_spec:
                    X_sub_list.append(X[idx, :])
                    y_sub_list.append(np.array([count]*len(idx)))
                    count += 1

            X_sub = X_sub_list[0]
            y_sub = y_sub_list[0]
            for i in range(1, len(X_sub_list)):
                X_sub = np.concatenate((X_sub, X_sub_list[i]), axis=0)
                y_sub = np.concatenate((y_sub, y_sub_list[i]), axis=0)

            self.spectra, self.labels = X_sub, y_sub

        self.n_classes = len(np.unique(self.labels))
        self.n_spectra, self.n_spec_dim = self.spectra.shape


class RRUFF_CMP(Mineral):
    """
    Parameters:
    -----------
        min_num_spec : default 1.
            Only samples which have more than min_num_spec will be included.
            Thus we can generate a subset which has more spectra per class.
    """
    def __init__(self, min_num_spec=1, preprocessing=None): #min_num_spec=1或者3 None
        self.min_num_spec = min_num_spec
        self.preprocessing = preprocessing
        self._load_from_disk()
        return super(RRUFF_CMP, self).__init__()

    def _load_from_disk(self):
        """Load preprocessed data from disk"""
        if self.preprocessing is not None:
            print(self.preprocessing)
            self.spectra = np.loadtxt(
                'datasets/RRUFF_CMP/cleaned/'+ self.preprocessing + \
                'prep_spec.txt')
        else:
            self.spectra = np.loadtxt(
                'datasets/RRUFF_CMP/cleaned/raw_spec.txt')

        self.labels = np.loadtxt(
                'datasets/RRUFF_CMP/cleaned/raw_labels.txt')

        with open('datasets/RRUFF_CMP/cleaned/names.txt', 'r') as f:
            self.mineral_names = [x.strip() for x in f.readlines()]
        self.mineral_names = [x.lower() for x in self.mineral_names]
        
        print(self.min_num_spec,"self.min_num_spec")
        if self.min_num_spec > 1:
            X, y = self.spectra, self.labels
            X_sub_list = []
            y_sub_list = []
            count = 0
            for i in range(int(np.max(y)+1)):
                idx = np.where(y==i)[0]
                if len(idx) >= self.min_num_spec:
                    X_sub_list.append(X[idx, :])
                    y_sub_list.append(np.array([count]*len(idx)))
                    count += 1

            X_sub = X_sub_list[0]
            y_sub = y_sub_list[0]
            for i in range(1, len(X_sub_list)):
                X_sub = np.concatenate((X_sub, X_sub_list[i]), axis=0)
                y_sub = np.concatenate((y_sub, y_sub_list[i]), axis=0)

            self.spectra, self.labels = X_sub, y_sub
      
        self.n_classes = len(np.unique(self.labels))
        print("self.n_classes",self.n_classes)
        self.n_spectra, self.n_spec_dim = self.spectra.shape

class Lorentz_18(Mineral):
    """
    Parameters:
    -----------
        min_num_spec : default 1.
            Only samples which have more than min_num_spec will be included.
            Thus we can generate a subset which has more spectra per class.
    """
    def __init__(self, min_num_spec=1, preprocessing=None):
        self.min_num_spec = min_num_spec
        self.preprocessing = preprocessing
        self._load_from_disk()
        return super(Lorentz_18, self).__init__()

    def _load_from_disk(self):
        """Load preprocessed data from disk"""
        
        ff = torch.load('datasets/RRUFF_CMP/cleaned/lorentz49_642.pth','r')   #打开h5文件
        self.labels = ff['lable'][:].astype(np.int)    #<HDF5 dataset "label": shape (887,), type "<f8">
        self.spectra= ff['x'][:]                       #<HDF5 dataset "spectra": shape (887, 1024), type "<f8">

        with open('datasets/RRUFF_CMP/cleaned/names.txt', 'r') as f:
            self.mineral_names = [x.strip() for x in f.readlines()]
        self.mineral_names = [x.lower() for x in self.mineral_names]

        if self.min_num_spec > 1:
            X, y = self.spectra, self.labels
            X_sub_list = []
            y_sub_list = []
            count = 0
            for i in range(int(np.max(y)+1)):
                idx = np.where(y==i)[0]
                if len(idx) >= self.min_num_spec:
                    X_sub_list.append(X[idx, :])
                    y_sub_list.append(np.array([count]*len(idx)))
                    count += 1

            X_sub = X_sub_list[0]
            y_sub = y_sub_list[0]
            for i in range(1, len(X_sub_list)):
                X_sub = np.concatenate((X_sub, X_sub_list[i]), axis=0)
                y_sub = np.concatenate((y_sub, y_sub_list[i]), axis=0)

            self.spectra, self.labels = X_sub, y_sub
            print("#########################################ok")

        self.n_classes = len(np.unique(self.labels))
        self.n_spectra, self.n_spec_dim = self.spectra.shape
        print("data",self.spectra.shape)
        print(self.labels.shape)

class RRUFF_USGS(Mineral):
    """
    Parameters:
    -----------
        min_num_spec : default 1.
            Only samples which have more than min_num_spec will be included.
            Thus we can generate a subset which has more spectra per class.
    """
    def __init__(self, min_num_spec=1, preprocessing=None):
        self.min_num_spec = min_num_spec
        self.preprocessing = preprocessing
        self._load_from_disk()
        return super(RRUFF_USGS, self).__init__()

    def _load_from_disk(self):
        """Load preprocessed data from disk"""
        
        ff = h5py.File('datasets/RRUFF_CMP/cleaned/usgs.hdf5','r')   #打开h5文件
        self.labels = ff['label'][:].astype(np.int)    #<HDF5 dataset "label": shape (887,), type "<f8">
        self.spectra= ff['spectra_norm'][:]      #<HDF5 dataset "spectra": shape (887, 1024), type "<f8">

        with open('datasets/RRUFF_CMP/cleaned/names.txt', 'r') as f:
            self.mineral_names = [x.strip() for x in f.readlines()]
        self.mineral_names = [x.lower() for x in self.mineral_names]

        if self.min_num_spec >= 1:
            X, y = self.spectra, self.labels
            X_sub_list = []
            y_sub_list = []
            count = 0
            for i in range(int(np.max(y)+1)):
                idx = np.where(y==i)[0]
                if len(idx) >= self.min_num_spec:
                    X_sub_list.append(X[idx, :])
                    y_sub_list.append(np.array([count]*len(idx)))
                    count += 1

            X_sub = X_sub_list[0]
            y_sub = y_sub_list[0]
            for i in range(1, len(X_sub_list)):
                X_sub = np.concatenate((X_sub, X_sub_list[i]), axis=0)
                y_sub = np.concatenate((y_sub, y_sub_list[i]), axis=0)

            self.spectra, self.labels = X_sub, y_sub
            print("#########################################ok")

        self.n_classes = len(np.unique(self.labels))
        self.n_spectra, self.n_spec_dim = self.spectra.shape
        print("data",self.spectra.shape)
        print(self.labels.shape)

class RRUFF_USGS1(Mineral):
    """
    Parameters:
    -----------
        min_num_spec : default 1.
            Only samples which have more than min_num_spec will be included.
            Thus we can generate a subset which has more spectra per class.
    """
    def __init__(self, min_num_spec=1, preprocessing=None):
        self.min_num_spec = min_num_spec
        self.preprocessing = preprocessing
        self._load_from_disk()
        return super(RRUFF_USGS1, self).__init__()

    def _load_from_disk(self):
        """Load preprocessed data from disk"""
        
        ff = h5py.File('datasets/RRUFF_CMP/cleaned/usgs1.hdf5','r')   #打开h5文件
        self.labels = ff['label'][:].astype(np.int)    #<HDF5 dataset "label": shape (887,), type "<f8">
        self.spectra= ff['spectra_als'][:]      #<HDF5 dataset "spectra": shape (887, 1024), type "<f8">

        with open('datasets/RRUFF_CMP/cleaned/names.txt', 'r') as f:
            self.mineral_names = [x.strip() for x in f.readlines()]
        self.mineral_names = [x.lower() for x in self.mineral_names]

        if self.min_num_spec > 1:
            X, y = self.spectra, self.labels
            X_sub_list = []
            y_sub_list = []
            count = 0
            for i in range(int(np.max(y)+1)):
                idx = np.where(y==i)[0]
                if len(idx) >= self.min_num_spec:
                    X_sub_list.append(X[idx, :])
                    y_sub_list.append(np.array([count]*len(idx)))
                    count += 1

            X_sub = X_sub_list[0]
            y_sub = y_sub_list[0]
            for i in range(1, len(X_sub_list)):
                X_sub = np.concatenate((X_sub, X_sub_list[i]), axis=0)
                y_sub = np.concatenate((y_sub, y_sub_list[i]), axis=0)

            self.spectra, self.labels = X_sub, y_sub
            print("#########################################ok")

        self.n_classes = len(np.unique(self.labels))
        self.n_spectra, self.n_spec_dim = self.spectra.shape
        print("data",self.spectra.shape,self.n_classes)
        print(self.labels.shape)
#------------------------------------------------------------------------------
# A few functions for debugging
#------------------------------------------------------------------------------
def visualize_for_debugging():

    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    #mrl = MINERAL107(preprocessing=False)
    mrl = RRUFF_ExUo(min_num_spec=5)
    X_train, y_train, X_test, y_test = mrl.get_train_test_sets(
        splitting_scheme='leave_one_out', 
        data_augmentation=False,
        fixed_split_for_debugging=True)

    for label in range(int(np.max(y_test)+1)):
        specs = mrl.get_spectra(label)
        #idx = np.where(y == label)[0]
        #specs = X[idx, :]
        n = specs.shape[0]
        for i in range(n-1):
            plt.plot(specs[i,:], 'k')
        y = specs[n-1, :]
        plt.plot(y, 'b')
        plt.grid('on')
        plt.xlim([0, specs.shape[1]])
        plt.ylim([0, 1.2])
        plt.savefig('Datasets/RRUFF_ExUo/cleaned/Figures/spec_'+str(label) \
                    +'.png')
        #plt.show()
        plt.clf()

def cross_compare_datasets():
    ds_exuo = RRUFF_ExUo(min_num_spec=1)
    ds_2 = RRUFF_UrUo(min_num_spec=1)

    ds = ds_exuo
    for i in range(ds.n_classes):
        x = ds.get_spectra(label_or_name=i)
        x_name = ds.names(i)
        print(x_name)
        for i in range(x.shape[0]):
            plt.plot(x[i,:], label=x_name, color='b')

        try:
            la = ds_2.mineral_names.index(x_name)
        except ValueError:
            continue
        
        y = ds_2.get_spectra(label_or_name=la)
        for j in range(y.shape[0]):
            tmp = y[j,:]
            plt.plot(tmp, label=x_name, color='r')
        y_name = ds_2.names(la)
        print(y_name)

        plt.grid('on')
        plt.xlim([0, x.shape[1]])
        plt.ylim([0, 1.2])
        plt.xlabel('Indexes of Raman Shift')
        plt.ylabel('Magnitude')
        #plt.legend()
        plt.savefig('./Datasets/RRUFF_UrUo/CmpFigures/'+x_name+'.png')
        plt.clf()

#=============================================================================
# RRUFF excellent unoriented
MINERAL_1700 = RRUFF_ExUo

#=============================================================================
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    #cross_compare_datasets()
    #exit()

    #mrl = MINERAL107(preprocessing=False)
    ds = RRUFF_ExUo(min_num_spec=1)
    #ds = RRUFF_LR(min_num_spec=1)
    #ds = RRUFF_UrUo(min_num_spec=1)
    #ck = ChemKitchen()
    print(ds.n_classes, ds.n_spectra)

    for i in range(ds.n_classes):
        x = ds.get_spectra(label_or_name=i)
        x_name = ds.names(i)
        print(x_name)
        for i in range(x.shape[0]):
            plt.plot(x[i,:], label=x_name)

        plt.grid('on')
        plt.xlim([0, x.shape[1]])
        plt.ylim([0, 1.2])
        plt.xlabel('Indexes of Raman Shift')
        plt.ylabel('Magnitude')
        #plt.legend()
        plt.savefig('./Datasets/RRUFF_UrUo/Figures/'+x_name+'.png')
        #plt.show()
        plt.clf()



    exit()
    x1 = mrl.get_spectra(label_or_name='dixenite')
    x2 = mrl.get_spectra(label_or_name='arakiite')
    plt.subplot(211)
    for i in range(x1.shape[0]):
        plt.plot(x1[i,:])
    plt.subplot(212)
    for i in range(x2.shape[0]):
        plt.plot(x2[i, :])
    plt.show()
    exit()

    mrl = RRUFF_ExUo(min_num_spec=5)
    X_train, y_train, X_test, y_test = mrl.get_train_test_sets(
        splitting_scheme='leave_one_out',
        data_augmentation=False,
        fixed_split_for_debugging=False)


    from collections import Counter
    y = np.concatenate((y_train, y_test), axis=0)
    indexes = []
    for i in range(int(np.max(y)+1)):
        idx = np.where(y==i)[0]
        indexes.append(len(idx))

    print(np.sort(indexes))

   