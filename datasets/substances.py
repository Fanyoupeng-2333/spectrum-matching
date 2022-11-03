from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import numpy as np
import numbers
from numpy.random import randint
from numpy.random import permutation

np.random.seed(0)

class SubstanceDB(object):
    def __init__(self):
        pass 

    def __str__(self):
        return 'SubstanceDB'

    def _load_from_disk(self):
        pass

    def names(self, label=None):
        """Find the corresponding name given a label

        Parameters:
        -----------
        label : 1d array or a list

        Returns:
        --------
        name : str
            return the corresponding name.
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
            print(label)
        else:
            print('Not valid argument')
            return

        indexes_label = np.where(self.labels == label)[0]
        return self.spectra[indexes_label, :]

    def get_data(self, copy=False):
        """Get data with label

        Parameters:
        -----------
        label : 1d array or a list

        Returns:
        --------
        X : numpy array. Each row corresponds a spectrum
            return all the spectra

        y : numpy array
            return the corresponding labels
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
                                                 X[idx[0:diff%n_idx],:]), 
                                                axis=0)
                    y_idx_copy = np.concatenate((y_idx_copy, 
                                                 y[idx[0:diff%n_idx],:]), 
                                                axis=0)

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
        if len(y_aug_mix)!=0 and False:
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
        if True:
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
            X_aug, y_aug = self.augument_data(X=X_train, y=y_train,
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
                    ridx = range(len(idx[0]))
                else:
                    ridx = permutation(range(len(idx[0])))

                if 0 < n_subset and n_subset < 1:
                    ridx = ridx[0:int(round(n_subset*len(idx[0])))]
                else:
                    ridx = ridx[0:n_subset]
                sub_indexes.append(idx[0][ridx].tolist())

            sub_indexes = sum(sub_indexes, [])
            rest_indexes = [x for x in range(self.n_spectra) 
                                if x not in sub_indexes]

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
                return X_test, y_test, X_train, y_train, train_status


class ChemKitchen(SubstanceDB):
    """A chemical dataset which contains 123 kinds of minerals and around 500
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
        return super(ChemKitchen, self).__init__()

    def _load_from_disk(self):
        """Load preprocessed data from disk"""
        if self.preprocessing is not None:
            print(self.preprocessing)
            self.spectra = np.loadtxt(
                'Datasets/ChemKitchen/cleaned/chemkitchen_spec_mod.txt')
        else:
            self.spectra = np.loadtxt(
                'Datasets/ChemKitchen/cleaned/chemkitchen_spec_raw.txt')
        self.labels = np.loadtxt(
            'Datasets/ChemKitchen/cleaned/chemkitchen_labels.txt')
        self.status = None
        with open('Datasets/ChemKitchen/cleaned/chemkitchen_names.txt', 
                  'r') as f:
            self.mineral_names = [x.strip() for x in f.readlines()]
        self.mineral_names = [x.lower() for x in self.mineral_names]

        self.n_classes = len(np.unique(self.labels))
        self.n_spectra, self.n_spec_dim = self.spectra.shape
