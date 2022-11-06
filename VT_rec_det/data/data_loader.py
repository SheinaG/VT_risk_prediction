import random

import numpy as np
from base_VT_parser import *
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence


# class Generator(Sequence):
#     """
#     This class is for getting data from a database that saves 24h ECG per person and chooses windows from 24h ECG
#     This was too slow so is not suppose to be in use
#     """
#     def __init__(self, db_parser,window_size,dataset,dataset_patients_list,patients, batch_size,return_binary,return_global_label, shuffle = True):
#         self.db_parser = db_parser
#         self.window_size = window_size
#         self.dataset = dataset
#         self.dataset_patients_list = dataset_patients_list
#         self.patient_list = patients
#
#         self.number_of_samples_dict = {}
#         self.batch_size = batch_size
#         self.return_binary =return_binary
#         self.return_global_label = return_global_label
#         self.shuffle = shuffle
#         self.labels = []
#
#
#         indexes = self.calc_index_tuples()
#         self.total_number_of_samples = len(list(indexes))
#         if shuffle:
#             indexes = np.random.permutation(indexes)
#
#         self.batch_indexes = indexes[:(len(indexes) // self.batch_size) * self.batch_size].reshape(-1, self.batch_size,2)
#
#         self.get_labels()
#
#         print(f'DEBUG self.labels = {self.labels}')
#         print(f'DEBUG len(self.labels) = {len(self.labels)}')
#         print(f'DEBUG self.total_number_of_samples = {self.total_number_of_samples}')
#
#
#     def __len__(self):
#
#         return  int(np.floor(self.total_number_of_samples/ self.batch_size))
#
#     def __getitem__(self, item):
#
#         indexes = self.batch_indexes[item]
#         masks = self.db_parser.return_masks_dict(self.patient_list)
#         batch_x = []
#         #
#         # pat_ids = indexes[:, 0].astype(int)
#         # window_ids = indexes[:, 1].astype(int)
#         # pat_idx = [np.where(self.dataset_patients_list == pat_ids[i]) for i in range(len(indexes))]
#         # pat_idx = np.array(pat_idx).squeeze().squeeze()
#
#         for i in range(len(indexes)):
#             pat_id = indexes[i][0]
#             window_id = indexes[i][1]
#             pat_idx = np.where(self.dataset_patients_list ==int(pat_id))
#             # print(f'i ={i} pat_id =  {pat_id} pat_idx = {pat_idx} ')
#             ecg_pat = self.dataset[pat_idx]
#
#             # remove zero padding
#             ecg_orig_len = self.db_parser.recording_time[pat_id] * self.db_parser.actual_fs
#             ecg_pat = np.squeeze(ecg_pat, axis=0)
#             ecg_pat = ecg_pat[:int(ecg_orig_len)]
#             ecg_pat = ecg_pat[:(len(ecg_pat) // self.window_size) * self.window_size].reshape(-1, self.window_size)[masks[pat_id]]
#             batch_x.append(ecg_pat[window_id])
#
#         batch_x = np.array(batch_x)
#         if self.return_binary:
#             batch_y = np.array(tuple(self.db_parser.af_win_lab_dict[elem[0]][self.window_size][masks[elem[0]]][elem[1]] for elem in indexes))
#         else: #TODO check what should be the output if the data is not binary look at UVAF_wave parser funcgtions used for loading data
#             batch_y = np.concatenate(
#                     tuple(self.db_parser.win_lab_dict[elem[0]][self.window_size][masks[elem[0]]] for elem in  indexes), axis=0)
#
#
#         if self.return_global_label:
#             global_lab = np.array(tuple((self.db_parser.af_pat_lab_dict[elem[0]] * np.ones(np.sum(masks[elem[0]])))[elem[1]] for elem in indexes))
#             return batch_x, batch_y, global_lab
#         else:
#             return batch_x, batch_y
#
#
#     def calc_index_tuples(self):
#         masks = self.db_parser.return_masks_dict(self.patient_list)
#         num_duplicates = {pat: np.sum(masks[pat]) for pat in masks.keys()}
#         window_indexes = np.concatenate(tuple(range(num_duplicates[elem]) for elem in self.patient_list), axis=0)
#         patients_duplicate = np.concatenate(tuple(elem * np.ones(num_duplicates[elem], dtype=object) for elem in self.patient_list), axis=0)
#
#         index_tuples = np.concatenate((np.expand_dims(patients_duplicate,axis=1) , np.expand_dims(window_indexes, axis=1)),axis = 1)
#         return  index_tuples
#
#     def get_labels(self):
#         masks = self.db_parser.return_masks_dict(self.patient_list)
#
#         for i in range(len(self.batch_indexes)):
#             indexes = self.batch_indexes[i]
#
#             batch_y = np.array(tuple(self.db_parser.af_win_lab_dict[elem[0]][self.window_size][masks[elem[0]]][elem[1]] for elem in indexes))
#
#             if i == 0:
#                 self.labels = batch_y
#             else:
#                 self.labels = np.concatenate((self.labels, batch_y))


class Generator(Sequence):

    def __init__(self, db_parser, window_size, dataset, patients, batch_size, return_binary, return_global_label,
                 shuffle=True, history=0, type=None):
        self.db_parser = db_parser
        self.window_size = window_size

        self.dataset_patients_list = dataset[:, -3:-1].astype(int)
        self.patient_list = patients

        self.number_of_samples_dict = {}
        self.batch_size = batch_size
        self.return_binary = return_binary
        self.return_global_label = return_global_label
        self.shuffle = shuffle
        self.dataset = dataset  # [:,:-3]

        self.indexes = np.concatenate(
            tuple(np.where(self.dataset_patients_list[:, 0] == int(p)) for p in self.patient_list), axis=1)
        self.indexes = np.squeeze(self.indexes, axis=0)

        assert len(self.indexes) > 0, f'did not find any data in dataset'
        # TODO check that there is data from every patient
        check_windows_from_each_pat_added(self.dataset_patients_list, self.patient_list)
        self.total_number_of_samples = (len(self.indexes) // self.batch_size) * self.batch_size
        self.indexes = self.indexes[:self.total_number_of_samples]

        self.labels = self.return_binary_lab()  # self.return_binary_lab() #dataset[:,-1:].astype(int)
        # Added for ArNet
        self.type = type
        self.history = history
        self.prec = self.db_parser.return_preceeding_windows(
            pat_list=self.patient_list)  # Todo check that it corresponds to the data
        self.global_lab = self.return_global_lab()
        self.non_binary_lab = self.return_non_binary_lab()
        self.ids = self.db_parser.return_patient_ids(pat_list=self.patient_list)
        self.prec = self.prec[:self.total_number_of_samples]
        self.ids = self.ids[:self.total_number_of_samples]

        self.start_time = self.return_start_time()

        self.subset_labels = np.squeeze(self.labels[self.indexes])  # Length of the samples actually used
        if shuffle:
            self.indexes = np.random.permutation(self.indexes)

    def __len__(self):
        return int(np.floor(self.total_number_of_samples / self.batch_size))

    def __getitem__(self, item):
        inds = (self.indexes[item * self.batch_size:(item + 1) * self.batch_size])
        ind = np.sort(inds)

        if self.history:
            batch_x = [np.concatenate((np.zeros(
                (self.history - min(self.history, self.prec[i]), self.window_size)),
                                       self.dataset[(i - min(self.history, self.prec[i]) + 1):(i + 1)]),
                axis=0)
                for i in ind]
        else:
            batch_x = self.dataset[ind][:, :-3]

        if self.type == 'LSTM':
            batch_x = np.concatenate(tuple([x.reshape(1, self.history, -1) for x in batch_x]),
                                     axis=0)  # Row Stack for LSTM: (Samples, Timesteps, Features)

        batch_y = self.labels[ind]
        return batch_x, np.array(batch_y).squeeze()

    def on_epoch_end(self):
        if self.shuffle:
            print('Shuffling indexes')
            self.indexes = shuffle(self.indexes, random_state=random.randint(0, 100))

    def return_global_lab(self):
        g_lab = np.empty((
                             len(self.dataset_patients_list)))  # Todo understand if this needs to be in the size of all the dataset in hdft or it can be only of the size of the patinet list data
        for p in self.patient_list:
            ind = np.where(self.dataset_patients_list[:, 0] == int(p))
            g_lab[ind] = self.db_parser.af_pat_lab_dict[p]
        g_lab = g_lab[self.indexes]
        return g_lab

    def return_non_binary_lab(self):
        nb_lab = np.empty((len(self.dataset_patients_list)))
        masks = self.db_parser.return_masks_dict(pat_list=self.patient_list)

        for p in self.patient_list:
            ind = np.where(self.dataset_patients_list[:, 0] == int(p))
            assert len(self.db_parser.win_lab_dict[p][self.window_size][masks[p]]) == len(
                ind[0]), f'database has different number of windows then the labels'
            nb_lab[ind] = self.db_parser.win_lab_dict[p][self.window_size][masks[p]]
        nb_lab = nb_lab[self.indexes]
        return nb_lab

    def return_binary_lab(self):
        b_lab = np.empty((len(self.dataset_patients_list)))
        masks = self.db_parser.return_masks_dict(pat_list=self.patient_list)

        for p in self.patient_list:
            ind = np.where(self.dataset_patients_list[:, 0] == int(p))
            assert len(self.db_parser.af_win_lab_dict[p][self.window_size][masks[p]]) == len(
                ind[0]), f'database has different number of windows then the labels'
            b_lab[ind] = self.db_parser.af_win_lab_dict[p][self.window_size][masks[p]]
        b_lab = b_lab[self.indexes]
        return b_lab

    def return_start_time(self):
        s_time = np.empty((len(self.dataset_patients_list)))

        for p in self.patient_list:
            ind = np.where(self.dataset_patients_list[:, 0] == int(p))
            s_time[ind] = self.db_parser.return_start_time(p)

        s_time = s_time[self.indexes]
        return s_time
