import pathlib

import numpy as np


class all_set(Dataset):
    def __init__(self, task, win_len=6, transform=ToTensor()):
        if task in ['train', 'val', 'train_part', 'test']:
            DL_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/train/')

        data_filename = DL_path / str('X_' + task + "_on_disk.npy")
        idx_filename = DL_path / str("idx_" + task + ".npy")
        label_filename = DL_path / str("labels_" + task + ".npy")

        self.win_len = win_len
        self.first_targets = np.load(label_filename)[::win_len]
        self.indexes = np.load(idx_filename)[::win_len]
        self.database = np.load(data_filename, mmap_mode='c')
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        start = idx * self.win_len
        stop = (idx + 1) * self.win_len
        ecg_win = self.database[start:stop, :].reshape([1, self.win_len * 10 * 200])
        label = self.targets[idx]
        if self.transform:
            ecg_win = self.transform(ecg_win)
        return ecg_win, label


class one_set(Dataset):
    def __init__(self, task, win_len=6, transform=ToTensor(), shuffle=False):
        if task in ['train', 'val', 'train_part', 'test']:
            DL_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/train/')

        data_filename = DL_path / str('X_' + task + "_on_disk.npy")
        idx_filename = DL_path / str("idx_" + task + ".npy")
        label_filename = DL_path / str("labels_" + task + ".npy")

        self.win_len = win_len
        self.targets_all = np.load(label_filename)[::win_len]
        self.indexes_all = np.load(idx_filename)[::win_len]
        self.database_all = np.load(data_filename, mmap_mode='c')
        self.transform = transform
        self.indexes = []
        self.targets = []
        self.shuffle = shuffle

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx_all):
        idx = self.indexes[idx_all]
        start = idx * self.win_len
        stop = (idx + 1) * self.win_len
        ecg_win = self.database_all[start:stop, :].reshape([1, self.win_len * 10 * 200])
        label = self.targets_all[idx]
        if self.transform:
            ecg_win = self.transform(ecg_win)
        return ecg_win, label

    def init_epoch(self, epoch_idx):
        epoch_idx = epoch_idx % 40
        epoch_idxs_ordered = np.where(self.indexes_all[:, 1] == str(epoch_idx * self.win_len))[0]

        if self.shuffle:
            np.random.seed(epoch_idx)
            np.random.shuffle(epoch_idxs_ordered)
            self.indexes = epoch_idxs_ordered
            self.targets = self.targets_all[epoch_idxs_ordered]
        else:
            self.indexes = epoch_idxs_ordered
            self.targets = self.targets_all[epoch_idxs_ordered]
