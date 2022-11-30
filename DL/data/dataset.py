import random

from DL.DL_utiles.base_packages import *


class all_set(Dataset):
    def __init__(self, task, win_len=6, transform=transforms.ToTensor()):
        if task in ['train', 'val', 'train_part', 'test']:
            DL_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/train/')

        data_filename = DL_path / str('X_' + task + "_on_disk.npy")
        idx_filename = DL_path / str("idx_" + task + ".npy")
        label_filename = DL_path / str("labels_" + task + ".npy")

        self.win_len = win_len
        self.targets = np.load(label_filename)[::win_len]
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
    def __init__(self, task, win_len=6, transform=transforms.ToTensor(), shuffle=False):
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

    def init_epoch(self, epoch_idx, eval_num=1):
        epoch_idx = epoch_idx % 40
        for i in range(eval_num):
            epoch_idxs_ordered_part = np.where(self.indexes_all[:, 1] == str(epoch_idx * self.win_len))[0]
            if i == 0:
                epoch_idxs_ordered = epoch_idxs_ordered_part
            else:
                epoch_idxs_ordered = np.concatenate(epoch_idxs_ordered, epoch_idxs_ordered_part)

        if self.shuffle:
            np.random.seed(epoch_idx)
            np.random.shuffle(epoch_idxs_ordered)
            self.indexes = epoch_idxs_ordered
            self.targets = self.targets_all[epoch_idxs_ordered]
        else:
            self.indexes = epoch_idxs_ordered
            self.targets = self.targets_all[epoch_idxs_ordered]


class overfit_set(Dataset):
    def __init__(self, task, win_len=6, transform=transforms.ToTensor(), size=1):
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
        self.size = size

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

    def init_epoch(self, epoch_idx=0, eval_num=1):
        epoch_idx = epoch_idx % 40
        epoch_idxs_ordered = np.where(self.indexes_all[:, 1] == str(epoch_idx * self.win_len))[0]
        targets = self.targets_all[epoch_idxs_ordered]
        p_idx = epoch_idxs_ordered[targets == 1]
        n_idx = epoch_idxs_ordered[targets == 0]
        self.indexes = p_idx[:self.size].tolist() + n_idx[:self.size].tolist()
        self.targets = self.targets_all[self.indexes]


class t_ansamble_set(Dataset):
    def __init__(self, task, win_len=6, transform=transforms.ToTensor(), size=1, ensamble_num=0):
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
        epoch_idxs_ordered = np.where(self.indexes_all[:, 1] == '0')[0]
        win_nums = np.diff(np.concatenate(epoch_idxs_ordered, np.array(len(epoch_idxs_ordered))))
        targets = self.targets_all[epoch_idxs_ordered]
        p_idx = self.indexes_all[self.targets_all == 1]
        n_idx = epoch_idxs_ordered[targets == 0]
        n_start = n_idx[ensamble_num * 30]
        n_end = n_idx[(ensamble_num + 1) * 30]
        n_rel = self.indexes_all[n_start:n_end]
        self.first_idexes = epoch_idxs_ordered[targets == 0].tolist() + n_idx.tolist()
        self.indexes_model = p_idx.tolist() + n_rel.tolist()
        self.targets_model = np.concatenate([np.ones([1, len(p_idx.tolist())]), np.zeros([1, len(n_rel.tolist())])],
                                            axis=1).squeeze()
        self.win_lens = win_nums[self.indexes_model]
        self.max_win = max(self.win_lens)
        self.indexes = []
        self.targets = []

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx_all):
        idx = self.indexes[idx_all]
        start = idx * self.win_len
        stop = (idx + 1) * self.win_len
        ecg_win = self.database_all[start:stop, :].reshape([1, self.win_len * 10 * 200])
        label = self.targets[idx_all]
        if self.transform:
            ecg_win = self.transform(ecg_win)
        return ecg_win, label

    def init_epoch(self, epoch_idx=0):
        epoch_idxes = self.first_idexes + epoch_idx
        epoch_idxes[self.win_lens < epoch_idx] = epoch_idxes[self.win_lens < epoch_idx] % self.win_lens[
            self.win_lens < epoch_idx]
        self.indexes = self.indexes_model[epoch_idxes]
        self.targets = self.targets_model[epoch_idxes]
