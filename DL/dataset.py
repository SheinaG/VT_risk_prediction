import pathlib

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class multi_set(Dataset):
    def __init__(self, task, len_w=180, transform=ToTensor()):
        self.len_w = len_w
        if task in ['train', 'val']:
            DL_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/train/')
        if task in ['test']:
            DL_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/test/')
        data_filename = DL_path / str('X_' + task + "_on_disk.npy")
        idx_filename = DL_path / str("idx" + task + ".npy")
        label_filename = DL_path / str("labels" + task + ".npy")

        self.labels = np.load(label_filename)[::len_w]
        self.indexes = np.load(idx_filename)[::len_w]
        self.database = np.load(data_filename, mmap_mode='c')
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        start = idx * 180
        stop = (idx + 1) * 180
        ecg_win = self.database[start:stop, :].reshape([1, len_w * 10 * 200])
        label = self.labels[idx]
        if self.transform:
            ecg_win = self.transform(ecg_win)
        return ecg_win, label
