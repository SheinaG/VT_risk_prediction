import pathlib

import h5py
from tsai.all import *


def data_transfer(win_len, X, y, fs=200):
    if win_len == 10:
        return X, y
    if win_len == 60:
        X_t = X.reshape([-1, win_len * fs])
        y_t = y[::6]
        return X_t, y_t
    if win_len == 1800:
        X_t = X.reshape([-1, win_len * fs])
        y_t = y[::180]
        return X_t, y_t


def load_on_disk(task='test'):
    DL_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/train/')
    file_hp = DL_path / str("data_" + task + ".hdf5")
    X_file = h5py.File(file_hp, "r")
    X = X_file.get('VT_DL')
    X_od = create_empty_array(X.shape, fname=str('X_' + task + '_on_disk'), path=DL_path, mode='r+')

    chunksize = 10_000
    pbar = progress_bar(range(math.ceil(len(X) / chunksize)))
    start = 0
    for i in pbar:
        end = start + chunksize
        X_od[start:end, :] = X[start:end, :]
        start = end


if __name__ == '__main__':
    load_on_disk(task='val')
