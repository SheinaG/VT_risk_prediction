import os.path
import pathlib
import sys

from sklearn.metrics import roc_auc_score
from tsai.all import *

sys.path.append('/home/sheina/tsai/')
from create_data import data_transfer

sys.path.append('/home/sheina/tsai/utiles/')
sys.path.append('/home/sheina/tsai/wgt15_60/')
sys.path.append('/home/sheina/tsai/wgt15_60/')

import utiles.consts as cts

my_setup()

# load train test and eval

# train a fully connected layer - train on train - 20 epochs, criterion - validation auroc
# names = ['FCN', 'ResNet', 'xresnet1d34', 'ResCNN', 'LSTM1bF', 'LSTM2bF', 'LSTM3bF', 'LSTM1bT', 'LSTM2bT', 'LSTM3bT',
#          'LSTM_FCN', 'LSTM_FCNs', 'InceptionTime', 'XceptionTime', 'OmniScaleCNN', 'mWDN']

path_ = '/home/sheina/tsai/'
bs = 8
DL_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL')

y_train = np.load(cts.TRAIN_DL_DIR / "labels_train_part.npy")
y_val = np.load(cts.TRAIN_DL_DIR / "labels_val.npy")
y_test = np.load(cts.TEST_DL_DIR / "labels_test.npy")
X_val_on_disk = np.load(cts.TRAIN_DL_DIR / 'X_val_on_disk.npy', mmap_mode='c')
X_train_on_disk = np.load(cts.TRAIN_DL_DIR / 'X_train_part_on_disk.npy', mmap_mode='c')
X_test_on_disk = np.load(cts.TEST_DL_DIR / 'X_test_on_disk.npy', mmap_mode='c')

for sec in [1800]:
    X_val, y_val_ = data_transfer(sec, X_val_on_disk, y_val)
    X_train, y_train_ = data_transfer(sec, X_train_on_disk, y_train)
    X_test, y_test_ = data_transfer(sec, X_test_on_disk, y_test)

    # ds_name = 'NATOPS'
    # X, y, splits = get_UCR_data(ds_name, return_split=False)

    bs = 64
    X, y, splits = combine_split_data([X_train, X_val], [y_train_, y_val_])
    print(X.shape)
    tfms = [None, [Categorize()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[bs, bs * 2])
    valid_dl = dls.valid
    train_dl = dls.train
    test_ds = dls.dataset.add_test(to3d(X_test))
    test_dl = valid_dl.new(test_ds)

    archs = [(FCN, {}), (ResNet, {}), (TCN, {}), (xresnet1d34, {}), (InceptionTime, {})]
    names = ['FCN', 'ResNet', 'TCN', 'xresnet1d34', 'InceptionTime']

    for i, (arch, k) in enumerate(archs[:7]):
        w_p = path_ + 'wgt15_' + str(sec) + '/models/' + names[i] + '.pth'
        model = create_model(arch, dls=dls, pretrained=True, weights_path=w_p, **k)
        learn = Learner(dls, model)
        train_probs, *_ = learn.get_preds(dl=train_dl, save_preds=None)
        valid_probs, *_ = learn.get_preds(dl=valid_dl, save_preds=None)
        test_probs, *_ = learn.get_preds(dl=test_dl, save_preds=None)
        train_auroc = roc_auc_score(y_train_, train_probs[:, 1])
        valid_auroc = roc_auc_score(y_val_, valid_probs[:, 1])
        test_auroc = roc_auc_score(y_test_, test_probs[:, 1])

        if not os.path.exists(DL_path / str('probs' + str(sec)) / names[i]):
            os.makedirs(DL_path / str('probs' + str(sec)) / names[i])

        np.save(DL_path / str('probs' + str(sec)) / names[i] / 'train.npy', train_probs[:, 1])
        np.save(DL_path / str('probs' + str(sec)) / names[i] / 'val.npy', valid_probs[:, 1])
        np.save(DL_path / str('probs' + str(sec)) / names[i] / 'test.npy', test_probs[:, 1])

        print(train_auroc)
        print(valid_auroc)
        print(test_auroc)

# eval on test
# per_window + per pateient
