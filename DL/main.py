import pathlib
import sys

from tsai.all import *

sys.path.append('/home/sheina/tsai/')
from create_data import data_transfer

my_setup()
# test
# preapare my dataset:

train_DL_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/train')

y_train = np.load(train_DL_path / "labels_train_part.npy")
y_val = np.load(train_DL_path / "labels_val.npy")
X_val_on_disk = np.load(train_DL_path / 'X_val_on_disk.npy', mmap_mode='c')
X_train_on_disk = np.load(train_DL_path / 'X_train_part_on_disk.npy', mmap_mode='c')

X_val, y_val = data_transfer(1800, X_val_on_disk, y_val)
X_train, y_train = data_transfer(1800, X_train_on_disk, y_train)

# ds_name = 'NATOPS'
# X, y, splits = get_UCR_data(ds_name, return_split=False)

bs = 2
X, y, splits = combine_split_data([X_train, X_val], [y_train, y_val])
print(X.shape)
tfms = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[bs, bs * 2])
#
archs = [(FCN, {}), (ResNet, {}), (TCN, {}), (xresnet1d34, {}),
         (LSTM, {'n_layers': 1, 'bidirectional': False}), (LSTM, {'n_layers': 1, 'bidirectional': True}),
         (LSTM_FCN, {}), (LSTM_FCN, {'shuffle': False}), (InceptionTime, {}), (mWDN, {'levels': 4, 'wavelet': 'db'}),
         ]

names = ['FCN', 'ResNet', 'TCN', 'xresnet1d34', 'LSTM1bF', 'LSTM1bT',
         'LSTM_FCN', 'LSTM_FCNs', 'InceptionTime', 'mWDN']

results = pd.DataFrame(columns=['arch', 'hyperparams', 'train loss', 'valid loss', 'accuracy', 'time'])
for i, (arch, k) in enumerate(archs):
    model = create_model(arch, dls=dls, device="cpu", **k)
    print(model.__class__.__name__)
    learn = Learner(dls, model, metrics=accuracy, cbs=SaveModel(fname=names[i]),
                    loss_func=CrossEntropyLossFlat(weight=Tensor([1, 10])))
    start = time.time()
    learn.fit_one_cycle(5, 1e-3)
    elapsed = time.time() - start
    vals = learn.recorder.values[-1]
    results.loc[i] = [arch.__name__, k, vals[0], vals[1], vals[2], int(elapsed)]
    results.sort_values(by='accuracy', ascending=False, ignore_index=True, inplace=True)
    display(results)
