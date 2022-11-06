import pathlib
import pathlib
import sys

from sklearn.metrics import roc_auc_score
from tsai.all import *

sys.path.append('/home/sheina/tsai/')
from win_to_rec_fc import Net2
import torch.optim as optim
from datetime import datetime
from collections import Counter
import torch.nn as nn

MAX_WIN = 55


def Tens_Trans(X_arr):
    X_arr = np.expand_dims(X_arr, axis=1)
    return Tensor(X_arr)


def organize_win_probabilities(idx_list, x_as_list, y):
    n_win = list(Counter(idx_list).values())
    data = np.zeros([len(n_win), MAX_WIN])
    j_1 = 0
    j_2 = 0
    y_p = []
    for i, n in enumerate(n_win):
        j_2 = j_2 + n_win[i]
        x_p = x_as_list[j_1:j_2]
        y_p.append(y[j_1])
        median_i = np.median(np.asarray(x_p))
        data[i, :] = np.asarray(x_p + [median_i] * (MAX_WIN - n_win[i]))
        j_1 = j_2
    y_p = np.asarray(y_p)
    return data, y_p


def build_model():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # sys.path.append('/home/sheina/tsai/utiles/')
    # sys.path.append('/home/sheina/tsai/wgt15_60/')
    # sys.path.append('/home/sheina/tsai/wgt15_60/')

    import utiles.consts as cts
    # names = ['InceptionTime']
    DL_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL')

    idx_train = np.load(cts.TRAIN_DL_DIR / "idx_train.npy")
    idx_val = np.load(cts.TRAIN_DL_DIR / "idx_val.npy")
    idx_test = np.load(cts.TRAIN_DL_DIR / "idx_test.npy")

    sec = 60
    z = int(sec / 10)

    idx_train_ = idx_train[::z][:, 0]
    idx_val_ = idx_val[::z][:, 0]
    idx_test_ = idx_test[::z][:, 0]

    BM_dir = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/probs/')

    with open(BM_dir / 'probs_dict.pkl', 'rb') as f:
        probs_dict = pickle.load(f)

    train_probs = probs_dict['train']['prob'][:, 0]
    valid_probs = probs_dict['val']['prob'][:, 0]
    test_probs = probs_dict['test']['prob'][:, 0]

    train_y = probs_dict['train']['y']
    valid_y = probs_dict['val']['y']
    test_y = probs_dict['test']['y']

    X_train, y_tr = organize_win_probabilities(idx_train_.tolist(), train_probs.tolist(), train_y)
    X_val, y_v = organize_win_probabilities(idx_val_.tolist(), valid_probs.tolist(), valid_y)
    X_test, y_ts = organize_win_probabilities(idx_test_.tolist(), test_probs.tolist(), test_y)

    X_train_s = np.sort(X_train, axis=1)
    X_val_s = np.sort(X_val, axis=1)
    X_test_s = np.sort(X_test, axis=1)

    EPOCHS = 40
    model = Net2()
    best_vloss = 1_000_000.
    loss_fn = nn.CrossEntropyLoss(weight=Tensor([1, 36]))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ToTensors:
    X_train = Tens_Trans(X_train_s)
    X_val = Tens_Trans(X_val_s)
    X_test = Tens_Trans(X_test_s)

    X_y_dict = {}
    X_y_dict['X_test'] = X_test
    X_y_dict['X_train'] = X_train
    X_y_dict['X_val'] = X_val
    X_y_dict['y_train'] = y_tr
    X_y_dict['y_val'] = y_v
    X_y_dict['y_test'] = y_ts

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        model.train(True)
        outputs = model(X_train)
        loss = loss_fn(outputs[:, -1], Tensor(y_tr).long())
        loss.backward()
        optimizer.step()
        # print('  epoch {} loss: {}'.format(epoch, loss.item()))
        model.train(False)
        with torch.no_grad():
            running_vloss = 0.0
            voutputs = model(X_val)
            vloss = loss_fn(voutputs[:, -1], Tensor(y_v).long())
            print('LOSS train {} valid {}'.format(loss.item(), vloss.item()))
        if vloss.item() < best_vloss:
            train_out = outputs[:, -1]
            val_out = voutputs[:, -1]
            best_vloss = vloss.item()
            np.save('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/cls_per_p/1808/train_IT.npy', voutputs[:, -1])
            model_path = '/MLAIM/AIMLab/Sheina/databases/VTdb/DL/cls_per_p/1808/model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), str(model_path))
            best_model = model

    print(roc_auc_score(y_tr, train_out[:, -1].detach().numpy()))
    print(roc_auc_score(y_v, val_out[:, -1].detach().numpy()))

    # test
    test_out = best_model(X_test)[:, -1][:, -1].detach().numpy()
    print(roc_auc_score(y_ts, test_out, ))


def anls_model():
    # load_model:
    model = Net2()
    model_path = str('/MLAIM/AIMLab/Sheina/databases/VTdb/DL/cls_per_p/1808/' + 'model_20220818_115300_12')
    model.load_state_dict(torch.load(model_path))

    # load_dat:


if __name__ == '__main__':
    build_model()
