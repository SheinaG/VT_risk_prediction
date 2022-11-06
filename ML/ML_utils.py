from utils import consts as cts
from utils.base_packages import *

ML_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')


def sample_to_time(sample):
    sample_sec = sample / 200
    m, s = divmod(sample_sec, 60)
    h, m = divmod(m, 60)
    print(f'{h:.0f}:{m:.0f}:{s:.0f}')


def create_part_dataset(ids, y=[], path=ML_path, model=0):
    if model == 0:
        add_path = 'features.xlsx'
    elif model == 1:
        add_path = 'vt_wins/features.xlsx'
    ids_group = []
    n_win = []
    for i, id_ in enumerate(ids):
        if id_ in cts.bad_bsqi:
            continue
        new_p = np.array(pd.read_excel((path / id_ / add_path), engine='openpyxl'))

        new_p = new_p[:, 1:]
        new_p = choose_right_features(new_p)
        new_y = matlib.repmat(y[i], 1, new_p.shape[0])
        ids_group.append([id_] * new_p.shape[0])
        if i == 0:
            dataset = new_p
            y_d = new_y

        else:
            dataset = np.concatenate([dataset, new_p], axis=0)
            y_d = np.concatenate([y_d, new_y], axis=1)
        n_win = n_win + [new_p.shape[0]]
    return dataset, y_d, ids_group, n_win


def create_dataset(ids, y=[], path=ML_path, model=0, return_num=False, n_pools=10):
    pool = multiprocessing.Pool(n_pools)
    in_pool = (len(ids) // n_pools) + 1
    ids_pool, y_pool = [], []

    for j in range(n_pools):
        ids_pool += [ids[j * in_pool:min((j + 1) * in_pool, len(ids))]]
        y_pool += [y[j * in_pool:min((j + 1) * in_pool, len(ids))]]
    res = pool.starmap(create_part_dataset, zip(ids_pool, y_pool))
    for i, res_i in enumerate(res):
        if i == 0:
            dataset = res_i[0]
            y_d = res_i[1]
            ids_group = res_i[2]
            n_win = res_i[3]
        else:
            dataset = np.concatenate([dataset, res_i[0]], axis=0)
            y_d = np.concatenate([y_d, res_i[1]], axis=1)
            ids_group += res_i[2]
            n_win += res_i[3]
    pool.close()
    # if model ! = 0: #coose just the columns that you want the model to have
    # else
    np_dataset = dataset

    for i in np.arange(0, np_dataset.shape[0]):
        for j in np.arange(0, np_dataset.shape[1]):
            np_dataset[i, j] = float('%.2f' % (np_dataset[i, j]))
    ids_group = [item for sublist in ids_group for item in sublist]
    if return_num:
        return np_dataset, y_d.squeeze(), ids_group, n_win
    else:
        return np_dataset, y_d.squeeze(), ids_group


def choose_right_features(X, num_leads=1):
    X_out = X[:, :84]  # intervals lead 0
    X_out = np.concatenate([X_out, X[:, (84 * num_leads):(84 * num_leads + 48)]], axis=1)  # waves lead 0
    X_out = np.concatenate([X_out, X[:, -25:]], axis=1)  # 23 HRV + 2 demographic
    X_out = np.delete(X_out, np.arange(5, 132, 6), 1)
    return X_out


def model_features(X, model_type):
    if model_type == 1:  # just_hrv
        X_out = X[:, 110:-2]
    if model_type == 2:  # just_morph
        X_out = X[:, :110]
    if model_type == 3:  # morph and hrv
        X_out = X[:, :-2]
    if model_type == 4:  # all
        X_out = X
    return X_out


def normalize_ecg_98(ecg):
    q99, q1 = np.percentile(ecg, [99, 1])
    ecg = (ecg - q1) / (q99 - q1)
    return ecg


def norm_mean_std(ecg):
    q99, q1 = np.percentile(ecg, [99, 1])
    rob_ecg = ecg[(ecg < q99) & (ecg > q1)]
    mean_ = np.mean(rob_ecg)
    std_ = np.std(rob_ecg)
    ecg = (ecg - mean_) / std_
    return ecg


def norm_rms(ecg):
    q99, q1 = np.percentile(ecg, [99, 1])
    rob_ecg = ecg[(ecg < q99) & (ecg > q1)]
    R_dB = -6
    R = np.power(10, R_dB / 20)
    a = (len(rob_ecg) * np.power(R, 2)) / sum(np.power(ecg, 2))
    ecg = a * ecg
    return ecg


def split_ids(tr_uv_p, tr_uv_n):
    ids_train = cts.ids_VT[:tr_uv_p] + cts.ids_rbdb_VT + cts.ids_no_VT[:tr_uv_n] + cts.ids_rbdb_no_VT
    y_train = np.concatenate(
        [np.ones([1, len(cts.ids_VT[:tr_uv_p] + cts.ids_rbdb_VT)]),
         np.zeros([1, len(cts.ids_no_VT[:tr_uv_n] + cts.ids_rbdb_no_VT)])],
        axis=1).squeeze()

    ids_test = cts.ids_VT[tr_uv_p:] + cts.ids_no_VT[tr_uv_n:]
    y_test = np.concatenate([np.ones([1, len(cts.ids_VT[tr_uv_p:])]), np.zeros([1, len(cts.ids_no_VT[tr_uv_n:])])],
                            axis=1).squeeze()

    return ids_train, ids_test, y_train, y_test


def features_mrmr(X, features_model, features_mrmr):
    X_mrmr = np.zeros([X.shape[0], len(features_mrmr)])
    for i, featuer_s in enumerate(features_mrmr):
        index = features_model.index(featuer_s)
        X_mrmr[:, i] = X[:, index]
    return X_mrmr


def stat_selection(stat_t, x, y, num):
    features = x.columns
    p_values = np.ones([len(features), ])
    if stat_t == 'mannw':
        for i, feature in enumerate(features):
            x1 = np.asarray(x)[y == 1, i]
            x2 = np.asarray(x)[y == 0, i]
            try:
                stat, p = mannwhitneyu(x1, x2)
                p_values[i] = p
            except:
                p_values[i] = 1
        idx = np.argsort(p_values).squeeze()
        best_features = np.asarray(features)[idx]
        x_new = x[best_features[:num]]
    return x_new, best_features[:num]


def rc_scorer(estimator, X, y):
    y_hat = estimator.predict(X)
    if len(np.unique(y_hat)) == 2:
        return roc_auc_score(y, y_hat)
    else:
        return 0


def mrmr(X, y, MI, num):
    pd_mrmr = X
    if 'class' not in pd_mrmr.columns:
        pd_mrmr.insert(0, 'class', y)
    results = pymrmr.mRMR(pd_mrmr, MI, num)
    data_mrmr = np.asarray(pd_mrmr[results])
    return data_mrmr, results


def RFE_func(X, y, n_jobs, num):
    features = X.columns
    RF = RandomForestClassifier(max_depth=10, n_estimators=100, n_jobs=n_jobs, class_weight='balanced')
    selector = RFE(RF, n_features_to_select=num, step=1)
    selector = selector.fit(X, y)
    b_features = list(features[selector.support_])
    X_new = selector.transform(X)
    return X_new, b_features  # , selector.


def feature_selection(X_train_df, y_train, method='mrmr_MID', n_jobs=10, num=10):
    if method == 'RFE':
        X_new, features = RFE_func(X_train_df, y_train, n_jobs, num)
    if method == 'mrmr_MID':
        X_new, features = mrmr(X_train_df, y_train, 'MID', num)
    if method == 'mrmr_MIQ':
        X_new, features = mrmr(X_train_df, y_train, 'MIQ', num)
    if method == 'mannw':
        X_new, features = stat_selection('mannw', X_train_df, y_train, num)
    return X_new, features
