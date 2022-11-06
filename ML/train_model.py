import pathlib

import numpy as np
import pandas as pd
import sheina.bayesiansearch as bs
import sheina.consts as cts
from numpy import matlib
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

multiply = 1

ML_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')


def create_dataset(ids, y=[], path=ML_path, model=0):
    ids_group = []
    for i, id in enumerate(ids):
        new_p = np.array(pd.read_excel(path / id / 'features.xlsx', engine='openpyxl'))

        if multiply:
            if id in cts.ids_VT:
                new_p = matlib.repmat(new_p, multiply, 1)

        new_p = new_p[:, 1:]
        new_p = bs.choose_right_features(new_p)
        new_y = matlib.repmat(y[i], 1, new_p.shape[0])
        ids_group.append([id] * new_p.shape[0])
        if i == 0:
            dataset = new_p
            y_d = new_y
        else:
            dataset = np.concatenate([dataset, new_p], axis=0)
            y_d = np.concatenate([y_d, new_y], axis=1)
    # if model ! = 0: #coose just the columns that you want the model to have
    # else
    np_dataset = dataset

    for i in np.arange(0, np_dataset.shape[0]):
        for j in np.arange(0, np_dataset.shape[1]):
            np_dataset[i, j] = float('%.2f' % (np_dataset[i, j]))
    ids_group = [item for sublist in ids_group for item in sublist]

    return np_dataset, y_d.squeeze(), ids_group


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


def feature_selection(method, X, y):
    if method == 'chi2':
        df_X = pd.DataFrame(columns=X.columns)
        for j, feature_s in enumerate(X.columns):
            feature = X[feature_s]
            feature_c = feature[feature != -1]
            feature_c = np.expand_dims(feature_c, axis=1)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(feature_c)
            df_X[feature_s] = kmeans.predict(np.expand_dims(feature, axis=1))
        SB = SelectKBest(chi2, k=10)
        X_new = SB.fit_transform(df_X, y)
        jj = np.argsort(SB.scores_)
        b_features = list(X.columns[jj][0:10])
    if method == 'f_c':
        X_new = SelectKBest(f_classif, k=10).fit_transform(X, y)
    if method == 'mi':
        SB = SelectKBest(mutual_info_classif, k=10)
        X_new = SB.fit_transform(X, y)
        jj = np.argsort(SB.scores_)
        b_features = list(X.columns[jj][0:10])
    return X_new, b_features


def stat_selection(stat_t, x, y):
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
        x_new = x[best_features[:10]]
    return x_new, best_features[:10]


# train = ids_VT[:8] +ids_no_VT[:44]
# y_train = np.concatenate([np.ones([1, len(ids_VT[:8])]),np.zeros([1,len(ids_no_VT[:44])])], axis = 1).squeeze()
#
# test =  [ids_VT[8]] + ids_no_VT[44:]
# y_test = np.array([1,0])

# x_train, y_train, train_ids_group = create_dataset(train, y_train, ML_path )
# x_test, y_test, _ = create_dataset(test, y_test, path = ML_path)


# clf = LogisticRegression(random_state=0, cv = bs.loo_cv(train_ids_len)).fit(x_train, y_train.squeeze() )
# y_predict = clf.predict(x_test)


a = 5
