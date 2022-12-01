import numpy as np
import pandas as pd

from utils import consts as cts
from utils.base_packages import *


multiply = 1

ML_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')


def create_part_dataset(ids, y=[], path='', model=0, features_name='features_nd.xlsx', bad_bsqi=cts.bad_bsqi):
    if model == 0:
        add_path = features_name
    elif model == 1:
        add_path = 'vt_wins' / features_name
    ids_group = []
    n_win = []
    for i, id_ in enumerate(ids):
        if id_ in cts.bad_bsqi:
            j = 0
            continue
        new_p = np.array(pd.read_excel((path / id_ / add_path), engine='openpyxl'))

        if multiply:
            if id_ in cts.ids_VT:
                new_p = matlib.repmat(new_p, multiply, 1)

        new_p = new_p[:, 1:]
        new_p = choose_right_features(new_p)
        new_y = matlib.repmat(y[i], 1, new_p.shape[0])
        ids_group.append([id_] * new_p.shape[0])
        if (i == 0) or ((i == 1) and (j == 0)):
            dataset = new_p
            y_d = new_y
            j = 1

        if i > 0:
            if 'dataset' not in locals():
                a = 5
            dataset = np.concatenate([dataset, new_p], axis=0)
            y_d = np.concatenate([y_d, new_y], axis=1)
        n_win = n_win + [new_p.shape[0]]
    return dataset, y_d, ids_group, n_win


def create_dataset(ids, y=[], path='', model=0, return_num=False, n_pools=10, features_name='features_nd.xlsx',
                   bad_bsqi_ids=cts.bad_bsqi):
    pool = multiprocessing.Pool(n_pools)
    in_pool = (len(ids) // n_pools) + 1
    ids_pool, y_pool, pathes, models, features_names, bad_bsqi_ids_s = [], [], [], [], [], []

    for j in range(n_pools):
        ids_pool += [ids[j * in_pool:min((j + 1) * in_pool, len(ids))]]
        y_pool += [y[j * in_pool:min((j + 1) * in_pool, len(ids))]]
        pathes += [path]
        models += [model]
        features_names += [features_name]
        bad_bsqi_ids_s += [bad_bsqi_ids]
    res = pool.starmap(create_part_dataset, zip(ids_pool, y_pool, pathes, models, features_names, bad_bsqi_ids_s))
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


def model_features(X, model_type, with_dems=False):
    if with_dems:
        if model_type == 2:  # just_hrv
            X_out = X[:, 110:133]
        if model_type == 3:  # just_morph
            X_out = np.concatenate([X[:, :110], X[:, 133:138]], axis=1)
        if model_type == 4:  # morph and hrv
            X_out = X[:, :-6]
        if model_type == 5:  # all
            X_out = X
        if model_type == 1:  # Dem
            X_out = X[:, -6:]
    else:
        if model_type == 1:  # just_hrv
            X_out = X[:, 110:133]
        if model_type == 2:  # just_morph
            X_out = np.concatenate([X[:, :110], X[:, 133:138]], axis=1)
        if model_type == 3:  # morph and hrv
            X_out = X[:, :-2]
        if model_type == 4:  # all
            X_out = X
        if model_type == 5:  # Dem
            X_out = X[:, -2:]

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



def features_mrmr(X, features_model, features_mrmr, remove=0):
    if not remove:
        X_mrmr = np.zeros([X.shape[0], len(features_mrmr)])
        for i, featuer_s in enumerate(features_mrmr):
            index = features_model.index(featuer_s)
            X_mrmr[:, i] = X[:, index]
        return X_mrmr
    else:
        X_mrmr = pd.DataFrame(X, columns=features_model)
        X_mrmr = X_mrmr.drop(labels=features_mrmr, axis=1)
        return X_mrmr


def stat_selection(stat_t, x, y, f_n):
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
        x_new = x[best_features[:f_n]]
    return x_new, best_features[:f_n]


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
    return X_new, b_features


def remove_not_significant(x, y):
    features = x.columns
    p_values = np.ones([len(features), ])
    drop_list = []

    for i, feature in enumerate(features):
        x1 = np.asarray(x)[y == 1, i]
        x2 = np.asarray(x)[y == 0, i]
        try:
            stat, p = mannwhitneyu(x1, x2)
            p_values[i] = p
        except:
            p_values[i] = 1
        if p_values[i] >= 0.05:
            drop_list.append(features[i])

    x = x.drop(labels=drop_list, axis=1)

    return x, drop_list


def feature_selection_func(X_train_df, y_train, method='mrmr_MID', n_jobs=10, num=10):
    if method == 'RFE':
        X_new, features = RFE_func(X_train_df, y_train, n_jobs, num)
    if method == 'mrmr_MID':
        X_new, features = mrmr(X_train_df, y_train, 'MID', num)
    if method == 'mrmr_MIQ':
        X_new, features = mrmr(X_train_df, y_train, 'MIQ', num)
    if method == 'mannw':
        X_new, features = stat_selection('mannw', X_train_df, y_train, num)
    if method == 'mrmr':
        X_new, features = mrmr_from_matlab(X_train_df, y_train)
    return X_new, features


def split_to_group(ids_group):
    groups = np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/train_groups.npy', allow_pickle=True)
    cv_groups = []
    for id_ in ids_group:
        for i, group in enumerate(groups):
            if id_ in group:
                cv_groups.append(i)
    return cv_groups


def split_to_group_old(ids_group, list_ids_vt=cts.ids_tp, list_ids_no_VT=cts.ids_tn + cts.ids_vn, n_vt=12):
    cv_groups = []
    ratio = len(list_ids_no_VT) // len(list_ids_vt)
    max_ind = ratio * len(list_ids_vt)
    for id in ids_group:
        if id in list_ids_vt:
            indx = list_ids_vt.index(id)
            cv_groups.append(np.floor(indx / n_vt))
        if id in list_ids_no_VT:
            indx = list_ids_no_VT.index(id)
            if indx < max_ind:
                indx = np.floor(indx / (ratio * n_vt))
                cv_groups.append(indx)
            else:
                indx = np.floor((indx - max_ind) / n_vt)
                cv_groups.append(indx)
    return cv_groups


def choose_right_features(X, num_leads=1):
    X_out = np.delete(X, np.arange(5, 132, 6), 1)
    return X_out


def set_path(algo, dataset, typ, results_dir):
    if not os.path.exists(results_dir / "logo_cv" / dataset / str(algo + '_' + str(typ))):
        os.makedirs(results_dir / "logo_cv" / dataset / str(algo + '_' + str(typ)))
    return results_dir / "logo_cv" / dataset / str(algo + '_' + str(typ))


def maximize_f_beta(probas, y_true, beta=1):
    """ This function returns the decision threshold which maximizes the F_beta score.
    :param probas: The scores/probabilities returned by the model.
    :param y_true: The actual labels.
    :param beta: The beta value used to compute the score (i.e. balance between Se and PPV).
    :returns best_th: The threshold which optimizes the F_beta score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probas)
    fbeta = (1 + beta ** 2) * precision * recall / ((beta ** 2) * precision + recall)
    if np.any(np.isnan(fbeta)):
        fbeta[np.isnan(fbeta)] = sys.float_info.epsilon
    best_th = thresholds[np.argmax(fbeta)]
    return best_th


def maximize_Se_plus_Sp(probas, y_true, beta=1):
    """ This function returns the decision threshold which maximizes the Se + Sp Measure.
    :param probas: The scores/probabilities returned by the model.
    :param y_true: The actual labels.
    :param beta: The beta value used to compute the score (i.e. balance between Se and PPV).
    :returns best_th: The threshold which optimizes the F_beta score.
    """
    fpr, tpr, thresholds = roc_curve(y_true, probas)
    se, sp = tpr, 1 - fpr
    best_th = thresholds[np.argmin(np.abs(se - sp))]
    return best_th


def calc_confidence_interval_of_array(x, confidence=0.95):
    m = x.mean()
    s = x.std()
    confidence = confidence
    crit = np.abs(norm.ppf((1 - confidence) / 2))
    return m - s * crit / np.sqrt(len(x)), m + s * crit / np.sqrt(len(x))


def mrmr_from_matlab(x_train, y_train):
    features = list(x_train.columns)
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd(r'/home/sheina/VT_risk_prediction/ML/', nargout=0)
    X = matlab.single(np.asarray(x_train).tolist())
    Y = matlab.int8(y_train.astype('int').tolist())
    idx, scores = eng.matlab_mrmr(X, Y, nargout=2)
    scores_norm = np.asarray(scores[0]) / sum(scores[0])
    energy = 0
    new_features = []
    for idx_i in idx[0]:
        energy += scores_norm[int(idx_i) - 1]
        new_features.append(features[int(idx_i) - 1])
        if energy >= 0.95:
            break
    X_new = x_train[new_features]
    return X_new, new_features
