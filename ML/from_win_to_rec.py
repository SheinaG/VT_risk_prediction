from ML.ML_utils import *

exmp_features = pd.read_excel(cts.VTdb_path + 'ML_model/1601/features.xlsx', engine='openpyxl')
features_arr = np.asarray(exmp_features.columns[1:])
features_list = choose_right_features(np.expand_dims(features_arr, axis=0))

f2 = lambda x: list(map('{:.2f}'.format, x))
MAX_WIN = cts.MAX_WIN
NM = cts.NM


def rc_scorer(estimator, X, y):
    y_hat = estimator.predict(X)
    if len(np.unique(y_hat)) == 2:
        return roc_auc_score(y, y_hat)
    else:
        return 0


def organize_win_probabilities(n_win, x_as_list):
    data = np.zeros([len(n_win), MAX_WIN])
    j_1 = 0
    j_2 = 0
    for i, n in enumerate(n_win):
        j_2 = j_2 + n_win[i]
        x_p = x_as_list[j_1:j_2]
        median_i = np.median(np.asarray(x_p))
        if n_win[i] > MAX_WIN:
            x_p = x_as_list[j_1:j_1 + MAX_WIN]
            data[i, :] = np.asarray(x_p)
        else:
            data[i, :] = np.asarray(x_p + [median_i] * (MAX_WIN - n_win[i]))
        j_1 = j_2
    return data


def parse_hyperparameters(PATH, algo):
    opt = joblib.load(PATH / 'opt.pkl')
    params = list(opt.best_params_.values())
    params_dict = {}
    for i, hyp in enumerate(cts.hyp_list[algo]):
        params_dict[hyp] = params[i]
    return params_dict


def tev_LR(x, y, hyp_path, task):
    x.sort(axis=1)
    search_spaces = {
                        'penalty': ['none', 'l2'],
                        'tol': [1e-4, 1e-3, 1e-5],
                        'C': [0.1, 1, 10],
                        'random_state': [0, 3, 20],
                        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
                        'max_iter': [100, 1000]
                        # 'model__max_leaf_nodes': [None],
                        #    'model__max_samples': Real(0.3, 0.99),
                    },
    if task == 'train':
        # hyperparameters search
        lr = LR(random_state=42, class_weight='balanced')
        clf = GridSearchCV(lr, search_spaces, scoring=rc_scorer, n_jobs=10)
        clf.fit(x, y)
        prob = clf.predict_proba(x)[:, 1]
        with open((hyp_path / 'LR.pkl'), 'wb') as f:
            joblib.dump(clf, f)
    if task == 'test':
        LR_clf = joblib.load(hyp_path / 'LR.pkl')
        prob = LR_clf.predict_proba(x)[:, 1]
    return prob


def tev_RF(x_train, y_train, x_val, params_dict, algo):
    StSC = StandardScaler()
    StSc_fit = StSC.fit(x_train)
    x_val = StSc_fit.transform(x_val)
    x_train = StSc_fit.transform(x_train)
    clf = cts.class_funcs[algo](**params_dict, class_weight='balanced',
                                n_jobs=10)  # RandomForestClassifier(**params_dict, n_jobs=10, class_weight='balanced')
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_val)[:, 1]
    return y_pred


def split_and_collect(x_train, y_train, y_train_p, train_groups, n_win, hyp_path, algo):
    logo = LeaveOneGroupOut()
    params_dict = parse_hyperparameters(hyp_path, algo)
    y_pred = []
    for train_index, test_index in logo.split(x_train, y_train, groups=train_groups):
        x_tt, x_tv = x_train[train_index], x_train[test_index]
        y_tt, y_tv = y_train[train_index], y_train[test_index]
        y_tv_pred = tev_RF(x_tt, y_tt, x_tv, params_dict, algo)
        y_pred = y_pred + y_tv_pred.tolist()
    data = organize_win_probabilities(n_win, y_pred)
    proba = tev_LR(data, y_train_p, hyp_path, task='train')
    return proba


def opt_thresh(proba, y_true_p, save_path, task='train', model_type=1, algo='RF'):
    # the policy is finding the best thresh with Sp>=0.9
    proba0 = proba[y_true_p == 0]
    if task == 'train':
        thresh = np.percentile(proba0, 90)
        np.save(save_path / 'thresh.npy', thresh)
    if task == 'test' or task == 'ext_test':
        thresh = np.load(save_path / 'thresh.npy')
    Sp_ = len(proba0[proba0 < thresh]) / len(proba0)
    proba1 = proba[y_true_p == 1]
    Se_ = len(proba1[proba1 > thresh]) / len(proba1)
    return thresh, Sp_, Se_


def plot_results(prob, y_true, title, save_path, algo):
    plt.figure()
    plt.style.use('bmh')
    Se, Sp, AUROC = {}, {}, {}
    columns = ['Se', 'Sp', 'AUROC']
    res = {}
    if title == 'train':
        str_t = 'Roc curve for RBDB validation'
    if title == 'test':
        str_t = 'Roc curve for RBDB test'
    if title == 'ext_test':
        str_t = 'Roc curve for UVAF test'
    for i in range(NM):
        AUROC[i] = roc_auc_score(y_true, prob[i, :])
        th, Sp_, Se_ = opt_thresh(prob[i, :].T, y_true, save_path, task=title, model_type=i + 1, algo=algo)
        tpr_rf, fpr_rf, ths = roc_curve(y_true, prob[i, :])
        plt.plot(tpr_rf, fpr_rf, colors_six[i], label='model ' + str(i + 1) + ' (' + str(np.round(AUROC[i], 2)) + ')')
        ind_th = np.argsort(abs(ths - th))[0]
        plt.plot(tpr_rf[ind_th], fpr_rf[ind_th], colors_six[i], marker="+", markersize=15)
        plt.plot(tpr_rf, tpr_rf, 'k')
        plt.xlabel('1-Sp')
        plt.ylabel('Se')
        Se[i] = Se_
        Sp[i] = Sp_

    plt.legend(facecolor='white', framealpha=0.8, loc=4)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(str_t)
    plt.savefig(save_path / str(title + '.png'), dpi=400, transparent=True)
    plt.show()
    res['Se'] = Se
    res['Sp'] = Sp
    res['AUROC'] = AUROC

    results = pd.DataFrame.from_dict(res)
    # results = results.transpose()
    # results = results.set_axis(columns, axis= 1)
    results = results.apply(f2)
    results.to_excel(save_path / str(title + '_results.xlsx'))
    return res


def run_all_models(dataset, DATA_PATH, algo):
    save_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/') / dataset
    # train:

    train_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_VT_ids.npy'))
    val_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_val_no_VT_ids.npy'))
    train_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_no_VT_ids.npy'))
    y_train_p = np.concatenate([np.ones([1, len(train_vt)]), np.zeros([1, len(train_no_vt + val_no_vt)])],
                               axis=1).squeeze()
    x_train, y_train, train_ids_groups, n_win = train_model.create_dataset(train_vt + train_no_vt + val_no_vt,
                                                                           y_train_p,
                                                                           path=DATA_PATH, model=0, return_num=True)
    train_groups = bs.split_to_group(train_ids_groups, train_vt, train_no_vt + val_no_vt, n_vt=12)
    for i in range(NM):
        x_train_model = train_model.model_features(x_train, i + 1)
        hyp_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/') / dataset / str(
            algo + '_' + str(i + 1))
        prob = split_and_collect(x_train_model, y_train, y_train_p, train_groups, n_win, hyp_path, algo)
        if i == 0:
            prob_all = np.expand_dims(prob, axis=1).T
        else:
            prob_all = np.concatenate([prob_all, np.expand_dims(prob, axis=1).T], axis=0)
    # res = plot_results(prob_all, y_train_p[:-6], 'train', save_path, algo)

    # test
    test_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))
    test_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_no_VT_ids.npy'))
    y_test_p = np.concatenate([np.ones([1, len(test_vt)]), np.zeros([1, len(test_no_vt)])], axis=1).squeeze()
    x_test, y_test, _, n_win_test = train_model.create_dataset(test_vt + test_no_vt, y_test_p, path=DATA_PATH, model=0,
                                                               return_num=True)
    for i in range(NM):
        x_test_model = train_model.model_features(x_test, i + 1)
        hyp_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/') / dataset / str(
            'RF_' + str(i + 1))
        opt = joblib.load(hyp_path / 'opt.pkl')
        y_pred = opt.predict_proba(x_test_model)[:, 1].tolist()
        data = organize_win_probabilities(n_win, y_pred)
        prob = tev_LR(data, y_test_p, hyp_path, task='test')
        if i == 0:
            prob_all = np.expand_dims(prob, axis=1).T
        else:
            prob_all = np.concatenate([prob_all, np.expand_dims(prob, axis=1).T], axis=0)
    res = plot_results(prob_all, y_test_p, 'test', save_path)

    # ext_test
    test_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_VT_ids.npy'))
    test_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_non_VT_ids.npy'))
    y_test_p = np.concatenate([np.ones([1, len(test_vt)]), np.zeros([1, len(test_no_vt)])], axis=1).squeeze()
    x_test, y_test, _, n_win = train_model.create_dataset(test_vt + test_no_vt, y_test_p, path=DATA_PATH, model=0,
                                                          return_num=True)
    for i in range(NM):
        x_test_model = train_model.model_features(x_test, i + 1)
        hyp_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/') / dataset / str(
            algo + '_' + str(i + 1))
        opt = joblib.load(hyp_path / 'opt.pkl')
        y_pred = opt.predict_proba(x_test_model)[:, 1].tolist()
        data = organize_win_probabilities(n_win, y_pred)
        prob = tev_LR(data, y_test_p, hyp_path, task='test')
        if i == 0:
            prob_all = np.expand_dims(prob, axis=1).T
        else:
            prob_all = np.concatenate([prob_all, np.expand_dims(prob, axis=1).T], axis=0)
    res = plot_results(prob_all, y_test_p, 'ext_test', save_path)
    return prob_all


def run_all_models_fs(dataset, DATA_PATH):
    save_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/') / dataset
    # train:

    train_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_VT_ids.npy'))
    val_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_val_no_VT_ids.npy'))
    train_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_no_VT_ids.npy'))
    y_train_p = np.concatenate([np.ones([1, len(train_vt)]), np.zeros([1, len(train_no_vt + val_no_vt)])],
                               axis=1).squeeze()
    x_train, y_train, train_ids_groups, n_win = train_model.create_dataset(train_vt + train_no_vt + val_no_vt,
                                                                           y_train_p,
                                                                           path=DATA_PATH, model=0, return_num=True)
    train_groups = bs.split_to_group(train_ids_groups, train_vt, train_no_vt + val_no_vt, n_vt=12)
    for i in range(NM):
        x_train_model = train_model.model_features(x_train, i + 1)
        hyp_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/') / dataset / str(
            algo + '_' + str(i + 1))
        features = joblib.load(hyp_path / 'features.pkl')
        features_model = train_model.model_features(features_list, i + 1)
        x_train_model = train_model.features_mrmr(x_train_model, list(features_model[0]), list(features))
        prob = split_and_collect(x_train_model, y_train, y_train_p, train_groups, n_win, hyp_path)
        if i == 0:
            prob_all = np.expand_dims(prob, axis=1).T
        else:
            prob_all = np.concatenate([prob_all, np.expand_dims(prob, axis=1).T], axis=0)
    # res = plot_results(prob_all, y_train_p[:-6], 'train', save_path)

    # test
    test_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))
    test_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_no_VT_ids.npy'))
    y_test_p = np.concatenate([np.ones([1, len(test_vt)]), np.zeros([1, len(test_no_vt)])], axis=1).squeeze()
    x_test, y_test, _, n_win = train_model.create_dataset(test_vt + test_no_vt, y_test_p, path=DATA_PATH, model=0,
                                                          return_num=True)
    for i in range(NM):
        x_test_model = train_model.model_features(x_test, i + 1)
        hyp_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/') / dataset / str(
            algo + '_' + str(i + 1))
        opt = joblib.load(hyp_path / 'opt.pkl')
        StSc = joblib.load(hyp_path / 'StSC.pkl')
        features = joblib.load(hyp_path / 'features.pkl')
        features_model = train_model.model_features(features_list, i + 1)
        x_test_model = StSc.transform(x_test_model)
        x_test_model = train_model.features_mrmr(x_test_model, list(features_model[0]), list(features))
        y_pred = opt.predict_proba(x_test_model)[:, 1].tolist()
        data = organize_win_probabilities(n_win, y_pred)
        prob = tev_LR(data, y_test_p, hyp_path, task='test')
        if i == 0:
            prob_all = np.expand_dims(prob, axis=1).T
        else:
            prob_all = np.concatenate([prob_all, np.expand_dims(prob, axis=1).T], axis=0)
    res = plot_results(prob_all, y_test_p, 'test', save_path)

    # ext_test
    test_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_VT_ids.npy'))
    test_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_non_VT_ids.npy'))
    y_test_p = np.concatenate([np.ones([1, len(test_vt)]), np.zeros([1, len(test_no_vt)])], axis=1).squeeze()
    x_test, y_test, _, n_win = train_model.create_dataset(test_vt + test_no_vt, y_test_p, path=DATA_PATH, model=0,
                                                          return_num=True)
    for i in range(NM):
        x_test_model = train_model.model_features(x_test, i + 1)
        hyp_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/') / dataset / str(
            algo + '_' + str(i + 1))
        opt = joblib.load(hyp_path / 'opt.pkl')
        StSc = joblib.load(hyp_path / 'StSC.pkl')
        features = joblib.load(hyp_path / 'features.pkl')
        features_model = train_model.model_features(features_list, i + 1)
        x_test_model = StSc.transform(x_test_model)
        x_test_model = train_model.features_mrmr(x_test_model, list(features_model[0]), list(features))
        y_pred = opt.predict_proba(x_test_model)[:, 1].tolist()
        data = organize_win_probabilities(n_win, y_pred)
        prob = tev_LR(data, y_test_p, hyp_path, task='test')
        if i == 0:
            prob_all = np.expand_dims(prob, axis=1).T
        else:
            prob_all = np.concatenate([prob_all, np.expand_dims(prob, axis=1).T], axis=0)
    res = plot_results(prob_all, y_test_p, 'ext_test', save_path)

    return prob_all

    # test


def run_one_model(all_path, DATA_PATH, algo):
    # train:

    train_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_VT_ids.npy'))
    val_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_val_no_VT_ids.npy'))
    train_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_no_VT_ids.npy'))
    y_train_p = np.concatenate([np.ones([1, len(train_vt)]), np.zeros([1, len(train_no_vt + val_no_vt)])],
                               axis=1).squeeze()
    x_train, y_train, train_ids_groups, n_win = train_model.create_dataset(train_vt + train_no_vt + val_no_vt,
                                                                           y_train_p,
                                                                           path=DATA_PATH, model=0, return_num=True)
    test_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))
    test_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_no_VT_ids.npy'))
    y_test_p = np.concatenate([np.ones([1, len(test_vt)]), np.zeros([1, len(test_no_vt)])], axis=1).squeeze()
    x_test, y_test, _, n_win_test = train_model.create_dataset(test_vt + test_no_vt, y_test_p, path=DATA_PATH, model=0,
                                                               return_num=True)
    for i in range(1, NM + 1):
        model_path = all_path / str(algo + '_' + str(i))
        opt = joblib.load(model_path / 'opt.pkl')
        x_train_model = train_model.model_features(x_train, i, with_pvc=True)
        y_pred = opt.predict_proba(x_train_model)[:, 1].tolist()
        data = organize_win_probabilities(n_win, y_pred)
        prob = tev_LR(data, y_train_p, model_path, task='train')
        opt_thresh(prob, y_train_p, model_path, task='train', model_type=i, algo=algo)
        # res = plot_results(prob, y_train_p, 'train', save_path) #y_train_p[:-6]
        # test

        x_test_model = train_model.model_features(x_test, i, with_pvc=True)

        y_pred = opt.predict_proba(x_test_model)[:, 1].tolist()
        data = organize_win_probabilities(n_win_test, y_pred)
        prob = tev_LR(data, y_test_p, model_path, task='test')
        AUROC = roc_auc_score(y_test_p, prob)
        print(AUROC)
    # res = plot_results(prob_all, y_test_p, 'test', save_path)
    return prob


def plot_test(dataset, DATA_PATH, algo):
    # train:
    save_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/') / dataset

    # test
    test_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))
    test_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_no_VT_ids.npy'))
    y_test_p = np.concatenate([np.ones([1, len(test_vt)]), np.zeros([1, len(test_no_vt)])], axis=1).squeeze()
    x_test, y_test, _, n_win = train_model.create_dataset(test_vt + test_no_vt, y_test_p, path=DATA_PATH, model=0,
                                                          return_num=True)
    for i in range(NM):
        x_test_model = train_model.model_features(x_test, i + 1, with_pvc=True)
        hyp_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/') / dataset / str(
            algo + '_' + str(i + 1))
        opt = joblib.load(hyp_path / 'opt.pkl')
        y_pred = opt.predict_proba(x_test_model)[:, 1].tolist()
        data = organize_win_probabilities(n_win, y_pred)
        prob = tev_LR(data, y_test_p, hyp_path, task='test')
        if i == 0:
            prob_all = np.expand_dims(prob, axis=1).T
        else:
            prob_all = np.concatenate([prob_all, np.expand_dims(prob, axis=1).T], axis=0)
    res = plot_results(prob_all, y_test_p, 'test', save_path)

    return prob_all


if __name__ == '__main__':
    dataset = '22_10_XGB'
    algo = 'XGB'
    DATA_PATH = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')
    all_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/22_10_XGB/')
    # run_all_models(dataset, DATA_PATH, algo)
    # plot_test(dataset, DATA_PATH, algo)
    run_one_model(all_path, DATA_PATH, algo)
    # run_one_model(model_path, 1, DATA_PATH)
    plot_test(dataset, DATA_PATH, algo)
