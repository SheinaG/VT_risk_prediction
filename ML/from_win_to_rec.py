import sys

sys.path.append("/home/sheina/VT_risk_prediction/")
from ML.ML_utils import *
from utils.plots import *

exmp_features = pd.read_excel(cts.VTdb_path / 'ML_model/V720H339/features_nd.xlsx', engine='openpyxl')
features_arr = np.asarray(exmp_features.columns[1:])
features_list = choose_right_features(np.expand_dims(features_arr, axis=0))

f2 = lambda x: list(map('{:.2f}'.format, x))
MAX_WIN = cts.MAX_WIN

def list_duplicates_of(seq, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def divide_CI_groups(LR_probs):
    main_path = '/MLAIM/AIMLab/Shany/databases/rbafdb/documentation/RBAF_Holter_Info.xlsx'
    rbdb_info = pd.read_excel(main_path, engine='openpyxl')
    rbdb_info['holter_id'] = rbdb_info['holter_id'].astype(str)
    ids_db = []
    ids_hid = cts.ids_sp + cts.ids_sn
    for id_ in ids_hid:
        # find his patiant id:
        ids_db.append(rbdb_info[rbdb_info['holter_id'] == id_]['db_id'].values[0])

    p_db = ids_db[:10]
    n_db = ids_db[10:]
    n_db_o = list(set(n_db))
    n_p = 10
    n_n = len(n_db_o)
    n_n_CI = int(n_n * 0.8)
    list_groups_p = []
    list_groups_n = []
    for i in range(n_p):
        groupi = p_db.copy()
        groupi.remove(p_db[i])
        for j in range(i + 1, n_p):
            groupij = groupi.copy()
            groupij.remove(p_db[j])
            list_groups_p.append(groupij)
            list_groups_n.append(random.sample(n_db_o, n_n_CI))
    list_groups_db = []
    list_groups_hid = []
    LR_prob_list = []
    roc_list = []
    y_test_list = []

    for i in range(len(list_groups_p)):
        group_hid = []
        prob_LR_group = []
        list_groups_db.append(list_groups_p[i] + list_groups_n[i])
        for db_id in list_groups_db[i]:
            idx_db = list_duplicates_of(ids_db, db_id)
            for j in idx_db:
                group_hid.append(ids_hid[j])
                prob_LR_group.append(LR_probs[j])
        list_groups_hid.append(group_hid)
        LR_prob_list.append(prob_LR_group)
        y_test = list(np.concatenate([np.ones([1, 8]), np.zeros([1, len(prob_LR_group) - 8])], axis=1).squeeze())
        y_test_list.append(y_test)
        roc_list.append(roc_auc_score(y_test, prob_LR_group))

    return y_test_list, LR_prob_list


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
                    },
    if task == 'train':
        # hyperparameters search
        lr = LogisticRegression(random_state=42, class_weight='balanced')
        clf = GridSearchCV(lr, search_spaces, scoring=rc_scorer, n_jobs=1)
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


def opt_thresh_sp(proba, y_true_p, save_path, min_sp, task='train', algo='RF'):
    # the policy is finding the best thresh with Sp >= (min_sp)
    proba0 = proba[y_true_p == 0]
    if task == 'train':
        thresh = np.percentile(proba0, min_sp)
        np.save(save_path / str('thresh' + algo + '.npy'), thresh)
    if task == 'test' or task == 'ext_test':
        thresh = np.load(save_path / 'thresh' + algo + '.npy')
    Sp_ = len(proba0[proba0 < thresh]) / len(proba0)
    proba1 = proba[y_true_p == 1]
    Se_ = len(proba1[proba1 > thresh]) / len(proba1)
    return thresh, Sp_, Se_


def plot_results(prob, y_true, title, save_path, algo, method):
    plt.figure()
    plt.style.use('bmh')
    Se, Sp, AUROC = {}, {}, {}
    # columns = ['Se', 'Sp', 'AUROC']
    res = {}
    if title == 'train':
        str_t = 'Roc curve for RBDB validation'
    if title == 'test':
        str_t = 'Roc curve for RBDB test'
    if title == 'ext_test':
        str_t = 'Roc curve for UVAF test'
    for i in range(1, cts.NM + 1):
        AUROC[i] = roc_auc_score(y_true, prob[i - 1, :])
        tpr_rf, fpr_rf, ths = roc_curve(y_true, prob[i - 1, :])
        plt.plot(tpr_rf, fpr_rf, cts.colors[i - 1], label='model ' + str(i) + ' (' + str(np.round(AUROC[i], 2)) + ')')
        plt.plot(tpr_rf, tpr_rf, 'k')
        plt.xlabel('1-Sp')
        plt.ylabel('Se')
        # Se[i] = Se_
        # Sp[i] = Sp_

    plt.legend(facecolor='white', framealpha=0.8, loc=4)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(str_t)
    plt.savefig(save_path / str(title + '_' + method + '.png'), dpi=400, transparent=True)
    plt.show()
    res['AUROC'] = AUROC

    results = pd.DataFrame.from_dict(res)
    results = results.apply(f2)
    results.to_excel(save_path / str(title + '_' + method + '_results.xlsx'))
    return res


def run_one_model(all_path, DATA_PATH, algo, feature_selection=0, method='LR', methods=['mrmr']):
    # load data for all models

    y_train_p = np.concatenate([np.ones([1, len(cts.ids_tp)]), np.zeros([1, len(cts.ids_tn + cts.ids_vn)])],
                               axis=1).squeeze()
    x_train, y_train, train_ids_groups, n_win = create_dataset(cts.ids_tp + cts.ids_tn + cts.ids_vn, y_train_p,
                                                               path=DATA_PATH, model=0, return_num=True)
    y_test_p = np.concatenate([np.ones([1, len(cts.ids_sp)]), np.zeros([1, len(cts.ids_sn)])], axis=1).squeeze()
    x_test, y_test, _, n_win_test = create_dataset(cts.ids_sp + cts.ids_sn, y_test_p, path=DATA_PATH, model=0,
                                                   return_num=True)
    auroc_all = []
    for i in range(1, cts.NM + 1):
        # train
        features_model = list(model_features(features_list, i, with_dems=True)[0])
        model_path = all_path / str(algo + '_' + str(i))
        opt = joblib.load(model_path / 'opt.pkl')
        x_train_model = model_features(x_train, i, with_dems=True)
        if feature_selection:
            features_str = str('features' + methods[0] + '.pkl')
            try:
                features = joblib.load(model_path / features_str)
            except FileNotFoundError:
                features = features_model
            x_train_model = features_mrmr(x_train_model, features_model, list(features), remove=0)
        y_pred = opt.predict_proba(x_train_model)[:, 1].tolist()
        data = organize_win_probabilities(n_win, y_pred)

        # test
        x_test_model = model_features(x_test, i, with_dems=True)
        if feature_selection:
            x_test_model = features_mrmr(x_test_model, features_model, list(features), remove=0)
        y_pred = opt.predict_proba(x_test_model)[:, 1].tolist()
        data_test = organize_win_probabilities(n_win_test, y_pred)

        if method == 'LR':
            prob = tev_LR(data, y_train_p, model_path, task='train')
            opt_thresh_sp(prob, y_train_p, model_path, min_sp=90, task='train', algo=algo)
            prob = tev_LR(data_test, y_test_p, model_path, task='test')
            AUROC = roc_auc_score(y_test_p, prob)

        if method == 'median':
            medians = np.median(data_test, axis=0)
            AUROC = roc_auc_score(y_test_p, medians)

        auroc_all.append(AUROC)
    print(auroc_all)
    return prob


def plot_test(dataset, DATA_PATH, algo, method='LR', feature_selection=0, methods=['mrmr']):
    save_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/') / dataset

    # test
    y_test_p = np.concatenate([np.ones([1, len(cts.ids_sp)]), np.zeros([1, len(cts.ids_sn)])], axis=1).squeeze()
    _, y_test, _, n_win = create_dataset(cts.ids_sp + cts.ids_sn, y_test_p, path=DATA_PATH, model=0,
                                         return_num=True)
    fig = plt.figure()
    plt.style.use('bmh')
    for i in range(cts.NM):

        hyp_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/') / dataset / str(
            algo + '_' + str(i + 1))
        opt = joblib.load(hyp_path / 'opt.pkl')
        x_test_model = joblib.load(hyp_path / 'X_test.pkl')
        y_pred = opt.predict_proba(x_test_model)[:, 1].tolist()
        data = organize_win_probabilities(n_win, y_pred)
        if method == 'LR':
            prob = tev_LR(data, y_test_p, hyp_path, task='test')
            y_test_list, prob_list = divide_CI_groups(prob)
            low_auroc_i, high_auroc_i = roc_plot_envelope(prob_list, y_test_list, K_test=45, augmentation=1, typ=i + 1,
                                                          title='model ' + str(i), algo='LR',
                                                          majority_vote=False, soft_lines=False)
        if method == 'median':
            prob = np.median(data, axis=0)
            y_test_list, prob_list = divide_CI_groups(prob)
            low_auroc_i, high_auroc_i = roc_plot_envelope(prob_list, y_test_list, K_test=45, augmentation=1, typ=i + 1,
                                                          title='model ' + str(i), algo='median',
                                                          majority_vote=False, soft_lines=False)
        if i == 0:
            prob_all = np.expand_dims(prob, axis=1).T
        else:
            prob_all = np.concatenate([prob_all, np.expand_dims(prob, axis=1).T], axis=0)

    plt.legend(facecolor='white', framealpha=0.8, loc=4)
    plt.title('Receiving operating curve')
    plt.xlabel('1-Sp')
    plt.ylabel('Se')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.grid()
    plt.savefig(save_path / str('AUROC_per_patient_' + algo + '_' + method + '.png'), dpi=400, transparent=True)
    plt.show()

    return prob_all


if __name__ == '__main__':
    algo = 'RF'
    DATA_PATH = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')
    all_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/results/logo_cv/WL_60/')

    #
    run_one_model(all_path, DATA_PATH, algo, method='LR', methods=['ns'], feature_selection=0)
    plot_test('WL_60', DATA_PATH, algo, feature_selection=0, methods=['ns'])
    # run_one_model(model_path, 1, DATA_PATH)
    # plot_test(dataset, DATA_PATH, algo)
