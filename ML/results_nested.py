import os

import joblib
import matplotlib.pyplot as plt

from ML.ML_utils import *
from utils import consts as cts
from utils.base_packages import *
from utils.metrics import *
from utils.plots import *

from ML.from_win_to_rec import tev_RF, parse_hyperparameters, organize_win_probabilities

colors_six = ['#307DA6', '#A65730', '#6F30A6', '#A6304F', '#A69E30', '#30A640']
light_colors = ['#B0E7FF', '#FFD7B0', '#BFC0FF', '#EFB0DF', '#FFEEB0', '#C0FFD0']
exmp_features = pd.read_excel(cts.VTdb_path / 'ML_model/V720H339/features_nd.xlsx', engine='openpyxl')
features_arr = np.asarray(exmp_features.columns[1:])
features_list = choose_right_features(np.expand_dims(features_arr, axis=0))


# def results_per_WS(ids=cts.ids_sn + cts.ids_sp, ):
#     for id_ in ids:
#         a = 5


def calc_on_validation(train_part=True, algo='XGB', feature_selection=1, methods=['mrmr'], all_path=[],
                       bad_bsqi_ids=cts.bad_bsqi, features_name='features.xlsx', data_path=[]):
    calc_test = True
    # load train_data
    if train_part:
        y_train = np.concatenate([np.ones([1, len(cts.ids_tp_2)]), np.zeros([1, len(cts.ids_tn_part_2)])],
                                 axis=1).squeeze()
        x_train, y_train, _ = create_dataset(cts.ids_tp_2 + cts.ids_tn_part_2, y_train, path=data_path,
                                             model=0, features_name=features_name,
                                             bad_bsqi_ids=bad_bsqi_ids)

    else:
        y_train = np.concatenate([np.ones([1, len(cts.ids_tp_2)]), np.zeros([1, len(cts.ids_tn_2)])],
                                 axis=1).squeeze()
        x_train, y_train, _ = create_dataset(cts.ids_tp_2 + cts.ids_tn_2, y_train, path=data_path,
                                             model=0, features_name=features_name,
                                             bad_bsqi_ids=bad_bsqi_ids)

    y_val = np.concatenate([np.ones([1, len(cts.ids_vp_2)]), np.zeros([1, len(cts.ids_vn_2)])],
                           axis=1).squeeze()
    x_val, y_val, _ = create_dataset(cts.ids_vp_2 + cts.ids_vn_2, y_val, path=data_path,
                                     model=0, features_name=features_name,
                                     bad_bsqi_ids=bad_bsqi_ids)
    # Choose right features
    for i in range(1, cts.NM + 1):
        # train
        features_model = list(model_features(features_list, i, with_dems=True)[0])
        model_path = all_path / str(algo + '_' + str(i))
        # opt = joblib.load(model_path / 'opt.pkl')
        x_train_model = model_features(x_train, i, with_dems=True)
        x_val_model = model_features(x_val, i, with_dems=True)
        if feature_selection:
            features_str = str('features' + methods[0] + '.pkl')
            try:
                StSc_fit = joblib.load(model_path / 'StSC.pkl')
                x_train_model = StSc_fit.transform(x_train_model)
                x_val_model = StSc_fit.transform(x_val_model)
                ff = 1
                features = joblib.load(model_path / features_str)
            except FileNotFoundError:
                features = features_model
            x_train_model = features_mrmr(x_train_model, features_model, list(features), remove=0)
            x_val_model = features_mrmr(x_val_model, features_model, list(features), remove=0)
        # load params dict

        param_dict = parse_hyperparameters(model_path, algo)

        y_pred, clf = tev_RF(x_train=x_train_model, y_train=y_train, x_val=x_val_model, params_dict=param_dict,
                             algo=algo)
        if calc_test:
            x_test = joblib.load(model_path / 'X_test.pkl')
            y_test = joblib.load(model_path / 'y_test.pkl')
            y_test_pred = clf.predict_proba(x_test)[:, 1]
            roc = roc_auc_score(y_test, y_test_pred)
            print(roc)
        # calculate best val
        best_TH = maximize_f_beta(y_pred, y_val)
        # save thresh
        with open((model_path / str('thresh_' + algo + '.pkl')), 'wb') as f:
            joblib.dump(best_TH, f)


def rebuild_clf(x_train, y_train, x_test, y_test, params_dict, algo):
    clf = cts.class_funcs[algo](**params_dict, class_weight='balanced',
                                n_jobs=10)  # RandomForestClassifier(**params_dict, n_jobs=10, class_weight='balanced')
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)[:, 1]
    results_ts = eval(clf, x_test, y_test)
    return y_pred


def test_samples(y_pred, y_test, n_iter):
    pos_ind = list(np.where(y_test == 1)[0])
    neg_ind = list(np.where(y_test == 0)[0])
    pos_len = len(pos_ind)
    neg_len = len(neg_ind)

    y_pred_list = []
    y_test_list = []
    for i in range(n_iter):
        # choose randomly 80 present of the positive and negative windows:
        random.seed(i)
        pos_part = random.sample(pos_ind, int(pos_len * 0.8))
        neg_part = random.sample(neg_ind, int(neg_len * 0.8))
        y_pred_part = y_pred[pos_part + neg_part]
        y_test_part = y_test[pos_part + neg_part]

        y_pred_list.append(y_pred_part)
        y_test_list.append(y_test_part)
    return y_test_list, y_pred_list


def plot_violin(data, leg_str_arr, title, labels, save_path):
    plt.style.use('bmh')
    plt.rcParams.update({'font.size': 16})
    color = 'steelblue'
    colors = cts.violin_colors
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    violins = axe.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    for i, vp in enumerate(violins['bodies']):
        vp.set_facecolor(colors[i])
        vp.set_edgecolor(color)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = violins[partname]
        vp.set_edgecolor(color)
    axe.set_xlabel(title)
    quartile1 = []
    medians = []
    quartile3 = []
    for i in range(len(data)):
        quartile1_i, medians_i, quartile3_i = np.percentile(data[i], [25, 50, 75])
        quartile1.append(quartile1_i)
        medians.append(medians_i)
        quartile3.append(quartile3_i)
    inds = range(1, 5)
    axe.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    axe.vlines(inds, quartile1, quartile3, color=color, linestyle='-', lw=5)
    axe.set_ylabel(r'$probability\ density$')
    axe.legend(leg_str_arr)

    plt.tight_layout()
    plt.savefig(save_path / str(title + '.png'))
    plt.show()


def vt_segments(ids):
    x_vt, y_vt, ids = create_dataset(ids, np.ones([len(ids), ]), path=DATA_PATH, model=1, n_pools=1)
    return x_vt


pos = [0, 1, 2, 3, 4]
colors = ['#003049', '#245D7C', '#307DA6', '#5F6F77', '#773000']


def intrp_model(path, features_model, results_dir, feature_selection, features_str='', threshold=None):
    # load model:
    opt = joblib.load(path / 'opt.pkl')
    x_test = joblib.load(path / 'X_test.pkl')
    y_test = joblib.load(path / 'y_test.pkl')
    features = 0
    # create test set:

    if feature_selection:
        try:
            features = joblib.load(path / features_str)
        except FileNotFoundError:
            features = features_model[0]
    results_ts = eval(opt, x_test, y_test, threshold=threshold)[6]

    return opt, results_ts, x_test, y_test, features


def ext_test_set(opt_d, model_path, features, features_selection, features_model, n_splits):
    ext_test_vt = cts.ext_test_vt
    ext_test_no_vt = cts.ext_test_no_vt
    y_ext = np.concatenate([np.ones([1, len(ext_test_vt)]), np.zeros([1, len(ext_test_no_vt)])], axis=1).squeeze()

    x_ext, y_ext, ext_ids_groups = create_dataset(ext_test_vt + ext_test_no_vt, y_ext, path=DATA_PATH,
                                                  model=0, n_pools=10, features_name='features_n.xlsx')
    results_ext = {}
    for i in range(2, cts.NM):
        x_ext_i = model_features(x_ext, i)
        for split in range(n_splits):
            if features_selection:
                x_ext_i = features_mrmr(x_ext_i, list(features_model[i][0]), list(features[i]))
            results_ext[i][split] = eval(opt_d[i], x_ext_i, y_ext)[6]
    results = pd.DataFrame.from_dict(results_ext)
    results = results.transpose()
    results = results.set_axis(cts.METRICS, axis=1)
    results = results.apply(f2)
    results.to_excel(model_path / 'results_ext.xlsx')
    return results_ext


def plot_ML_learning_curve(all_path, algo='XGB'):
    plt.style.use('bmh')
    fig = plt.plot([1])
    for i in range(2, cts.NM + 1):
        model_path = all_path / str(algo + '_' + str(i))
        opt = joblib.load(model_path / 'opt.pkl')
        curve = opt.cv_results_['mean_test_score']
        x_axe = np.arange(1, len(curve) + 1)
        plt.plot(x_axe, curve, '.', linestyle="None", label='model ' + str(i))
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('AUROC score')
    plt.title('ML learning curve')
    plt.ylim([0.4, 0.8])
    plt.show()


def clac_probs(x_test, y_test, opt, model_path):
    # load ids
    ids_sp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))

    # create datasets
    for i in range(1, cts.NM + 1):
        X_vt = vt_segments(ids_sp)
        X_vt = model_features(X_vt, i, with_dems=True)
        X_vt_p = x_test[i][y_test[i] == 1, :]
        X_non_vt_p = x_test[i][y_test[i] == 0, :]

        # calc probabilities
        p_vt = opt[i].predict_proba(X_vt)[:, 1]
        p_vt_p = opt[i].predict_proba(X_vt_p)[:, 1]
        p_non_vt_p = opt[i].predict_proba(X_non_vt_p)[:, 1]

        p_all = np.concatenate([p_vt, p_vt_p, p_non_vt_p])
        plot_violin([p_vt, p_vt_p, p_non_vt_p],
                    ['VT segments', 'VT patients', 'Non VT patients'], 'model ' + str(i),
                    model_path)


def all_models(model_path, results_dir=cts.ML_RESULTS_DIR, dataset='rbdb_10', algo='RF', feature_selection=0,
               methods=['mrmr'], win_len=30):
    opt_d = {}
    results_d = {}
    path_d = {}
    features_model = {}
    x_test_d = {}
    features_d = {}
    y_test_d = {}
    features_str = ''
    n_splits = 10

    dataset_n = dataset
    with_ext_test = True
    exmp_features = pd.read_excel(cts.VTdb_path / 'ML_model/1020D818/features_nd.xlsx', engine='openpyxl')
    # exmp_features = pd.read_excel(cts.VTdb_path + 'ML_model/1419Ec09/features.xlsx', engine='openpyxl')
    features_arr = np.asarray(exmp_features.columns[1:])
    features_list = choose_right_features(np.expand_dims(features_arr, axis=0))
    if feature_selection:
        dataset = dataset + '_' + '_'.join(methods)
        features_str = str('features' + methods[0] + '.pkl')
    for i in range(1, cts.NM + 1):
        opt_d[i] = {}
        results_d[i] = {}
        path_d[i] = {}
        features_model[i] = {}
        x_test_d[i] = {}
        features_d[i] = {}
        y_test_d[i] = {}
        features_model[i] = model_features(features_list, i, with_dems=True)
        for split in range(n_splits):
            path_d[i][split] = pathlib.PurePath(model_path / str('split_' + str(split)) / str(algo + '_' + str(i)))
            # if os.path.exists(path_d[i][split] / str('thresh_' + algo + '.pkl')):
            # threshold = joblib.load(path_d[i][split] / str('thresh_' + algo + '.pkl'))
            opt_d[i][split], results_d[i][split], x_test_d[i][split], y_test_d[i][split], features_d[i][
                split] = intrp_model(path_d[i][split], features_model[i],
                                     results_dir, feature_selection,
                                     features_str, threshold=None)

    # train_val(opt_d, model_path, algo)
    # hyper_model(opt_d, model_path, algo)

    if with_ext_test:
        results_ext = ext_test_set(opt_d, model_path, features_d, feature_selection, features_model)
    results = pd.DataFrame.from_dict(results_d)
    results = results.transpose()
    results = results.set_axis(cts.METRICS, axis=1)
    results = results.apply(f2)
    results.to_excel(model_path / str('all_model_results_' + algo + '.xlsx'))

    for i in range(1, cts.NM + 1):
        all_auroc = []
        for split in range(n_splits):
            all_auroc.append(results_d[i][split])
        print(np.mean(all_auroc), np.std(all_auroc))

    prob_rf = {}
    for i in range(1, cts.NM + 1):
        prob_rf[i] = {}
        for split in range(n_splits):
            prob_rf[i][split] = opt_d[i][split].predict_proba(x_test_d[i][split])

    ## VT burden
    dict_VTs = {}
    for vt_id in cts.vt_ids:
        dict_VTs[vt_id] = []

    for split in range(n_splits):
        n_win_test = joblib.load(model_path / str('split_' + str(split)) / "n_win_test.pkl")
        test_vt_ids = list(np.load(cts.IDS_DIR / str('split_' + str(split)) / 'VT_test.npy'))
        data = organize_win_probabilities(n_win_test, list(prob_rf[4][split][:, 1]), win_len=win_len)
        burden = np.median(data, 1) / cts.MAX_WIN
        vt_burden = burden[:len(test_vt_ids)] - np.mean(burden[len(test_vt_ids):])
        for i, id_ in enumerate(test_vt_ids):
            dict_VTs[id_].append(vt_burden[i])

    # feture importance

    n_F_max = 10
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(30, 8))
    features_num = []
    for i in range(1, cts.NM + 1):
        importance_array = np.zeros([len(features_model[i][0]), ])
        axi = axes[i - 1]
        # axi = axes[int((i - 1) / 2), (i - 1) % 2]
        if feature_selection:
            n_F = np.minimum(n_F_max, len(features_d[i]))
            features_num.append(len(features_d[i]))
            features = list(features_d[i])
            importance_arrayi = opt_d[i].best_estimator_.named_steps.model.feature_importances_
            for j in range(len(importance_arrayi)):
                index = list(features_model[i][0]).index(features[j])
                importance_array[index] = importance_arrayi[j]

        else:
            n_F = np.minimum(n_F_max, len(features_model[i][0]))
            importance_array = opt_d[i].best_estimator_.named_steps.model.feature_importances_

        indices = np.argsort(importance_array)[::-1]
        axi.set_title(" model " + str(i))
        axi.barh(range(n_F), importance_array[indices[0:n_F]],
                 color="#307DA6", align="center")
        stic_list = []

        for j in range(n_F):
            if features_model[i][0][indices[j]] == 'gender':
                stic_list.append('sex')
            else:
                stic_list.append(features_model[i][0][indices[j]])
        axi.set_yticks(range(n_F))
        axi.set_yticklabels(stic_list, rotation='horizontal', fontsize=16)
        axi.invert_yaxis()
    plt.tight_layout()

    fig.savefig(model_path / str('importance_' + algo + '.png'), dpi=400, transparent=True)
    plt.show()

    prob_rf_train = {}

    fig = plt.figure()
    plt.style.use('bmh')
    rf_plot_all = []
    rf_point_all = []
    low_auroc = []
    high_auroc = []

    for i in range(1, cts.NM + 1):
        y_test_list, y_pred_list = test_samples(prob_rf[i][:, 1], y_test_d[i], 100)
        low_auroc_i, high_auroc_i = roc_plot_envelope(y_pred_list, y_test_list, K_test=100, augmentation=1, typ=i,
                                                      title='model ' + str(i),
                                                      majority_vote=False, soft_lines=False, algo=algo)

        low_auroc.append(low_auroc_i)
        high_auroc.append(high_auroc_i)
        # plt.plot(tpr_rf, tpr_rf, 'k')

    plt.legend(facecolor='white', framealpha=0.8, loc=4)
    # plt.title('Receiving operating curve')
    plt.xlabel('1-Sp')
    plt.ylabel('Se')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.grid()
    plt.savefig(model_path / str('AUROC_' + algo + '.png'), dpi=400, transparent=True)
    plt.show()


def eval_one_model(results_dir, path):
    opt = joblib.load(results_dir / path / 'opt.pkl')
    x_test = joblib.load(results_dir / path / 'X_test.pkl')
    y_test = joblib.load(results_dir / path / 'y_test.pkl')
    results_ts = eval(opt, x_test, y_test)
    return results_ts


if __name__ == '__main__':
    # eval_one_model(cts.ML_RESULTS_DIR, 'logo_cv/new_dem41_stand/RF_4/')
    # DATA_PATH = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/win_len/win_len_10/')
    DATA_PATH = cts.ML_path
    # plot_ML_learning_curve(all_path=cts.ML_RESULTS_DIR / "logo_cv" / 'split_2_30_vs_mrmr', algo='XGB')
    # calc_on_validation(train_part=True, algo='XGB', feature_selection=1, methods=['mrmr'],
    #                    all_path=cts.ML_RESULTS_DIR / "logo_cv" / 'split_2_30_vs_mrmr', bad_bsqi_ids=cts.bad_bsqi,
    #                    features_name='features_nd.xlsx', data_path=DATA_PATH)

    all_models(model_path=cts.ML_RESULTS_DIR / "logo_cv" / 'cn_mrmr', dataset='cn_mrmr',
               feature_selection=1,
               methods=['mrmr'], algo='XGB')
