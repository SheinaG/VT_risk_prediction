from ML.ML_utils import *
from utils import consts as cts
from utils.base_packages import *
from utils.metrics import *
from utils.plots import *

# DATA_PATH = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')
# path_all = cts.RESULTS_DIR/"logo_cv" / algo

colors_six = ['#307DA6', '#A65730', '#6F30A6', '#A6304F', '#A69E30', '#30A640']
light_colors = ['#B0E7FF', '#FFD7B0', '#BFC0FF', '#EFB0DF', '#FFEEB0', '#C0FFD0']
DATA_PATH = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')


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



f2 = lambda x: list(map('{:.2f}'.format, x))
pos = [0, 1, 2, 3, 4]
colors = ['#003049', '#245D7C', '#307DA6', '#5F6F77', '#773000']


def intrp_model(path, features_model, results_dir, feature_selection, features_str=''):
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
    results_ts = eval(opt, x_test, y_test)

    return opt, results_ts, x_test, y_test, features


def ext_test_set(opt_d, model_path, features, features_selection, features_model):
    ext_test_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_VT_ids.npy'))
    ext_test_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_non_VT_ids.npy'))
    y_ext = np.concatenate([np.ones([1, len(ext_test_vt)]), np.zeros([1, len(ext_test_no_vt)])], axis=1).squeeze()

    x_ext, y_ext, ext_ids_groups = create_dataset(ext_test_vt + ext_test_no_vt, y_ext, path=DATA_PATH,
                                                  model=0, n_pools=1)
    results_ext = {}
    for i in range(1, cts.NM + 1):
        x_ext_i = model_features(x_ext, i)
        if features_selection:
            x_ext_i = features_mrmr(x_ext_i, list(features_model[i][0]), list(features[i]))
        results_ext[i] = eval(opt_d[i], x_ext_i, y_ext)
    results = pd.DataFrame.from_dict(results_ext)
    results = results.transpose()
    results = results.set_axis(cts.METRICS, axis=1)
    results = results.apply(f2)
    results.to_excel(model_path / 'results_ext.xlsx')
    return results_ext


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
               methods=['mrmr']):
    opt_d = {}
    results_d = {}
    path_d = {}
    features_model = {}
    x_test_d = {}
    features_d = {}
    train_val_data_d = {}
    y_test_d = {}
    features_str = ''

    dataset_n = dataset
    with_ext_test = False
    exmp_features = pd.read_excel(cts.VTdb_path / 'ML_model/1020D818/features_nd.xlsx', engine='openpyxl')
    # exmp_features = pd.read_excel(cts.VTdb_path + 'ML_model/1419Ec09/features.xlsx', engine='openpyxl')
    features_arr = np.asarray(exmp_features.columns[1:])
    features_list = choose_right_features(np.expand_dims(features_arr, axis=0))

    for i in range(1, cts.NM + 1):
        if feature_selection:
            dataset = dataset + '_' + '_'.join(methods)
            features_str = str('features' + methods[0] + '.pkl')
        path_d[i] = pathlib.PurePath(model_path / str('RF_' + str(i)))
        features_model[i] = model_features(features_list, i, with_dems=True)
        opt_d[i], results_d[i], x_test_d[i], y_test_d[i], features_d[i] = intrp_model(path_d[i], features_model[i],
                                                                                      results_dir, feature_selection,
                                                                                      features_str)

    train_val(opt_d, model_path)
    hyper_model(opt_d, model_path)
    # clac_probs(x_test_d, y_test_d, opt_d, model_path)
    if with_ext_test:
        results_ext = ext_test_set(opt_d, model_path, features_d, feature_selection, features_model)
    results = pd.DataFrame.from_dict(results_d)
    results = results.transpose()
    results = results.set_axis(cts.METRICS, axis=1)
    results = results.apply(f2)
    results.to_excel(model_path / 'all_model_results.xlsx')

    plt.style.use('bmh')
    fig = plt.plot([1])
    AUROC = [results_d[1][6], results_d[2][6], results_d[3][6], results_d[4][6], results_d[5][6]]
    if with_ext_test:
        AUROC_ext = [results_ext[1][6], results_ext[2][6], results_ext[3][6], results_ext[4][6], results_ext[5][6]]
    x_axis = np.arange(cts.NM)
    plt.bar(x_axis - 0.2, AUROC, 0.4, color=(colors_six[0]), label='RBDB test set')
    if with_ext_test:
        plt.bar(x_axis + 0.2, AUROC_ext, 0.4, color=(light_colors[0]), label='UVAF dataset')
    stic_list = range(1, cts.NM + 1)
    plt.xticks(range(cts.NM), stic_list, fontsize='small')
    min_auc = np.min(AUROC)
    if with_ext_test:
        min_auc = np.min(AUROC + AUROC_ext)
    max_auroc = np.max(AUROC)
    if with_ext_test:
        max_auroc = np.max(AUROC + AUROC_ext)
    plt.ylim((min_auc - 0.1, max_auroc + 0.1))
    plt.ylabel('AUROC')
    plt.xlabel('Model #')
    plt.legend(loc='upper left')
    plt.title('AUROC per model')
    plt.savefig(model_path / 'auroc_bar.png', dpi=400, transparent=True)
    plt.show()

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
            stic_list.append(features_model[i][0][indices[j]])
        axi.set_yticks(range(n_F))
        axi.set_yticklabels(stic_list, rotation='horizontal', fontsize=16)
        axi.invert_yaxis()
    plt.tight_layout()

    fig.savefig(model_path / 'importance_rf.png', dpi=400, transparent=True)
    plt.show()

    prob_rf = {}
    prob_rf_train = {}

    for i in range(1, cts.NM + 1):
        prob_rf[i] = opt_d[i].predict_proba(x_test_d[i])
        # prob_rf_ext[i] = opt_d[i].predict_proba(x_test_d[i])
        # train_val_data_d[i] = train_validation_results(opt_d[i], i)

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
                                                      majority_vote=False, soft_lines=False)

        low_auroc.append(low_auroc_i)
        high_auroc.append(high_auroc_i)
        # plt.plot(tpr_rf, tpr_rf, 'k')
    plt.legend(facecolor='white', framealpha=0.8, loc=4)

    plt.title('Receiving operating curve')
    plt.xlabel('1-Sp')
    plt.ylabel('Se')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.grid()
    plt.savefig(model_path / 'AUROC.png', dpi=400, transparent=True)
    plt.show()


def eval_one_model(results_dir, path):
    opt = joblib.load(results_dir / path / 'opt.pkl')
    x_test = joblib.load(results_dir / path / 'X_test.pkl')
    y_test = joblib.load(results_dir / path / 'y_test.pkl')
    results_ts = eval(opt, x_test, y_test)
    return results_ts



if __name__ == '__main__':
   # eval_one_model(cts.ML_RESULTS_DIR, 'logo_cv/new_dem/RF_5/')

   all_models(model_path=cts.ML_RESULTS_DIR / "logo_cv" / 'new_dem', dataset='new_dem', feature_selection=0,
              methods=['mrmr'])
