import sys
import numpy as np

sys.path.append("/home/sheina/VT_risk_prediction/")
import train_model
import sheina.consts as cts
import pathlib
import sheina.bayesiansearch as bs
import joblib
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from metrics import eval
import pandas as pd

algo = 'RF'
N_model = [23, 110, 133, 135]
NM = 5

# DATA_PATH = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')
# path_all = cts.RESULTS_DIR/"logo_cv" / algo

colors_six = ['#307DA6', '#A65730', '#6F30A6', '#A6304F', '#A69E30', '#30A640']
light_colors = ['#B0E7FF', '#FFD7B0', '#BFC0FF', '#EFB0DF', '#FFEEB0', '#C0FFD0']
DATA_PATH = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')


def plot_violin(data, leg_str_arr, title, save_path):
    plt.style.use('bmh')
    plt.rcParams.update({'font.size': 16})
    color = 'steelblue'
    colors = ['dodgerblue', 'blue', 'darkcyan', 'royalblue']
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    violins = axe.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    for i, vp in enumerate(violins['bodies']):
        vp.set_facecolor(colors[i])
        vp.set_edgecolor(color)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = violins[partname]
        vp.set_edgecolor(color)
    axe.set_xlabel(title)
    quartile11, medians1, quartile31 = np.percentile(data[0], [25, 50, 75])
    quartile12, medians2, quartile32 = np.percentile(data[1], [25, 50, 75])
    quartile13, medians3, quartile33 = np.percentile(data[2], [25, 50, 75])
    quartile14, medians4, quartile34 = np.percentile(data[3], [25, 50, 75])
    inds = range(1, 5)
    axe.scatter(inds, [medians1, medians2, medians3, medians4], marker='o', color='white', s=30, zorder=3)
    axe.vlines(inds, [quartile11, quartile12, quartile13, quartile14], [quartile31, quartile32, quartile33, quartile34],
               color=color, linestyle='-', lw=5)
    axe.set_ylabel(r'$probability\ density$')
    axe.legend(leg_str_arr)
    # labels = [r'$C_0$', r'$C_1$']
    plt.tight_layout()
    plt.savefig(save_path / str(title + '.png'))
    plt.show()


def vt_segments(ids):
    x_vt, y_vt, ids = create_dataset(ids, np.ones([len(ids), ]), path=DATA_PATH, model=1, n_pools=1)
    return x_vt


def sus_vt(ids):
    x_vt, y_vt, ids = create_dataset(ids, np.zeros([len(ids), ]), path=DATA_PATH, model=0, n_pools=4)
    return x_vt


f2 = lambda x: list(map('{:.2f}'.format, x))
pos = [0, 1, 2, 3, 4]
colors = ['#003049', '#245D7C', '#307DA6', '#5F6F77', '#773000']


def intrp_model(path, features_model, results_dir, feature_selection):
    # load model:
    opt = joblib.load(path / 'opt.pkl')
    x_test = joblib.load(path / 'X_test.pkl')
    y_test = joblib.load(path / 'y_test.pkl')
    features = 0
    # create test set:

    if feature_selection:
        features = joblib.load(path / 'features.pkl')
    #     x_test = train_model.features_mrmr(x_test, list(features_model[0]), list(features))
    results_ts = eval(opt, x_test, y_test)

    return opt, results_ts, x_test, y_test, features


def hyper_model(opt_d, path):
    hyp_pd = pd.DataFrame(columns=opt_d[1].best_params_.keys())
    for i in range(1, NM + 1):
        best_hyp = pd.DataFrame(opt_d[i].best_params_, columns=opt_d[i].best_params_.keys(), index=[i])
        hyp_pd = hyp_pd.append(best_hyp)
    hyp_pd.to_excel(path / 'hyperparameters.xlsx')


def ext_test_set(opt_d, model_path, features, features_selection, features_model):
    ext_test_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_VT_ids.npy'))
    ext_test_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_non_VT_ids.npy'))
    y_ext = np.concatenate([np.ones([1, len(ext_test_vt)]), np.zeros([1, len(ext_test_no_vt)])], axis=1).squeeze()

    x_ext, y_ext, ext_ids_groups = train_model.create_dataset(ext_test_vt + ext_test_no_vt, y_ext, path=DATA_PATH,
                                                              model=0, n_pools=1)
    results_ext = {}
    for i in range(1, NM + 1):
        x_ext_i = train_model.model_features(x_ext, i)
        if features_selection:
            x_ext_i = train_model.features_mrmr(x_ext_i, list(features_model[i][0]), list(features[i]))
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
    md_test = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/md_test.npy'))

    # create datasets
    for i in range(1, NM + 1):
        X_vt = vt_segments(ids_sp)
        X_vt = train_model.model_features(X_vt, i, with_pvc=True)
        X_vt_sus_p = sus_vt(md_test)
        X_vt_sus_p = train_model.model_features(X_vt_sus_p, i, with_pvc=True)
        X_vt_p = x_test[i][y_test[i] == 1, :]
        X_non_vt_p = x_test[i][y_test[i] == 0, :]

        # calc probabilities
        p_vt = opt[i].predict_proba(X_vt)[:, 1]
        p_vt_sus_p = opt[i].predict_proba(X_vt_sus_p)[:, 1]
        p_vt_p = opt[i].predict_proba(X_vt_p)[:, 1]
        p_non_vt_p = opt[i].predict_proba(X_non_vt_p)[:, 1]

        p_all = np.concatenate([p_vt, p_vt_p, p_vt_sus_p, p_non_vt_p])
        plot_violin([p_vt, p_vt_p, p_vt_sus_p, p_non_vt_p],
                    ['VT segments', 'VT patients', 'Suspected as VT patients', 'Non VT patients'], 'model ' + str(i),
                    model_path)


def all_models(model_path, results_dir=cts.RESULTS_DIR, dataset='rbdb_10', algo='RF'):
    opt_d = {}
    results_d = {}
    path_d = {}
    features_model = {}
    x_test_d = {}
    features_d = {}
    train_val_data_d = {}
    y_test_d = {}
    feature_selection = 0
    method = '_RFE'
    dataset_n = dataset
    with_ext_test = False
    exmp_features = pd.read_excel(cts.VTdb_path + 'normalized/1020D818/features.xlsx', engine='openpyxl')
    # exmp_features = pd.read_excel(cts.VTdb_path + 'ML_model/1419Ec09/features.xlsx', engine='openpyxl')
    features_arr = np.asarray(exmp_features.columns[1:])
    features_list = bs.choose_right_features(np.expand_dims(features_arr, axis=0))

    ext_test_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_VT_ids.npy'))
    ext_test_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_non_VT_ids.npy'))
    ids_tn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_no_VT_ids.npy'))
    ids_sn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_no_VT_ids.npy'))
    ids_tp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_VT_ids.npy'))
    ids_sp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))
    ids_vn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_val_no_VT_ids.npy'))
    y_test = np.concatenate([np.ones([1, len(ids_sp)]), np.zeros([1, len(ids_sn)])], axis=1).squeeze()

    for i in range(1, NM + 1):
        if feature_selection:
            dataset_n = dataset + method
        path_d[i] = pathlib.PurePath(model_path / str('RF_' + str(i)))
        features_model[i] = train_model.model_features(features_list, i, with_pvc=True)
        opt_d[i], results_d[i], x_test_d[i], y_test_d[i], features_d[i] = intrp_model(path_d[i], features_model[i],
                                                                                      results_dir, feature_selection)

    # for i in range(1, 5):
    #     _ = plot_objective(opt_d[i].optimizer_results_[0],
    #                        dimensions=['criterion', 'max_depth','max_features','min_samples_leaf', 'n_estimators'],
    #                        n_minimum_search=int(1e8))
    #     plt.savefig(path_d[i] /str(str(i) + 'RF.png'))
    #     plt.show()
    train_val(opt_d, model_path)
    hyper_model(opt_d, model_path)
    clac_probs(x_test_d, y_test_d, opt_d, model_path)
    if with_ext_test:
        results_ext = ext_test_set(opt_d, model_path, features_d, feature_selection, features_model)
    results = pd.DataFrame.from_dict(results_d)
    results = results.transpose()
    results = results.set_axis(cts.METRICS, axis=1)
    results = results.apply(f2)
    results.to_excel(model_path / 'all_model_results.xlsx')

    plt.style.use('bmh')
    fig = plt.plot([1])
    AUROC = [results_d[1][6], results_d[2][6], results_d[3][6], results_d[4][6]]  # , results_d[5][6]]
    if with_ext_test:
        AUROC_ext = [results_ext[1][6], results_ext[2][6], results_ext[3][6], results_ext[4][6], results_ext[5][6]]
    x_axis = np.arange(NM)
    plt.bar(x_axis - 0.2, AUROC, 0.4, color=(colors_six[0]), label='RBDB test set')
    if with_ext_test:
        plt.bar(x_axis + 0.2, AUROC_ext, 0.4, color=(light_colors[0]), label='UVAF dataset')
    stic_list = range(1, NM + 1)
    plt.xticks(range(NM), stic_list, fontsize='small')
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
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 20))

    for i in range(1, 5):
        importance_array = np.zeros([len(features_model[i][0]), ])
        # axi = axes[i-1]
        axi = axes[int((i - 1) / 2), (i - 1) % 2]
        if feature_selection:
            n_F = np.minimum(n_F_max, len(features_d[i]))
            features = list(features_d[i])
            importance_arrayi = opt_d[i].best_estimator_.named_steps.model.feature_importances_
            for j in range(len(importance_arrayi)):
                index = list(features_model[i][0]).index(features[j])
                importance_array[index] = importance_arrayi[j]
        else:
            n_F = n_F_max
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

    for i in range(1, NM + 1):
        prob_rf[i] = opt_d[i].predict_proba(x_test_d[i])
        # prob_rf_ext[i] = opt_d[i].predict_proba(x_test_d[i])
        # train_val_data_d[i] = train_validation_results(opt_d[i], i)

    fig = plt.figure()
    plt.style.use('bmh')
    rf_plot_all = []
    rf_point_all = []

    for i in range(1, NM + 1):
        tpr_rf, fpr_rf, th = roc_curve(y_test_d[i], prob_rf[i][:, 1])
        rf_plot, = plt.plot(tpr_rf, fpr_rf, colors_six[i - 1])  ## add labels
        # rf_point, =plt.plot((1-results_d[i][3]).astype(float), (results_d[i][2]).astype(float), colors_six[i-1],marker="+",markersize=15)
        rf_plot_all.append(rf_plot)
        # rf_point_all.append(rf_point)
        plt.plot(tpr_rf, tpr_rf, 'k')
    plt.legend((rf_plot_all),
               ('RF model 1 (' + str(np.round(AUROC[0], 2)) + ')',
                'RF model 2 (' + str(np.round(AUROC[1], 2)) + ')',
                'RF model 3 (' + str(np.round(AUROC[2], 2)) + ')',
                'RF model 4 (' + str(np.round(AUROC[3], 2)) + ')',)
               # 'RF model 5 ('+str(np.round(AUROC[4], 2))+')')
               , facecolor='white', framealpha=0.8, loc=4)

    plt.title('Receiving operating curve')
    plt.xlabel('1-Sp')
    plt.ylabel('Se')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.grid()
    plt.savefig(model_path / 'AUROC.png', dpi=400, transparent=True)
    plt.show()


def train_val(opt_d, model_path):
    columns = ['AUROC train', 'AUROC validation']
    train_val = pd.DataFrame(columns=columns, index=range(1, len(opt_d) + 1))
    for j in opt_d:
        opt = opt_d[j]
        be = opt.best_index_
        mean_test = np.around(opt.cv_results_['mean_test_score'][be], 2)
        std_test = np.around(opt.cv_results_['std_test_score'][be], 2)
        mean_train = np.around(opt.cv_results_['mean_train_score'][be], 2)
        std_train = np.around(opt.cv_results_['std_train_score'][be], 2)
        train_ms = str(mean_train) + '±' + str(std_train)
        val_ms = str(mean_test) + '±' + str(std_test)
        train_val['AUROC train'][j] = train_ms
        train_val['AUROC validation'][j] = val_ms

    train_val.to_excel(model_path / 'all_model_train_val.xlsx')


def eval_one_model(results_dir, path):
    opt = joblib.load(results_dir / path / 'opt.pkl')
    x_test = joblib.load(results_dir / path / 'X_test.pkl')
    y_test = joblib.load(results_dir / path / 'y_test.pkl')
    # features = joblib.load(results_dir / path / 'features.pkl')
    # x_test = train_model.features_mrmr(x_test, list(features_list), list(features))
    results_ts = eval(opt, x_test, y_test)

    a = 5


if __name__ == '__main__':
    # eval_one_model(cts.RESULTS_DIR, 'logo_cv/22_10_XGB/XGB_4/')

    all_models(model_path=cts.RESULTS_DIR / "logo_cv" / '10_22', dataset='10_22')
