import sys
import numpy as np

sys.path.append("/home/sheina/armand_repo/")
sys.path.append("/home/sheina/armand_repo/sheina")
import train_model
import os
import utils.consts as cts
import pathlib
import bayesiansearch as bs

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from metrics import eval
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from bayesiansearch import set_path
from sklearn.linear_model import LogisticRegression as LR

algo = 'RF'
N_model = [23, 110, 133, 135]
NM = 5

# DATA_PATH = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')
# path_all = cts.RESULTS_DIR/"logo_cv" / algo

colors_six = ['#307DA6', '#A65730', '#6F30A6', '#A6304F', '#A69E30', '#30A640']
light_colors = ['#B0E7FF', '#FFD7B0', '#BFC0FF', '#EFB0DF', '#FFEEB0', '#C0FFD0']
DATA_PATH = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')


def vt_group(ids):
    x_vt, y_vt, ids = train_model.create_dataset(ids, np.ones([len(ids), ]), path=DATA_PATH, model=1)
    return x_vt, y_vt


f2 = lambda x: list(map('{:.2f}'.format, x))


def generate_frame(parts, colors, pos):
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        # for partname in ('cbars', 'cmedianns', 'cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor("#0D232E")
    for i in range(len(pos)):
        parts['bodies'][i].set_facecolor(colors[i])
        parts['bodies'][i].set_alpha(0.8)
    return


pos = [0, 1, 2, 3, 4]
colors = ['#003049', '#245D7C', '#307DA6', '#5F6F77', '#773000']


def intrp_model(path, features_model, results_dir, feature_selection):
    # load model:
    opt = joblib.load(results_dir / path / 'opt.pkl')
    x_test = joblib.load(results_dir / path / 'X_test.pkl')
    y_test = joblib.load(results_dir / path / 'y_test.pkl')
    features = 0
    # create test set:

    if feature_selection:
        features = joblib.load(results_dir / path / 'features.pkl')
        x_test = train_model.features_mrmr(x_test, list(features_model[0]), list(features))
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


def all_models(model_path, results_dir=cts.RESULTS_DIR, dataset='rbdb_10'):
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
        path_d[i] = set_path('XGB', dataset_n, i, results_dir=results_dir)
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
                'RF model 4 (' + str(np.round(AUROC[3], 2)) + ')',
                'RF model 5 (' + str(np.round(AUROC[4], 2)) + ')')
               , facecolor='white', framealpha=0.8, loc=4)

    plt.title('Receiving operating curve')
    plt.xlabel('1-Sp')
    plt.ylabel('Se')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.grid()
    plt.savefig(model_path / 'AUROC.png', dpi=400, transparent=True)
    plt.show()


# def train_validation_results(opt, typ):
#
#     features_train, y_train,train_ids_groups  = create_dataset(cts.ids_train, cts.y_train, path = DATA_PATH, model = 0)
#     features_train = model_features(features_train, typ)
#     cv_groups = split_to_group(train_ids_groups)
#     mux = pd.MultiIndex.from_product([['Train', 'Validation'], cts.METRICS])
#     train_val_data = pd.DataFrame(0, index = np.unique(cv_groups), columns = mux )
#     logo = LeaveOneGroupOut()
#     i= 0
#     for train_index, test_index in logo.split(features_train, y_train, cv_groups):
#         ftr, fts = features_train[train_index], features_train[test_index]
#         ytr, yts = y_train[train_index], y_train[test_index]
#         results_ftr = eval(opt, ftr, ytr)
#         results_fts = eval(opt, fts, yts)
#         train_val_data['Train'].loc[i] = np.asarray(results_ftr)
#         train_val_data['Validation'].loc[i] = np.asarray(results_fts)
#         i = i+1
#     #for col in train_val_data.columns:
#     #add: summery with median and IQR
#     #save it as exel, maybe outside
#     return train_val_data


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

    all_models(model_path=cts.RESULTS_DIR / "logo_cv" / '22_10_XGB', dataset='22_10_XGB')
