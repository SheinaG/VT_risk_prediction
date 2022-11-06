import pathlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metrics import eval
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, \
    auc

import bayesiansearch as bs
import train_model
import utils.consts as cts

exmp_features = pd.read_excel(cts.VTdb_path + 'ML_model/1601/features.xlsx', engine='openpyxl')
features_arr = np.asarray(exmp_features.columns[1:])
features_list = bs.choose_right_features(np.expand_dims(features_arr, axis=0))

colors_six = ['#307DA6', '#A65730', '#6F30A6', '#A6304F', '#A69E30', '#30A640']
light_colors = ['#B0E7FF', '#FFD7B0', '#BFC0FF', '#EFB0DF', '#FFEEB0', '#C0FFD0']

f2 = lambda x: list(map('{:.2f}'.format, x))


def reaper_dict(dict, path):
    new_dict = {}
    j = 0
    for key in dict:
        new_dict[j] = dict[key]
        j = j + 1
    with open(path, 'wb') as f:
        joblib.dump(new_dict, f)


def results_list(results_d, path):
    all_model_results = pd.DataFrame(columns=cts.METRICS)
    results = pd.DataFrame.from_dict(results_d)
    results = results.transpose()
    results = results.set_axis(cts.METRICS, axis=1)
    if len(results.index) > 9:
        results = results.drop(index=[9, 10])
    results = results.round(2)
    results.to_excel(path / 'model_results.xlsx')
    for result_s in results:
        data = np.asarray(results[result_s])
        mean = np.around(np.mean(data), 2)
        std = np.around(np.std(data), 2)
        all_model_results[result_s] = [str(mean) + '±' + str(std)]
    return all_model_results


def hyper_model(opt_d, path, k_test):
    hyp_pd = pd.DataFrame(columns=opt_d[0].best_params_.keys())
    for i in range(k_test):
        best_hyp = pd.DataFrame(opt_d[i].best_params_, columns=opt_d[i].best_params_.keys(), index=[i])
        hyp_pd = hyp_pd.append(best_hyp)
    hyp_pd.to_excel(path / 'hyperparameters.xlsx')


def plot_roc_bsqi(K_test, models, arc_roc, path):
    fig, axe = plt.subplots(nrows=4, ncols=1, figsize=(20, 10), sharex=True)
    for i in models:
        arc_roci = arc_roc[i]
        axe[i - 1].bar(range(K_test), list(arc_roci.values())[:K_test], color=colors_six[i - 1])
        axe[i - 1].set_xticks(range(len(cts.ids_VT)))
        axe[i - 1].set_xticklabels(cts.ids_VT, fontsize=20)
        axe[i - 1].set_ylim(((np.min(list(arc_roci.values())[:K_test])) - 0.1, 1))
        axe[i - 1].set_title('auroc model ' + str(i))
    plt.savefig(path / 'roc_per_rec.png')
    plt.show()


def nested_importance(K_test, opt_d, axe, model, features_d={}, mrmr=0):
    n_F = 10
    score_arrey = []
    features_model = list(train_model.model_features(features_list, model)[0])
    importance_array = np.zeros([K_test, len(features_model)])
    if mrmr:
        for i in range(K_test):
            features = features_d[i]
            importance_arrayi = opt_d[i].best_estimator_.named_steps.model.feature_importances_
            for j in range(len(importance_arrayi)):
                index = features_model.index(features[j])
                importance_array[i, index] = importance_arrayi[j]
    else:
        for i in range(K_test):
            importance_array[i, :] = opt_d[i].best_estimator_.named_steps.model.feature_importances_
    importance_mean = np.mean(importance_array, axis=0)
    importance_std = np.std(importance_array, axis=0)

    indices = np.argsort(importance_mean)[::-1]
    axe.set_title(" model " + str(model))
    axe.barh(range(n_F), importance_mean[indices[0:n_F]], color=colors_six[model - 1], align="center")
    stic_list = []
    axe.errorbar(importance_mean[indices[0:n_F]], range(n_F), xerr=importance_std[indices[0:n_F]], linestyle='None',
                 color='black')

    for j in range(n_F):
        stic_list.append(features_model[indices[j]])
    axe.set_yticks(range(n_F))
    axe.set_yticklabels(stic_list, rotation='horizontal', fontsize=16)
    axe.invert_yaxis()
    plt.tight_layout()


def roc_plot_envelope(y_preds, y_tests, K_test, augmentation, typ, title='', majority_vote=False, soft_lines=False):
    fprs = []
    tprs = []
    fprs_major = []
    tprs_major = []
    for i in range(K_test):
        # Plot of a ROC curve:
        y_true = y_tests[i]
        y_score = y_preds[i][:, 1]
        fpr, tpr, THs = roc_curve(y_true.ravel(), y_score.ravel())
        fprs.append(fpr)
        tprs.append(tpr)
        if majority_vote:
            fpr_major, tpr_major = majority_roc(THs, augmentation, y_score, y_true)
            fprs_major.append(fpr_major)
            tprs_major.append(tpr_major)

        if majority_vote and soft_lines:
            plt.plot(fpr_major, tpr_major, '--', lw=1, color='lightgray')
        elif soft_lines:
            plt.plot(fpr, tpr, '--', lw=1, color='lightgray')
    color = colors_six[typ - 1]
    fill_color = light_colors[typ - 1]
    if majority_vote:
        fpr_all, mean_tpr, max_tpr, min_tpr = get_K_rocs(K_test, fprs_major, tprs_major)
    else:
        fpr_all, mean_tpr, max_tpr, min_tpr = get_K_rocs(K_test, fprs, tprs)

    roc_auc = auc(fpr_all, mean_tpr)
    plt.plot(fpr_all, mean_tpr, lw=2, color=color, label='model ' + str(typ))
    # label = f'K={K_test}; (AUROC = {round(roc_auc, 3)})'
    plt.fill_between(fpr_all, min_tpr, max_tpr, color=fill_color)

    # Add coin flipping line:
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Sp')
    plt.ylabel('Se')
    plt.title(title)
    plt.legend(loc="lower right")


def majority_roc(THs, augmentation, y_score, y_true):
    # Reshape to work as an array - every different patient is a row
    y_score_shaped = np.reshape(y_score, (-1, augmentation))
    y_true_singles = np.reshape(y_true, (-1, augmentation))[:, 0]
    tpr = []
    fpr = []

    # compute Se and 1-Sp for each threshold
    for i, TH in enumerate(THs):
        thresholded = y_score_shaped > TH
        thresholded_sum = np.sum(thresholded, axis=1)
        y_pred_singles = (thresholded_sum > (augmentation / 2)) * 1  # preds by majority
        tn, fp, fn, tp = confusion_matrix(y_true_singles, y_pred_singles).ravel()
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    # Set limits between 0 and 1:
    tpr[0] = 0
    tpr[-1] = 1
    fpr[0] = 0
    fpr[-1] = 1

    return np.array(tpr), np.array(fpr)


def get_K_rocs(K_test, fprs, tprs):
    """
    Get the roc parameter suits for the roc mean and envelope
    """
    lengths = []
    for i in range(K_test):
        lengths.append(fprs[i].shape)

    max_l = np.max(lengths)
    max_index = np.argmax(lengths)

    tprs_interp = np.zeros((K_test, max_l))
    fpr_all = fprs[max_index]

    for i in range(K_test):
        f = interp1d(fprs[i], tprs[i])
        tprs_interp[i, :] = f(fpr_all)

    mean_tpr = np.mean(tprs_interp, axis=0)
    max_tpr = np.max(tprs_interp, axis=0)
    min_tpr = np.min(tprs_interp, axis=0)

    return fpr_all, mean_tpr, max_tpr, min_tpr


def nested_results(results_dir):
    algo = 'RF'
    K_test = 9
    auroc_model = {}
    mrmr = 1
    dataset = 'mannw'
    path_all = cts.RESULTS_DIR / "logo_cv" / dataset
    results_pd = pd.DataFrame(columns=cts.METRICS)
    for i in range(1, 5):

        path = bs.set_path(algo, dataset, i, results_dir)
        y_test_d = joblib.load(path / 'y_test_d.pkl')
        y_pred_d = {}
        features_d = {}
        x_test_d = joblib.load(path / 'X_test_d.pkl')
        opt_d = joblib.load(path / 'opt_d.pkl')
        if mrmr:
            features_d = joblib.load(path / 'features_d.pkl')
        results_d = {}
        hyper_model(opt_d, path, K_test)
        auroc_rec = {}
        for j in opt_d:
            opt = opt_d[j]
            features_model = train_model.model_features(features_list, i)
            if mrmr:
                x_test = train_model.features_mrmr(x_test_d[j], list(features_model[0]), features_d[j])
            else:
                x_test = x_test_d[j]
            y_pred_d[j] = opt.predict_proba(x_test)
            auroc_rec[j] = roc_auc_score(y_test_d[j], y_pred_d[j][:, 1])
            results_d[j] = eval(opt, x_test, y_test_d[j])
        results_model = results_list(results_d, path)
        results_pd = results_pd.append(results_model)

        auroc_model[i] = auroc_rec
        fig = plt.figure()
        roc_plot_envelope(y_pred_d, y_test_d, K_test=K_test, augmentation=1, typ=i, title='model ' + str(i),
                          majority_vote=False, soft_lines=True)
        plt.savefig(path / 'auroc_model.png')

    plot_roc_bsqi(K_test, range(1, 5), auroc_model, path_all)
    results_pd.to_excel(path_all / 'results_all.xlsx')

    fig1, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))
    for i in range(1, 5):
        path = bs.set_path(algo, dataset, i, results_dir)
        if mrmr:
            features_d = joblib.load(path / 'features_d.pkl')
        opt_d = joblib.load(path / 'opt_d.pkl')
        nested_importance(K_test, opt_d, axes[i - 1], i, features_d, mrmr=mrmr)
    fig1.savefig(path_all / 'importance_rf.png', dpi=400, transparent=True)
    plt.show()


if __name__ == '__main__':
    bsqi_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/')

    nested_results(results_dir=cts.RESULTS_DIR)
