from sklearn.datasets import load_digits
from sklearn.svm import SVC
import os
import sys
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import StratifiedKFold
# import utils.consts as cts
import sheina.consts as cts
import sklearn.utils.class_weight as skl_cw
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
# import datetime
from sklearn.metrics import balanced_accuracy_score

# from train_prediction_model import type, dataset

search_spaces_RF = {
    'model__n_estimators': Integer(10, 300),
    'model__criterion': ['gini', 'entropy'],
    'model__max_depth': Integer(2, 50),
    'model__min_samples_leaf': Integer(1, 100),
    'model__max_features': [0.05, 0.1, 0.3, 0.5, 0.8, 1],
    # 'model__max_leaf_nodes': [None],
    #    'model__max_samples': Real(0.3, 0.99),
}
search_spaces_XGB = {
    'model__n_estimators': Integer(10, 300),
    'model__max_depth': Integer(2, 50),
    'model__reg_alpha': Integer(40, 180),
    'model__reg_lambda': [0, 1],
    'model__eta': [0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1],
    'model__gamma': [0, 1, 10, 30, 50, 80, 100],
    'model__min_child_weight': [1, 10, 30, 50, 80, 100],
    'model__colsample_bytree': [0.05, 0.1, 0.3, 0.5, 0.8, 1],
    'model__subsample': [0.05, 0.1, 0.3, 0.5, 0.8, 1],
    'model__scale_pos_weight': [0, 1, 10, 30, 50, 80, 100],
}


def split_to_group(ids_group, list_ids_vt, list_ids_no_VT, n_vt=2):
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


def hyperparamaters_comb(dict_vals):
    """ This function returns all the possible combinations of hyperparameters.
    Returns a list of dictionaries. Each dict contains the name and the value of the hyperparameter
    for the given combination.
    :param dict_vals: Dictionnary containing as keys the different hyperparameter names and as values the different values they take.
    :returns dict_comb: The output list."""
    hyper_names = list(dict_vals.keys())
    n_hyper = len(hyper_names)
    combinations = list(itertools.product(*list(dict_vals.values())))
    dict_comb = [{hyper_names[i]: comb[i] for i in range(n_hyper)} for comb in combinations]
    return dict_comb


def stratify(train_pat_features, train_pat_labels, train_pat_ids, normalize, n_folds):
    skf = StratifiedKFold(n_folds)
    split_pat = list(skf.split(train_pat_features, train_pat_labels))
    datasets_valid = []
    for train_index, valid_index in split_pat:
        _, X_valid = train_pat_features[train_index], train_pat_features[valid_index]
        _, y_valid = train_pat_labels[train_index], train_pat_labels[valid_index]
        # TODO: add normalization for the validation set
        if normalize:
            standartization = StandardScaler()
            X_valid = standartization.fit_transform(X_valid)
        datasets_valid.append(
            (train_pat_ids[valid_index], X_valid, y_valid,))
    return datasets_valid


def set_path(algo, dataset, typ, results_dir):
    if not os.path.exists(results_dir / "logo_cv" / dataset / str(algo + '_' + str(typ))):
        os.makedirs(results_dir / "logo_cv" / dataset / str(algo + '_' + str(typ)))
    return results_dir / "logo_cv" / dataset / str(algo + '_' + str(typ))


def save_model(final_dict, clf, path):
    with open(path, 'wb') as file:
        final_dict['classifier'] = clf
        pickle.dump(final_dict, file)


def load_model(path):
    with open(path, 'rb') as file:
        clf_dict = pickle.load(file)
    return clf_dict


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


# def loo_cv(ids_len): #leave one out cros validation generator
#     n = len(ids_len)
#     start = 0
#     end = 0
#     for i, n_len in enumerate(ids_len):
#         start =  end
#         end = end + n_len
#         idx = np.arange(start, end, dtype=int)
#         idx_list = [idx]*n
#         yield idx_list

def rc_scorer(estimator, X, y):
    y_hat = estimator.predict(X)
    if len(np.unique(y_hat)) == 2:
        return roc_auc_score(y, y_hat)
    else:
        return 0


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


def vt_bayesianCV(train_pat_features, train_pat_labels, algo, groups, normalize=False,
                  weighting=True, n_jobs=20, typ=1, dataset='both', results_dir=cts.RESULTS_DIR):
    if weighting:
        clf = cts.class_funcs[algo](class_weight='balanced', n_jobs=n_jobs)
    else:
        clf = cts.class_funcs[algo](n_jobs=n_jobs)
    if normalize:
        standartization = StandardScaler()
    pipe = Pipeline([
        ('normalize', standartization), ('model', clf)
    ])
    logo = LeaveOneGroupOut()
    if algo == 'RF':
        search_space = search_spaces_RF
    elif algo == 'XGB':
        search_space = search_spaces_XGB
    opt = BayesSearchCV(
        pipe,
        search_spaces=search_space,

        # search_spaces={
        #     'model__C': Real(0.1, 1000),
        # },
        scoring=rc_scorer,
        n_iter=600,

        cv=logo.split(train_pat_features, train_pat_labels, groups=groups),
        return_train_score=True, verbose=1, n_jobs=n_jobs)

    # callback handler
    def on_step(res):
        path = set_path(algo, dataset, typ, results_dir)
        if res.func_vals.shape[0] == 1:
            results_c = pd.DataFrame(
                columns=['combination', 'score'])
        else:
            results_c = pd.read_csv(path / 'results.csv')
            results_c = results_c.set_axis(results_c['Unnamed: 0'], axis='index')
            results_c = results_c.drop(columns=['Unnamed: 0'])
        # c_iter = res.x_iters[-1]
        # score =res.func_vals[-1]
        # c_iter = c_iter.append(score)
        results_c.loc[results_c.shape[0]] = [res.x_iters[-1], res.func_vals[-1]]
        results_c.to_csv(path / 'results.csv')

    opt.fit(train_pat_features, train_pat_labels, eval_metric='auc')  # callback=on_step
    delattr(opt, 'cv')
    # save your model or results
    path = set_path(algo, dataset, typ, results_dir)

    plt.show()

    return opt
