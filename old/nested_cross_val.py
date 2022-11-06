import pathlib
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import utils.consts as cts
from sklearn.model_selection import LeaveOneGroupOut
import bayesiansearch as bs
from ML import train_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Integer
import joblib
import matplotlib.pyplot as plt

exmp_features = pd.read_excel(cts.VTdb_path + 'ML_model/1601/features.xlsx', engine='openpyxl')
features_arr = np.asarray(exmp_features.columns[1:])
features_list = bs.choose_right_features(np.expand_dims(features_arr, axis=0))


def train_nested_model(DATA_PATH, results_dir, typ):
    algo = 'RF'
    n_jobs = 10
    n_iter = 100
    feature_selection = 1
    opt_d = {}
    X_test_d = {}
    y_test_d = {}
    features_d = {}
    ids_VT = cts.ids_VT + cts.ids_rbdb_VT
    ids_no_VT = cts.ids_rbdb_no_VT + cts.ids_no_VT
    y_all = np.concatenate([np.ones([1, len(ids_VT)]), np.zeros([1, len(ids_no_VT)])], axis=1).squeeze()

    # create dataset ( one VT each grop)
    X, y, ids_groups = train_model.create_dataset(ids_VT + ids_no_VT, y_all, path=DATA_PATH, model=0)
    X = train_model.model_features(X, typ)
    cv_groups = bs.split_to_group(ids_groups, ids_VT, ids_no_VT, n_vt=1)
    logo = LeaveOneGroupOut()
    i = 0
    path = bs.set_path(algo, 'mannw', typ, results_dir)

    # for on groups
    for train_index, test_index in logo.split(X, y, cv_groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_test_d[i] = y_test
        features_model = list(train_model.model_features(features_list, typ)[0])
        if feature_selection:
            if os.path.isfile(path / 'features_d.pkl'):
                features = joblib.load(path / 'features_d.pkl')
                X_test_d[i] = X_test
            else:
                StSC = StandardScaler()
                StSc_fit = StSC.fit(X_train)
                X_stsc_test = StSc_fit.transform(X_test)
                X_stsc_train = StSc_fit.transform(X_train)
                X_df = pd.DataFrame(X_stsc_train, columns=features_model)

                X_test_d[i] = X_stsc_test
                X_train_m, features = train_model.stat_selection('mannw', X_df, y_train)
                print(features)
                X_train = X_train_m
        else:
            X_test_d[i] = X_test
        ids_train = [ids_groups[index] for index in train_index]
        ids_uni_train = np.unique(ids_train)
        ids_VT_train = [i for i in ids_uni_train if i in ids_VT]
        ids_no_vt_train = [i for i in ids_uni_train if i in ids_no_VT]
        cv_train = bs.split_to_group(ids_train, ids_VT_train, ids_no_vt_train, n_vt=2)
        clf = cts.class_funcs[algo](class_weight='balanced', n_jobs=n_jobs)
        standartization = StandardScaler()
        pipe = Pipeline([('normalize', standartization), ('model', clf)])

        logoi = LeaveOneGroupOut()
        opt = BayesSearchCV(
            pipe,
            search_spaces={
                'model__n_estimators': Integer(10, 100),
                'model__criterion': ['gini', 'entropy'],
                'model__max_depth': Integer(3, 15),
                'model__min_samples_leaf': Integer(1, 20),
                'model__max_features': [0.05, 0.1, 0.3, 0.5, 0.8, 1],
            },

            scoring=bs.rc_scorer,
            n_iter=n_iter,
            cv=logoi.split(X_train, y_train, groups=cv_train),
            return_train_score=True, verbose=1, n_jobs=n_jobs)

        # callback handler
        def on_step(res):
            path = bs.set_path(algo, 'nested', typ, results_dir)
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

        opt.fit(X_train, y_train)  # callback=on_step
        delattr(opt, 'cv')
        # save your model or results
        #
        if feature_selection:
            features_d[i] = features
        opt_d[i] = opt
        if i == 8:
            break
        else:
            i = i + 1
    with open((path / 'opt_d.pkl'), 'wb') as f:
        joblib.dump(opt_d, f)
    with open((path / 'X_test_d.pkl'), 'wb') as f:
        joblib.dump(X_test_d, f)
    with open((path / 'y_test_d.pkl'), 'wb') as f:
        joblib.dump(y_test_d, f)
    if feature_selection:
        with open((path / 'features_d.pkl'), 'wb') as f:
            joblib.dump(features_d, f)


if __name__ == '__main__':
    DATA_PATH = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')
    results_dir = cts.RESULTS_DIR
    # train_nested_model(DATA_PATH, results_dir, typ=1)
    # train_nested_model(DATA_PATH, results_dir, typ = 2)
    # train_nested_model(DATA_PATH, results_dir, typ=3)
    train_nested_model(DATA_PATH, results_dir, typ=4)
