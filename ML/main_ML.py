import sys
sys.path.append("/home/sheina/VT_risk_prediction/")
import ML.bayesiansearch as bs
from ML.ML_utils import *
from utils import consts as cts
from utils.base_packages import *

# exmp_features = pd.read_excel( cts.VTdb_path + 'ML_model/1601/features.xlsx', engine='openpyxl')
exmp_features = pd.read_excel(cts.VTdb_path / 'ML_model/V720H339/features_nd.xlsx', engine='openpyxl')
features_arr = np.asarray(exmp_features.columns[1:])
features_list = choose_right_features(np.expand_dims(features_arr, axis=0))


def rebuild_clf(x_train, y_train, x_test, y_test, params_dict, algo):
    clf = cts.class_funcs[algo](**params_dict, class_weight='balanced',
                                n_jobs=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)[:, 1]
    results_ts = eval(clf, x_test, y_test)
    return y_pred


def train_by_V_ratio():
    y_train = np.concatenate([np.ones([1, len(cts.ids_tp)]), np.zeros([1, len(cts.ids_tn + cts.ids_vn)])],
                             axis=1).squeeze()
    y_test = np.concatenate([np.ones([1, len(cts.ids_sp)]), np.zeros([1, len(cts.ids_sn)])], axis=1).squeeze()

    x_train, y_train, train_ids_groups = create_dataset(cts.ids_tp + cts.ids_tn + cts.ids_vn, y_train, path=cts.ML_path,
                                                        model=0)
    x_test, y_test, test_ids_groups = create_dataset(cts.ids_sp + cts.ids_sn, y_test, path=cts.ML_path, model=0)
    v_train = x_train[:, -10]
    v_test = x_test[:, -10]


def train_prediction_model(DATA_PATH, results_dir, model_type, dataset, methods='', feature_selection=0, n_jobs=10,
                           algo='RF', features_name='features_nd.xlsx'):
    features_model = list(model_features(features_list, model_type, with_dems=True)[0])
    f_n = cts.num_selected_features_model[model_type - 1]

    y_train = np.concatenate([np.ones([1, len(cts.ids_tp)]), np.zeros([1, len(cts.ids_tn + cts.ids_vn)])],
                             axis=1).squeeze()
    y_test = np.concatenate([np.ones([1, len(cts.ids_sp)]), np.zeros([1, len(cts.ids_sn)])], axis=1).squeeze()

    # create dataset ( VT each grop)
    x_train, y_train, train_ids_groups = create_dataset(cts.ids_tp + cts.ids_tn + cts.ids_vn, y_train, path=DATA_PATH,
                                                        model=0, features_name=features_name)
    x_train = model_features(x_train, model_type, with_dems=True)
    train_groups = split_to_group(train_ids_groups)
    x_test, y_test, test_ids_groups = create_dataset(cts.ids_sp + cts.ids_sn, y_test, path=DATA_PATH, model=0,
                                                     features_name=features_name)
    x_test = model_features(x_test, model_type, with_dems=True)

    if feature_selection:
        dataset = dataset + '_' + '_'.join(methods)
        StSC = StandardScaler()
        StSc_fit = StSC.fit(x_train)
        X_stsc_train = StSc_fit.transform(x_train)
        x_test = StSc_fit.transform(x_test)
        X_df = pd.DataFrame(X_stsc_train, columns=features_model)
        features_all = features_model
        for method in methods:
            if method == 'ns':
                X_df, removed_features = remove_not_significant(X_df, y_train)
                x_test = features_mrmr(x_test, list(features_all), list(removed_features), remove=1)
                features_new = x_test.columns
            else:
                X_df, features_new = feature_selection_func(X_df, y_train, method, n_jobs=n_jobs, num=f_n)
                x_test = features_mrmr(x_test, list(features_all), list(features_new), remove=0)
            x_train = X_df
            path = set_path(algo, dataset, model_type, results_dir)
            with open((path / str('features' + method + '.pkl')), 'wb') as f:
                joblib.dump(features_new, f)
            features_all = features_new
        with open((path / 'StSC.pkl'), 'wb') as f:
            joblib.dump(StSc_fit, f)
        print(features_new)

    path = set_path(algo, dataset, model_type, results_dir)
    opt = bs.bayesianCV(x_train, y_train, algo, normalize=1, groups=train_groups,
                        weighting=True, n_jobs=n_jobs, typ=model_type, results_dir=results_dir, dataset=dataset)

    with open((path / 'opt.pkl'), 'wb') as f:
        joblib.dump(opt, f)
    with open((path / 'X_test.pkl'), 'wb') as f:
        joblib.dump(x_test, f)
    with open((path / 'y_test.pkl'), 'wb') as f:
        joblib.dump(y_test, f)


if __name__ == "__main__":
    # train_by_V_ratio()
    for i in range(2, cts.NM + 1):
        train_prediction_model(cts.ML_path, cts.ML_RESULTS_DIR, model_type=i, dataset='new_dem41',
                               methods=['mrmr'],
                               n_jobs=15, feature_selection=1, algo='XGB')
