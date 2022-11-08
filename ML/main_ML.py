import ML.bayesiansearch as bs
from ML.ML_utils import *
from utils import consts as cts
from utils.base_packages import *

# exmp_features = pd.read_excel( cts.VTdb_path + 'ML_model/1601/features.xlsx', engine='openpyxl')
exmp_features = pd.read_excel(cts.VTdb_path / 'ML_model/V720H339/features_nd.xlsx', engine='openpyxl')
features_arr = np.asarray(exmp_features.columns[1:])
features_list = choose_right_features(np.expand_dims(features_arr, axis=0))


def train_by_V_ratio():
    ids_tn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_no_VT_ids.npy'))
    ids_sn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_no_VT_ids.npy'))
    ids_tp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_VT_ids.npy'))
    ids_sp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))
    ids_vn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_val_no_VT_ids.npy'))

    y_train = np.concatenate([np.ones([1, len(ids_tp)]), np.zeros([1, len(ids_tn + ids_vn)])], axis=1).squeeze()
    y_test = np.concatenate([np.ones([1, len(ids_sp)]), np.zeros([1, len(ids_sn)])], axis=1).squeeze()

    # create dataset ( VT each grop)
    x_train, y_train, train_ids_groups = create_dataset(ids_tp + ids_tn + ids_vn, y_train, path=DATA_PATH,
                                                        model=0)
    v_train = x_train[:, -5]
    x_test, y_test, test_ids_groups = create_dataset(ids_sp + ids_sn, y_test, path=DATA_PATH, model=0)
    v_test = x_test[:, -5]


def train_prediction_model(DATA_PATH, results_dir, model_type, dataset, method=''):
    algo = 'RF'
    n_jobs = 4
    feature_selection = 1
    features_model = list(model_features(features_list, model_type, with_pvc=True)[0])

    f_n = cts.num_selected_features_model[model_type - 1]

    y_train = np.concatenate([np.ones([1, len(cts.ids_tp)]), np.zeros([1, len(cts.ids_tn + cts.ids_vn)])],
                             axis=1).squeeze()
    y_test = np.concatenate([np.ones([1, len(cts.ids_sp)]), np.zeros([1, len(cts.ids_sn)])], axis=1).squeeze()

    # create dataset ( VT each grop)
    x_train, y_train, train_ids_groups = create_dataset(cts.ids_tp + cts.ids_tn + cts.ids_vn, y_train, path=DATA_PATH,
                                                        model=0)
    x_train = model_features(x_train, model_type, with_pvc=True)
    train_groups = split_to_group(train_ids_groups, cts.ids_tp, cts.ids_tn + cts.ids_vn, n_vt=12)
    x_test, y_test, test_ids_groups = create_dataset(cts.ids_sp + cts.ids_sn, y_test, path=DATA_PATH, model=0)
    x_test = model_features(x_test, model_type, with_pvc=True)

    if feature_selection:
        dataset = dataset + '_' + method
        path = set_path(algo, dataset, model_type, results_dir)
        StSC = StandardScaler()
        StSc_fit = StSC.fit(x_train)
        x_test = StSc_fit.transform(x_test)
        X_stsc_train = StSc_fit.transform(x_train)
        with open((path / 'StSC.pkl'), 'wb') as f:
            joblib.dump(StSc_fit, f)
        X_df = pd.DataFrame(X_stsc_train, columns=features_model)
        X_train_m, features_new = feature_selection_func(X_df, y_train, method, n_jobs=n_jobs, num=f_n)
        print(features_new)
        x_train = X_train_m
        with open((path / 'features.pkl'), 'wb') as f:
            joblib.dump(features_new, f)
        x_test = features_mrmr(x_test, list(features_model), list(features_new))

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
    DATA_PATH = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')
    results_dir = cts.RESULTS_DIR
    train_prediction_model(DATA_PATH, results_dir, model_type=2, dataset='22_10', method='manww')
    # train_prediction_model(DATA_PATH, results_dir, model_type=2, dataset='rbdb_10', method = 'RFE')
    # train_prediction_model(DATA_PATH, results_dir, model_type=3, dataset='rbdb_10', method = 'RFE')
    # train_prediction_model(DATA_PATH, results_dir, model_type=4, dataset='rbdb_10', method = 'RFE')
