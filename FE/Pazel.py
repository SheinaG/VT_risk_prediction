from utils.base_packages import *
import utils.consts as cts

from FE.features_per_window import *
from FE.window_fe import *
from FE.statistical_test import *
from ML.main_ML import train_prediction_model
from ML.results import all_models
from ML.from_win_to_rec import *
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    win_len_n = 'win_len_60'
    n_pools = 1

    dataset = 'rbdb'
    win_len = 60
    ecg_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / dataset
    bsqi_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / win_len_n
    fiducials_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / 'fiducials'
    features_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/') / 'win_len' / win_len_n
    results_dir = cts.ML_RESULTS_DIR / 'logo_cv' / win_len_n
    data_path = cts.VTdb_path
    algo = 'RF'
    n_jobs = 10
    # fe_process(ids, dataset, ecg_path, bsqi_path, fiducials_path, features_path, win_len)
    # fe_dataset(ids, n_pools, dataset, win_len, ecg_path, bsqi_path, fiducials_path, features_path)
    # features_per_window(dataset, cts.ids_sp + cts.ids_tp, data_path, features_path, vt_wins=1, win_len=win_len)
    #
    # ids = cts.ids_tn + cts.ids_sn
    # fe_process(ids, dataset, ecg_path, bsqi_path, fiducials_path, features_path, win_len)
    # fe_dataset(ids, n_pools, dataset, win_len, ecg_path, bsqi_path, fiducials_path, features_path)
    # features_per_window(dataset, ids, data_path, features_path, vt_wins=0, win_len=win_len)
    bad_bsqi = ['J621F4c4', '5921D0ce', '4018Fcff']
    not_exist = cts.ids_vn + cts.ids_sn + cts.ids_tp + cts.ids_sp
    ids = not_exist
    Not_exist_list = []
    # fe_process(ids, dataset, ecg_path, bsqi_path, fiducials_path, features_path, win_len)
    # features_per_window(dataset, ids, data_path, features_path, vt_wins=0, win_len=win_len)
    for id_ in ids:
        p_dir = pathlib.PurePath(features_path / id_)  # ML_model
        res = df_replace_nans(p_dir, 'features.xlsx', 'mean')
        if res == -1:
            Not_exist_list.append(id_)
    fe_process(Not_exist_list, dataset, ecg_path, bsqi_path, fiducials_path, features_path, win_len)
    features_per_window(dataset, Not_exist_list, data_path, features_path, vt_wins=0, win_len=win_len)

    Not_exist_list2 = []
    for id_ in ids:
        p_dir = pathlib.PurePath(features_path / id_)  # ML_model
        res = df_replace_nans(p_dir, 'features.xlsx', 'mean')
        if res == -1:
            Not_exist_list2.append(id_)

    for i in range(1, cts.NM + 1):
        train_prediction_model(features_path, cts.ML_RESULTS_DIR, model_type=i, dataset='WL_60',
                               methods=['ns'],
                               n_jobs=10, feature_selection=0, algo=algo, features_name='features.xlsx')
