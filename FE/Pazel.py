import sys
sys.path.append("/home/sheina/VT_risk_prediction/")
from ML.main_ML import train_prediction_model
from ML.from_win_to_rec import *

warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == '__main__':
    win_len_n = 'win_len_10'
    n_pools = 10

    dataset = 'rbdb'
    win_len = 10
    ecg_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / dataset
    bsqi_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / win_len_n
    fiducials_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / 'fiducials'
    features_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/') / 'win_len' / win_len_n
    results_dir = cts.ML_RESULTS_DIR / 'logo_cv' / win_len_n
    data_path = cts.VTdb_path
    algo = 'RF'
    ids = cts.ids_sp + cts.ids_vn + cts.ids_tp + cts.ids_sn + cts.ids_tn

    # fe_dataset(cts.ids_sn, n_pools, dataset, win_len, ecg_path, bsqi_path, fiducials_path, features_path, stand=0)
    # features_per_window(dataset, cts.ids_sp + cts.ids_tp, data_path, features_path, features_path, vt_wins=1, win_len=win_len)
    # features_per_window(dataset, cts.ids_sn, data_path, features_path,features_path, vt_wins=0, win_len=win_len)

    # for id_ in ids:
    #     if os.path.exists(features_path / id_ / 'features_stand.xlsx'):
    #         old_name = str(features_path / id_ / 'features_stand.xlsx')
    #         new_name = str(features_path / id_ / 'features.xlsx')
    #         os.rename(old_name, new_name)
    #     if os.path.exists(features_path / id_ / 'features_stand.xlsx'):
    #         old_name = str(features_path / id_ / 'bm_features_stand.xlsx')
    #         new_name = str(features_path / id_ / 'bm_features.xlsx')
    #         os.rename(old_name, new_name)
    # #
    # Not_exist_list = []
    # for id_ in ids:
    #     p_dir = pathlib.PurePath(features_path / id_)  # ML_model
    #     res = df_replace_nans(p_dir, 'features.xlsx', 'mean')
    #     if res == -1:
    #         Not_exist_list.append(id_)

    a = 5

    for i in range(1, cts.NM + 1):
        train_prediction_model(features_path, cts.ML_RESULTS_DIR, model_type=i, dataset='WL_10',
                               methods=['ns'],
                               n_jobs=10, feature_selection=0, algo=algo, features_name='features.xlsx',
                               bad_bsqi_ids=cts.bad_bsqi_10)
