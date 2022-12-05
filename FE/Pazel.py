import sys

sys.path.append("/home/sheina/VT_risk_prediction/")
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
    win_len_n = 'win_len_10'
    n_pools = 1

    dataset = 'rbdb'
    win_len = 10
    ecg_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / dataset
    bsqi_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / win_len_n
    fiducials_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / 'fiducials'
    features_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/') / 'win_len' / win_len_n
    results_dir = cts.ML_RESULTS_DIR / 'logo_cv' / win_len_n
    data_path = cts.VTdb_path
    algo = 'RF'
    n_jobs = 10
    ids = cts.ids_sp + cts.ids_vn + cts.ids_tp + cts.ids_sp + cts.ids_tn
    fe_dataset(ids, n_pools, dataset, win_len, ecg_path, bsqi_path, fiducials_path, features_path, stand=0)
    # features_per_window(dataset, cts.ids_sp + cts.ids_tp, data_path, features_path, vt_wins=1, win_len=win_len)
    #
    # for id_ in ids:
    #     p_dir = pathlib.PurePath(features_path / id_)  # ML_model
    #     res = df_replace_nans(p_dir, 'features.xlsx', 'mean')
    #     if res == -1:
    #         Not_exist_list.append(id_)

    # for i in range(3, cts.NM + 1):
    #     train_prediction_model(features_path, cts.ML_RESULTS_DIR, model_type=i, dataset='WL_60',
    #                            methods=['ns'],
    #                            n_jobs=10, feature_selection=0, algo=algo, features_name='features.xlsx',
    #                            bad_bsqi_ids=bad_bsqi_60)
