from utils.base_packages import *
import utils.consts as cts

from FE.features_per_window import *
from FE.window_fe import *
from FE.statistical_test import *
from ML.main_ML import train_prediction_model
from ML.results import all_models
from ML.from_win_to_rec import *

if __name__ == '__main__':
    win_len_10 = 'win_len_60'
    n_pools = 2
    ids = cts.ids_sp
    dataset = 'rbdb'
    win_len = 60
    ecg_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / dataset
    bsqi_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / win_len_10
    fiducials_path = ecg_path / 'fiducials'
    features_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/') / 'win_len' / win_len_10
    results_dir = cts.ML_RESULTS_DIR / 'logo_cv' / win_len_10
    data_path = cts.VTdb_path
    algo = 'RF'
    n_jobs = 10
    fe_process(ids, dataset, ecg_path, bsqi_path, fiducials_path, features_path, win_len)
    # fe_dataset(ids, n_pools, dataset, win_len, ecg_path, bsqi_path, fiducials_path, features_path)
    # features_per_window(dataset, ids, data_path, features_path, vt_wins=1, win_len=30)
