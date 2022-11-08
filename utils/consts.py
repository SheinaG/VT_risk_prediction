import pathlib

import numpy as np
from utils.base_packages import *
from xgboost import XGBClassifier

from utils import metrics

# male ==1

# pathes:
VTdb_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/')
path_to_rbdb = pathlib.PurePath('/MLAIM/AIMLab/Shany/databases/rbafdb/')
rbdb_doc_path = path_to_rbdb / 'documentation'
PRE_PROCESSED_DIR = VTdb_path / 'preprocessed_data'
ids_path = VTdb_path / 'IDS'
DL_path = VTdb_path / 'DL'

REANNOTATION_DIR = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/rbdb/ann_VT/data')

# pebm:
INTS = ['Pwave_int', 'PR_int', 'PR_seg', 'PR_int2', 'QRS_int', 'QT_int', 'Twave_int', 'TP_seg'
    , 'RR_int', 'R_dep', 'QTc_b', 'QTc_frid', 'QTc_fra', 'QTc_hod']
WAVES = ['Pwave', 'Twave', 'Rwave', 'STamp', 'Parea', 'Tarea', 'QRSarea', 'Jpoint']

# hrv:
IMPLEMENTED_FEATURES = np.array(['cosEn', 'AFEv', 'OriginCount', 'IrrEv', 'PACEv', 'AVNN', 'minRR', 'medHR',
                                 'SDNN', 'SEM', 'PNN20', 'PNN50', 'RMSSD', 'CV', 'SD1', 'SD2', 'sq_map_intercept',
                                 'sq_map_linear', 'sq_map_quadratic', 'PIP', 'IALS', 'PSS', 'PAS'])
# r-peaks:
ANNOTATION_TYPES = np.array(['epltd0', 'xqrs', 'gqrs', 'rqrs', 'jqrs', 'wqrs', 'wavedet', 'wrqrs'])

bad_bsqi = ['1021', 'H8208813', '2A21F10e', 'F520H114', 'L620D996', 'M918Ccc4', 'N8218967']

duplicated_ids = ['8520F416', 'H520818b']

num_features_model = [2, 23, 110, 133, 135]
num_selected_features_model = [2, 5, 10, 15, 15]

NM = 5

class_funcs = {
    'RF': RandomForestClassifier,
    'LR': LogisticRegression,
    'XGB': XGBClassifier
}

opt_thresh_policies = {
    'Se_Sp': metrics.maximize_Se_plus_Sp,
    'F_beta': metrics.maximize_f_beta,
}

METRICS = np.array(['Accuracy', 'F1-Score', 'Se', 'Sp', 'PPV', 'NPV', 'AUROC'])

# Paths definition
REPO_DIR_POSIX = pathlib.PurePath('/MLAIM/AIMLab/home/sheina/VT_risk_prediction/')
BASE_DIR = pathlib.PurePath('/MLAIM/AIMLab/')
DATA_DIR = BASE_DIR / "Sheina" / "databases"
TRAIN_DL_DIR = BASE_DIR / "Sheina" / "databases" / "VTdb" / "DL" / "train"
TEST_DL_DIR = BASE_DIR / "Sheina" / "databases" / "VTdb" / "DL" / "test"
MATLAB_TEST_VECTORS_DIR = DATA_DIR / "Matlab_test_vectors"
WQRS_PROG_DIR = pathlib.PurePath('/usr/local/bin/wqrs')
PARSING_PROJECT_DIR = REPO_DIR_POSIX / "parsing"
IDS_DIR = BASE_DIR / "Sheina" / "databases" / "VTdb" / "IDS"
RESULTS_DIR = BASE_DIR / "Sheina" / "databases" / "VTdb" / "DL" / "results"
MODELS_DIR = BASE_DIR / "Sheina" / "databases" / "VTdb" / "DL" / "models"
ML_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')

colors_six = ['#307DA6', '#A65730', '#6F30A6', '#A6304F', '#A69E30', '#30A640']
light_colors = ['#B0E7FF', '#FFD7B0', '#BFC0FF', '#EFB0DF', '#FFEEB0', '#C0FFD0']

EPLTD_FS = 200
rhythms_head = '_ecg_start_0_end_3_n_leads_3_rhythms.txt'
HDF5_DATASET = 'VT_DL'

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
    'model__max_depth': Integer(2, 200),
    'model__reg_alpha': Integer(40, 400),
    'model__reg_lambda': [0, 1],
    'model__eta': [0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1],
    'model__gamma': [0, 1, 10, 30, 50, 80, 100],
    'model__min_child_weight': [1, 10, 30, 50, 80, 100, 200, 300, 500, 800],
    'model__colsample_bytree': [0.05, 0.1, 0.3, 0.5, 0.8, 1],
    'model__subsample': [0.05, 0.1, 0.3, 0.5, 0.8, 1],
    'model__scale_pos_weight': [0, 1, 10, 30, 50, 80, 100, 200, 300, 500, 800],
}

hyp_list = {'XGB': ['colsample_bytree', 'eta', 'gamma', 'max_depth', 'min_child_weight', 'n_estimators', 'reg_alpha',
                    'reg_lambda',
                    'scale_pos_weight', 'subsample'],
            'RF': ['criterion', 'max_depth', 'max_features', 'min_samples_leaf', 'n_estimators']}
MAX_WIN = 55

ext_test_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_VT_ids.npy'))
ext_test_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_non_VT_ids.npy'))
ids_tn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_no_VT_ids.npy'))
ids_sn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_no_VT_ids.npy'))
ids_tp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_VT_ids.npy'))
ids_sp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))
ids_vn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_val_no_VT_ids.npy'))
ids_VT = ids_sp + ids_tp + ext_test_vt
