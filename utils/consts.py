import pathlib

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import metrics

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

colors_six = ['#307DA6', '#A65730', '#6F30A6', '#A6304F', '#A69E30', '#30A640']
light_colors = ['#B0E7FF', '#FFD7B0', '#BFC0FF', '#EFB0DF', '#FFEEB0', '#C0FFD0']

EPLTD_FS = 200
rhythms_head = '_ecg_start_0_end_3_n_leads_3_rhythms.txt'
HDF5_DATASET = 'VT_DL'
