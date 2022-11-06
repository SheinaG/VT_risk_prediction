import pathlib

from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import os, shutil
import sheina.consts as cts
import sheina.bayesiansearch as bs
from sheina import train_model, train_prediction_model, vt_per_window, results
import vt_pebm as f_vt

experiment_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/experiments/std_mean/')
org_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/')
features_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/')
results_dir = experiment_path / 'results'
algo = 'RF'


# filtering + normalization:
def scaling_ecg(ids, dataset):
    for id in ids:
        raw_lead = np.load(org_path / dataset / id / 'ecg_0.npy')
        raw_lead = train_model.norm_mean_std(raw_lead)
        np.save(experiment_path / dataset / id / 'scaled_ecg.npy', raw_lead)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


if __name__ == '__main__':
    # shutil.copytree(org_path, experiment_path)
    # scaling_ecg(cts.ids_no_VT + cts.ids_VT, 'uvafdb')
    # scaling_ecg(cts.ids_rbdb_no_VT + cts.ids_rbdb_VT, 'rbdb')
    # shutil.copytree(features_path/ 'VTn', experiment_path / 'VTn')
    # shutil.copytree(features_path / 'VTp', experiment_path / 'VTp')
    # f_vt.calculate_pebm( cts.ids_rbdb_VT, 'rbdb', 1, experiment_path, experiment_path)
    # f_vt.calculate_pebm( cts.ids_rbdb_no_VT, 'rbdb', 0, experiment_path, experiment_path)
    # f_vt.calculate_pebm( cts.ids_VT, 'uvafdb', 1, experiment_path, experiment_path)
    # f_vt.calculate_pebm( cts.ids_no_VT, 'uvafdb', 0, experiment_path, experiment_path)
    # vt_per_window.features_per_window('rbdb', experiment_path, experiment_path/'ML_model')
    # vt_per_window.features_per_window('uvafdb', experiment_path, experiment_path/'ML_model')
    # train_prediction_model.train_prediction_model(experiment_path/'ML_model', results_dir, 1, 'mean_std')
    # train_prediction_model.train_prediction_model(experiment_path/'ML_model', results_dir,  2, 'mean_std')
    # train_prediction_model.train_prediction_model(experiment_path/'ML_model', results_dir, 3, 'mean_std')
    # train_prediction_model.train_prediction_model(experiment_path/'ML_model', results_dir,  4, 'mean_std')
    results.all_models(experiment_path / 'ML_model', results_dir / "logo_cv" / algo, results_dir=results_dir,
                       dataset='mean_std')
