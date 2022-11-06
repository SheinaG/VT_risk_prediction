import os
import pathlib

import numpy as np
import pandas as pd


def features_per_window(name, ids, data_path, save_path, vt_wins=0, win_len=30):
    if name == 'rbdb':
        s = ''
        ts = 5 * 60
    if name == 'uvafdb':
        s = ''
        ts = 0

    demographic_VT_xl = pd.read_excel(data_path / 'VTp' / str('demographic_features_' + name + '.xlsx'),
                                      engine='openpyxl')
    demographic_no_VT_xl = pd.read_excel(data_path / 'VTn' / str('demographic_features_' + name + '.xlsx'),
                                         engine='openpyxl')
    demographic_VT_xl = demographic_VT_xl.set_axis(demographic_VT_xl['Unnamed: 0'], axis='index')
    demographic_VT_xl = demographic_VT_xl.drop(columns=['Unnamed: 0'])
    demographic_no_VT_xl = demographic_no_VT_xl.set_axis(demographic_no_VT_xl['Unnamed: 0'], axis='index')
    demographic_no_VT_xl = demographic_no_VT_xl.drop(columns=['Unnamed: 0'])
    demographic_xl = demographic_VT_xl.append(demographic_no_VT_xl)
    demographic_xl = demographic_xl[~demographic_xl.index.duplicated()]

    bad_ids = []
    fs = 200
    if vt_wins:
        segments = pd.read_excel(data_path / 'VTp' / str('segments_array_' + name + '.xlsx'), engine='openpyxl')

    for id_ in ids:
        isExist = os.path.exists(save_path / str(id_) / 'features_n.xlsx')
        if isExist:
            continue
        notExist = os.path.exists(save_path / str(id_) / 'hrv_features.xlsx')
        if not notExist:
            bad_ids.append(id_)
            print(id_)
            continue
        notExist = os.path.exists(save_path / str(id_) / 'bm_features.xlsx')
        if not notExist:
            print(id_)
            continue
        hrv_vt = pd.read_excel(save_path / str(id_) / 'hrv_features.xlsx', engine='openpyxl')
        hrv_vt = hrv_vt.set_axis(hrv_vt['Unnamed: 0'], axis='index')
        hrv_vt = hrv_vt.drop(columns=['Unnamed: 0'])
        bm_vt = pd.read_excel(save_path / str(id_) / 'bm_features.xlsx', engine='openpyxl')
        bm_vt = bm_vt.set_axis(bm_vt['Unnamed: 0'], axis='index')
        bm_vt = bm_vt.drop(columns=['Unnamed: 0'])
        pvc_vt = pd.read_excel(pathlib.PurePath(save_path) / str(id_) / 'pvc_features.xlsx', engine='openpyxl')
        pvc_vt = pvc_vt.set_axis(pvc_vt['win'], axis='index')
        pvc_vt = pvc_vt.drop(columns=['win', 'Unnamed: 0'])

        if vt_wins:
            isExist = os.path.exists(save_path / id_ / 'vt_wins')
            if not isExist:
                os.makedirs(save_path / id_ / 'vt_wins')
            bm_vt_win = pd.DataFrame(columns=bm_vt.columns)
            hrv_vt_win = pd.DataFrame(columns=hrv_vt.columns)
            pvc_vt_win = pd.DataFrame(columns=pvc_vt.columns)
            VT_win = []
            segments_id = segments[segments['holter_id'] == id_]
            segments_id = segments_id.set_axis(range(len(segments_id)), axis='index')

            for i in np.arange(0, np.shape(segments_id)[0]):
                starti = (segments_id['start'][i] + ts) // (win_len * 60)
                endi = (segments_id['end'][i] + ts) // (win_len * 60)

                # load vt dataframes:

                win_name_b = 'patiant_' + str(id_) + '_win_' + str(int(starti))
                win_name_h = str(id_) + '_num_' + str(int(starti))
                if win_name_b in VT_win:
                    continue
                if not (win_name_b in bm_vt.index):
                    continue
                win = bm_vt.loc[win_name_b]
                bm_vt = bm_vt.drop([win_name_b], axis=0)
                pvc_win = pvc_vt.loc[win_name_b]
                pvc_vt = pvc_vt.drop([win_name_b], axis=0)
                hrv_win = hrv_vt.loc[win_name_h]
                hrv_vt = hrv_vt.drop([win_name_h], axis=0)
                bm_vt_win = bm_vt_win.append(win)
                hrv_vt_win = hrv_vt_win.append(hrv_win)
                pvc_vt_win = pvc_vt_win.append(pvc_win)
                VT_win.append(win_name_b)

                if starti != endi:
                    win_name_b = 'patiant_' + str(id_) + '_win_' + str(int(starti))
                    win_name_h = str(id_) + '_num_' + str(int(starti))
                if win_name_b in VT_win:
                    continue
                if not (win_name_b in bm_vt.index):
                    continue
                win = bm_vt.loc[win_name_b]
                bm_vt = bm_vt.drop([win_name_b], axis=0)
                pvc_win = pvc_vt.loc[win_name_b]
                pvc_vt = pvc_vt.drop([win_name_b], axis=0)
                hrv_win = hrv_vt.loc[win_name_h]
                hrv_vt = hrv_vt.drop([win_name_h], axis=0)
                bm_vt_win = bm_vt_win.append(win)
                hrv_vt_win = hrv_vt_win.append(hrv_win)
                pvc_vt_win = pvc_vt_win.append(pvc_win)
                VT_win.append(win_name_b)

            if len(bm_vt_win) > 0:
                dem_vt = pd.DataFrame(demographic_xl.loc[s + id_])

                dem_vt_patient = pd.concat([dem_vt] * len(bm_vt_win), axis=1).transpose()
                dem_vt_patient = dem_vt_patient.set_axis(bm_vt_win.index, axis='index')
                hrv_vt_win = hrv_vt_win.set_axis(bm_vt_win.index, axis='index')
                vt_features = pd.concat([bm_vt_win, hrv_vt_win, pvc_vt_win, dem_vt_patient], axis=1, join='inner')
                vt_features.to_excel(save_path / id_ / 'vt_wins' / 'features_n.xlsx')

        dem_ = pd.DataFrame(demographic_xl.loc[s + id_]).transpose()
        dem_patient = pd.concat([dem_] * len(bm_vt), axis=0)
        dem_patient = dem_patient.set_axis(bm_vt.index, axis='index')
        hrv_vt = hrv_vt.set_axis(bm_vt.index, axis='index')
        no_vt_features = pd.concat([bm_vt, hrv_vt, pvc_vt, dem_patient], axis=1, join='inner')
        no_vt_features.to_excel(save_path / id_ / 'features_n.xlsx')

    a = 5


if __name__ == '__main__':
    data_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/')
    # ML_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/win_len/1/')
    ML_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')
    ecg_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/')

    ids_tn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_no_VT_ids.npy'))
    ids_sn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_no_VT_ids.npy'))
    ids_tp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_VT_ids.npy'))
    ids_sp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))
    ids_vn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_val_no_VT_ids.npy'))

    md_test = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/md_test.npy'))

    features_per_window('rbdb', ids_tp + ids_sp, data_path, ML_path, vt_wins=1, win_len=1)
    features_per_window('rbdb', ids_tn + ids_sn + ids_vn, data_path, ML_path, vt_wins=0, win_len=1)
    # features_per_window('rbdb', test_no_vt, data_path, ML_path, vt_wins=0)
    # features_per_window('rbdb', train_no_vt, data_path, ML_path, vt_wins=0)

    # features_per_window('uvafdb', ext_test_vt, data_path, ML_path, vt_wins=1)
    # features_per_window('uvafdb', ext_test_no_vt, data_path, ML_path, vt_wins=0)
