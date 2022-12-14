import sys

sys.path.append("/home/sheina/VT_risk_prediction/")
from utils import consts as cts
from utils.base_packages import *


def features_per_window(dataset, ids, data_path, features_path, pvc_path, vt_wins=0, win_len=30):
    if dataset == 'rbdb':
        s = ''
        ts = 5 * 60
    if dataset == 'uvafdb':
        s = ''
        ts = 0

    demographic_VT_xl = pd.read_excel(data_path / 'VTp' / str('demographic_features_' + dataset + '.xlsx'),
                                      engine='openpyxl')
    demographic_no_VT_xl = pd.read_excel(data_path / 'VTn' / str('demographic_features_' + dataset + '.xlsx'),
                                         engine='openpyxl')
    demographic_VT_xl = demographic_VT_xl.set_axis(demographic_VT_xl['Unnamed: 0'], axis='index')
    demographic_VT_xl = demographic_VT_xl.drop(columns=['Unnamed: 0'])
    demographic_no_VT_xl = demographic_no_VT_xl.set_axis(demographic_no_VT_xl['Unnamed: 0'], axis='index')
    demographic_no_VT_xl = demographic_no_VT_xl.drop(columns=['Unnamed: 0'])
    demographic_xl = demographic_VT_xl.append(demographic_no_VT_xl)
    demographic_xl = demographic_xl[~demographic_xl.index.duplicated()]
    path_to_new_dem = '/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/new_dem.xlsx'
    new_dem_xl = pd.read_excel(path_to_new_dem, engine='openpyxl')
    new_dem_xl = new_dem_xl.set_axis(new_dem_xl['holter id'], axis='index')
    new_dem_xl = new_dem_xl.drop_duplicates(subset=['holter id'], keep='first')
    new_dem_xl = new_dem_xl.drop(columns=['holter id', 'Unnamed: 0'])

    bad_ids = []
    fs = 200
    if vt_wins:
        segments = pd.read_excel(data_path / 'VTp' / str('segments_array_' + dataset + '.xlsx'), engine='openpyxl')

    for id_ in ids:
        isExist = os.path.exists(features_path / str(id_) / 'features_n.xlsx')

        notExist = os.path.exists(features_path / str(id_) / 'hrv_features.xlsx')
        if not notExist:
            bad_ids.append(id_)
            print(id_)
            continue
        notExist = os.path.exists(features_path / str(id_) / 'bm_features.xlsx')
        if not notExist:
            print(id_)
            continue
        hrv_vt = pd.read_excel(features_path / str(id_) / 'hrv_features.xlsx', engine='openpyxl')
        hrv_vt = hrv_vt.set_axis(hrv_vt['Unnamed: 0'], axis='index')
        hrv_vt = hrv_vt.drop(columns=['Unnamed: 0'])
        bm_vt = pd.read_excel(features_path / str(id_) / 'bm_features.xlsx', engine='openpyxl')
        bm_vt = bm_vt.set_axis(bm_vt['Unnamed: 0'], axis='index')
        bm_vt = bm_vt.drop(columns=['Unnamed: 0'])
        try:
            pvc_vt = pd.read_excel(pvc_path / str(id_) / 'pvc_features.xlsx', engine='openpyxl')
            pvc_vt = pvc_vt.set_axis(pvc_vt['win'], axis='index')
            pvc_vt = pvc_vt.drop(columns=['win', 'Unnamed: 0'])
        except:
            print(id_)
            continue

        if vt_wins:
            isExist = os.path.exists(features_path / id_ / 'vt_wins')
            if not isExist:
                os.makedirs(features_path / id_ / 'vt_wins')
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
                new_dem_vt = pd.DataFrame(new_dem_xl.loc[s + id_])
                dem_vt_patient = pd.concat([dem_vt] * len(bm_vt_win), axis=1).transpose()
                new_dem_vt_patient = pd.concat([new_dem_vt] * len(bm_vt_win), axis=1).transpose()
                dem_vt_patient = dem_vt_patient.set_axis(bm_vt_win.index, axis='index')
                new_dem_vt_patient = new_dem_vt_patient.set_axis(bm_vt_win.index, axis='index')
                hrv_vt_win = hrv_vt_win.set_axis(bm_vt_win.index, axis='index')
                vt_features = pd.concat([bm_vt_win, hrv_vt_win, pvc_vt_win, dem_vt_patient, new_dem_vt_patient], axis=1,
                                        join='inner')  # , new_dem_vt_patient
                vt_features.to_excel(features_path / id_ / 'vt_wins' / 'features_nd.xlsx')

        dem_ = pd.DataFrame(demographic_xl.loc[s + id_]).transpose()
        new_dem_ = pd.DataFrame(new_dem_xl.loc[s + id_]).transpose()
        dem_patient = pd.concat([dem_] * len(bm_vt), axis=0)
        new_dem_patient = pd.concat([new_dem_] * len(bm_vt), axis=0)
        dem_patient = dem_patient.set_axis(bm_vt.index, axis='index')
        new_dem_patient = new_dem_patient.set_axis(bm_vt.index, axis='index')
        hrv_vt = hrv_vt.set_axis(bm_vt.index, axis='index')
        no_vt_features = pd.concat([bm_vt, hrv_vt, pvc_vt, dem_patient, new_dem_patient], axis=1,
                                   join='inner')  # , new_dem_patient
        no_vt_features.to_excel(features_path / id_ / 'features_nd.xlsx')

    return bad_ids


if __name__ == '__main__':
    win_len_n = 'win_len_120'
    n_pools = 10

    dataset = 'rbdb'
    win_len = 30
    ecg_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / dataset
    bsqi_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / dataset
    fiducials_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / 'fiducials'
    features_path = cts.ML_path
    results_dir = cts.ML_RESULTS_DIR / 'logo_cv' / win_len_n
    pvc_path = features_path
    ML_path = cts.VTdb_path
    algo = 'RF'
    missing_ids = ['UVA2643', 'UVA0978', 'UVA2135', 'UVA1267', 'UVA0429', 'UVA0704', 'UVA0102', 'UVA1581', 'UVA2437',
                   'UVA1867']
    features_per_window(dataset, cts.vt_ids, ML_path, features_path, pvc_path, vt_wins=0, win_len=win_len)
    # features_per_window(dataset, cts.ext_test_vt, ML_path, features_path, pvc_path, vt_wins=1, win_len=win_len)
    # bad_ids = features_per_window('rbdb', cts.ids_tn + cts.ids_sn + cts.ids_vn, ML_path, features_path, features_path,
    #                               vt_wins=0,
    #                               win_len=120)
    # from FE.window_fe import *

    # fe_dataset(bad_ids, n_pools, dataset, win_len, ecg_path, bsqi_path, fiducials_path, features_path, stand=0)
    # features_per_window('rbdb', ['5620G109'], data_path, ML_path, pvc_path, vt_wins=0,
    #                     win_len=10)
    # features_per_window('rbdb', test_no_vt, data_path, ML_path, vt_wins=0)
    # features_per_window('rbdb', train_no_vt, data_path, ML_path, vt_wins=0)

    # features_per_window('uvafdb', ext_test_vt, data_path, ML_path, vt_wins=1)
    # features_per_window('uvafdb', ext_test_no_vt, data_path, ML_path, vt_wins=0)
