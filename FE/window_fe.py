import os.path
from itertools import repeat

import joblib
import pandas as pd
from pebm import Preprocessing as Pre
from pebm.ebm import Biomarkers as Obm
from pebm.ebm import FiducialPoints as Fp
from scipy.io import loadmat

import utils.feature_comp as fc
from ML.ML_utils import *
from parsing.base_VT_parser import *


def calculate_bsqi(ids, dataset, save_path, win_len=10):
    fs = 200
    starti = 5 * 60 * fs
    db = VtParser()
    for id in ids:
        # if os.path.exists(save_path / dataset / id / str('bsqi_' + str(win_len) + '.npy')):
        #     continue

        raw_lead = np.load(save_path / dataset / id / 'ecg_0.npy')

        xqrs_lead = db.parse_annotation(id, type='xqrs')
        xqrs_lead = xqrs_lead[(xqrs_lead >= starti)] - starti
        epltd_lead = db.parse_annotation(id, type='epltd0')
        epltd_lead = epltd_lead[(epltd_lead >= starti)] - starti
        bsqi_list = []
        i = 0
        start_win = 0
        end_win = start_win + win_len * 60 * fs
        # windoing:
        while end_win < len(raw_lead):
            epltd_lead_win = epltd_lead[(epltd_lead >= start_win) & (epltd_lead < end_win)] - start_win
            xqrs_lead_win = xqrs_lead[(xqrs_lead >= start_win) & (xqrs_lead < end_win)] - start_win
            signal_win_lead = raw_lead[start_win:end_win]
            if len(xqrs_lead_win) < 10 or len(epltd_lead_win) < 10:
                if len(bsqi_list):
                    bsqi_list.append(0)
                else:
                    bsqi_list = [0]
                i = i + 1
                start_win = end_win
                end_win = start_win + win_len * 60 * fs
                continue

            pre = Pre.Preprocessing(signal_win_lead, fs)
            bsqi = pre.bsqi(epltd_lead_win, xqrs_lead_win)
            if len(bsqi_list):
                bsqi_list.append(bsqi)
            else:
                bsqi_list = [bsqi]
            i = i + 1
            start_win = end_win
            end_win = start_win + win_len * 60 * fs

        np.save(save_path / dataset / id / str('bsqi_' + str(win_len) + '.npy'), np.asarray(bsqi_list))


def preprocess_ecg(ids, fs, dataset, save_path, plot=0):
    db = VtParser()
    if dataset == 'rbdb':
        lead_n = 1
        notc_freq = 50  # power line in israel
    if dataset == 'uvafdb':
        lead_n = 0
        notc_freq = 60  # power line in usa
    for id in ids:
        # isExist = os.path.exists(save_path / 'preprocessed_data'/dataset / id )
        # if not isExist:
        #     os.makedirs(save_path / 'preprocessed_data' / dataset / id )
        isExistecg = os.path.exists(save_path / dataset / id / 'ecg_0.npy')
        # if isExistecg:
        #     continue

        raw_lead = db.parse_raw_rec(id, start=0, end=-1)
        # raw_lead = raw_ecg[0]
        if dataset == 'rbdb':
            raw_lead = raw_lead[5 * 60 * fs:]
        raw1 = raw_lead  # cut the first five minuts
        pre = Pre.Preprocessing(raw_lead, fs)
        raw_lead = pre.bpfilt()
        if plot:
            plt.rcParams.update({'font.size': 10})
            start = 35 * 60 * fs
            stop = start + 5 * fs * 60
            raw_lead = raw_lead[start:stop]
            f_axis = np.linspace(-(fs / 2), fs / 2, len(raw_lead))
            # freq domain filter

            fft_signal_c = np.abs(fftshift(fft(raw1[start:stop])))
            fft_fsig_c = np.abs(fftshift(fft(raw_lead)))
            plt.figure()
            plt.style.use('bmh')
            plt.plot(f_axis, fft_signal_c, color=colors_six[0], label='Raw signal')
            plt.plot(f_axis, fft_fsig_c, color=colors_six[1], linewidth=1, label='Filtered signal')
            plt.legend(prop={'size': 14})
            plt.xlabel('f[Hz]')
            plt.ylabel('mV')
            plt.xlim([-100, 100])
            plt.title('Bandpass filter')
            plt.tight_layout()
            plt.savefig('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/filter_respons/bandpass.png')
            plt.show()

        raw_lead = pre.notch(notc_freq)

        if plot:
            raw_lead = raw_lead[start:stop]
            f_l = int((len(raw_lead) / 2) + np.round(((notc_freq - 5) * len(raw_lead) / fs)))
            f_h = int((len(raw_lead) / 2) + np.round(((notc_freq + 5) * len(raw_lead) / fs)))

            fft_signal_c = np.abs(fftshift(fft(raw1[start:stop])))
            fft_fsig_c = np.abs(fftshift(fft(raw_lead)))

            plt.figure()
            plt.style.use('bmh')
            plt.plot(f_axis[f_l:f_h], fft_signal_c[f_l:f_h], color=colors_six[0], label='Raw signal')
            plt.plot(f_axis[f_l:f_h], fft_fsig_c[f_l:f_h], color=colors_six[1], linewidth=1, label='Filtered signal')
            plt.xlabel('f[Hz]')
            plt.ylabel('mV')
            plt.xlim([notc_freq - 5, notc_freq + 5])
            plt.legend(prop={'size': 14})
            plt.title('Notch filter')
            plt.tight_layout()
            plt.savefig('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/filter_respons/notch.png')
            plt.show()

        raw2 = raw_lead
        # filterd lead
        raw_lead1 = normalize_ecg_98(raw_lead)
        raw_lead2 = norm_rms(raw_lead)
        raw_lead3 = norm_mean_std(raw_lead)
        fp = Fp.FiducialPoints(raw_lead, fs)
        epltd_lead = fp.epltd

        np.save(save_path / dataset / id / 'ecg_0.npy', raw_lead)
        np.save(save_path / dataset / id / 'epltd_0.npy', epltd_lead)
        if plot == 1:
            start = 35 * 60 * fs
            stop = start + 1 * fs * 10
            fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 10), sharey=True)
            axes[0].plot(np.arange(start, stop) / fs, raw1[start:stop])
            axes[0].set_title('raw ecg')
            axes[0].set_xlabel('time[seconds]')
            axes[1].plot(np.arange(start, stop) / fs, raw2[start:stop])
            axes[1].set_title('filtered ecg')
            axes[1].set_xlabel('time[seconds]')
            axes[2].plot(np.arange(start, stop) / fs, raw_lead1[start:stop])
            axes[2].set_title('normalized min max ecg')
            axes[2].set_xlabel('time[seconds]')
            axes[3].plot(np.arange(start, stop) / fs, raw_lead3[start:stop])
            axes[3].set_title('standardized std mean ecg')
            axes[3].set_xlabel('time[seconds]')
            axes[4].plot(np.arange(start, stop) / fs, raw_lead2[start:stop])
            axes[4].set_title('standardized RMS ecg')
            axes[4].set_xlabel('time[seconds]')

            isExist = os.path.exists(save_path / 'figures' / 'normalization')
            if not isExist:
                os.makedirs(save_path / 'figures' / 'normalization')
            fig.savefig(save_path / 'figures' / 'normalization' / str(id + '_min_35.png'))


def calculate_hrv(ids, dataset, cls, ecg_path, features_path, win_len=30):
    features_vt = defaultdict(lambda: defaultdict(dict))
    feats = cts.IMPLEMENTED_FEATURES
    fs = 200
    if cls:
        cls_s = 'VTp'
    else:
        cls_s = 'VTn'

    for id in ids:
        # isExist = os.path.exists('/MLAIM/AIMLab/Sheina/databases/VTdb/win_len/' + str(win_len) + '/' + str(id) + str('/hrv_features.xlsx'))
        # if isExist:
        #     continue
        # load:
        bsqi = np.load(ecg_path / dataset / id / str('bsqi_' + str(win_len) + '.npy'))
        raw_lead = np.load(ecg_path / dataset / id / 'ecg_0.npy')
        epltd_lead = np.load(ecg_path / dataset / id / 'epltd_0.npy')
        start_win = 0
        end_win = start_win + fs * win_len * 60
        win = 0
        isExist = os.path.exists('/MLAIM/AIMLab/Sheina/databases/VTdb/win_len/' + str(win_len) + '/' + str(id) + '/')
        if not isExist:
            os.makedirs('/MLAIM/AIMLab/Sheina/databases/VTdb/win_len/' + str(win_len) + '/' + str(id) + '/')
        while end_win < len(raw_lead):
            if bsqi[win] < 0.8:
                win = win + 1
                start_win = end_win
                end_win = start_win + fs * win_len * 60
                continue
            epltd_lead_win = epltd_lead[(epltd_lead >= start_win) & (epltd_lead < end_win)]
            if len(epltd_lead_win) < 2:
                win = win + 1
                start_win = end_win
                end_win = start_win + fs * win_len * 60
                continue
            rr_lead_win = epltd_lead_win[1:] - epltd_lead_win[:-1]
            rrt = rr_lead_win / fs
            start = time.time()
            for i, feat in enumerate(feats):
                func = getattr(fc, 'comp_' + feat)

                features_vt[id][win][feat] = np.array(func(rrt))
            end = time.time()
            print('HRV features computed in ' + str(end - start) + ' for win number ' + str(win))
            start_win = end_win
            end_win = start_win + fs * win_len * 60
            win = win + 1

    vt_df = pd.DataFrame()
    mapper = {}
    for id in features_vt:

        names = list(features_vt[id].keys())
        for i, name in enumerate(names):
            mapper[name] = id + '_num_' + str(name)
        vt_df = pd.DataFrame.from_dict(features_vt[id], orient='index')
        vt_df = vt_df.rename(index=mapper)
        vt_df.to_excel(
            '/MLAIM/AIMLab/Sheina/databases/VTdb/win_len/' + str(win_len) + '/' + str(id) + str('/hrv_features.xlsx'))


def calculate_pebm(ids, dataset, cls, ecg_path, features_path, win_len=30):
    if cls:
        cls_s = 'VTp'
    else:
        cls_s = 'VTn'

    fs = np.uint8(200)

    matlab_pat = '/usr/local/MATLAB/R2021a/'
    for id in ids:
        # isExist = os.path.exists('/MLAIM/AIMLab/Sheina/databases/VTdb/win_len/' + str(win_len) + '/' + str(id) + str('/bm_features.xlsx'))
        # if isExist:
        #     continue
        pebm_feat = {}
        bsqi = np.load(ecg_path / dataset / id / str('bsqi_' + str(win_len) + '.npy'))
        raw_lead = np.load(ecg_path / dataset / id / 'ecg_0.npy')
        epltd_lead = np.load(ecg_path / dataset / id / 'epltd_0.npy')
        fiducials = joblib.load(ecg_path / 'fiducials' / id / 'fiducials.pkl')
        i = 0
        start_win = 0
        end_win = start_win + win_len * 60 * fs
        qrs_all = fiducials[0]['qrs']
        while end_win < len(raw_lead):
            if bsqi[i] < 0.8:
                i = i + 1
                start_win = end_win
                end_win = start_win + win_len * 60 * fs
                continue

            fiducials_win = defaultdict(lambda: defaultdict(dict))
            r_qrs_ind = np.where((qrs_all < end_win) & (qrs_all > start_win))
            for fid_point in fiducials[0]:
                all_fid = fiducials[0][fid_point]
                win_fid = all_fid[r_qrs_ind[0][1]:r_qrs_ind[0][-2]] - start_win
                fiducials_win[0][fid_point] = win_fid

            # create fiducial win
            signali = raw_lead[start_win:end_win]

            pebm_feat_win = {}
            bm = Obm.Biomarkers(signali, fs, fiducials_win)
            raw_intervals, interval_stat = bm.intervals()
            raw_waves, wave_stat = bm.waves()
            for interval in interval_stat:
                for stat in interval_stat[interval]:
                    pebm_feat_win[stat + '_' + interval] = interval_stat[interval][stat]
            for wave in wave_stat:
                for stat in wave_stat[wave]:
                    pebm_feat_win[stat + '_' + wave] = wave_stat[wave][stat]

            pebm_feat['patiant_' + id + '_win_' + str(i)] = pebm_feat_win

            i = i + 1
            start_win = end_win
            end_win = start_win + win_len * 60 * fs

        bm = pd.DataFrame.from_dict(pebm_feat)
        bm = bm.transpose()
        bm.to_excel(
            '/MLAIM/AIMLab/Sheina/databases/VTdb/win_len/' + str(win_len) + '/' + str(id) + str('/bm_features.xlsx'))

        print(str(id))
    a = 5


def fe_dataset(ids, n_pools, dataset, win_len):
    pool = multiprocessing.Pool(n_pools)
    in_pool = (len(ids) // n_pools) + 1
    ids_pool, dataset_pool = [], []
    for j in range(n_pools):
        ids_pool += [ids[j * in_pool:min((j + 1) * in_pool, len(ids))]]
    dataset_pool = [dataset] * n_pools
    pool.starmap(fe_process, zip(ids_pool, dataset_pool, repeat(win_len)))
    pool.close()


def calculate_pvc_features(ids, win_len):
    ecg_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/')
    fs = 200
    m_id = []
    for id_ in ids:
        pvc_path = '/home/sheina/PVC/mats/'
        pvc_head = '_ECG_heartbeat_classifier.mat'
        bd_head = '_QRS_detection.mat'
        isExist = os.path.exists(pvc_path + id_ + pvc_head)
        bsqi = np.load(ecg_path / 'rbdb' / id_ / str('bsqi_0.npy'))
        if not isExist:
            m_id.append(id_)

        # elif os.path.exists('/MLAIM/AIMLab/Sheina/databases/VTdb/win_len/' + str(win_len) + '/' + str(id_) + str('/pvc_features.xlsx')):
        #     continue
        else:
            HBC_mat = loadmat(pvc_path + id_ + pvc_head)
            bd_mat = loadmat(pvc_path + id_ + bd_head)
            HBC_array = HBC_mat['anntyp']
            bd_array = bd_mat['epltd_I'][0][0][0].squeeze()
            if len(bd_array) - len(HBC_array) == 1:
                bd_array = bd_array[:-1]

            start = 0
            i = 0
            stop = fs * 60 * win_len
            df = pd.DataFrame(columns=['win', 'N_ratio', 'S_ratio', 'V_ratio', 'F_ratio', 'U_ratio'])
            try:
                while stop <= bd_array[-1]:
                    win_pvc = HBC_array[(bd_array < stop) & (bd_array > start)]

                    if len(win_pvc) == 0:
                        a = 5
                    if bsqi[i] < 0.8:
                        a = 5
                    else:
                        N_ratio = len(win_pvc[win_pvc == 'N']) / len(win_pvc)
                        S_ratio = len(win_pvc[win_pvc == 'S']) / len(win_pvc)
                        V_ratio = len(win_pvc[win_pvc == 'V']) / len(win_pvc)
                        F_ratio = len(win_pvc[win_pvc == 'F']) / len(win_pvc)
                        U_ratio = len(win_pvc[win_pvc == 'U']) / len(win_pvc)
                        new_win = pd.DataFrame(
                            [['patiant_' + id_ + '_win_' + str(i), N_ratio, S_ratio, V_ratio, F_ratio, U_ratio]],
                            columns=['win', 'N_ratio', 'S_ratio', 'V_ratio', 'F_ratio', 'U_ratio'])
                        df = df.append(new_win, ignore_index=True)

                    i += 1
                    start = stop
                    stop = stop + fs * 60 * win_len
                # df.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/win_len/' + str(win_len) + '/' + str(id_) + str('/pvc_features.xlsx'))
                df.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/' + str(id_) + str('/pvc_features.xlsx'))
            except:
                print(id_)

    print(m_id)
    np.save('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/no_PVCs_IDS.npy', m_id)


def four_pvc_in_a_row(ids):
    ecg_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/')
    fs = 200
    m_id = {}
    for id_ in ids:
        pvc_path = '/home/sheina/PVC/mats/'
        pvc_head = '_ECG_heartbeat_classifier.mat'
        bd_head = '_QRS_detection.mat'
        isExist = os.path.exists(pvc_path + id_ + pvc_head)
        bsqi = np.load(ecg_path / 'rbdb' / id_ / str('bsqi_0.npy'))

        # elif os.path.exists('/MLAIM/AIMLab/Sheina/databases/VTdb/win_len/' + str(win_len) + '/' + str(id_) + str('/pvc_features.xlsx')):
        #     continue

        HBC_mat = loadmat(pvc_path + id_ + pvc_head)
        bd_mat = loadmat(pvc_path + id_ + bd_head)
        HBC_array = HBC_mat['anntyp']
        bd_array = bd_mat['epltd_I'][0][0][0].squeeze()
        list_pvc = []
        aa = np.where(HBC_array == 'V')
        for ja in aa[0]:
            flag = 0
            if ja + 1 in aa[0]:
                flag += 1
            if ja + 2 in aa[0]:
                flag += 1
            if flag == 2:
                list_pvc.append(bd_array[ja])

        m_id[id_] = list_pvc

    np.save('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/ann.pkl', m_id)


def clear_from_bad_bsqi(ids_paths, bsqi_path):
    IDS_main = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/')
    bad_IDS = list(np.load(IDS_main / 'low_bsqi_IDS.npy'))
    no_PVCs_IDS = list(np.load(IDS_main / 'no_PVCs_IDS.npy'))
    for path_ in ids_paths:
        ids = list(np.load(IDS_main / path_))
        for id_ in ids:
            try:
                bsqi = np.load(bsqi_path / id_ / 'bsqi_0.npy')
                g_b = bsqi[bsqi >= 0.8]
                if len(g_b) < 2:
                    if id_ not in bad_IDS:
                        bad_IDS.append(id_)

                    ids.remove(id_)
            except:
                print(id_)
            if id_ in no_PVCs_IDS:
                ids.remove(id_)
        np.save(IDS_main / path_, ids)
    np.save(IDS_main / 'low_bsqi_IDS.npy', bad_IDS)


def fe_process(ids, dataset, win_len):
    fs = 200
    save_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/')
    features_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/')

    preprocess_ecg(ids, fs, dataset, save_path, plot=0)
    calculate_bsqi(ids, dataset, save_path, win_len=win_len)
    calculate_hrv(ids, dataset, 0, save_path, features_path, win_len=win_len)
    calculate_pebm(ids, dataset, 0, save_path, features_path, win_len=win_len)
    calculate_pvc_features(ids, win_len)


def df_replace_nans(path_df, name_df, manner):
    if not os.path.exists(path_df / name_df):
        return
    df = pd.read_excel(path_df / name_df, engine='openpyxl')
    print(path_df)
    # clean :

    if 'Unnamed: 0.1' in list(df.columns):
        df = df.drop(columns=['Unnamed: 0'])
        df = df.set_axis(df['Unnamed: 0.1'], axis='index')
        df = df.drop(columns=['Unnamed: 0.1'])
    if 'Unnamed: 0' in list(df.columns):
        df = df.set_axis(df['Unnamed: 0'], axis='index')
        df = df.drop(columns=['Unnamed: 0'])

    for col in df.columns:
        data = df[col]
        data = np.asarray(data)
        if manner == 'mean':
            data[np.isnan(data)] = np.mean(data[~np.isnan(data)])
        df[col] = data
        # df[abs(df) > 1e6] = 1e6
        # df[abs(df) < 1e-6] = 1e-6
    df.to_excel(path_df / name_df)


def run_on_dir(ids):
    for id_ in ids:
        p_dir = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/win_len/1/' + id_)  # ML_model
        df_replace_nans(p_dir, 'features_n.xlsx', 'mean')


def calculate_fiducials_per_rec(ids, ecg_path, dataset):
    for i, id_ in enumerate(ids):
        fs = 200
        raw_lead = np.load(ecg_path / dataset / id_ / 'ecg_0.npy')
        epltd_lead = np.load(ecg_path / dataset / id_ / 'epltd_0.npy')
        matlab_path = '/usr/local/MATLAB/R2021a/'
        fp = Fp.FiducialPoints(raw_lead, fs)
        fiducials = fp.wavedet(matlab_path, epltd_lead)
        path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/fiducials/')
        if not os.path.exists(path / id_):
            os.makedirs(path / id_)
        with open((path / id_ / 'fiducials.pkl'), 'wb') as f:
            joblib.dump(fiducials, f)
        print(i, id_)


if __name__ == '__main__':
    ids_tn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_no_VT_ids.npy'))
    ids_sn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_no_VT_ids.npy'))
    ids_tp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_VT_ids.npy'))
    ids_sp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))
    ids_vn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_val_no_VT_ids.npy'))

    UVAF_sp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_VT_ids.npy'))
    UVAF_sn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_non_VT_ids.npy'))
    md_test = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/md_test.npy'))

    # calculate_pvc_features(md_test, 30)
    four_pvc_in_a_row(ids_tp + ids_sp)
    # run_on_dir(ids_sn+ids_sp+ids_tp+ids_vn+ids_tn)
# calculate_pvc_features(['1021Cd9d'], win_len = 1)
# fe_dataset(ids_tn, dataset='rbdb', n_pools =10, win_len =1)
# fe_process(aa, dataset='rbdb', win_len=1)
# preprocess_ecg(['1419Fc22'], 200, 'rbdb', pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/'), plot=0)
# calculate_bsqi(['1419Fc22'], 'rbdb',pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') , win_len=1)
# calculate_pvc_features(ids_sn+ids_sp+ids_tp+ids_vn+ids_tn,  win_len =1)
# ids_paths = ['RBDB_train_no_VT_ids.npy', 'RBDB_test_no_VT_ids.npy', 'RBDB_train_VT_ids.npy','RBDB_test_VT_ids.npy', 'RBDB_val_no_VT_ids.npy']
# bsqi_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/normalized/')
# clear_from_bad_bsqi(ids_paths, bsqi_path)
