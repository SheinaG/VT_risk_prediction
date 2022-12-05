import os.path
import sys
sys.path.append("/home/sheina/VT_risk_prediction/")
sys.path.append("/home/sheina/ecg-kit/ecg-kit/examples")
from ML.ML_utils import *
from utils.base_packages import *
from parsing.base_VT_parser import VtParser


def ids_to_groups(ids, n_pools=10):
    lead2 = []
    other_p = []
    pool = multiprocessing.Pool(n_pools)
    in_pool = (len(ids) // n_pools) + 1
    ids_pool, dataset_pool = [], []
    for j in range(n_pools):
        ids_pool += [ids[j * in_pool:min((j + 1) * in_pool, len(ids))]]
    res = pool.starmap(raw_to_mat, zip(ids_pool))
    pool.close()
    for resi in res:
        lead2 += resi[0]
    a = 5


def raw_to_mat(ids):
    db = VtParser()
    ecg_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/')
    lead2ids = []
    fs = 200
    for id_ in ids:
        # if os.path.exists('/home/sheina/PVC/mats/' + id_ + '.mat'):
        #     continue
        # first_lead
        raw_lead = np.expand_dims(np.load(ecg_path / 'normalized' / id_ / 'ecg_0.npy'), axis=1)
        epltd_lead = np.expand_dims(np.load(ecg_path / 'normalized' / id_ / 'epltd_0.npy'), axis=1)
        epltd = {'epltd_I': {'time': epltd_lead}}
        # second lead
        raw_rec_new = db.parse_raw_rec(id_, lead=1)
        # raw_rec_new = raw_rec_new[5 * 60 * fs:]
        pre = Pre.Preprocessing(raw_rec_new, fs)
        raw_rec_new = pre.bpfilt()
        raw_rec_new = pre.notch(n_freq=50)
        raw_rec_new = normalize_ecg_98(raw_rec_new)
        raw_lead = np.concatenate([raw_lead, np.expand_dims(raw_rec_new, axis=1)], axis=1)
        # if exists third lead:
        try:
            raw_rec_new = db.parse_raw_rec(id_, lead=2)
            raw_rec_new = raw_rec_new[5 * 60 * fs:]
            pre = Pre.Preprocessing(raw_rec_new, fs)
            raw_rec_new = pre.bpfilt()
            raw_rec_new = pre.notch(n_freq=50)
            raw_lead = np.concatenate([raw_lead, np.expand_dims(raw_rec_new, axis=1)], axis=1)
            dic_ = {'signal': raw_lead, 'header': {'recname': id_, 'freq': fs, 'nsamp': len(raw_lead), 'nsig': 3,
                                                   'gain': [1, 1, 1], 'adczero': [0, 0, 0], 'units': ['mV', 'mV', 'mV'],
                                                   'desc': ['I', 'II', 'III'], 'btime': '00:00:00',
                                                   'bdate': '29/10/1996'}}

        except:
            dic_ = {'signal': raw_lead, 'header': {'recname': id_, 'freq': fs, 'nsamp': len(raw_lead), 'nsig': 2,
                                                   'gain': [1, 1], 'adczero': [0, 0], 'units': ['mV', 'mV'],
                                                   'desc': ['I', 'III'], 'btime': '00:00:00', 'bdate': '29/10/1996'}}
            lead2ids += [id_]
        savemat('/home/sheina/PVC/mats/' + id_ + '.mat', dic_)
        savemat('/home/sheina/PVC/mats/' + id_ + '_QRS_detection.mat', epltd)

    return lead2ids


def run_pvcs(ids):
    for id_ in ids:
        example_file = id_ + '.mat'
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.cd(r'/home/sheina/ecg-kit/ecg-kit/examples', nargout=0)
        eng.hbc(example_file, nargout=0)


def clear_place(ids):
    pvc_path = '/home/sheina/PVC/mats/'
    pvc_head = '_ECG_heartbeat_classifier.mat'
    for id_ in ids:
        if os.path.exists(pvc_path + id_ + pvc_head):
            if os.path.exists(pvc_path + id_ + '.mat'):
                os.remove(pvc_path + id_ + '.mat')


if __name__ == '__main__':
    # raw_to_mat(cts.ext_test_vt +cts.ext_test_no_vt)
    # clear_place(cts.ids_tn + cts.ids_sp)
    # ids_to_groups(cts.ext_test_vt +cts.ext_test_no_vt)
    run_pvcs(cts.ext_test_vt + cts.ext_test_no_vt)
    # pvc_path = '/home/sheina/PVC/'
    # savemat(pvc_path + 'ids.mat', {'ids': cts.ext_test_vt + cts.ext_test_no_vt})
