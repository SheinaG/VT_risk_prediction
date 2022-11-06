import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from sheina.plot_ecg import plot_ecg_fig_MD
from uvafdb_parser import *


def explor_VT_segmentes_statistics():
    ids = ['1601', '1630', '1769', '1804', '2045', '2223', '2523', '2588', '2606']

    Nid = len(ids)

    db = UVAFDB_Parser(load_ectopics=False, load_on_start=True)
    # ids = np.load('/home/sheina/ids_VT.npy', allow_pickle=True)
    dir_p = pathlib.PurePath('/home/sheina/VT_3_2')

    new_line = 0
    arr = np.zeros([10000, 4])

    for id in ids:
        arr, new_line = db.append_rhythms_array(id, new_line, arr)
    arr = arr[0:new_line, :]
    mean_len = np.mean(arr[:, 3])
    median_len = np.median(arr[:, 3])
    q75, q25 = np.percentile(arr[:, 3], [75, 25])
    iqr_ = q75 - q25
    len_per_patient = np.zeros(Nid)
    count_per_patient = np.zeros(Nid)

    i = 0
    for j, id in enumerate(ids):
        while np.int(id) == arr[i, 0]:
            len_per_patient[j] = len_per_patient[j] + arr[i, 3]
            count_per_patient[j] = count_per_patient[j] + 1
            i = i + 1
            if i >= new_line:
                break
        if i >= new_line:
            break
    a = 5

    MIN, MAX = .1, 1000

    pl.figure()
    pl.hist(arr[:, 3], bins=10 ** np.linspace(np.log10(MIN), np.log10(MAX), 50))
    pl.gca().set_xscale("log")
    pl.title('Length of VT segments')
    pl.xlabel('time[sec]')
    pl.savefig('/MLAIM/AIMLab/Sheina/databases/VTdb/uvdb/hist_vt_seg.png')

    pl.figure()
    pl.title('Length of all VT segments per patient')
    pl.xlabel('patient #')
    pl.ylabel('time[sec]')
    pl.bar(ids, len_per_patient)
    pl.savefig('/MLAIM/AIMLab/Sheina/databases/VTdb/uvdb/len_vt_seg.png')

    pl.figure()
    pl.title('Num of VT segments per patient')
    pl.xlabel('patient #')
    pl.bar(ids, count_per_patient)
    pl.savefig('/MLAIM/AIMLab/Sheina/databases/VTdb/uvdb/num_vt_seg.png')

    aa1 = db.parse_raw_ecg(ids[6], start=0, end=-1, type='epltd0', lead=[0, 1, 2])
    ecg_long = np.transpose(aa1[0][np.int(arr[20, 1] * 200):np.int(arr[20, 2] * 200)])

    plot_ecg_fig_MD(
        ecg=ecg_long,
        lead_index=['I', 'II', 'III'],
        sample_rate=200,
        title='Holter',
        columns=1,
    )
    VT_list = []
    for id in db.rlab_dict:
        if 14 in db.rlab_dict[id]:
            VT_list.append(id)


def plot_statistics(dataset):
    DY = 365
    plt.style.use('seaborn-deep')
    es_no_VT = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTn/excel_sheet_no_VT' + dataset + '.xlsx',
                             engine='openpyxl')
    es_VT = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTp/excel_sheet_VT' + dataset + '.xlsx',
                          engine='openpyxl')

    # gender statistics per patient

    # 1. remove redandence recording:
    es_VT = es_VT.drop_duplicates(subset="Patient ID")
    es_no_VT.drop_duplicates(subset="Patient ID")

    # 2.calculate statistics per patient

    F_VT = len(es_VT["Patient ID"][es_VT[es_VT["Gender"] == "F"].index.values])
    F_no_VT = len(es_no_VT["Patient ID"][es_no_VT[es_no_VT["Gender"] == "F"].index.values])
    M_VT = len(es_VT["Patient ID"][es_VT[es_VT["Gender"] == "M"].index.values])
    M_no_VT = len(es_no_VT["Patient ID"][es_no_VT[es_no_VT["Gender"] == "M"].index.values])

    Age_VT = (np.asarray(es_VT["Age (days)"]) + np.asarray(es_VT["Days from First"])) // DY
    Age_no_VT = (np.asarray(es_no_VT["Age (days)"]) + np.asarray(es_no_VT["Days from First"])) // DY

    Age_all = np.concatenate([Age_VT, Age_no_VT], axis=0)
    bins = np.linspace(np.min(Age_all), np.max(Age_all), np.max(Age_all) - np.min(Age_all) + 1)

    plt.figure()
    plt.hist([Age_VT, Age_no_VT], bins, label=['VT', 'no VT'])
    plt.legend(loc='upper right')
    plt.show()

    pl.figure()


if __name__ == '__main__':
    plot_statistics('')
    # plot_statistics('_rbdb')
