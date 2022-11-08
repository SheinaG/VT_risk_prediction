import numpy as np
import pandas as pd

import utils.consts as cts
# from parsing.base_VT_parser import VtParser
from utils.base_packages import *


def create_segments_array(plot=1):
    db = RBAFDB_Parser(load_on_start=True)
    segments_array = pd.DataFrame(columns=['holter_id', 'start', 'end', 'len'])
    j = 0
    fs = 200
    ids_rbdb_VT = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))
    for id in ids_rbdb_VT:
        rhythm_df = db.parse_reference_rhythm(id)
        for i in rhythm_df.index:
            segments_array.loc[j] = 0
            segments_array.loc[j]['holter_id'] = id
            segments_array.loc[j]['start'] = rhythm_df.loc[i]['Beginning']
            segments_array.loc[j]['end'] = rhythm_df.loc[i]['End']
            segments_array.loc[j]['len'] = segments_array.loc[j]['end'] - segments_array.loc[j]['start']
            j = j + 1
    a = 5
    segments_array.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTp/segments_array_rbdb.xlsx')
    if plot:
        raw_ecg = db.parse_raw_ecg(segments_array['holter_id'][1], start=0, end=-1, type='epltd0', lead=1)
        raw_lead1 = raw_ecg[0]
        raw_ecg = db.parse_raw_ecg(segments_array['holter_id'][1], start=0, end=-1, type='epltd0', lead=2)
        raw_lead2 = raw_ecg[0]
        raw_ecg = db.parse_raw_ecg(segments_array['holter_id'][1], start=0, end=-1, type='epltd0', lead=3)
        raw_lead3 = raw_ecg[0]
        raw_lead = np.stack([raw_lead1, raw_lead2, raw_lead3], axis=0)
        start_samp, end_samp = int(segments_array['start'][1] * fs), int(segments_array['end'][1] * fs)
        part_to_plot = raw_lead[:, start_samp: end_samp]
        plot_ecg_fig_MD(part_to_plot, ['Ch. 1', 'Ch. 2', 'Ch. 3'], title='', row_height=8)
        plt.savefig('/MLAIM/AIMLab/Sheina/databases/VTdb/figures/1419Fc22_seg_2.png', dpi=400, transperant=True)
        raw_ecg = db.parse_raw_ecg(segments_array['holter_id'][4], start=0, end=-1, type='epltd0', lead=1)
        raw_lead1 = raw_ecg[0]
        raw_ecg = db.parse_raw_ecg(segments_array['holter_id'][4], start=0, end=-1, type='epltd0', lead=2)
        raw_lead2 = raw_ecg[0]
        raw_ecg = db.parse_raw_ecg(segments_array['holter_id'][4], start=0, end=-1, type='epltd0', lead=3)
        raw_lead3 = raw_ecg[0]
        raw_lead = np.stack([raw_lead1, raw_lead2, raw_lead3], axis=0)
        start_samp, end_samp = int(segments_array['start'][4] * fs), int(segments_array['end'][4] * fs)
        part_to_plot = raw_lead[:, start_samp - 600: end_samp + 600]
        plot_ecg_fig_MD(part_to_plot, ['Ch. 1', 'Ch. 2', 'Ch. 3'], title='', row_height=4)
        plt.savefig('/MLAIM/AIMLab/Sheina/databases/VTdb/figures/1318A93a_seg_1.png', dpi=400, transperant=True)

        part_to_plot = raw_lead[:, start_samp - 600: end_samp + 600]
        plot_ecg_fig_MD(part_to_plot, ['Ch. 1', 'Ch. 2', 'Ch. 3'], title='', row_height=4)
        plt.savefig('/MLAIM/AIMLab/Sheina/databases/VTdb/figures/1318A93a_seg_1.png', dpi=400, transperant=True)


def export_patient_ids():
    path_rbdb_info = '/MLAIM/AIMLab/Shany/databases/rbafdb/documentation/RBAF_Holter_Info.xlsx'
    path_rbdb_ann = '/MLAIM/AIMLab/Shany/databases/rbafdb/documentation/reports/Holter_labels_final.xlsx'

    test_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))
    train_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_VT_ids.npy'))
    VT_Holters = test_vt + train_vt
    no_VT_Holters = []
    md_test = []

    xl_info = pd.read_excel(path_rbdb_info, engine='openpyxl')
    xl_aa = pd.read_excel(path_rbdb_ann, engine='openpyxl')
    xl_info['holter_id'] = xl_info['holter_id'].astype(str)

    xl_work = pd.DataFrame(columns=xl_info.columns)
    #
    for id in xl_aa['Holter_id']:
        row_n = xl_info[xl_info["holter_id"] == id]
        xl_work = xl_work.append(row_n)

    h_work = list(xl_work["holter_id"])
    p_work = list(xl_work["db_id"])
    Nid = len(VT_Holters)

    patient_list = []
    excel_sheet_VT = pd.DataFrame(columns=xl_info.columns)
    excel_sheet_no_VT = pd.DataFrame(columns=xl_info.columns)
    excel_patients = pd.DataFrame(columns=xl_info.columns)

    all_sus_VT = xl_aa[xl_aa['VT'] == 1]
    all_VTs = list(all_sus_VT['Holter_id']) + VT_Holters
    all_VTs = list(set(all_VTs))

    for id_ in all_VTs:
        excel_sheet_VT = excel_sheet_VT.append(xl_info[xl_info["holter_id"] == id_])
        xl_info = xl_info.drop(xl_info[xl_info["holter_id"] == id_].index.values)
        if id_ in h_work:
            xl_work = xl_work.drop(xl_work[xl_work["holter_id"] == id_].index.values)
        if id_ not in VT_Holters:
            md_test.append(id_)

        patient_list.append(
            excel_sheet_VT["db_id"][excel_sheet_VT[excel_sheet_VT["holter_id"] == id_].index.values].values[0])

    patient_list = list(set(patient_list))

    for patient in patient_list:
        excel_patients = excel_patients.append(xl_info[xl_info["db_id"] == patient])
        xl_info = xl_info.drop(xl_info[xl_info["db_id"] == patient].index.values)
        if id in h_work:
            xl_work = xl_work.drop(xl_work[xl_work["db_id"] == patient].index.values)
    print(xl_info.shape)

    # ids_vt_aa = xl_aa["Holter_id"][xl_aa["VT"] ==1]
    #
    # for id in ids_vt_aa:
    #     row_n = xl_work[xl_work["db_id"] == id].index.values
    #     xl_work = xl_work.drop(row_n)
    # print(xl_work.shape)

    excel_sheet_VT = excel_sheet_VT.append(excel_patients)
    # duplicated_ids:
    excel_sheet_VT = excel_sheet_VT.drop(excel_sheet_VT[excel_sheet_VT["holter_id"] == 'H520818b'].index.values)
    excel_sheet_VT = excel_sheet_VT.drop(excel_sheet_VT[excel_sheet_VT["holter_id"] == '8520F416'].index.values)
    md_test.remove('H520818b')

    data = excel_sheet_VT["age_at_recording"]
    data[np.isnan(data)] = np.median(data[~np.isnan(data)])
    excel_sheet_VT["age_at_recording"] = data
    print(xl_work.shape)

    right_age = xl_work[xl_work["age_at_recording"] > 18]
    right_age = right_age[right_age["age_at_recording"] < 100]
    excel_sheet_no_VT = right_age
    excel_sheet_no_VT.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTn/excel_sheet_no_VT_rbdb.xlsx')
    excel_sheet_VT.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTp/excel_sheet_VT_rbdb.xlsx')


def recording_len(PATH_TO_DICT, ids_='all'):
    with open(PATH_TO_DICT / 'len_dict.pkl', 'rb') as f:
        data = pickle.load(f)
    if ids_ == 'all':
        len_array = np.asarray(list(data.values()))
        median = np.median(len_array)
        m, s = divmod(median, 60)
        h, m = divmod(m, 60)
        p75 = np.percentile(len_array, 75)
        p25 = np.percentile(len_array, 25)
        m75, s75 = divmod(p75, 60)
        h75, m75 = divmod(m75, 60)
        m25, s25 = divmod(p25, 60)
        h25, m25 = divmod(m25, 60)
        print(f'{h:.0f}:{m:.0f}:{s:.0f}')
        print(f'{h75:.0f}:{m75:.0f}:{s75:.0f}')
        print(f'{h25:.0f}:{m25:.0f}:{s25:.0f}')


def train_val_test_split():
    xl_no_vt = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTn/excel_sheet_no_VT_rbdb.xlsx', engine='openpyxl')
    xl_no_vt = xl_no_vt.drop_duplicates(subset='holter_id')
    print(xl_no_vt.shape)
    test_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_no_VT_ids.npy'))
    for id_ in test_no_vt:
        if not xl_no_vt[xl_no_vt["holter_id"] == id_].index.values:
            test_no_vt.remove(id_)
            print(id_)
        xl_no_vt = xl_no_vt.drop(xl_no_vt[xl_no_vt["holter_id"] == id_].index.values)
    # np.save('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_no_VT_ids.npy', test_no_vt)
    xl_no_vt = xl_no_vt.sort_values('db_id')
    db_ids_val = list(set(list(xl_no_vt["db_id"])))[:165]
    xl_val = pd.DataFrame(columns=xl_no_vt.columns)
    for id_ in db_ids_val:
        xl_val = xl_val.append(xl_no_vt[xl_no_vt["db_id"] == id_])
        xl_no_vt = xl_no_vt.drop(xl_no_vt[xl_no_vt["db_id"] == id_].index.values)
    val_no_vt_h = list(xl_val['holter_id'])
    train_val_no_vt_h = list(xl_no_vt['holter_id'])
    # np.save('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_val_no_VT_ids.npy', val_no_vt_h)
    # np.save('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_no_VT_ids.npy', train_val_no_vt_h)
    a = 5


def rbdb_new_dem(ids):
    main_path = '/MLAIM/AIMLab/Shany/databases/rbafdb/documentation/RBAF_Holter_Info.xlsx'
    rbdb_info = pd.read_excel(main_path, engine='openpyxl')
    rbdb_info['holter_id'] = rbdb_info['holter_id'].astype(str)
    ids_db = []
    for id_ in ids:
        # find his patiant id:
        ids_db.append(rbdb_info[rbdb_info['holter_id'] == id_]['db_id'].values[0])

    n_ids_db = len(list(set(ids_db)))

    main_path = '/MLAIM/AIMLab/Shany/databases/rbafdb/documentation/RBAF_Holter_Info_mdclone.xlsx'
    rbdb_mdclone = pd.read_excel(main_path, engine='openpyxl')
    new_features = ['bmi', 'smoking', 'pacemaker', 'ablation']
    dem_holters = pd.DataFrame(columns=['holter id'] + new_features)
    # all_features = rbdb_mdclone.columns
    # interesting_features = ['weight-result numeric','height-result numeric','smoking-diagnosis', 'pacemaker-procedure date', 'history of ablation ever-event date']
    for i, id_ in enumerate(ids):
        holter_date = rbdb_info[rbdb_info['holter_id'] == id_]['recording_date'].values[0]
        ablation_date = rbdb_mdclone[rbdb_mdclone['db_id'] == ids_db[i]]['history of ablation ever-event date'].values[
            0]
        if not (np.isnat(ablation_date)):
            if ablation_date < holter_date:
                ablation = 1
            else:
                ablation = 0

        else:
            ablation = 0

        pacemaker_date = rbdb_mdclone[rbdb_mdclone['db_id'] == ids_db[i]]['pacemaker-procedure date'].values[0]

        if not (np.isnat(pacemaker_date)):
            if pacemaker_date < holter_date:
                pacemaker = 1
            else:
                pacemaker = 0
        else:
            pacemaker = 0

        weight = rbdb_mdclone[rbdb_mdclone['db_id'] == ids_db[i]]['weight-result numeric'].values[0]
        height = rbdb_mdclone[rbdb_mdclone['db_id'] == ids_db[i]]['height-result numeric'].values[0]

        if (np.isnan(weight)) or (np.isnan(height)):
            bmi = 0
        elif height < 3:
            bmi = weight / np.power(height, 2)
        elif height < 100:
            bmi = 0
        elif height < weight:
            bmi = height / np.power(weight / 100, 2)
        else:
            bmi = weight / np.power(height / 100, 2)

        if bmi > 100:
            a = 5

        smoking_diagnosis = rbdb_mdclone[rbdb_mdclone['db_id'] == ids_db[i]]['smoking-diagnosis'].values[0]

        try:
            np.isnan(smoking_diagnosis)
            smoking = 0
        except:
            smoking = 1

        dem_holters_i = pd.DataFrame([[id_, bmi, smoking, pacemaker, ablation]], columns=['holter id'] + new_features,
                                     index=[i])
        dem_holters = dem_holters.append(dem_holters_i)

    n_no_bmi = len(dem_holters[dem_holters['bmi'] == 0])
    mean_bmi = np.sum(dem_holters[dem_holters['bmi'] != 0]['bmi'].values)

    dem_holters.loc[dem_holters['bmi'] == 0, 'bmi'] = mean_bmi / len(dem_holters[dem_holters['bmi'] != 0])
    dem_holters.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/new_dem.xlsx')
    a = 5

    return

def append_rhythms_array(self, pat, new_line, arr):
    start = 0
    end = -1
    rhythms = np.array([int(i) for i in self.rlab_dict[pat][self.window_size]])
    periods = np.concatenate(([0], np.where(np.diff(rhythms.astype(int)))[0] + 1, [len(rhythms) - 1]))
    start_idx, end_idx = periods[:-1], periods[1:]
    final_rhythms = rhythms[start_idx]
    mask_rhythms = (final_rhythms == 14)  # We do not keep NSR as rhythm (keep just the VT rethem)
    raw_rrt = self.rrt_dict[pat]
    rrt = raw_rrt[:(len(raw_rrt) // self.window_size) * self.window_size].reshape(-1, self.window_size)
    start_events, end_events = rrt[start_idx, 0], rrt[end_idx, 0]
    mask_int_events = np.logical_and(start_events < end, end_events > start)
    mask_rhythms = np.logical_and(mask_rhythms, mask_int_events)
    start_events, end_events = start_events[mask_rhythms] - start, end_events[mask_rhythms] - start
    end_events[end_events > (end - start)] = end - start
    for i in range(len(start_events)):
        arr[new_line + i, :] = [pat, start_events, end_events, end_events - start_events]
    return arr, new_line + i + 1


if __name__ == '__main__':
    # db = VtParser()
    # segments_array = pd.DataFrame(columns=['holter_id', 'start', 'end', 'len'])
    # j = 0
    # for id in cts.ids_rbdb_VT:
    #     rhythm_df = db.parse_reference_rhythm(id)
    #     for i in rhythm_df.index:
    #         segments_array.loc[j] = 0
    #         segments_array.loc[j]['holter_id'] = id
    #         segments_array.loc[j]['start'] = rhythm_df.loc[i]['Beginning']
    #         segments_array.loc[j]['end'] = rhythm_df.loc[i]['End']
    #         segments_array.loc[j]['len'] = segments_array.loc[j]['end'] - segments_array.loc[j]['start']
    #         j = j + 1
    # a = 5

    ids_tn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_no_VT_ids.npy'))
    ids_sn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_no_VT_ids.npy'))
    ids_tp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_VT_ids.npy'))
    ids_sp = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))
    ids_vn = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_val_no_VT_ids.npy'))

    rbdb_new_dem(ids_tn + ids_sn + ids_tp + ids_sp + ids_vn)
