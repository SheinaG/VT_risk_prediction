import numpy as np
import pandas as pd
import sheina.consts as cts

from parsing.base_VT_parser import VtParser


def export_patient_ids():
    path_rbdb_info = '/MLAIM/AIMLab/Shany/databases/rbafdb/documentation/RBAF_Holter_Info.xlsx'
    path_rbdb_ann = '/MLAIM/AIMLab/Shany/databases/rbafdb/documentation/reports/Holter_labels_final.xlsx'

    test_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_test_VT_ids.npy'))
    train_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/RBDB_train_VT_ids.npy'))
    VT_Holters = test_vt + train_vt

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

    for id in VT_Holters:
        excel_sheet_VT = excel_sheet_VT.append(xl_info[xl_info["holter_id"] == id])
        xl_info = xl_info.drop(xl_info[xl_info["holter_id"] == id].index.values)
        if id in h_work:
            xl_work = xl_work.drop(xl_work[xl_work["holter_id"] == id].index.values)
        patient_list.append(
            excel_sheet_VT["db_id"][excel_sheet_VT[excel_sheet_VT["holter_id"] == id].index.values].values[0])

    patient_list = list(set(patient_list))

    for patient in patient_list:
        excel_patients = excel_patients.append(xl_info[xl_info["db_id"] == patient])
        xl_info = xl_info.drop(xl_info[xl_info["db_id"] == patient].index.values)
        if id in h_work:
            xl_work = xl_work.drop(xl_work[xl_work["db_id"] == patient].index.values)
    print(xl_info.shape)

    excel_sheet_VT = excel_sheet_VT.append(excel_patients)
    # duplicated_ids:
    excel_sheet_VT = excel_sheet_VT.drop(excel_sheet_VT[excel_sheet_VT["holter_id"] == 'H520818b'].index.values)
    excel_sheet_VT = excel_sheet_VT.drop(excel_sheet_VT[excel_sheet_VT["holter_id"] == '8520F416'].index.values)

    data = excel_sheet_VT["age_at_recording"]
    data[np.isnan(data)] = np.median(data[~np.isnan(data)])
    excel_sheet_VT["age_at_recording"] = data
    print(xl_work.shape)

    right_age = xl_work[xl_work["age_at_recording"] > 18]
    right_age = right_age[right_age["age_at_recording"] < 100]
    excel_sheet_no_VT = right_age
    excel_sheet_no_VT.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTn/excel_sheet_no_VT_rbdb.xlsx')
    excel_sheet_VT.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTp/excel_sheet_VT_rbdb.xlsx')


if __name__ == '__main__':
    db = VtParser()
    segments_array = pd.DataFrame(columns=['holter_id', 'start', 'end', 'len'])
    j = 0
    for id in cts.ids_rbdb_VT:
        rhythm_df = db.parse_reference_rhythm(id)
        for i in rhythm_df.index:
            segments_array.loc[j] = 0
            segments_array.loc[j]['holter_id'] = id
            segments_array.loc[j]['start'] = rhythm_df.loc[i]['Beginning']
            segments_array.loc[j]['end'] = rhythm_df.loc[i]['End']
            segments_array.loc[j]['len'] = segments_array.loc[j]['end'] - segments_array.loc[j]['start']
            j = j + 1
    a = 5


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
