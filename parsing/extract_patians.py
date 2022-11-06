from base_packages import *

from utils import consts as cts


def extract_patients():
    ids_VT = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_VT_ids.npy'))
    Nid = len(ids_VT)

    ids_no_VT = []
    patient_list = []
    # db = UVAFDB_Parser(load_ectopics=False, load_on_start=True)
    excel_sheet_path = cts.DATA_DIR / "uvfdb" / "uvfdb_rr" / "UVA_Holter_Info.xlsx"
    excel_sheet = pd.read_excel(excel_sheet_path, engine='openpyxl')
    excel_sheet_VT = pd.DataFrame(columns=excel_sheet.columns)
    excel_sheet_no_VT = pd.DataFrame(columns=excel_sheet.columns)
    excel_patients = pd.DataFrame(columns=excel_sheet.columns)

    for id in ids_VT:
        excel_sheet_VT = excel_sheet_VT.append(excel_sheet[excel_sheet["Holter ID"] == "UVA" + id])
        excel_sheet = excel_sheet.drop(excel_sheet[excel_sheet["Holter ID"] == "UVA" + id].index.values)
        patient_list.append(
            int(excel_sheet_VT["Patient ID"][excel_sheet_VT[excel_sheet_VT["Holter ID"] == "UVA" + id].index.values]))

    patient_list = list(set(patient_list))

    for patient in patient_list:
        excel_patients = excel_patients.append(excel_sheet[excel_sheet["Patient ID"] == patient])
        excel_sheet = excel_sheet.drop(excel_sheet[excel_sheet["Patient ID"] == patient].index.values)

    excel_sheet_VT = excel_sheet_VT.append(excel_patients)

    right_age = excel_sheet[excel_sheet["Age at First"] > 18]
    right_age = right_age[right_age["Age at First"] < 100]
    excel_sheet_no_VT = right_age
    excel_sheet_no_VT.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTn/excel_sheet_no_VT_uvafdb.xlsx')
    excel_sheet_VT.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTp/excel_sheet_VT_uvafdb.xlsx')
    non_VT_ids = list(excel_sheet_no_VT['Holter ID'])
    VT_ids = list(excel_sheet_VT['Holter ID'])
    np.save('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_VT_ids.npy', VT_ids)
    np.save('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_non_VT_ids.npy', non_VT_ids)


def statistic_uvafdb():
    excel_sheet_path = cts.DATA_DIR / "uvfdb" / "uvfdb_rr" / "UVA_Holter_Info.xlsx"
    excel_sheet = pd.read_excel(excel_sheet_path, engine='openpyxl')
    recording_num = excel_sheet.shape[0]
    patient_num = len(set(list(excel_sheet['Patient ID'])))
    print('Recording #: ' + str(recording_num))
    print('patient #: ' + str(patient_num))
    ra_es = excel_sheet[excel_sheet['Age at First'] > 18]
    ra_recording_num = ra_es.shape[0]
    ra_patient_num = len(set(list(ra_es['Patient ID'])))
    print('Adults:')
    print('Recording #: ' + str(ra_recording_num))
    print('patient #: ' + str(ra_patient_num))
    es_pa = ra_es.drop_duplicates(subset='Patient ID')
    age_pa = np.asarray(es_pa['Age at First'])
    median_age = np.median(age_pa)
    q75, q25 = np.percentile(age_pa, [75, 25])
    iqr_ = q75 - q25
    print('The median age is: ' + str(median_age) + '(' + str(iqr_) + ')')
    gender_pa = np.asarray(pd.get_dummies(es_pa['Gender'])['M'])
    male_precent = sum(gender_pa) / len(gender_pa)
    print('male percent is: ' + str(male_precent))

    # After choosing and exckuding Bad BSQI recordings
    VT_xlsx = pd.DataFrame(columns=excel_sheet.columns)
    non_VT_xlsx = pd.DataFrame(columns=excel_sheet.columns)
    VT_ids = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_VT_ids.npy'))
    non_VT_ids = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_non_VT_ids.npy'))
    for id_ in VT_ids:
        VT_xlsx = VT_xlsx.append(excel_sheet[excel_sheet["Holter ID"] == id_])
    for id_ in non_VT_ids:
        non_VT_xlsx = non_VT_xlsx.append(excel_sheet[excel_sheet["Holter ID"] == id_])

    Used_recordings = VT_xlsx.append(non_VT_xlsx)
    ra_recording_num = Used_recordings.shape[0]
    ra_patient_num = len(set(list(Used_recordings['Patient ID'])))
    print('Adults:')
    print('Recording #: ' + str(ra_recording_num))
    print('patient #: ' + str(ra_patient_num))
    es_pa = Used_recordings.drop_duplicates(subset='Patient ID')
    age_pa = np.asarray(es_pa['Age at First'])
    median_age = np.median(age_pa)
    q75, q25 = np.percentile(age_pa, [75, 25])
    iqr_ = q75 - q25
    print('The median age is: ' + str(median_age) + '(' + str(iqr_) + ')')
    gender_pa = np.asarray(pd.get_dummies(es_pa['Gender'])['M'])
    male_precent = sum(gender_pa) / len(gender_pa)
    print('male percent is: ' + str(male_precent))

    a = 5


if __name__ == "__main__":
    # extract_patients()
    statistic_uvafdb()
    # db = UVAFDB_Parser(load_ectopics=False, load_on_start=True)
    # path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/bad_bsqi/')
    # for rec in cts.bad_bsqi:
    #     db.export_ecg(rec, path)

# def export_ecg(self, pat, directory=cts.ERROR_ANALYSIS_DIR, start=0, end=-1, force=False, ann_type='epltd0',
#                         export_rhythms=False, n_leads=1):
#     """ This function exports a given recording to the physiozoo format for further investigation.
#     The files are exported under a .txt format with the proper headers to be read by the PhysioZoo software.
#     The raw ECG as well as the peaks are exported. If provided, the AF events are exported as well.
#     :param pat: The patient ID to export.
#     :param directory: The directory under which the files will be saved.
#     :param start: The beginning timestamp for the ECG portion to be saved.
#     :param end: The ending timestamp for the ECG portion to be saved.
#     :param force: If True, generate the files anyway, otherwise verifies if the file has already been exported under the directory
#     :param export_rhythms: If True, exports a file with the different rhythms over the recording."""
#
#     ecg_header = ['---\n',
#                   'Mammal:            human\n',
#                   'Fs:                ' + str(cts.EPLTD_FS) + '\n',
#                   'Integration_level: electrocardiogram\n',
#                   '\n'
#                   'Channels:\n',
#                   '\n'
#                   '    - type:   electrography\n',
#                   '      name:   Data\n',
#                   '      unit:   mV\n',
#                   '      enable: yes\n',
#                   '\n',
#                   '---\n',
#                   '\n'
#                   ]
#
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
#     ecgs = np.concatenate(tuple(
#         [self.parse_raw_ecg(pat, start=start, end=end, type=ann_type, lead=lead)[0].reshape(-1, 1) for lead in
#          range(n_leads)]), axis=1)
#     if end == -1:
#         end = self.recording_time[pat]
#
#     ecg_full_path = directory / (
#                 pat + '_ecg_start_' + str(start) + '_end_' + str(end) + '_n_leads_' + str(n_leads) + '.txt')
#
#     # Raw ECG
#     join_func = lambda x: ' '.join(['%.4f' % i for i in x])
#     ecg_str = np.apply_along_axis(join_func, 1, ecgs)
#     if not os.path.exists(ecg_full_path) or force:
#         with open(ecg_full_path, 'w+') as ecg_file:
#             ecg_file.writelines(ecg_header)
#             ecg_file.write('\n'.join(ecg_str))

# def export_res_rythems(self, pat, directory=cts.ERROR_ANALYSIS_DIR, start=0, end=-1, force=False, ann_type='epltd0',
#                         export_rhythms=False,n_leads=1 ):
#     """ This function exports a given recording to the physiozoo format for further investigation.
#     The files are exported under a .txt format with the proper headers to be read by the PhysioZoo software.
#     The raw ECG as well as the peaks are exported. If provided, the AF events are exported as well.
#     :param pat: The patient ID to export.
#     :param directory: The directory under which the files will be saved.
#     :param start: The beginning timestamp for the ECG portion to be saved.
#     :param end: The ending timestamp for the ECG portion to be saved.
#     :param force: If True, generate the files anyway, otherwise verifies if the file has already been exported under the directory
#     :param export_rhythms: If True, exports a file with the different rhythms over the recording."""
#
#
#
#     sig_qual_header = ['---\n',
#                        'type: quality annotation\n',
#                        'source file: ',  # To be completed by filename
#                        '\n',
#                        '---\n',
#                        '\n',
#                        'Beginning\tEnd\t\tClass\n']
#
#     rhythms_header = copy.deepcopy(sig_qual_header)
#     rhythms_header[1] = 'type: rhythms annotation\n'
#
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
#     if end == -1:
#         end = self.recording_time[pat]
#
#     rhythms_full_path = directory / (pat + '_res_rhythms_start_' + str(start) + '_end_' + str(end) + '.txt')
#
#
#     # Rhythms
#     if export_rhythms:
#         if not os.path.exists(rhythms_full_path) or force:
#             with open(rhythms_full_path, 'w+') as rhythms_file:
#                 rhythms_header[2] = 'source file: ' + pat + '_rhythms.txt\n'
#                 rhythms_file.writelines(rhythms_header)
#                 rhythms = np.array([int(i) for i in self.rlab_dict[pat][self.window_size]])
#                 periods = np.concatenate(([0], np.where(np.diff(rhythms.astype(int)))[0] + 1, [len(rhythms) - 1]))
#                 start_idx, end_idx = periods[:-1], periods[1:]
#                 final_rhythms = rhythms[start_idx]
#                 mask_rhythms = (final_rhythms == 14 ) # We do not keep NSR as rhythm (keep just the VT rethem)
#                 raw_rrt = self.rrt_dict[pat]
#                 rrt = raw_rrt[:(len(raw_rrt) // self.window_size) * self.window_size].reshape(-1, self.window_size)
#                 start_events, end_events = rrt[start_idx, 0], rrt[end_idx, 0]
#                 mask_int_events = np.logical_and(start_events < end, end_events > start)
#                 mask_rhythms = np.logical_and(mask_rhythms, mask_int_events)
#                 start_events, end_events = start_events[mask_rhythms] - start, end_events[mask_rhythms] - start
#                 end_events[end_events > (end - start)] = end - start
#                 final_rhythms = final_rhythms[mask_rhythms]
#                 final_rhythms_str = np.array([self.rhythms[i][1:] for i in final_rhythms])
#                 rhythms_file.write('\n'.join(
#                     ['%.5f\t%.5f\t%s' % (start_events[i], end_events[i], final_rhythms_str[i]) for i in
#                      range(len(start_events))]))
#
#
