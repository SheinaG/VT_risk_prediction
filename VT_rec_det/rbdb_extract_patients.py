import random

from utils.base_packages import *
from parsing.base_VT_parser import *
from utils.plot_ecg import plot_ecg_fig_MD



def create_segments_array(plot=1):
    db = VtParser()
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
    list_no_vt_db = list(set(list(xl_no_vt["db_id"])))
    random.shuffle(list_no_vt_db)
    test_no_vt = list_no_vt_db[:130]
    val_no_vt = list_no_vt_db[130:260]
    train_no_vt = list_no_vt_db[260:]
    lists = [test_no_vt, val_no_vt, train_no_vt]
    list_names = ['test_no_vt', 'val_no_vt', 'train_no_vt']

    for i, list_ in enumerate(lists):
        list_no_vt = []
        for id_ in list_:
            list_no_vt.append(xl_no_vt[xl_no_vt["db_id"] == id_]["holter_id"].values[0])
            xl_no_vt = xl_no_vt.drop(xl_no_vt[xl_no_vt["db_id"] == id_].index.values)
        np.save('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/' + list_names[i] + '_2.npy', list_no_vt)

    a = 5


def split_train_to_k_folders(k_folders=5, split_way=1):
    """

    """
    VT_ids = cts.ids_tp
    non_VT_ids = cts.ids_tn + cts.ids_vn
    main_path = '/MLAIM/AIMLab/Shany/databases/rbafdb/documentation/RBAF_Holter_Info.xlsx'
    rbdb_info = pd.read_excel(main_path, engine='openpyxl')
    ids_db_VT = []
    ids_db_non_VT = []
    for id_ in VT_ids:
        ids_db_VT.append(rbdb_info[rbdb_info['holter_id'] == id_]['db_id'].values[0])
    for id_ in non_VT_ids:
        ids_db_non_VT.append(rbdb_info[rbdb_info['holter_id'] == id_]['db_id'].values[0])
    ids_db_VT = list(set(ids_db_VT))
    ids_db_VT.sort()
    ids_db_non_VT = list(set(ids_db_non_VT))
    ids_db_non_VT.sort()
    if split_way == 1:
        VT_g1, VT_g2, VT_g3, VT_g4 = ids_db_VT[:10], ids_db_VT[10:18] + ids_db_VT[26:28], ids_db_VT[18:26], ids_db_VT[
                                                                                                            28:]
        nVT_g1, nVT_g2, nVT_g3, nVT_g4 = ids_db_non_VT[:371], ids_db_non_VT[371:742], ids_db_non_VT[
                                                                                      742:1113], ids_db_non_VT[1113:]
        groups = []
        groups.append(VT_g1 + nVT_g1)
        groups.append(VT_g2 + nVT_g2)
        groups.append(VT_g3 + nVT_g3)
        groups.append(VT_g4 + nVT_g4)

    if split_way == 2:
        groups = []
        for i in range(k_folders):
            groups.append([])
        all_dbs = ids_db_VT + ids_db_non_VT
        for id_ in all_dbs:
            ind = all_dbs.index(id_)
            j = int(ind % 4)
            groups[j].append(id_)

    if split_way == 3:
        groups = []
        VT_in_grop = np.ceil(len(ids_db_VT) / k_folders)
        ratio = np.ceil(len(ids_db_non_VT) / len(ids_db_VT))
        for i in range(k_folders):
            groups.append([])
        # all_dbs = ids_db_VT + ids_db_non_VT
        for i, id_ in enumerate(ids_db_VT):
            j = int(np.floor(i / VT_in_grop))
            groups[j].append(id_)
        for i, id_ in enumerate(ids_db_non_VT):
            j = int(np.floor(i / (VT_in_grop * ratio)))
            try:
                groups[j].append(id_)
            except:
                print(j, i, ratio)

    groups_ids = []
    for group in groups:
        group_id = []
        for id_ in group:
            group_id.append(list(rbdb_info[rbdb_info['db_id'] == id_]['holter_id']))
        group_id_flat = [item for sublist in group_id for item in sublist]
        for id_ in group_id_flat:
            if id_ not in VT_ids + non_VT_ids:
                group_id_flat.remove(id_)
        print(len(group_id_flat))
        groups_ids.append(group_id_flat)


    np.save('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/train_groups' + str(k_folders) + str(split_way) + '.npy',
            groups_ids)

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


def calculate_demographic(name):
    if name == 'uvafdb':

        excel_sheet_no_VT = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTn/excel_sheet_no_VT_uvafdb.xlsx',
                                          engine='openpyxl')
        excel_sheet_VT = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTp/excel_sheet_VT_uvafdb.xlsx',
                                       engine='openpyxl')
        demographic_no_VT_xl = pd.DataFrame(columns=['age', 'gender'], index=list(excel_sheet_no_VT['Holter ID']))
        demographic_VT_xl = pd.DataFrame(columns=['age', 'gender'], index=list(excel_sheet_VT['Holter ID']))
        demographic_no_VT_xl['age'] = (np.asarray(excel_sheet_no_VT["Age (days)"]) + np.asarray(
            excel_sheet_no_VT["Days from First"])) // DY
        demographic_VT_xl['age'] = (np.asarray(excel_sheet_VT["Age (days)"]) + np.asarray(
            excel_sheet_VT["Days from First"])) // DY
        demographic_no_VT_xl['gender'] = np.asarray(pd.get_dummies(excel_sheet_no_VT['Gender'])['M'])
        demographic_VT_xl['gender'] = np.asarray(pd.get_dummies(excel_sheet_VT['Gender'])['M'])
        demographic_VT_xl.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTp/demographic_features_uvafdb.xlsx')
        demographic_no_VT_xl.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTn/demographic_features_uvafdb.xlsx')
    elif name == 'rbdb':
        excel_sheet_no_VT = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTn/excel_sheet_no_VT_rbdb.xlsx',
                                          engine='openpyxl')
        excel_sheet_VT = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTp/excel_sheet_VT_rbdb.xlsx',
                                       engine='openpyxl')
        demographic_no_VT_xl = pd.DataFrame(columns=['age', 'gender'], index=list(excel_sheet_no_VT['holter_id']))
        demographic_VT_xl = pd.DataFrame(columns=['age', 'gender'], index=list(excel_sheet_VT['holter_id']))
        demographic_no_VT_xl['age'] = np.asarray(excel_sheet_no_VT["age_at_recording"])
        demographic_VT_xl['age'] = np.asarray(excel_sheet_VT["age_at_recording"])
        demographic_no_VT_xl['gender'] = np.asarray(pd.get_dummies(excel_sheet_no_VT['sex'])['M'])
        demographic_VT_xl['gender'] = np.asarray(pd.get_dummies(excel_sheet_VT['sex'])['M'])
        demographic_VT_xl.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTp/demographic_features_rbdb.xlsx')
        demographic_no_VT_xl.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTn/demographic_features_rbdb.xlsx')


def rhythms_array(ids):
    segments_array = pd.DataFrame(columns=['holter_id', 'start', 'end'])
    i = 0
    start_events, end_events = [], []
    for id_ in ids:
        path = '/MLAIM/AIMLab/Sheina/databases/VTdb/rean_rbdb/'
        txt_file = path + id_ + '_ecg_start_0_end_3_n_leads_3_rhythms.txt'
        try:
            with open(txt_file) as f:
                lines = f.readlines()
        except FileNotFoundError:
            txt_file = path + id_ + '_ecg_start_0_end_2_n_leads_2_rhythms.txt'
            with open(txt_file) as f:
                lines = f.readlines()
        f = 0
        for line in lines:
            if line == 'Beginning\tEnd\t\tClass\n':
                f = 1
            elif f == 1:
                split_line = line.split('\t')
                start_events.append(split_line[0])
                end_events.append(split_line[1])
                segments_array = segments_array.append(
                    pd.DataFrame([[id_, split_line[0], split_line[1]]], columns=segments_array.columns))
                i += 1
    segments_array.to_excel('/MLAIM/AIMLab/Sheina/databases/VTdb/VTp/segments_array_rbdb')


if __name__ == '__main__':
    # rhythms_array(cts.ids_conf + cts.ids_sp)
    # split_train_to_k_folders(k_folders=38, split_way=3)
    # train_val_test_split()
    rbdb_new_dem(cts.ids_vp + cts.ids_vn + cts.ids_tp + cts.ids_tn + cts.ids_sp + cts.ids_sn)
