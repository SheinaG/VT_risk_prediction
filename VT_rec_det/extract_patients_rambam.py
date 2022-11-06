import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metrics import *
from rbaf_parser import *
from sheina.consts import *
from sklearn.metrics import roc_curve, auc


def extract_by_reports():
    excel_sheet_path_all = rbdb_doc_path + 'RBAF_Holter_Info.xlsx'
    es_all = pd.read_excel(excel_sheet_path_all, engine='openpyxl')

    taged_vt_holters = es_all[
        es_all['diagnosis_merged'] == 'Paroxysmal ventricular tachycardia ']  # diagnosis_prometheus,diagnosis_merged

    db = RBAFDB_Parser(load_on_start=True)
    holter_list = set(taged_vt_holters['Holter_ID'])  # add here the holter ids you need to extract examples for
    for holter in holter_list:
        directory = pathlib.PurePath("/MAIM/AIMLab/Sheina/databases/rbdb/PZ_format")  # where to save the examples
        if not os.path.exists(directory):
            os.makedirs(directory)
        db.export_to_physiozoo(holter, directory=directory, export_rhythms=True, force=True,
                               n_leads=db.get_num_leads(holter))


def extract_by_keywords():
    db = RBAFDB_Parser(load_on_start=True)
    holter_labels = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/Holter_label.xlsx', engine='openpyxl')
    keywords = pd.read_excel(rbdb_doc_path + 'reports/RBAF_reports.xlsx', engine='openpyxl')

    vt_holters = holter_labels[holter_labels['(VT'] > 80]

    HSVH = pd.DataFrame(columns=['vt score', 'text a', 'text b', 'sheina_test'],
                        index=vt_holters['Unnamed: 0'])  # high scored VT holters

    for i, holter in enumerate(vt_holters['Unnamed: 0']):
        j = int(keywords['holter_id'][keywords['holter_id'] == holter].index.values)
        HSVH['text a'][i] = keywords['טקסט מתוך סיכום'][j]
        HSVH['text b'][i] = keywords['טקסט מתוך תוצאות'][j]

        jvt = int(vt_holters['Unnamed: 0'][vt_holters['Unnamed: 0'] == holter].index.values)
        HSVH['vt score'][i] = vt_holters['(VT'][jvt]
        directory = pathlib.PurePath("/MLAIM/AIMLab/Sheina/databases/rbdb/PZ_format")  # where to save the examples
        if not os.path.exists(directory):
            os.makedirs(directory)
        db.export_to_physiozoo(holter, directory=directory, export_rhythms=False, force=True,
                               n_leads=db.get_num_leads(holter))

    aaa = 5


def extract_by_keywords_shany():
    holter_labels = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/Holter_label.xlsx', engine='openpyxl')
    shany_ids = np.load('/MLAIM/AIMLab/Shany/databases/rbafdb/documentation/reports/control_group.npy')
    holter_labels['Unnamed: 0'] = holter_labels['Unnamed: 0'].astype(str)
    for i, holter in enumerate(shany_ids):
        j = int(holter_labels['Unnamed: 0'][holter_labels['Unnamed: 0'] == str(holter)].index.values)
        print(holter_labels['AFIB'][j])


def create_test_annotations():
    keywords = pd.read_excel(rbdb_doc_path + 'reports/RBAF_reports.xlsx', engine='openpyxl')
    holter_labels = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/Holter_label.xlsx', engine='openpyxl')
    holter_list = list(holter_labels['Unnamed: 0'])
    cosen_holters = random.choices(np.asarray(holter_list), k=1000)
    holter_labels = holter_labels[holter_labels['Unnamed: 0'].isin(cosen_holters)]
    test_annotations = pd.DataFrame(columns=['text a', 'text b'],
                                    index=holter_labels['Unnamed: 0'])
    for i, holter in enumerate(holter_labels['Unnamed: 0']):
        j = int(keywords['holter_id'][keywords['holter_id'] == holter].index.values)
        test_annotations['text a'][i] = keywords['טקסט מתוך סיכום'][j]
        test_annotations['text b'][i] = keywords['טקסט מתוך תוצאות'][j]
    test_annotations['AFIB'] = 0
    test_annotations['VT'] = 0
    test_annotations1 = test_annotations[0:500]
    test_annotations2 = test_annotations[500:1000]
    test_annotations1.to_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/keyword_ans/test_ann_sheinaaa.xlsx')
    test_annotations2.to_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/keyword_ans/test_ann_noam.xlsx')

    a = 5


def create_gold_standard():
    # validation:
    df_val_s = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/keyword_ans/test_ann_sheina.xlsx', engine='openpyxl')
    df_val_n = pd.read_excel('/MLAIM/AIMLab/Noam/fromSheina/test_ann_noam_ann.xlsx', engine='openpyxl').drop(
        columns=['Unnamed: 0', 'Unnamed: 6'])
    df_val_n.rename(columns={'Unnamed: 0.1': 'Unnamed: 0'}, inplace=True)

    df_val_diff = pd.concat([df_val_s, df_val_n]).drop_duplicates(keep=False)

    # test:
    df_test_s = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/keyword_ans/test_ann_sheina2.xlsx',
                              engine='openpyxl')
    df_test_n = pd.read_excel('/MLAIM/AIMLab/Noam/fromSheina/test_ann_noam_ann_2.xlsx', engine='openpyxl')

    df_test_diff = pd.concat([df_test_s, df_test_n]).drop_duplicates(keep=False)
    a = 5


def analyse_keywords():
    holter_labels = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/Holter_label.xlsx', engine='openpyxl')
    holter_labels_new = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/Holter_label_new_1_3.xlsx',
                                      engine='openpyxl')
    val = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/keyword_ans/test_ann_sheina.xlsx', engine='openpyxl')
    holter_labels_new['Unnamed: 0'] = holter_labels_new['Unnamed: 0'].astype(str)
    holter_labels['Unnamed: 0'] = holter_labels['Unnamed: 0'].astype(str)
    val['Unnamed: 0'] = val['Unnamed: 0'].astype(str)
    val_holters = holter_labels[holter_labels['Unnamed: 0'].isin(np.asarray(val['Unnamed: 0']))]
    val_holters_new = holter_labels_new[holter_labels_new['Unnamed: 0'].isin(np.asarray(val['Unnamed: 0']))]

    # annotation results:
    num_afib = np.sum(val['AFIB'])
    percent_afib = num_afib / 5

    num_vt = np.sum(val['VT'])
    percent_vt = num_vt / 5

    # AFIB:
    data_C_val = np.asarray(val['AFIB'])
    data_T_val = np.asarray(val_holters['AFIB'] / 100)
    data_T_new_val = np.asarray(val_holters_new['AFIB'] / 100)

    data_C_val_VT = np.asarray(val['VT'])
    data_T_val_VT = np.asarray(val_holters['VT'] / 100)
    data_T_new_val_VT = np.asarray(val_holters_new['VT'] / 100)

    fig, ax = plt.subplots()
    fpr, tpr, thresholds = roc_curve(data_C_val, data_T_val)
    fpr_new, tpr_new, thresholds_new = roc_curve(data_C_val, data_T_new_val)
    roc_auc = auc(fpr, tpr)
    roc_auc_new = auc(fpr_new, tpr_new)
    plt.plot(fpr, tpr, label='old model')
    plt.plot(fpr_new, tpr_new, label='new model')
    plt.xlabel('false positive ratio')
    plt.ylabel('true positive ratio')
    plt.title('AFIB Receiver operating characteristic')
    plt.legend()

    index_thr = np.argmax(tpr > 0.92)
    index_thr_new = np.argmax(tpr_new > 0.92)
    thr = thresholds[index_thr]
    thr_new = thresholds[index_thr_new]

    plt.plot(fpr_new[index_thr_new], tpr_new[index_thr_new], marker="*", markersize=8, markeredgecolor="red",
             markerfacecolor="red")
    plt.plot(fpr[index_thr], tpr[index_thr], marker="*", markersize=8, markeredgecolor="red",
             markerfacecolor="red")

    plt.show()

    y = data_C_val
    y_hat = (data_T_new_val > thr_new) * 1

    print(confusion_matrix(y, y_hat))

    fp_idx_val = np.where(y_hat - y_hat * y > 0)
    fn_idx_val = np.where(y - y_hat * y > 0)

    fig, ax = plt.subplots()
    fpr, tpr, thresholds = roc_curve(data_C_val_VT, data_T_val_VT)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.xlabel('false positive ratio')
    plt.ylabel('true positive ratio')
    plt.title('VT Receiver operating characteristic')
    index_thr = np.argmax(tpr > 0.92)
    thr = thresholds[index_thr]
    plt.plot(fpr[index_thr], tpr[index_thr], marker="*", markersize=8, markeredgecolor="red",
             markerfacecolor="red")
    fig.savefig('/MLAIM/AIMLab/Sheina/databases/rbdb/keyword_ans/VT_ROC.png')
    plt.show()

    # test:
    test = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/keyword_ans/test_ann_sheina2.xlsx', engine='openpyxl')
    test['Unnamed: 0'] = test['Unnamed: 0'].astype(str)
    test_holters = holter_labels[holter_labels['Unnamed: 0'].isin(np.asarray(test['Unnamed: 0']))]
    test_holters_new = holter_labels_new[holter_labels_new['Unnamed: 0'].isin(np.asarray(test['Unnamed: 0']))]
    data_C_test = np.asarray(test['AFIB'])
    data_T_new_test = np.asarray(test_holters_new['AFIB'] / 100)

    y = data_C_test
    y_hat = (data_T_new_test > thr_new) * 1

    print(confusion_matrix(y, y_hat))

    data_C_test_VT = np.asarray(test['VT'])
    data_T_test_VT = np.asarray(test_holters['VT'] / 100)
    data_T_new_test_VT = np.asarray(test_holters_new['VT'] / 100)

    y = data_C_test_VT
    y_hat = (data_T_test_VT >= 1) * 1

    print(confusion_matrix(y, y_hat))

    Holter_label_final = pd.DataFrame(0, columns=['Holter_id', 'AFIB', 'VT', 'manually'], index=holter_labels.index)
    Holter_label_final['Holter_id'] = holter_labels['Unnamed: 0']
    Holter_label_final['AFIB'] = (holter_labels_new['AFIB'] > thr_new * 100) * 1
    Holter_label_final['VT'] = (holter_labels['VT'] >= 100) * 1
    Holter_label_final[holter_labels['Unnamed: 0'].isin(np.asarray(test['Unnamed: 0']))]['AFIB'] = test_holters['AFIB']
    Holter_label_final[holter_labels['Unnamed: 0'].isin(np.asarray(test['Unnamed: 0']))]['VT'] = test_holters['VT']
    Holter_label_final['manually'] = (holter_labels['Unnamed: 0'].isin(np.asarray(test['Unnamed: 0']))) * 1
    Holter_label_final[holter_labels['Unnamed: 0'].isin(np.asarray(val['Unnamed: 0']))]['AFIB'] = val_holters['AFIB']
    Holter_label_final[holter_labels['Unnamed: 0'].isin(np.asarray(val['Unnamed: 0']))]['VT'] = val_holters['VT']
    Holter_label_final['manually'] = Holter_label_final['manually'] + (
        holter_labels['Unnamed: 0'].isin(np.asarray(val['Unnamed: 0']))) * 1
    Holter_label_final.to_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/Holter_labels_final.xlsx')

    a = 5


def rbdb_statistics():
    Holter_label_final = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/Holter_labels_final.xlsx',
                                       engine='openpyxl')
    holter_ids = Holter_label_final['Holter_id']
    db = RBAFDB_Parser(load_on_start=True)
    all_ids = list(db.parse_available_ids().astype(str))
    zz = []
    not_at_all = []

    for i in holter_ids:
        try:
            all_ids.remove(i)
        except:
            zz.append(i)

    miss_ids = all_ids

    main_path = '/MLAIM/AIMLab/Shany/databases/rbafdb/documentation/RBAF_Holter_Info.xlsx'
    rbdb_info = pd.read_excel(main_path, engine='openpyxl')
    rbdb_info['holter_id'] = rbdb_info['holter_id'].astype(str)
    miss_info = pd.DataFrame()
    for indx_miss, j in enumerate(miss_ids):
        try:
            ind = int(rbdb_info['holter_id'][rbdb_info['holter_id'] == str(j)].index.values)
            miss_info = miss_info.append(rbdb_info.iloc[ind], ignore_index=True)
        except:
            not_at_all.append(j)

    age_all = rbdb_info['age_at_recording'].dropna()
    miss_info.to_excel('/MLAIM/AIMLab/Shany/databases/rbafdb/documentation/Holters_miss_annotations.xlsx')
    a = 5


if __name__ == '__main__':
    analyse_keywords()
    # db = RBAFDB_Parser(load_on_start=True)
    # HSVH = pd.read_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/HSVH.xlsx', engine='openpyxl')
    # for i, holter in enumerate(HSVH['Unnamed: 0']):
    #     directory = pathlib.PurePath("/MLAIM/AIMLab/Sheina/databases/rbdb/PZ_format")   # where to save the examples
    #     if not os.path.exists(directory/ holter):
    #         os.makedirs(directory/ holter)
    #     db.export_to_physiozoo(holter, directory=directory/ holter, export_rhythms=False, force=True, n_leads=db.get_num_leads(holter))

    a = 5
