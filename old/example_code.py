import numpy as np
import pandas as pd
import sheina.consts as cts
from pebm import Preprocessing as Pre
from pebm.ebm import FiducialPoints as Fp
from rbaf_parser import *

##epltd and signal quality


fs = 200
win_quality = 30
db = RBAFDB_Parser(load_on_start=True)
# calculate_hrv(db, 'rbdb')
ids = cts.ids_rbdb_no_VT[-2:]
for id in ids:
    bsqi_xl = pd.DataFrame(0, index=range(0, 80), columns=['lead 1', 'lead 2', 'lead 3']).astype('float')
    for lead in range(1, 4):
        try:
            raw_ecg_i = db.parse_raw_ecg(id, start=0, end=-1, type='epltd0', lead=lead)
        except:
            continue
        raw_ecg_lead = raw_ecg_i[0][5 * 60 * fs:]
        fp = Fp.FiducialPoints(raw_ecg_lead, fs)
        epltd_lead = fp.epltd()
        jqrs_lead = fp.jqrs()
        np.save('/MLAIM/AIMLab/Sheina/databases/rbdb/epltd/' + id + '_' + str(lead) + '.npy', epltd_lead)
        win_len = 30
        i = 0
        start_win = 0
        end_win = start_win + win_len * 60 * fs
        bsqi_dict = {}
        # windoing:
        while end_win < len(raw_ecg_i[0]):
            epltd_lead_win = epltd_lead[(epltd_lead >= start_win) & (epltd_lead < end_win)] - start_win
            jqrs_lead_win = jqrs_lead[(jqrs_lead >= start_win) & (jqrs_lead < end_win)] - start_win
            signal_win_lead = raw_ecg_lead[start_win:end_win]
            pre = Pre.Preprocessing(signal_win_lead, fs)

            bsqi = pre.bsqi(epltd_lead_win, jqrs_lead_win)
            bsqi_xl.loc[i]['lead ' + str(lead)] = bsqi
            i = i + 1
            start_win = end_win
            end_win = start_win + win_len * 60 * fs

    bsqi_xl.to_excel('/MLAIM/AIMLab/Sheina/databases/rbdb/epltd/bsqi' + str(id) + '.xlsx')
