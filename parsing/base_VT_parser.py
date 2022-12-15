import os
import pathlib

import numpy as np
import pandas as pd
import wfdb

import utils.consts as cts
import utils.dat_reader as dr
import utils.data_processing as dp


class VtParser:

    def __init__(self):

        self.fs = 200
        self.window_size = self.fs * 10
        self.bsqi_window_size = self.fs * 30 * 60
        self.ecg_path = []
        self.ex_d = {}
        self.raw_ecg_path = []
        self.base_lead = []
        self.st = []

        self.train_n = list(np.load(cts.IDS_DIR / 'RBDB_train_no_VT_ids.npy'))
        self.train_p = list(np.load(cts.IDS_DIR / 'RBDB_train_VT_ids.npy'))
        self.test_n = list(np.load(cts.IDS_DIR / 'RBDB_test_no_VT_ids.npy'))
        self.test_p = list(np.load(cts.IDS_DIR / 'RBDB_test_VT_ids.npy'))
        self.val_n = list(np.load(cts.IDS_DIR / 'RBDB_val_no_VT_ids.npy'))
        self.ext_test_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_VT_ids.npy'))
        self.ext_test_no_vt = list(np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/UVAF_non_VT_ids.npy'))
        self.rbdb_ids = self.train_n + self.train_p + self.test_n + self.test_p + self.val_n + cts.bad_bsqi
        self.uvafdb_ids = self.ext_test_vt + self.ext_test_no_vt

    def dataset_split(self, rec):
        if rec in self.rbdb_ids:
            self.dataset = 'rbdb'
            self.fts = 5
            self.orig_fs = 128
            self.ecg_files_name = ['rawecg1', 'rawecg2', 'rawecg3']
        elif rec in self.uvafdb_ids:
            self.dataset = 'uvafdb'
            self.fts = 0
        self.ecg_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data') / self.dataset

        if self.dataset == 'rbdb':
            self.raw_ecg_path = pathlib.PurePath('/MLAIM/AIMLab/Shany/databases/rbafdb/dataset')
            self.generated_anns_path = cts.BASE_DIR / "Shany" / "Annotations" / "RBAFDB"
            self.base_lead = 1
            self.st = ''
            self.fts = 5 * 60 * 200
            if os.path.exists(cts.IDS_DIR / 'ids_ints.xlsx'):
                self.ids_int = pd.read_excel(cts.IDS_DIR / 'ids_ints.xlsx', engine='openpyxl')
            else:
                self.ids_int = self.create_rdbd_int()
            if os.path.exists(cts.PRE_PROCESSED_DIR / 'rbdb' / 'excluded_win.npy'):
                self.ex_d_rbdb = np.load(cts.PRE_PROCESSED_DIR / 'rbdb' / 'excluded_win.npy',
                                         allow_pickle=True).item()
            else:
                self.ex_d_rbdb = self.calculate_excluded_win_dict('rbdb')
            self.ex_d = self.ex_d_rbdb

        if self.dataset == 'uvafdb':
            self.raw_ecg_path = cts.DATA_DIR / "uvfdb"
            self.generated_anns_path = cts.DATA_DIR / "Annotations" / "UVFDB"

            self.base_lead = 0
            self.st = ''
            self.fts = 0
            if os.path.exists(cts.PRE_PROCESSED_DIR / 'uvafdb' / 'excluded_win.npy'):
                self.ex_d_uvafdb = np.load(cts.PRE_PROCESSED_DIR / 'uvafdb' / 'excluded_win.npy',
                                           allow_pickle=True).item()
            else:
                self.ex_d_rbdb = self.calculate_excluded_win_dict('uvafdb')
            self.ex_d = self.ex_d_uvafdb
        self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)

    def parse_pp_rec(self, rec):
        self.dataset_split(rec)
        norm_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/rbdb/')
        ecg = np.load(norm_path / rec / 'ecg_0.npy')
        return ecg

    def parse_raw_rec(self, id, lead=0, start=0, end=-1):
        self.dataset_split(id)
        if self.dataset == 'rbdb':
            record = self.parse_raw_rec_rbdb(id, lead=lead)
        if self.dataset == 'uvafdb':
            record = self.parse_raw_rec_uvafdb(id, lead=lead)
        return record

    def parse_raw_rec_uvafdb(self, rec, lead=0, start=0, end=-1):
        self.dataset_split(rec)
        lead = self.base_lead
        record = self._read_rf(self.raw_ecg_path / (self.st + rec + '.rf'), lead=lead)
        ecg = record
        if end == -1:
            end = int(len(ecg) / self.fs)
        start_sample = start * cts.EPLTD_FS
        end_sample = end * cts.EPLTD_FS
        ecg = ecg[start_sample:end_sample]
        return ecg

    def parse_raw_rec_rbdb(self, patient_id, start=0, end=-1, lead=0):
        record = dr.read_ecg_file(self.raw_ecg_path / (patient_id + '.FUL') / str(self.ecg_files_name[lead] + '.dat'))
        record = dp.resample_by_interpolation(record, self.orig_fs, self.fs)
        if end == -1:
            end = int(len(record) / self.fs)
        start_sample = int(start * self.fs)
        end_sample = int(end * self.fs)
        record = record[start_sample:end_sample]
        return record

    def _read_rf(self, file, lead=0):
        """ This function reads the raw ECG files, which are given in an encoded (.rf) format.
        :param file: The path to the .rf file."""
        n_chans = 3
        n_bits_per_chan = 10
        f = open(self.raw_ecg_path / file, "rb")
        A = np.fromfile(f, dtype=np.uint32)
        masks_abs_val = [0x1ff, 0x7fc00, 0x1ff00000]
        masks_sign = [0x200, 0x80000, 0x20000000]
        ecgs = np.zeros((len(A), n_chans))
        ecgs_sign = np.zeros((len(A), n_chans))
        for i in range(n_chans):
            ecgs[:, i] = np.bitwise_and(A, masks_abs_val[i]) >> n_bits_per_chan * i
            ecgs_sign[:, i] = np.bitwise_and(A, masks_sign[i]) >> (n_bits_per_chan * (i + 1) - 1)
            ecgs[ecgs_sign[:, i] == 1, i] -= 2 ** (n_bits_per_chan - 1)
        Vptp = 5.0
        ecgs *= (Vptp / 2 ** n_bits_per_chan)  # Conversion from A/D value to [mV]
        return ecgs[:, lead]

    def return_masks_dict(self, rec_list):
        """ The mask can be false for specific window in two cases:
                1. The bsqi is lower than 0.8.
                2. The window include a VT segments.
        """
        self.dataset_split(rec_list[0])
        mask_dict = self.ex_d
        len_dict = {}
        for p in mask_dict:
            len_dict[p] = len(mask_dict[p])
        return mask_dict, len_dict

    def get_labels(self, rec_list):
        labels = {}
        for rec_list in rec_list:
            if rec_list in self.test_p + self.train_p + self.ext_test_vt:
                y = 1
            else:
                y = 0
            labels[rec_list] = y
        return labels

    def rbdb_ids_to_int(self, ids_list):
        int_list = []
        ids_int = self.ids_int
        for id_ in ids_list:
            int_ = ids_int['int'][ids_int['ids'] == int_]
            int_list.append(id_)
        return int_list

    def rbdb_int_to_ids(self, int_list):
        ids_list = []
        ids_int = self.ids_int
        for int_ in int_list:
            id_ = ids_int['ids'][ids_int['int'] == int_]
            ids_list.append(int_)
        return ids_list

    def create_rdbd_int(self):
        train_n = list(np.load(cts.IDS_DIR / 'RBDB_train_no_VT_ids.npy'))
        train_p = list(np.load(cts.IDS_DIR / 'RBDB_train_VT_ids.npy'))
        test_n = list(np.load(cts.IDS_DIR / 'RBDB_test_no_VT_ids.npy'))
        test_p = list(np.load(cts.IDS_DIR / 'RBDB_test_VT_ids.npy'))
        val_n = list(np.load(cts.IDS_DIR / 'RBDB_val_no_VT_ids.npy'))
        list_ids = train_n + train_p + test_n + test_p + val_n
        ids_int = pd.DataFrame()
        list_ints = range(1, len(list_ids) + 1)
        list_int_str = [str(int).zfill(4) for int in list_ints]
        ids_int = ids_int.assign(ids=list_ids)
        ids_int = ids_int.assign(int=list_int_str)
        ids_int.to_excel(cts.IDS_DIR / 'ids_ints.xlsx')

    def parse_annotation(self, id, type):
        self.dataset_split(id)
        if self.dataset == 'rbdb':
            ann = self.parse_annotation_rbdb(id, type)
        if self.dataset == 'uvafdb':
            ann = self.parse_annotation_uvafdb(id, type)
        return ann

    def parse_annotation_uvafdb(self, id, type="epltd0", lead=1):
        self.dataset_split(id)
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        return wfdb.rdann(str(self.generated_anns_path / type / id[3:7]), type).sample

    def parse_annotation_rbdb(self, id, type="epltd0", lead=1):
        self.dataset_split(id)
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        return wfdb.rdann(str(self.generated_anns_path / type / str(lead) / id), type).sample

    def parse_ids(self, task='train'):
        if task == 'train':
            train_n = list(np.load(cts.IDS_DIR / 'RBDB_train_no_VT_ids.npy'))
            train_p = list(np.load(cts.IDS_DIR / 'RBDB_train_VT_ids.npy'))[10:]
            train_df = pd.DataFrame(columns=['holter_id', 'th_id', 'y'])
            y = np.asarray([0] * len(train_n) + [1] * len(train_p))
            train_df['holter_id'] = np.asarray(train_n + train_p)
            train_df['th_id'] = np.asarray(range(len(train_n + train_p)))
            train_df['y'] = y
            return train_df
        if task == 'test':
            test_n = list(np.load(cts.IDS_DIR / 'RBDB_test_no_VT_ids.npy'))
            test_p = list(np.load(cts.IDS_DIR / 'RBDB_test_VT_ids.npy'))
            test_df = pd.DataFrame(columns=['holter_id', 'th_id', 'y'])
            y = np.asarray([0] * len(test_n) + [1] * len(test_p))
            test_df['holter_id'] = np.asarray(test_n + test_p)
            test_df['th_id'] = np.asarray(range(len(test_n + test_p)))
            test_df['y'] = y
            return test_df
        if task == 'all':
            train_n = list(np.load(cts.IDS_DIR / 'RBDB_train_no_VT_ids.npy'))
            train_p = list(np.load(cts.IDS_DIR / 'RBDB_train_VT_ids.npy'))
            test_n = list(np.load(cts.IDS_DIR / 'RBDB_test_no_VT_ids.npy'))
            test_p = list(np.load(cts.IDS_DIR / 'RBDB_test_VT_ids.npy'))
            val_n = list(np.load(cts.IDS_DIR / 'RBDB_val_no_VT_ids.npy'))
            all_df = pd.DataFrame(columns=['holter_id', 'th_id', 'y'])
            y = np.asarray([0] * len(test_n + train_n) + [1] * len(test_p + train_p))
            all_df['holter_id'] = np.asarray(test_n + train_n + test_p + train_p)
            all_df['th_id'] = np.asarray(range(len(test_n + train_n + test_p + train_p)))
            all_df['y'] = y
            return all_df
        if task == 'train_part':
            train_n = list(np.load(cts.IDS_DIR / 'RBDB_train_no_VT_ids.npy'))[:150]
            train_p = list(np.load(cts.IDS_DIR / 'RBDB_train_VT_ids.npy'))[10:]
            train_df = pd.DataFrame(columns=['holter_id', 'th_id', 'y'])
            y = np.asarray([0] * len(train_n) + [1] * len(train_p))
            train_df['holter_id'] = np.asarray(train_n + train_p)
            train_df['th_id'] = np.asarray(range(len(train_n + train_p)))
            train_df['y'] = y
            return train_df
        if task == 'val':
            train_n = list(np.load(cts.IDS_DIR / 'RBDB_val_no_VT_ids.npy'))
            train_p = list(np.load(cts.IDS_DIR / 'RBDB_train_VT_ids.npy'))[:10]
            train_df = pd.DataFrame(columns=['holter_id', 'th_id', 'y'])
            y = np.asarray([0] * len(train_n) + [1] * len(train_p))
            train_df['holter_id'] = np.asarray(train_n + train_p)
            train_df['th_id'] = np.asarray(range(len(train_n + train_p)))
            train_df['y'] = y
            return train_df
        if task == 'test_part':
            test_n = list(np.load(cts.IDS_DIR / 'RBDB_test_no_VT_ids.npy'))
            test_p = list(np.load(cts.IDS_DIR / 'RBDB_test_VT_ids.npy'))
            test_df = pd.DataFrame(columns=['holter_id', 'th_id', 'y'])
            y = np.asarray([0] * len(test_n) + [1] * len(test_p))
            test_df['holter_id'] = np.asarray(test_n + test_p)
            test_df['th_id'] = np.asarray(range(len(test_n + test_p)))
            test_df['y'] = y
            return test_df
        if task == 'train_part_pat':
            train_n = list(np.load(cts.IDS_DIR / 'RBDB_train_no_VT_ids.npy'))[:500]
            train_p = list(np.load(cts.IDS_DIR / 'RBDB_train_VT_ids.npy'))[10:]
            train_df = pd.DataFrame(columns=['holter_id', 'th_id', 'y'])
            y = np.asarray([0] * len(train_n) + [1] * len(train_p))
            train_df['holter_id'] = np.asarray(train_n + train_p)
            train_df['th_id'] = np.asarray(range(len(train_n + train_p)))
            train_df['y'] = y
            return train_df
        if task == 'val_part':
            train_n = list(np.load(cts.IDS_DIR / 'RBDB_val_no_VT_ids.npy'))[:80]
            train_p = list(np.load(cts.IDS_DIR / 'RBDB_train_VT_ids.npy'))[:10]
            train_df = pd.DataFrame(columns=['holter_id', 'th_id', 'y'])
            y = np.asarray([0] * len(train_n) + [1] * len(train_p))
            train_df['holter_id'] = np.asarray(train_n + train_p)
            train_df['th_id'] = np.asarray(range(len(train_n + train_p)))
            train_df['y'] = y
            return train_df

    def calculate_excluded_win_dict(self, dataset):
        all_df = self.parse_ids('all')

        ex_d = {}
        segments = pd.read_excel(cts.VTdb_path / 'VTp' / str('segments_array_' + dataset + '.xlsx'), engine='openpyxl')
        event_holters = list(set(list(segments['holter_id'])))
        for h_id in list(all_df['holter_id']):
            bsqi = np.load(cts.VTdb_path / 'preprocessed_data' / dataset / h_id / 'bsqi_0.npy')
            mask_windows = bsqi > 0.8
            if h_id in event_holters:
                segments_id = segments[segments['holter_id'] == h_id]
                start_windows = list((np.asarray(segments_id['start']) + (self.fts / self.fs)) // (30 * 60))
                stop_windows = list((np.asarray(segments_id['start']) + (self.fts / self.fs)) // (30 * 60))
                event_windows = list(set(start_windows + stop_windows))
                for win in event_windows:
                    mask_windows[int(win)] = False
            ex_d[h_id] = mask_windows
        np.save(cts.PRE_PROCESSED_DIR / dataset / 'excluded_win.npy', ex_d)
        return


if __name__ == '__main__':
    db = VtParser()
    db.dataset_split('1419Fc22')
    db.calculate_excluded_win_dict('rbdb')
