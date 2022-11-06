from base_parser import *

random.seed(cts.SEED)


# TODO: Verify SQI problem on the LTAF.

class LTAFDB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True):

        super(LTAFDB_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overriden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- #
        """

        """Missing records"""
        self.missing_ecg = np.array([])

        """Helper variables"""
        self.window_size = window_size

        """Variables relative to the ECG signals."""
        self.orig_fs = 128
        self.actual_fs = 200
        self.n_leads = 1
        self.name = "LTAFDB"
        self.ecg_format = "wfdb"
        self.rhythms = np.array(['(N', '(AFIB', '(AB', '(AFL', '(B', '(BII', '(IVR', '(NOD',
                                 '(P', '(PREX', '(SBR', '(SVTA', '(T', '(VFL', '(VT', '(J', 'MISSB',
                                 'PSE', 'MB', 'M'])
        self.rhythms_dict = {self.rhythms[i]: i for i in range(len(self.rhythms))}

        """Variables relative to the different paths"""
        self.raw_ecg_path = cts.DATA_DIR / "af_long_term"
        self.orig_anns_path = cts.DATA_DIR / "af_long_term"
        self.generated_anns_path = cts.DATA_DIR / "Annotations" / self.name
        self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)
        self.filename = "LTAF.pkl"
        self.main_path = cts.PREPROCESSED_DATA_DIR / "LTAF"
        self.window_size = window_size

        if os.path.exists(self.main_path):
            parsed_patients = self.parsed_patients()
            test_pat = parsed_patients[0]
            self.window_sizes = np.array([int(x[:-4]) for x in os.listdir(self.main_path / test_pat / "features")])

        """ Checking the parsed window sizes and setting the window size. The data corresponding to the window size
        requested will be loaded into the system."""

        if load_on_start:
            self.set_window_size(self.window_size)

        self.anns_path = cts.DATA_DIR / "af_long_term"
        self.raw_ecg_path = cts.DATA_DIR / "af_long_term"
        self.all_ecgs = self.parse_available_ids()

    def parse_available_ids(self):
        with open(self.raw_ecg_path / 'RECORDS', 'r') as f:
            records = np.array([x[:-1] for x in f.readlines()])
        return records

    def parse_annotation(self, id, type='epltd0'):
        return wfdb.rdann(str(self.generated_anns_path / type / id), type).sample

    def parse_reference_annotation(self, id):
        ann = wfdb.rdann(str(self.raw_ecg_path / id), 'atr')
        if id == 64:  # ID 64 notes did not present the label AFIB at the beginning.
            ann.aux_note[0] = '(AFIB'
        rhythm = i_o.pad_rhythm(np.array(ann.aux_note), missing=['', '\x01 Aux'])
        rhythm = np.array([self.rhythms_dict[i] for i in rhythm])
        return ann.sample, rhythm

    def record_to_wfdb(self, id):
        record = wfdb.rdrecord(str(self.raw_ecg_path / id))
        ecg = record.p_signal[:, 0]
        ecg_resampled = signal.resample(ecg, int(len(ecg) * self.actual_fs / self.orig_fs))
        wfdb.wrsamp(str(id), fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=ecg_resampled.reshape(-1, 1), fmt=['16'])

    def parse_raw_ecg(self, patient_id, start=0, end=-1, type='epltd0', lead=0):
        record = wfdb.rdrecord(str(self.raw_ecg_path / patient_id))
        ecg = record.p_signal[:, 0]
        ecg = signal.resample(ecg, int(len(ecg) * self.actual_fs / self.orig_fs))
        ann = self.parse_annotation(patient_id, type=type)
        if end == -1:
            end = int(len(ecg) / self.actual_fs)
        start_sample = start * self.actual_fs
        end_sample = end * self.actual_fs
        ecg = ecg[start_sample:end_sample]
        ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
        ann -= start_sample
        return ecg, ann

    def parse_ahi(self, id):
        self.ahi_dict[id] = None  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = None  # This data is not available for this dataset.

    def parse_demographic_features(self, id):
        pass


if __name__ == '__main__':
    db = LTAFDB_Parser(load_on_start=True)
    db.parse_raw_data(patient_list=['51'])
    # for pat in db.signal_quality_dict.keys():
    #     for win in cts.BASE_WINDOWS:
    #         curr_sig_qual = copy.deepcopy(db.signal_quality_dict[pat][win])
    #         del db.signal_quality_dict[pat][win]
    #         db.signal_quality_dict[pat][win] = {}
    #         db.signal_quality_dict[pat][win]['xqrs'] = curr_sig_qual
    #
    # print("Adding PIP")
    # db.add_feature('PIP')
    # print("Adding PSS")
    # db.add_feature('PSS')
    # print("Adding PAS")
    # db.add_feature('PAS')
    # print("Adding IALS")
    # db.add_feature('IALS')
    #
    # for pat in db.features_dict.keys():
    #     for win in cts.BASE_WINDOWS:
    #         np.save(db.main_path / pat / 'signal_quality' / str(win), db.signal_quality_dict[pat][win])
    #         np.save(db.main_path / pat / 'features' / str(win), db.features_dict[pat][win])
    # #db.parse_raw_data(total_run=True)
