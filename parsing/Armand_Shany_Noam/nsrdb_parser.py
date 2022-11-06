from base_parser import *

random.seed(cts.SEED)


class NSRDB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True):

        super(NSRDB_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overriden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- #
        """

        """ Missing records """
        self.missing_ecg = np.array([])

        """Helper variables"""
        self.window_size = window_size

        """ Variables relative to the ECG signals. """
        self.orig_fs = 128
        self.actual_fs = 200
        self.n_leads = 1
        self.name = "NSRDB"
        self.ecg_format = "wfdb"
        self.rhythms = np.array(['(N', '(AFIB', '(AB', '(AFL', '(B', '(BII', '(IVR', '(NOD',
                                 '(P', '(PREX', '(SBR', '(SVTA', '(T', '(VFL', '(VT', '(J'])
        self.rhythms_dict = {self.rhythms[i]: i for i in range(len(self.rhythms))}

        """ Variables relative to the different paths """
        self.raw_ecg_path = cts.DATA_DIR / "nsrdb"
        self.orig_anns_path = cts.DATA_DIR / "nsrdb"
        self.generated_anns_path = cts.DATA_DIR / "Annotations" / self.name
        self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)
        self.filename = "NSRDB.pkl"
        self.main_path = cts.PREPROCESSED_DATA_DIR / "NSRDB"
        self.window_size = window_size

        """ Checking the parsed window sizes and setting the window size. The data corresponding to the window size
        requested will be loaded into the system."""
        if load_on_start:
            if os.path.exists(self.main_path):
                parsed_patients = self.parsed_patients()
                test_pat = parsed_patients[0]
                self.window_sizes = np.array([int(x[:-4]) for x in os.listdir(self.main_path / test_pat / "features")])
            self.set_window_size(self.window_size)

        self.anns_dir = cts.DATA_DIR / "nsrdb"
        self.raw_ecg_path = cts.DATA_DIR / "nsrdb"

    """
    # ------------------------------------------------------------------------- #
    # ----- Parsing functions: have to be overridden by the child classes ----- #
    # ------------------------------------------------------------------------- #
    """

    def parse_available_ids(self):
        with open(self.raw_ecg_path / 'RECORDS', 'r') as f:
            records = np.array([x[:-1] for x in f.readlines()])
        return records

    def parse_annotation(self, id, type='epltd0'):
        return wfdb.rdann(str(self.generated_anns_path / type / id), type).sample

    def record_to_wfdb(self, id):
        record = wfdb.rdrecord(str(self.raw_ecg_path / id))
        ecg = record.p_signal[:, 0]
        ecg_resampled = signal.resample(ecg, int(len(ecg) * self.actual_fs / self.orig_fs))
        wfdb.wrsamp(str(id), fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=ecg_resampled.reshape(-1, 1), fmt=['16'])
        return ecg

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

    def parse_reference_annotation(self, id):
        ann = wfdb.rdann(str(self.raw_ecg_path / id), 'atr')
        rhythm = np.array(['(N' for _ in range(len(ann.sample))])
        rhythm = np.array([self.rhythms_dict[i] for i in rhythm])
        return ann.sample, rhythm

    def parse_ahi(self, id):
        self.ahi_dict[id] = None  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = None  # This data is not available for this dataset.

    def parse_demographic_features(self, id):
        pass


if __name__ == '__main__':
    db = NSRDB_Parser(load_on_start=False)
    db.generate_rqrs_annotations(pat_list=['18177', ])
    db.parse_raw_data(patient_list=db.parsed_patients())
    # db.generate_annotations(types=('xqrs', ), force=True)
