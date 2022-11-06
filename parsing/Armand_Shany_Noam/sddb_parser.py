from base_parser import *

random.seed(cts.SEED)


class SDDB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True):

        super(SDDB_Parser, self).__init__()

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
        self.orig_fs = 250
        self.actual_fs = 250
        self.n_leads = 3
        self.name = "SDDB"
        self.ecg_format = "wfdb"
        self.rhythms = np.array(['(N', '(AFIB', '(AB', '(AFL', '(B', '(BII', '(IVR', '(NOD',
                                 '(P', '(PREX', '(SBR', '(SVTA', '(T', '(VFL', '(VT', '(J'])
        self.rhythms_dict = {self.rhythms[i]: i for i in range(len(self.rhythms))}

        """ Variables relative to the different paths """
        self.raw_ecg_path = cts.DATA_DIR / "sddb"
        self.orig_anns_path = cts.DATA_DIR / "sddb"
        self.generated_anns_path = cts.DATA_DIR / "Annotations" / self.name
        # self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)
        self.filename = None
        # self.main_path = cts.PREPROCESSED_DATA_DIR / "NSRDB"
        # self.window_size = window_size

        """ Checking the parsed window sizes and setting the window size. The data corresponding to the window size
        requested will be loaded into the system."""
        # if load_on_start:
        #     if os.path.exists(self.main_path):
        #         parsed_patients = self.parsed_patients()
        #         test_pat = parsed_patients[0]
        #         self.window_sizes = np.array([int(x[:-4]) for x in os.listdir(self.main_path / test_pat / "features")])
        #     self.set_window_size(self.window_size)
        #
        # self.anns_dir = cts.DATA_DIR / "nsrdb"
        # self.raw_ecg_path = cts.DATA_DIR / "nsrdb"

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
        if os.path.exists(str(self.raw_ecg_path / id) + '.atr'):
            ann_rhythm = wfdb.rdann(str(self.raw_ecg_path / id), 'atr')

        else:
            ann_rhythm = wfdb.rdann(str(self.raw_ecg_path / id), 'ari')
        rhythm_samp = ann_rhythm.sample
        rhythm = ann_rhythm.chan
        return rhythm_samp, rhythm

    def parse_ahi(self, id):
        self.ahi_dict[id] = None  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = None  # This data is not available for this dataset.

    def parse_demographic_features(self, id):
        pass

    def export_to_physiozoo(self, pat, directory=cts.ERROR_ANALYSIS_DIR, start=0, end=-1, force=False,
                            ann_type='epltd0', export_rhythms=False, n_leads=1):
        """ This function exports a given recording to the physiozoo format for further investigation.
        The files are exported under a .txt format with the proper headers to be read by the PhysioZoo software.
        The raw ECG as well as the peaks are exported. If provided, the AF events are exported as well.
        :param pat: The patient ID to export.
        :param directory: The directory under which the files will be saved.
        :param start: The beginning timestamp for the ECG portion to be saved.
        :param end: The ending timestamp for the ECG portion to be saved.
        :param force: If True, generate the files anyway, otherwise verifies if the file has already been exported under the directory
        :param export_rhythms: If True, exports a file with the different rhythms over the recording."""

        ecg_header = ['---\n',
                      'Mammal:            human\n',
                      'Fs:                ' + str(cts.EPLTD_FS) + '\n',
                      'Integration_level: electrocardiogram\n',
                      '\n'
                      'Channels:\n']
        channels = [['\n'
                     '    - type:   electrography\n',
                     '      name:   data' + str(i) + '\n',
                     '      unit:   mV\n',
                     '      enable: yes\n'] for i in range(1, n_leads + 1)]
        end_header = ['\n',
                      '---\n',
                      '\n']
        [ecg_header.extend(channels[i]) for i in range(n_leads)]
        ecg_header.extend(end_header)

        peaks_header = ['---\n',
                        'Mammal:            human\n',
                        'Fs:                ' + str(self.actual_fs) + '\n',
                        'Integration_level: electrocardiogram\n',
                        '\n'
                        'Channels:\n',
                        '\n'
                        '    - type:   peak\n',
                        '      name:   interval\n',
                        '      unit:   index\n',
                        '      enable: yes\n',
                        '\n',
                        '---\n',
                        '\n'
                        ]

        sig_qual_header = ['---\n',
                           'type: quality annotation\n',
                           'source file: ',  # To be completed by filename
                           '\n',
                           '---\n',
                           '\n',
                           'Beginning\tEnd\t\tClass\n']

        rhythms_header = copy.deepcopy(sig_qual_header)
        rhythms_header[1] = 'type: rhythms annotation\n'

        if not os.path.exists(directory):
            os.makedirs(directory)

        ecgs = np.concatenate(tuple(
            [self.parse_raw_ecg(pat, start=start, end=end, type=ann_type, lead=lead)[0].reshape(-1, 1) for lead in
             range(1, n_leads + 1)]), axis=1)
        ann = self.parse_raw_ecg(pat, start=start, end=end, type=ann_type)[1]
        if end == -1:
            end = len(ecgs[0] / self.actual_fs)

        ecg_full_path = directory / (
                    pat + '_ecg_start_' + str(start) + '_end_' + str(end) + '_n_leads_' + str(n_leads) + '.txt')
        peaks_full_path = directory / (
                    pat + '_peaks_start_' + str(start) + '_end_' + str(end) + '_' + str(ann_type) + '.txt')
        rhythms_full_path = directory / (pat + '_rhythms_start_' + str(start) + '_end_' + str(end) + '.txt')
        # Raw ECG
        join_func = lambda x: ' '.join(['%.2f' % i for i in x])
        ecg_str = np.apply_along_axis(join_func, 1, ecgs)
        if not os.path.exists(ecg_full_path) or force:
            with open(ecg_full_path, 'w+') as ecg_file:
                ecg_file.writelines(ecg_header)
                ecg_file.write('\n'.join(ecg_str))

        # Peaks
        if not os.path.exists(peaks_full_path) or force:
            with open(peaks_full_path, 'w+') as peaks_file:
                peaks_file.writelines(peaks_header)
                peaks_file.write('\n'.join(['%d' % i for i in ann]))

        # Rhythms
        if export_rhythms:
            if not os.path.exists(rhythms_full_path) or force:
                with open(rhythms_full_path, 'w+') as rhythms_file:
                    rhythms_header[2] = 'source file: ' + pat + '_rhythms.txt\n'
                    rhythms_file.writelines(rhythms_header)
                    rhythms_samp, rhythms = np.array(self.parse_reference_annotation(pat))
                    periods = np.concatenate(([0], np.where(np.diff(rhythms.astype(int)))[0] + 1, [len(rhythms) - 1]))
                    start_idx, end_idx = periods[:-1], periods[1:]
                    final_rhythms = rhythms[start_idx]
                    mask_rhythms = final_rhythms > 0
                    rhythms_samp_t = rhythms_samp / self.actual_fs  # We do not keep NSR as rhythm
                    start_events, end_events = rhythms_samp_t[start_idx], rhythms_samp_t[end_idx]
                    mask_int_events = np.logical_and(start_events < end, end_events > start)
                    mask_rhythms = np.logical_and(mask_rhythms, mask_int_events)
                    start_events, end_events = start_events[mask_rhythms] - start, end_events[mask_rhythms] - start
                    end_events[end_events > (end - start)] = end - start
                    final_rhythms = final_rhythms[mask_rhythms]
                    final_rhythms_str = np.array([self.rhythms[i][1:] for i in final_rhythms])
                    rhythms_file.write('\n'.join(
                        ['%.5f\t%.5f\t%s' % (start_events[i], end_events[i], final_rhythms_str[i]) for i in
                         range(len(start_events))]))
