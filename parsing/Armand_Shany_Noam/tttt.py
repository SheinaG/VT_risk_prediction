from base_parser import *

warnings.filterwarnings('ignore')


class UVAFDB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True, load_ectopics=True):

        super(UVAFDB_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overriden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- #
        """
        """Missing records"""
        self.missing_ecg = np.array(['0280', '0308'])

        """ Helper variables"""
        self.window_size = window_size

        """Variables relative to the ECG signals."""
        self.orig_fs = cts.EPLTD_FS
        self.actual_fs = cts.EPLTD_FS
        self.n_leads = 3
        self.name = "UVAFDB"
        self.ecg_format = "rf"
        self.rhythms = np.array(['(N', '(AFIB', '(AB', '(AFL', '(B', '(BII', '(IVR', '(NOD',
                                 '(P', '(PREX', '(SBR', '(SVTA', '(T', '(VFL', '(VT', '(J',
                                 '(PAT', '(AT', '(VTS', '(AIVRS', '(IVRS', '(AIVR'])

        self.rhythms_dict = {self.rhythms[i]: i for i in range(len(self.rhythms))}

        """Variables relative to the different paths"""
        self.raw_ecg_path = cts.DATA_DIR / "uvfdb"
        self.orig_anns_path = None
        self.generated_anns_path = cts.DATA_DIR / "Annotations" / self.name
        self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)
        self.main_path = cts.PREPROCESSED_DATA_DIR / "UVAFDB"

        """ Checking the parsed window sizes and setting the window size. The data corresponding to the window size
        requested will be loaded into the system."""

        self.window_size = window_size
        test_pat = self.parsed_patients()[0]
        self.window_sizes = np.array([int(x[:-4]) for x in os.listdir(self.main_path / test_pat / "features")])
        if load_on_start:
            if os.path.exists(self.main_path):
                self.set_window_size(self.window_size)
        self.is_ectopic = {}
        self.n_ectopics = {}

        """
        # ------------------------------------------------------------------------------- #
        # ---------------- Local variables (relevant only for the UVAFDB) --------------- #
        # ------------------------------------------------------------------------------- #
        """

        self.beats_bea = np.array(['NORMAL', 'PVC', 'APC', 'AESC', 'VESC', 'PACE', 'PFUS',
                                   # The different rhythms present across the dataset.
                                   'UNKNOWN', 'UNCLASS', 'SUBTYPE', 'RHYTHM', 'AUX', 'SUB', 'ARFCT',
                                   'VFON', 'FLWAV', 'VFOFF', 'RONT',
                                   'FUSION'])  # Like the SQI step, the rate of missing annotations below which the patient is excluded.
        self.bea_path = cts.DATA_DIR / "uvfdb" / "uvfdb_rr" / "BEA"
        self.excel_sheet_path = cts.DATA_DIR / "uvfdb" / "uvfdb_rr" / "UVA Holter Info.xlsx"
        self.excel_sheet = pd.read_excel(self.excel_sheet_path)

        # Quick and dirty
        if load_ectopics:
            self.load_ectopics([self.window_size])

    """
    # ------------------------------------------------------------------------- #
    # ----- Parsing functions: have to be overridden by the child classes ----- #
    # ------------------------------------------------------------------------- #
    """
    """ These functions are documented in the base parser."""

    def parse_available_ids(self):
        return np.array([file[3:7] for file in os.listdir(str(self.raw_ecg_path))])

    def parse_reference_annotation(self, id):
        _, _, _, beat, _, _, rhythm, _ = self.readbea(self.bea_path / ("UVA" + id + '.bea'))
        beat = ((beat / cts.N_MS_IN_S) * self.actual_fs).astype(int)
        rhythm = np.array([self.rhythms_dict[i] for i in rhythm])
        return beat, rhythm

    def parse_annotation(self, id, type="epltd0"):
        if type not in self.annotation_types:
            raise IOError("The requested annotation does not exist.")
        return wfdb.rdann(str(self.generated_anns_path / type / id), type).sample

    def record_to_wfdb(self, id):
        file = self.raw_ecg_path / ("UVA" + id + ".rf")
        record = self._read_rf(file)
        ecg_raw = record[:, 0]
        wfdb.wrsamp(id, fs=cts.EPLTD_FS, units=['mV'],
                    sig_name=['V5'], p_signal=ecg_raw.reshape(-1, 1), fmt=['16'])
        return ecg_raw

    def parse_raw_ecg(self, patient_id, start=0, end=-1, type='epltd0', lead=0):
        record = self._read_rf(self.raw_ecg_path / ('UVA' + patient_id + '.rf'), lead=lead)
        ecg = record
        ann = self.parse_annotation(patient_id, type=type)
        if end == -1:
            end = int(len(ecg) / self.actual_fs)
        start_sample = start * cts.EPLTD_FS
        end_sample = end * cts.EPLTD_FS
        ecg = ecg[start_sample:end_sample]
        ann = ann[np.where(np.logical_and(ann >= start_sample, ann < end_sample))]
        ann -= start_sample
        return ecg, ann

    def parse_demographic_features(self, id):
        age = float(self.excel_sheet[self.excel_sheet["Holter ID"] == "UVA" + id]["Age at First"])
        gender = float(
            self.excel_sheet[self.excel_sheet["Holter ID"] == "UVA" + id]["Gender"] == 'F')  # True: Female, False: Male
        for win in self.loaded_window_sizes:
            self.features_dict[id][win]['Age'] = age
            self.features_dict[id][win]['Gender'] = gender

    def parse_ahi(self, id):
        self.ahi_dict[id] = np.nan  # This data is not available for this dataset.

    def parse_odi(self, id):
        self.odi_dict[id] = np.nan  # This data is not available for this dataset.

    """
    # ------------------------------------------------------------------------- #
    # ---------------- Functions relative to this dataset only ---------------- #
    # ------------------------------------------------------------------------- #
    """

    def load_ectopics(self, wins=None):
        """ This function loads the number of ectopic beats per window for the UVAF dataset.
            :param wins: The windows for which the number of ectopics should be loaded.
        """
        print("Loading ectopics")
        if wins is None:
            wins = self.window_sizes
        for pat in self.parsed_patients():
            self.n_ectopics[pat] = {}
            for win in wins:
                self.n_ectopics[pat][win] = np.load(self.main_path / pat / "n_ectopics" / (str(win) + ".npy"))

    def generate_beats_hist(self, figsize=(15, 15), remove_N=True):
        """ This function generates a bar plot with the number of beats for the most represented rhythms in the dataset."""
        max_y = 0.5 * 1e8
        jump = 2
        beats_threshold = 1e6
        self.extract_reann_pat()
        beats_table_reann, beats_table_not_reann = np.zeros((len(self.reann_pat), len(self.rhythms)),
                                                            dtype=int), np.zeros(
            (len(self.not_reann_pat), len(self.rhythms)), dtype=int)
        pat_table_reann, pat_table_not_reann = np.zeros((len(self.reann_pat), len(self.rhythms)), dtype=int), np.zeros(
            (len(self.not_reann_pat), len(self.rhythms)), dtype=int)

        for i, pat in enumerate(self.reann_pat):
            rhythms, counts = np.unique(self.rlab_dict[pat], return_counts=True)
            for j, rhy in enumerate(rhythms):
                beats_table_reann[i, int(rhy)] = counts[j]
                pat_table_reann[i, int(rhy)] = 1

        for i, pat in enumerate(self.not_reann_pat):
            rhythms, counts = np.unique(self.rlab_dict[pat], return_counts=True)
            for j, rhy in enumerate(rhythms):
                beats_table_not_reann[i, int(rhy)] = counts[j]
                pat_table_not_reann[i, int(rhy)] = 1

        beats_per_rhythm_reann = np.sum(beats_table_reann, axis=0)
        beats_per_rhythm_not_reann = np.sum(beats_table_not_reann, axis=0)

        beats_table = np.concatenate((beats_table_reann, beats_table_not_reann), axis=0)
        pat_table = np.concatenate((pat_table_reann, pat_table_not_reann), axis=0)
        patients_per_rhythm = np.sum(pat_table, axis=0)
        beats_per_rhythm = np.sum(beats_table, axis=0)

        idx_sort = np.argsort(beats_per_rhythm)[::-1]
        beats_per_rhythm = beats_per_rhythm[idx_sort]
        rhythms = self.rhythms[idx_sort]
        beats_per_rhythm_reann = beats_per_rhythm_reann[idx_sort]
        beats_per_rhythm_not_reann = beats_per_rhythm_not_reann[idx_sort]
        patients_per_rhythm = patients_per_rhythm[idx_sort]

        mask = beats_per_rhythm.copy() > beats_threshold
        beats_per_rhythm = beats_per_rhythm[mask]
        patients_per_rhythm = patients_per_rhythm[mask]

        beats_per_rhythm_not_reann = beats_per_rhythm_not_reann[mask]
        beats_per_rhythm_reann = beats_per_rhythm_reann[mask]

        rhythms = rhythms[mask]
        beats_per_rhythm_disp = ['{:.1e}'.format(float(x)) for x in beats_per_rhythm]

        # Scaling first element ((N)
        beats_per_rhythm_reann[0] = beats_per_rhythm_reann[0] * max_y / beats_per_rhythm[0]
        beats_per_rhythm_not_reann[0] = beats_per_rhythm_not_reann[0] * max_y / beats_per_rhythm[0]

        if remove_N:
            beats_per_rhythm_reann = beats_per_rhythm_reann[1:]  # Removing the (N class
            beats_per_rhythm_not_reann = beats_per_rhythm_not_reann[1:]
            rhythms = rhythms[1:]
            beats_per_rhythm = beats_per_rhythm[1:]
            beats_per_rhythm_disp = beats_per_rhythm_disp[1:]
            patients_per_rhythm = patients_per_rhythm[1:]
            max_y = beats_per_rhythm_reann[0] + beats_per_rhythm_not_reann[0] + 2000000
        # Plotting Beats repartition
        fig, axes = graph.create_figure(figsize=figsize)
        p1 = axes[0][0].bar(np.arange(0, 2 * len(rhythms), jump), beats_per_rhythm_not_reann,
                            tick_label=[r[1:] for r in rhythms],
                            label='No manual correction')
        p2 = axes[0][0].bar(np.arange(0, 2 * len(rhythms), jump), beats_per_rhythm_reann,
                            tick_label=[r[1:] for r in rhythms],
                            bottom=beats_per_rhythm_not_reann, label='Manually corrected')
        for i, rhy in enumerate(rhythms):
            if 0 < beats_per_rhythm[i] < max_y:
                plt.text(jump * i - 0.5 + 0.1, beats_per_rhythm[i] + 1000000,
                         'n=' + beats_per_rhythm_disp[i] + ',\np=' + str(patients_per_rhythm[i])
                         , fontsize=24)
            elif beats_per_rhythm[i] > max_y:
                plt.text(jump * (i + 1) - 1, 0.9 * max_y,
                         'n=' + beats_per_rhythm_disp[i] + ',\np=' + str(patients_per_rhythm[i])
                         , fontsize=24)
        graph.complete_figure(fig, axes, savefig=True, x_titles=[['Rhythm types']], y_titles=[['Count']],
                              y_lim=[[[0, max_y]]], main_title='UVAFDB_beats_distribution')

    def generate_events_hist(self):
        """ This function generates a histogram of events lengths per patient label category across the dataset."""
        AF_events_lengths = [[], [], [], []]
        for pat in self.all_patients():
            is_af = (self.rlab_dict[pat] == cts.WINDOW_LABEL_AF).reshape(-1)
            is_af = np.array([0, *is_af.tolist(), 0])
            starts = np.where(np.diff(is_af) == 1)[0] + 1
            ends = np.where(np.diff(is_af) == -1)[0]
            label = self.af_pat_lab_dict[pat]
            if label == cts.PATIENT_LABEL_OTHER_CVD:
                label = cts.PATIENT_LABEL_NON_AF  # Merging Other CVD with Non-AF
            AF_events_lengths[label].extend((ends - starts).tolist())
        fig, axes = graph.create_figure(subplots=(1, 2))
        all_data = np.concatenate(tuple(AF_events_lengths))
        print("Number of events: " + str(len(all_data)))
        print("Number of events smaller than 60 seconds: " + str(np.sum(all_data < 60)))
        max_val = np.round(np.percentile(all_data, 98))
        max_val += (500 - max_val % 500)
        bins_first = np.arange(0, 60, 10)
        bins_second = np.arange(100, max_val, 500)
        bins = np.concatenate((bins_first, bins_second))

        for i, lab in enumerate([cts.PATIENT_LABELS[1], cts.PATIENT_LABELS[0]]):
            data = AF_events_lengths[lab]
            hist, bin_edges = np.histogram(data, bins)
            axes[0][0].bar(range(0, len(hist) * 2, 2), hist, width=2 * (1 - i / len(cts.PATIENT_LABELS)),
                           color=cts.COLORS[lab],
                           label=cts.GLOBAL_LAB_TITLES[lab])

        for i, lab in enumerate(cts.PATIENT_LABELS[2:4]):
            data = AF_events_lengths[lab]
            hist, bin_edges = np.histogram(data, bins)
            axes[0][1].bar(range(0, len(hist) * 2, 2), hist, width=2 * (1 - i / len(cts.GLOBAL_LAB_TITLES)),
                           color=cts.COLORS[lab],
                           label=cts.GLOBAL_LAB_TITLES[lab])

        graph.complete_figure(fig, axes, xticks_fontsize=12,
                              x_ticks=[[[2 * (0.5 + i) for i, j in enumerate(hist)]] * 2],
                              x_ticks_labels=[[['%d' % (bins[i + 1]) for i, j in enumerate(hist)]] * 2],
                              x_titles=[['Events lengths (in number of beats)'] * 2], xlabel_fontsize=20,
                              y_titles=[['Count', '']], savefig=True, main_title='UVAFDB_Events_lengths')

    def generate_af_burden_hist(self, figsize=(15, 10)):
        """ This function generates a histogram of the AF burden per patient label category across the dataset."""
        AF_Burdens = [[], [], [], []]
        for pat in self.af_burden_dict.keys():
            lab = self.af_pat_lab_dict[pat]
            if lab == cts.PATIENT_LABEL_OTHER_CVD:
                lab = cts.PATIENT_LABEL_NON_AF  # Merging Other CVD with Non-AF
            AF_Burdens[lab].append(self.af_burden_dict[pat])

        fig, axes = graph.create_figure(figsize=figsize)
        labels = np.array([cts.PATIENT_LABEL_AF_MILD, cts.PATIENT_LABEL_AF_MODERATE, cts.PATIENT_LABEL_AF_SEVERE])
        bins = np.append([0, cts.AF_MODERATE_THRESHOLD * 100], np.arange(10, 101, 5))
        rwidths = np.array([1, 1, 1, 1])[::-1]
        n_points = np.sum([len(x) for x in AF_Burdens[1:]])
        for i, lab in enumerate(labels[::-1]):
            data = AF_Burdens[lab]
            weights = np.ones_like(data) / n_points
            axes[0][0].hist(np.array(data) * 100,
                            bins=bins, label=cts.GLOBAL_LAB_TITLES[lab],
                            color=cts.COLORS[lab], rwidth=rwidths[i], weights=weights)
        graph.complete_figure(fig, axes, x_titles=[['AF Burden (%)']], y_titles=[['Proportion of AF Patients']],
                              xlim=[[[0, 1]]],
                              savefig=True, main_title='UVAFDB_AF_Burden_hist', x_lim=[[[0, 100]]])

    def generate_features_hist(self, feats_names=cts.SELECTED_FEATURES):
        """ This function generates a histogram of features per patient label category across the dataset.
        :param feats_names: The list of features to include in the histograms subplot."""
        n_feats_per_plot = 8
        n_rows = 4
        n_cols = 2
        n_bins = 40
        X, y = self.return_features(feats_list=feats_names, return_binary=False)
        y[y >= cts.WINDOW_LABEL_OTHER] = cts.WINDOW_LABEL_OTHER
        n_plots = (X.shape[1] // n_feats_per_plot) + 1
        for k in range(n_plots):
            fig, axes = graph.create_figure(subplots=(n_rows, n_cols), figsize=(30, 15))

            start, end = k * n_feats_per_plot, min((k + 1) * n_feats_per_plot, len(feats_names))
            for n, idx in enumerate(np.arange(start, end, 1)):
                i, j = n // n_cols, n % n_cols
                lim1 = 0.9 * np.percentile(X[:, idx], 1)
                lim2 = 1.1 * np.percentile(X[:, idx], 98)
                bins = np.linspace(lim1, lim2, n_bins)
                for lab in cts.WINDOW_LABELS[:-1]:  # One label in the plot
                    weights = np.ones_like(X[y == lab, idx]) / float(
                        len(X[y == lab, idx]))
                    axes[i][j].hist(X[y == lab, idx],
                                    bins=bins, rwidth=1 - lab / (2 * len(cts.WINDOW_LAB_TITLES)),
                                    label=cts.WINDOW_LAB_TITLES[lab], color=cts.COLORS[lab],
                                    weights=weights)

            put_legend = np.zeros((n_rows, n_cols), dtype=bool)
            put_legend[0][1] = True
            fig.subplots_adjust(left=0.05)
            fig.subplots_adjust(bottom=0.05)
            if end - start == n_feats_per_plot:
                x_titles = np.array(feats_names[start:end]).reshape(n_rows, n_cols)
            else:
                x_titles = '' * np.ones((n_rows, n_cols), dtype=object)
                i, j = 0, 0
                for n, idx in enumerate(np.arange(start, end, 1)):
                    x_titles[i, j] = feats_names[idx]
                    j = (j + 1) % n_cols
                    i = (n + 1) // n_rows
            graph.complete_figure(fig, axes, x_titles=x_titles,
                                  y_titles=[['Density', '']] * n_rows, legend_fontsize=18, ylabel_fontsize=18,
                                  xlabel_fontsize=16,
                                  main_title='UVAFDB_features_' + str(self.window_size) + '_beats_plot_number_' + str(
                                      k),
                                  savefig=True, put_legend=put_legend, xticks_fontsize=16, yticks_fontsize=16)

    def extract_reann_pat(self):
        """ This function looks into the Excel report present in the directory of the .BEA files to report which patients have been reannotated."""
        # First extracting reannotated patients from UVA Info file
        res = pd.read_excel(self.excel_sheet_path)
        self.reann_pat = np.setdiff1d(np.array(res[res['Comments'].notnull()]['Holter ID'].apply(lambda x: x[-4:])),
                                      self.missing_ecg)
        self.not_reann_pat = np.setdiff1d(np.array(res[res['Comments'].isnull()]['Holter ID'].apply(lambda x: x[-4:])),
                                          self.missing_ecg)
        self.reann_pat = np.intersect1d(self.reann_pat, self.parsed_ecgs)
        self.not_reann_pat = np.intersect1d(self.not_reann_pat, self.parsed_ecgs)

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

    def bea_spliter(self, line):
        """ Helper function the read properly the different lines in the .bea files which contain the annotations from the Holter.
        :param line: The input line belonging to the .bea file.
        :returns res: The line with an extension for the rhythm in case it was missing."""
        res = line.split()
        if len(res) == 2:
            res.append(None)
        return res

    # Need to return (beat, rhythm) to have the actual annotation. The rest can be derived from there.
    def readbea(self, beafile, remove_special=False):
        """ Reads a bea file and returns the different important elements.
        :param beafile: The path to the .bea file.
        :param remove_special: If True, removes some of the beats with a given label according to Lake&Moorman's convention. If False, leaves the beats as-is.
        :returns rr: The RR intervals.
        :returns rlab: The labels corresponding to the RR intervals.
        :returns rrt: The timestamp corresponding to each RR interval.
        """
        try:
            with open(beafile, 'r') as file:
                lines = np.array(file.readlines())
                splited = np.array(list(map(self.bea_spliter, lines)))
                splited[:, 0] = splited[:, 0].astype(int)
                beat = splited[:, 0]
                label = splited[:, 1]
                rhythm = splited[:, 2]
                not_none = np.where(rhythm != None)[0]
                not_none = np.append(not_none, len(rhythm))
                diffs = np.diff(not_none)
                sing_rhy = rhythm[not_none[:-1]]
                rhythm[not_none[0]:] = np.repeat(sing_rhy, diffs)
                rhythm[0:not_none[0]] = sing_rhy[0]
                rhythm_lab = np.array(list(map(lambda x: self.rhythms_dict[x], rhythm))).reshape(-1, 1)
        except IOError:
            sys.exit(1)

        bnum = [0, 2, 3, 4, 5, 6, 7, 8, 8, 12, 10, 11, 12, 13, 20, 21, 22, 9, 9]
        # bnum = [0, 2, 3, 4, 5, 6, 7, 8, 8, 9,  10, 11, 12, 13, 20, 21, 22, 9, 9] # Correct version. Currently using the same version as Lake and Moorman
        bnum_dict = {self.beats_bea[i]: bnum[i] for i in range(len(bnum))}
        lab = np.array(list(map(lambda x: bnum_dict[x], label))).reshape(-1, 1)

        if remove_special:
            mask_good = (lab < 10).reshape(-1)
            if np.sum(mask_good) >= 2:
                good_beats = beat[mask_good]
                rr = np.diff(good_beats)
                rrt = good_beats[0] + np.cumsum(rr)
                rlab = rhythm_lab[mask_good][1:]  # [1:] to select the second extremity of the RR interval for the beat
        else:
            rr = np.diff(beat)
            rrt = beat[0] + np.cumsum(rr)
            rlab = rhythm_lab[1:]

        mask = (lab == 11).reshape(-1)  # Label AUX
        rtime = beat[mask]

        # For lab, converting to binary label only: True for Ectopic (APC, PVC), False otherwise
        lab = lab.reshape(-1)
        lab = np.logical_or(lab == 1, lab == 2)
        return rr, rrt, rlab, beat, lab, label, rhythm, rtime

    def export_res_rythems(self, pat, directory=cts.ERROR_ANALYSIS_DIR, start=0, end=-1, force=False, ann_type='epltd0',
                           export_rhythms=False, n_leads=1):
        """ This function exports a given recording to the physiozoo format for further investigation.
        The files are exported under a .txt format with the proper headers to be read by the PhysioZoo software.
        The raw ECG as well as the peaks are exported. If provided, the AF events are exported as well.
        :param pat: The patient ID to export.
        :param directory: The directory under which the files will be saved.
        :param start: The beginning timestamp for the ECG portion to be saved.
        :param end: The ending timestamp for the ECG portion to be saved.
        :param force: If True, generate the files anyway, otherwise verifies if the file has already been exported under the directory
        :param export_rhythms: If True, exports a file with the different rhythms over the recording."""

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

        if end == -1:
            end = self.recording_time[pat]

        rhythms_full_path = directory / (pat + '_res_rhythms_start_' + str(start) + '_end_' + str(end) + '.txt')

        # Rhythms
        if export_rhythms:
            if not os.path.exists(rhythms_full_path) or force:
                with open(rhythms_full_path, 'w+') as rhythms_file:
                    rhythms_header[2] = 'source file: ' + pat + '_rhythms.txt\n'
                    rhythms_file.writelines(rhythms_header)
                    rhythms = np.array([int(i) for i in self.rlab_dict[pat]])
                    periods = np.concatenate(([0], np.where(np.diff(rhythms.astype(int)))[0] + 1, [len(rhythms) - 1]))
                    start_idx, end_idx = periods[:-1], periods[1:]
                    final_rhythms = rhythms[start_idx]
                    mask_rhythms = final_rhythms > 0  # We do not keep NSR as rhythm
                    raw_rrt = self.rrt_dict[pat]
                    # rrt = raw_rrt[:(len(raw_rrt) // self.window_size) * self.window_size].reshape(-1, self.window_size)
                    # start_events, end_events = rrt[start_idx, 0], rrt[end_idx, 0]
                    # mask_int_events = np.logical_and(start_events < end, end_events > start)
                    # mask_rhythms = np.logical_and(mask_rhythms, mask_int_events)
                    # start_events, end_events = start_events[mask_rhythms] - start, end_events[mask_rhythms] - start
                    # end_events[end_events > (end - start)] = end - start
                    start_idx, end_idx = start_idx[mask_rhythms], end_idx[mask_rhythms]
                    start_rrt = raw_rrt[start_idx]
                    end_rrt = raw_rrt[end_idx]
                    # final_rrt =final_rrt[mask_rhythms]

                    final_rhythms = final_rhythms[mask_rhythms]
                    final_rhythms_str = np.array([self.rhythms[i][1:] for i in final_rhythms])
                    rhythms_file.write('\n'.join(
                        ['%.5f\t%.5f\t%s' % (start_rrt[i], end_rrt[i], final_rhythms_str[i]) for i in
                         range(len(start_idx))]))

    def export_ecg(self, pat, directory=cts.ERROR_ANALYSIS_DIR, start=0, end=-1, force=False, ann_type='epltd0',
                   export_rhythms=False, n_leads=1):
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
                      'Channels:\n',
                      '\n'
                      '    - type:   electrography\n',
                      '      name:   Data\n',
                      '      unit:   mV\n',
                      '      enable: yes\n',
                      '\n',
                      '---\n',
                      '\n'
                      ]

        if not os.path.exists(directory):
            os.makedirs(directory)

        ecgs = np.concatenate(tuple(
            [self.parse_raw_ecg(pat, start=start, end=end, type=ann_type, lead=lead)[0].reshape(-1, 1) for lead in
             range(n_leads)]), axis=1)
        if end == -1:
            end = self.recording_time[pat]

        ecg_full_path = directory / (
                pat + '_ecg_start_' + str(start) + '_end_' + str(end) + '_n_leads_' + str(n_leads) + '.txt')

        # Raw ECG
        join_func = lambda x: ' '.join(['%.4f' % i for i in x])
        ecg_str = np.apply_along_axis(join_func, 1, ecgs)
        if not os.path.exists(ecg_full_path) or force:
            with open(ecg_full_path, 'w+') as ecg_file:
                ecg_file.writelines(ecg_header)
                ecg_file.write('\n'.join(ecg_str))


if __name__ == '__main__':
    db = UVAFDB_Parser(load_ectopics=False, load_on_start=False)
    db.parse_raw_data(patient_list=['1610', ])
    # db.set_sqi_test_ann('wavedet')
    # db.print_summary()
    # #db.generate_wrqrs_annotations(pat_list=db.parsed_patients())
    # wins = [60, 70, 80, 90, 100, 110, 120]
    # db.create_pool()
    # for win in wins:
    #     print(win)
    #     for pat in db.parsed_patients()[2400:2700]:
    #         print(pat)
    #         db.load_patient_from_disk(pat)
    #         db._sqi(pat, win, test_ann='wavedet')
    #         np.save(db.main_path / pat / 'signal_quality' / str(win), db.signal_quality_dict[pat][win])
    # #db.generate_af_burden_hist(figsize=(10, 10))
    # #db.generate_features_hist(np.array(['cosEn', 'AFEv', 'OriginCount', 'IrrEv', 'PIP', 'PNN50', 'SD1', 'SD2']))
    # #db.parse_elem_data('0011')
    # # db.load_patient_from_disk('0001')
    # #db.export_to_physiozoo('0001', export_rhythms=True)
    # # db.create_pool()
    # # pat_list = db.parsed_patients()
    # # for pat in pat_list:
    # #     print(pat)
    # #     vals = np.load(db.main_path / pat / 'pat_values.npy', allow_pickle=True)
    # #     ecg, _ = db.parse_raw_ecg(pat)
    # #     vals[5] = len(ecg) / db.actual_fs
    # #     np.save(db.main_path / pat / 'pat_values', vals)
    # ################################################################################
    # # pat_list = db.parsed_patients()[2500:]
    # # for win in cts.BASE_WINDOWS:
    # #     db.loaded_window_sizes = np.append(db.loaded_window_sizes, win)
    # #     for pat in pat_list:
    # #         print(pat)
    # #         db.load_patient_from_disk(pat, [win])
    # #         del db.signal_quality_dict[pat][win]
    # #         del db.features_dict[pat][win]['sqi']
    # #         for ann_type in ['xqrs', 'gqrs', 'rqrs']:
    # #             db._sqi(pat, win, test_ann=ann_type)
    # #         np.save(db.main_path / pat / 'signal_quality' / str(win), db.__dict__['signal_quality' + '_dict'][pat][win])
    # #         np.save(db.main_path / pat / 'features' / str(win), db.__dict__['features' + '_dict'][pat][win])
    # #
    #
