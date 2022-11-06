from base_parser import *

random.seed(cts.SEED)
np.random.seed(cts.SEED)


class AFDB_Parser(BaseParser):

    def __init__(self, window_size=60, load_on_start=True):

        super(AFDB_Parser, self).__init__()

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overriden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- #
        """

        """Missing records"""
        self.missing_ecg = np.array(['00735', '03665'])

        """Helper variables"""
        self.window_size = window_size

        """Variables relative to the ECG signals."""
        self.orig_fs = 250
        self.actual_fs = cts.EPLTD_FS
        self.n_leads = 1
        self.name = "AFDB"
        self.ecg_format = "wfdb"
        self.rhythms = np.array(['(N', '(AFIB', '(AB', '(AFL', '(B', '(BII', '(IVR', '(NOD',
                                 '(P', '(PREX', '(SBR', '(SVTA', '(T', '(VFL', '(VT', '(J'])
        self.rhythms_dict = {self.rhythms[i]: i for i in range(len(self.rhythms))}

        """Variables relative to the different paths."""
        self.raw_ecg_path = cts.DATA_DIR / "afdb"
        self.orig_anns_path = cts.DATA_DIR / "afdb"
        self.generated_anns_path = cts.DATA_DIR / "Annotations" / self.name
        self.annotation_types = np.intersect1d(np.array(os.listdir(self.generated_anns_path)), cts.ANNOTATION_TYPES)
        self.filename = "AFDB.pkl"
        self.main_path = cts.PREPROCESSED_DATA_DIR / "AFDB"
        self.window_size = window_size
        if os.path.exists(self.main_path):
            parsed_patients = self.parsed_patients()
            test_pat = parsed_patients[0]
            self.window_sizes = np.array([int(x[:-4]) for x in os.listdir(self.main_path / test_pat / "features")])
        """ Checking the parsed window sizes and setting the window size. The data corresponding to the window size
        requested will be loaded into the system."""
        if load_on_start:
            self.set_window_size(self.window_size)

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
        if type == 'manual':
            return wfdb.rdann(str(self.raw_ecg_path / id, 'qrs')).sample
        else:
            return wfdb.rdann(str(self.generated_anns_path / type / id), type).sample

    def parse_reference_annotation(self, id):
        ann = self.parse_annotation(id)
        ann_rhythm = wfdb.rdann(str(self.raw_ecg_path / id), 'atr')
        rhythm_samp = ann_rhythm.sample
        rhythm_names = ann_rhythm.aux_note
        rhythm = np.ndarray(shape=len(ann), dtype=object)
        for j in range(len(rhythm_samp)):
            if j < len(rhythm_samp) - 1:
                rhythm[np.where(np.logical_and(ann > rhythm_samp[j], ann <= rhythm_samp[j + 1]))[0]] = rhythm_names[j]
            else:
                rhythm[np.where(ann > rhythm_samp[j])[0]] = rhythm_names[j]
        rhythm = np.array([self.rhythms_dict[i] for i in rhythm])
        return ann.sample, rhythm

    def record_to_wfdb(self, id):
        record = wfdb.rdrecord(str(self.raw_ecg_path / id))
        ecg = record.p_signal[:, 0]
        ecg_resampled = signal.resample(ecg, int(len(ecg) * self.actual_fs / self.orig_fs))
        wfdb.wrsamp(str(id), fs=self.actual_fs, units=['mV'],
                    sig_name=['V5'], p_signal=ecg_resampled.reshape(-1, 1), fmt=['16'])
        return ecg

    def parse_raw_ecg(self, patient_id, start=0, end=-1, type="epltd0", lead=0):
        record = wfdb.rdrecord(self.raw_ecg_path / patient_id)
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
                              savefig=True, main_title='AFDB_AF_Burden_hist', x_lim=[[[0, 100]]])

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
                                  main_title='AFDB_features_' + str(self.window_size) + '_beats_plot_number_' + str(k),
                                  savefig=True, put_legend=put_legend, xticks_fontsize=16, yticks_fontsize=16)


if __name__ == '__main__':
    db = AFDB_Parser(load_on_start=True)
    db.parse_raw_data()
    a = 5
