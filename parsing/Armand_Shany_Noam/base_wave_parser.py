from parsing.base_parser import *


class BaseWaveParser(BaseParser):
    """Class defining a base parser. All the parsers relative to the different datasets will inherit this class.
    This class offers the possibility to load the dataset onto the system, with different window sizes, features, sqi.
    It further provide tools to display elementary statistics on the datasets, such as the time in AF, the total recording time,
    display of the ECG signals for a given segment, and more.
    Each child dataset has its own init function, in which the following the paths and the information relative to the
    specific dataset will be set, to ensure the dataset will be loaded properly."""

    def __init__(self):
        super(BaseWaveParser, self).__init__()
        """The purpose of the __init__ function for the BaseParser class
        is to define the different dictionnaries that will be found among the different parsers.
        Note: Depending on the dataset, some of those dictionnaries may be empty. Data extraction is taken care of
        apart in each dataset with the elementary parsing functions whose signature can be found below. """

        """
        # ------------------------------------------------------------------------------- #
        # ----------------------- To be overriden in child classes ---------------------- #
        # ------------------------------------------------------------------------------- # """

        """
        # ------------------------------------------------------------------------------- #
        # ------------------------- Overriden variables end here ------------------------ #
        # ------------------------------------------------------------------------------- #
        """

        """Raw data. Used further to derive metrics and features."""

        """Dictionaries obtained after computation through the methods."""

        """ Labels dicts."""

        """ Patients and exclusions. """

        """ General variables for signal processing/Filtering """

    """
    # ------------------------------------------------------------------------- #
    # ----- Parsing functions: have to be overridden by the child classes ----- #
    # ------------------------------------------------------------------------- #
    """

    # ------------------------------------------------------------------------- #
    # ------------------------ Computational functions ------------------------ #
    # ------------------------------------------------------------------------- #
    def number_of_windows(self, pat, win):
        """
        :param pat: patient's id
        :param win: window size in number of samples per window
        :return: number of windows
        """
        return int(self.recording_time[pat] * self.actual_fs // win)

    def parse_elem_data(self, pat, reannotated=False):
        """ This function is responsible for extracting the basic raw data for the given id.
        """
        dicts_to_fill = [self.start_windows_dict, self.end_windows_dict, self.av_prec_windows_dict,
                         self.n_excluded_windows_dict, self.mask_rr_dict]
        for dic in dicts_to_fill:
            if pat not in dic.keys():
                dic[pat] = {}
        sec_int = 0  # Security interval for RR exclusion (number of beats to exclude around a problematic RR

        # Extracting data based on Automatic annotation (default EPLTD)
        ecg, ann = self.parse_raw_ecg(pat)
        self.recording_time[pat] = len(ecg) / self.actual_fs
        rr = np.diff(ann) / self.actual_fs
        start_rr, end_rr = ann[:-1] / self.actual_fs, ann[1:] / self.actual_fs

        # Extracting the reference annotation
        ref_ann, ref_rhythm = self.parse_reference_annotation(pat, reannotated=reannotated)
        ref_rr = np.diff(ref_ann) / self.actual_fs
        start_ref_rr, end_ref_rr = ref_ann[:-1] / self.actual_fs, ref_ann[1:] / self.actual_fs
        ref_rlab = ref_rhythm[1:]  # To have the same dimension as ref_rr

        # First filtering - Removing RR intervals at which equal zero
        mask = ref_rr > consts.RR_OUTLIER_THRESHOLD_INF
        ref_rr, start_ref_rr, end_ref_rr, ref_rlab = ref_rr[mask], start_ref_rr[mask], end_ref_rr[mask], ref_rlab[mask]
        ref_interbeats = np.append(np.insert((start_ref_rr + end_ref_rr) / 2, 0, max(0, start_ref_rr[0] - 1)),
                                   end_ref_rr[-1] + 1.0)

        # Second filtering - Based on missing reference annotations
        mask_sup = ref_rr <= consts.RR_OUTLIER_THRESHOLD_SUP
        irreg_rr_indices = np.where(~mask_sup)[0]
        self.excluded_portions_dict[pat] = np.array(
            [[ref_interbeats[x - sec_int], ref_interbeats[x + sec_int + 2]] for x in irreg_rr_indices])
        if len(self.excluded_portions_dict[pat]) > 0:
            self.excluded_portions_dict[pat] = np.concatenate(
                ([[0, start_ref_rr[0]]], self.excluded_portions_dict[pat]),
                axis=0)  # Beggining of recordings without any label are considered excluded
        else:
            self.excluded_portions_dict[pat] = np.array([[0, start_ref_rr[0]]])

        # Building interpolator for the label
        ref_rr, start_ref_rr, end_ref_rr, ref_rlab = ref_rr[mask_sup], start_ref_rr[mask_sup], end_ref_rr[mask_sup], \
                                                     ref_rlab[mask_sup]
        f = interp.interp1d(end_ref_rr, ref_rlab, kind='nearest', fill_value="extrapolate")

        # Saving raw data
        self.rlab_dict[pat] = f(start_rr)
        self.rr_dict[pat] = rr
        self.rrt_dict[pat] = start_rr

        for win in self.window_sizes:
            num_of_windows = self.number_of_windows(pat, win)
            rrt_idx_start = [np.argmax(self.rrt_dict[pat] >= i * win / self.actual_fs) for i in range(num_of_windows)]
            rrt_idx_end = [np.argmin(self.rrt_dict[pat] < (i + 1) * win / self.actual_fs) for i in
                           range(num_of_windows)]

            mask_start = np.logical_or.reduce(tuple(
                [np.logical_and(self.rrt_dict[pat][rrt_idx_start] > x[0], self.rrt_dict[pat][rrt_idx_start] <= x[1]) for
                 x in
                 self.excluded_portions_dict[
                     pat]]))  # The window begins in an excluded portion.
            mask_end = np.logical_or.reduce(tuple(
                [np.logical_and(self.rrt_dict[pat][rrt_idx_end] > x[0], self.rrt_dict[pat][rrt_idx_end] <= x[1]) for x
                 in
                 self.excluded_portions_dict[
                     pat]]))  # The window ends in an excluded portion.
            mask_between = np.logical_or.reduce(tuple(
                [np.logical_and(self.rrt_dict[pat][rrt_idx_start] <= x[0], self.rrt_dict[pat][rrt_idx_end] > x[1]) for x
                 in
                 self.excluded_portions_dict[
                     pat]]))  # The window contains an excluded portion.
            final_mask = np.logical_not(
                np.logical_or.reduce((mask_start, mask_end, mask_between)))  # We don't want any of the three
            self.start_windows_dict[pat][win] = rrt_idx_start
            self.end_windows_dict[pat][win] = rrt_idx_end
            self.mask_rr_dict[pat][win] = final_mask
            # Depend on the mask applied on the windows
            self.n_excluded_windows_dict[pat][win] = (~final_mask).sum()
            self.av_prec_windows_dict[pat][win] = dp.cumsum_reset(final_mask)

    def _sqi(self, id, win, test_ann='xqrs'):
        """ Computes the Signal Quality Index (SQI) of each window.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        :param win: The window size (in number of samples) along which the raw recording is divided.
        :param test_ann: The test annotation to use for the computation of the bsqi function (in utils.feature_comp module)"""

        if id not in self.signal_quality_dict.keys():
            self.signal_quality_dict[id] = {}
        if win not in self.signal_quality_dict[id].keys():
            self.signal_quality_dict[id][win] = {}
        raw_rrt = self.rrt_dict[id]

        num_of_windows = self.number_of_windows(id, win)

        rrt = raw_rrt * self.actual_fs
        refqrs = [rrt[self.start_windows_dict[id][win][i]:self.end_windows_dict[id][win][i]] for i in
                  range(num_of_windows)]

        testqrs = self.parse_annotation(id, type=test_ann)

        testqrs = [testqrs[np.where(
            np.logical_and(testqrs > i * win, testqrs <= (i + 1) * win))] for i in
                   range(num_of_windows)]

        self.signal_quality_dict[id][win][test_ann] = np.array(self.pool.starmap(fc.bsqi,
                                                                                 zip(refqrs, testqrs,
                                                                                     self.agw * np.ones(
                                                                                         len(testqrs)),
                                                                                     self.actual_fs * np.ones(
                                                                                         len(testqrs)))))

    def parse_raw_data(self, window_sizes=np.array([6000]), gen_ann=False, feats=cts.IMPLEMENTED_FEATURES,
                       patient_list=None, test_anns=None, reannotated=False):
        """ This function is responsible of performing all the necessary computations for the dataset, among which:
        all the features according to the input, the sqi, the labels per window, the ahi, the odi, and the demographic features if available.
        This function has usualy a long running time (at least for the big databases).
        :param window_sizes: The different window sizes along which the windows are derived. In number of samples per window
        :param get_ann: If True, calls the function gen_ann in Force mode, and runs all the annotations.
        :param feats: List of the features to compute. The function 'comp_" + feature name must be implemented in the utils.feature_comp module.
        :param total_run: if True, runs the function on all the IDs, otherwise runs on the non-parsed IDs only.
        :param test_anns: All the test annotations upon which SQI needs to run. If None, uses all the available annotations.
        """
        self.window_sizes = window_sizes
        self.generate_annotations(pat_list=patient_list, force=gen_ann, types=test_anns)
        self.parsed_ecgs = self.parsed_patients()
        if patient_list is None:
            patient_list = self.parse_available_ids()

        self.create_pool()
        for i, id in enumerate(patient_list):
            print("Parsing patient number " + id)
            ecg, ann = self.parse_raw_ecg(id)
            self.curr_ecg, self.curr_id = ecg, id
            if len(ann) > self.min_annotation_len:
                self.parse_elem_data(id,
                                     reannotated=reannotated)  # First deriving rr, rlab, rrt dictionnaries and other basic elements.
                for win in window_sizes:  # Running on all the required windows
                    self.window_sizes = np.unique(np.append(self.window_sizes, win))
                    self.loaded_window_sizes = np.append(self.loaded_window_sizes, win)
                    # self._features(id, win, feats=feats)    # Computing all the features
                    if test_anns is None:
                        test_anns = self.annotation_types[self.annotation_types != self.sqi_ref_ann]
                    for ann_type in test_anns:  # Computing SQI
                        self._sqi(id, win, test_ann=ann_type)
                        print(self.signal_quality_dict[id][win][ann_type])
                    self._win_lab(id, win)  # Generating label for each widow
                    self._af_win_lab(id, win)  # Generating binary AF label for each window.
                # self.parse_demographic_features(id)         # Parsing demographics
                self._af_pat_lab(id)  # Generating patient label among the different categories based on AF burden
                self.parse_ahi(id)  # Computing AHI
                self.parse_odi(id)  # Computing ODI
                self.save_patient_to_disk(id)  # Saving
            else:
                print("Corrupted ECG.")
                self.corrupted_ecg = np.append(self.corrupted_ecg,
                                               id)  # If the recording presents less than 1000 peaks, it is considered as corrupted and is not considered.
        # except:
        #    print('Error')
        #    continue
        self.destroy_pool()
