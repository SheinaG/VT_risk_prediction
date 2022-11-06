import sys

import h5py
import numpy as np
import scipy.interpolate as interp

sys.path.append('/home/b.noam/')

import Noam_Repo.AF_DETECT.utils.consts as consts
import Noam_Repo.AF_DETECT.parsing.feature_comp as fc
import Noam_Repo.AF_DETECT.data.data_processing as dp
from Noam_Repo.AF_DETECT.parsing.uvafdb_parser import UVAFDB_Parser


class UVAFDB_Wave_Parser(UVAFDB_Parser):

    def __init__(self, window_size=6000, load_on_start=False, load_ectopics=True, windows_shifted=False):

        super(UVAFDB_Wave_Parser, self).__init__(window_size=window_size, load_on_start=load_on_start,
                                                 load_ectopics=load_ectopics, windows_shifted=windows_shifted,
                                                 base_path=consts.PREPROCESSED_WAVE_DATA_DIR)

        self.main_path = consts.PREPROCESSED_WAVE_DATA_DIR / ("UVAFDB" + ("_shifted" if windows_shifted else ""))

        """ Variables with different meaning then parent class"""
        self.start_windows_dict = {}  # Key: Patient ID, Value: Windows starts (index of first rr in window)
        self.end_windows_dict = {}  # Key: Patient ID, Value: Windows ends (index of last rr in window)

    def number_of_windows(self, pat, win):
        """
        :param pat: patient's id
        :param win: window size in number of samples per window
        :return: number of windows
        """
        return int(self.recording_time[pat] * self.actual_fs // win)

    def parse_raw_data(self, window_sizes=[6000], gen_ann=False, feats=consts.IMPLEMENTED_FEATURES,
                       patient_list=None, test_anns=None, reannotate=False):
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
                self.parse_pat_elem_data(id,
                                         reannotate=reannotate)  # First deriving rr, rlab, rrt dictionnaries and other basic elements.
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

    def parse_pat_elem_data(self, pat, reannotate=False):
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
        ref_ann, ref_rhythm = self.parse_reference_annotation(pat, reannotated=reannotate)
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

    def _win_lab(self, id, win):
        """ Computes the label of a window. The label is computed based on the most represented label over the window.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        :param win: The window size (in number of  SAMPLES) along which the raw recording is divided.
        """
        if id not in self.win_lab_dict.keys():
            self.win_lab_dict[id] = {}
        raw_rlab = self.rlab_dict[id]
        num_of_windows = self.number_of_windows(id, win)
        rlab = [raw_rlab[self.start_windows_dict[id][win][i]:self.end_windows_dict[id][win][i]] for i in
                range(num_of_windows)]
        counts = []
        count_nan = []
        for j in range(num_of_windows):
            win_count_lab = np.array([np.sum((rlab[j] == i), axis=0) for i in range(len(self.rhythms))]).astype(float)
            win_count_nan = np.sum(np.isnan(rlab[j]), axis=0).astype(float)
            counts.append(win_count_lab)
            count_nan.append(win_count_nan)

        counts = np.transpose(np.array(counts))
        count_nan = np.transpose(np.array(count_nan))

        max_lab_count = np.max(counts, axis=0).astype(float)
        self.win_lab_dict[id][win] = np.argmax(counts, axis=0).astype(float)
        self.win_lab_dict[id][win][count_nan > max_lab_count] = np.nan

    def _af_pat_lab(self, id):
        """ Computes the AF Burden and the global label for a given patient. The AF Burden is computed as the time
        spent on AF divided by the total time of the recording. The different categories of patients are: Non-AF (Time in AF
        does not exceed 30 [sec], Mild AF (Time in AF above 30 [sec] and AFB under 4%), Moderate AF (AFB between 4 and 80%),
        and Severe AF (AFB between 80 and 100%). If the burden of a given pathology for a patient is over 50%, we flag him as a patient
        suffering from another CVD (label cts.PATIENT_LABEL_OTHER_CVD). As a convention, for windows, 0 is the label for NSR, 1 for AF, and above
        2 for other rhythms.
        :param id: The patient ID. Assumed to be in the list of IDs present in the database.
        """
        # Using minimal window size to have the higher granularity

        raw_rr = self.rr_dict[id]
        raw_rlab = self.rlab_dict[id]

        if np.all(np.isnan(raw_rlab)):  # Case where the labels are not available (like in SHHS)
            self.af_burden_dict[id] = np.nan
            self.af_pat_lab_dict[id] = np.nan
            self.other_cvd_burden_dict[id] = np.nan
            self.missing_af_label = np.append(self.missing_af_label, id)
        else:
            time_in_af = raw_rr[raw_rlab == consts.WINDOW_LABEL_AF].sum()  # Deriving time in AF.
            self.af_burden_dict[id] = time_in_af / self.recording_time[id]  # Computing AF Burden.
            self.other_cvd_burden_dict[id] = np.sum(raw_rlab > consts.WINDOW_LABEL_AF) / self.recording_time[
                id]  # Computing Other CVD Burden.
            if self.af_burden_dict[id] > consts.AF_SEVERE_THRESHOLD:  # Assessing the class according to the guidelines
                self.af_pat_lab_dict[id] = consts.PATIENT_LABEL_AF_SEVERE
            elif self.af_burden_dict[id] > consts.AF_MODERATE_THRESHOLD:
                self.af_pat_lab_dict[id] = consts.PATIENT_LABEL_AF_MODERATE
            elif time_in_af > consts.AF_MILD_THRESHOLD:
                self.af_pat_lab_dict[id] = consts.PATIENT_LABEL_AF_MILD
            elif self.other_cvd_burden_dict[id] > 0.5:
                self.af_pat_lab_dict[id] = consts.PATIENT_LABEL_OTHER_CVD
            else:
                self.af_pat_lab_dict[id] = consts.PATIENT_LABEL_NON_AF

    def get_ecg_from_hdf5(self, database, patient_list, pat):
        """

        :param pat: patient's id we want to extract raw ecg from database in hdf5_file.
        :return: patient's raw ecg after zero padding is removed
        """
        pat_idx = np.where(patient_list == int(pat))
        print(f' patient =  {pat} pat_idx = {pat_idx} ')
        ecg_pat = database[pat_idx]
        print('start zero padding')
        # remove zero padding
        ecg_orig_len = self.recording_time[pat] * self.actual_fs
        ecg_pat = np.squeeze(ecg_pat, axis=0)
        ecg_pat = ecg_pat[:int(ecg_orig_len)]
        print('end zero padding')
        return ecg_pat

    def return_masks_dict(self, pat_list=None, exclude_low_sqi_win=True, win_thresh=consts.SQI_WINDOW_THRESHOLD):
        """

        :param pat_list: The list of patients for whom the features should be returned.
        :param exclude_low_sqi_win:  If true, exclude the sqi windows under 'win_thresh' (default 0.8).
        :param win_thresh: exclusion threshold under which a window is excluded by the SQI criterion.
        :return: dictionary which the keys are the patients and the values are arrays of if we need to mask each window or not
        """
        if pat_list is None:
            correct_pat_list = self.non_corrupted_ecg_patients()
        else:
            correct_pat_list = np.array([elem for elem in pat_list])
            if len(correct_pat_list) == 0:
                return np.array([[]])  # Returning empty arrays in case of empty lists.

        if exclude_low_sqi_win:
            masks = {pat: np.logical_and(self.mask_rr_dict[pat][self.window_size],
                                         self.signal_quality_dict[pat][self.window_size][
                                             self.sqi_test_ann] >= win_thresh) for pat in correct_pat_list}
        else:
            masks = {pat: self.mask_rr_dict[pat][self.window_size] for pat in correct_pat_list}
        return masks

    def return_start_time(self, patient, exclude_low_sqi_win=True, win_thresh=consts.SQI_WINDOW_THRESHOLD):
        """
        :param patient:  patient number
        :param exclude_low_sqi_win:  If true, exclude the sqi windows under 'win_thresh' (default 0.8).
        :param win_thresh: exclusion threshold under which a window is excluded by the SQI criterion.
        :return: This function returns the seconds from the start of the examination of each window that has a sqi bigger then win_threshold
        """
        n_of_windows = self.number_of_windows(patient, self.window_size)
        window_time = self.window_size // self.actual_fs
        start_time = np.array(range(0, n_of_windows * window_time, window_time))
        mask = self.return_masks_dict(pat_list=[patient], exclude_low_sqi_win=exclude_low_sqi_win,
                                      win_thresh=win_thresh)
        start_time = start_time[mask[patient]]
        return start_time

    # def return_ecg_data(self,hdf5file, pat_list=None, return_binary=True, return_global_label=False, exclude_low_sqi_win=True, win_thresh=consts.SQI_WINDOW_THRESHOLD):
    #     """Concatenates all the ecg windows contained in the whole dataset and returns them in the form X, y.
    #     :param pat_list: The list of patients for whom the features should be returned.
    #     :param return_binary: If true, returns the binary AF label, otherwise the general rhythm label (af_win_lab vs. win_lab)
    #     :param return_global_label: If true, returns as well the global label (patient label) for each patient.
    #     :param exclude_low_sqi_win: If true, exclude the sqi windows under 'win_thresh' (default 0.8).
    #     :param win_thresh: Exclusion threshold under which a window is excluded by the SQI criterion.
    #     :return X: ecg intervals matrix (n_windows, window_size).
    #     :return y: AF label per window. (n_samples, )"""
    #
    #     f = h5py.File(hdf5file,"r")
    #     database =  f[consts.HDF5_DATASET]
    #     database_patient_list = database.attrs[consts.HDF5_PATIENT_IDS]
    #
    #     if pat_list is None:
    #         correct_pat_list = self.non_corrupted_ecg_patients()
    #     else:
    #         correct_pat_list = np.array([elem for elem in pat_list])
    #         if len(correct_pat_list) == 0:
    #             return np.array([[]]), np.array([])     # Returning empty arrays in case of empty lists.
    #
    #     if exclude_low_sqi_win:
    #         masks = {pat: np.logical_and(self.mask_rr_dict[pat][self.window_size],
    #                                      self.signal_quality_dict[pat][self.window_size][self.sqi_test_ann] >= win_thresh) for pat in correct_pat_list}
    #     else:
    #         masks = {pat: self.mask_rr_dict[pat][self.window_size] for pat in correct_pat_list}
    #
    #     X = None
    #     self.create_pool()
    #     patients_added_list = []
    #     while len(patients_added_list) < len(correct_pat_list):
    #         X_temp = None
    #         if len(correct_pat_list) > (len(patients_added_list)+5):
    #             current_patient_list = correct_pat_list[len(patients_added_list):len(patients_added_list)+100]
    #         else:
    #             current_patient_list = correct_pat_list[len(patients_added_list):-1]
    #         for elem in current_patient_list:
    #             print(elem)
    #             ecg = self.get_ecg_from_hdf5(database,database_patient_list,elem)
    #             print('done with get_ecg_from_hdf5')
    #             # ecg_ref, ann = self.parse_raw_ecg(elem)
    #             # #pass data through notch filter
    #             # ecg = notch_filter(ecg, elem, consts.ELECTRICITY_FREQ, self.actual_fs)
    #
    #             ecg = ecg[:(len(ecg) // self.window_size) * self.window_size].reshape(-1, self.window_size)[masks[elem]]
    #             print('start concat')
    #             if X_temp is None:
    #                 X_temp = ecg
    #             else:
    #                 X_temp = np.concatenate((X_temp,ecg), axis=0)
    #             print('end concat')
    #             patients_added_list.append(elem for elem in current_patient_list)
    #         if X is None:
    #             X = X_temp
    #         else:
    #             X = np.concatenate((X, X_temp), axis = 0)
    #     self.destroy_pool()
    #     if return_binary:
    #         y = np.concatenate(tuple(self.af_win_lab_dict[elem][self.window_size][masks[elem]] for elem in correct_pat_list), axis=0)
    #     else:
    #         y = np.concatenate(tuple(self.win_lab_dict[elem][self.window_size][masks[elem]] for elem in correct_pat_list), axis=0)
    #
    #     if return_global_label:
    #         global_lab = np.concatenate(tuple(self.af_pat_lab_dict[elem] * np.ones(np.sum(masks[elem])) for elem in correct_pat_list), axis=0)
    #         return X, y, global_lab
    #     else:
    #         return X, y
    def return_ecg_data(self, hdf5file, pat_list=None, return_binary=True, return_global_label=False,
                        exclude_low_sqi_win=True, win_thresh=consts.SQI_WINDOW_THRESHOLD):
        """Concatenates all the ecg windows contained in the whole dataset and returns them in the form X, y.
        :param pat_list: The list of patients for whom the features should be returned.
        :param return_binary: If true, returns the binary AF label, otherwise the general rhythm label (af_win_lab vs. win_lab)
        :param return_global_label: If true, returns as well the global label (patient label) for each patient.
        :param exclude_low_sqi_win: If true, exclude the sqi windows under 'win_thresh' (default 0.8).
        :param win_thresh: Exclusion threshold under which a window is excluded by the SQI criterion.
        :return X: ecg intervals matrix (n_windows, window_size).
        :return y: AF label per window. (n_samples, )"""

        f = h5py.File(hdf5file, "r")
        database = f[consts.HDF5_DATASET]
        dataset_patients_list = database[:, -3:-1].astype(int)

        if pat_list is None:
            correct_pat_list = self.non_corrupted_ecg_patients()
        else:
            correct_pat_list = np.array([elem for elem in pat_list])
            if len(correct_pat_list) == 0:
                return np.array([[]]), np.array([])  # Returning empty arrays in case of empty lists.

        if exclude_low_sqi_win:
            masks = {pat: np.logical_and(self.mask_rr_dict[pat][self.window_size],
                                         self.signal_quality_dict[pat][self.window_size][
                                             self.sqi_test_ann] >= win_thresh) for pat in correct_pat_list}
        else:
            masks = {pat: self.mask_rr_dict[pat][self.window_size] for pat in correct_pat_list}

        X = database[:, :-3]
        indexes = np.concatenate(tuple(np.where(dataset_patients_list[:, 0] == int(p)) for p in correct_pat_list),
                                 axis=1)
        indexes = np.squeeze(indexes, axis=0)
        X = X[indexes]

        if return_binary:
            y = np.concatenate(
                tuple(self.af_win_lab_dict[elem][self.window_size][masks[elem]] for elem in correct_pat_list), axis=0)
        else:
            y = np.concatenate(
                tuple(self.win_lab_dict[elem][self.window_size][masks[elem]] for elem in correct_pat_list), axis=0)

        if return_global_label:
            global_lab = np.concatenate(
                tuple(self.af_pat_lab_dict[elem] * np.ones(np.sum(masks[elem])) for elem in correct_pat_list), axis=0)
            return X, y, global_lab
        else:
            return X, y

    def calculate_number_of_windows_in_pat_list(self, pat_list, win):
        n_of_windows = 0
        for p in pat_list:
            n_of_windows += self.number_of_windows(pat=p, win=win)
        return n_of_windows


if __name__ == "__main__":
    UVAF_W_db = UVAFDB_Wave_Parser(window_size=6000, load_ectopics=False, load_on_start=True)
    # UVAF_W_db.parse_raw_data(window_sizes=[6000], patient_list=['0614'], test_anns=['xqrs'])

    train_pat = np.load(UVAF_W_db.main_path / "UVAFDB_train_pat.npy", allow_pickle=True)
    val_pat = np.load(UVAF_W_db.main_path / "UVAFDB_val_pat.npy", allow_pickle=True)
    test_pat = np.load(UVAF_W_db.main_path / "UVAFDB_test_pat.npy", allow_pickle=True)

    pat_list = np.concatenate((train_pat, val_pat))
    pat_list = np.concatenate((pat_list, test_pat))

    num = UVAF_W_db.calculate_number_of_windows_in_pat_list(pat_list, win=6000)
    print('done')

    gender_train = []
    age_train = []
    for p in test_pat:
        g, a = UVAF_W_db.parse_demographic_features(p)
        gender_train.append(g)
        age_train.append(a)
    gender_train = np.array(gender_train)
    age_train = np.array(age_train)
    female = np.sum(gender_train)
    mean_age = np.mean(age_train)
    std_age = np.std(age_train)
    print('done')

    # short_train_pat_list = train_pat[:10]
    # short_test_pat_list = test_pat[:10]
    # short_val_pat_list = val_pat[:10]
    UVAF_W_db.parse_raw_data(window_sizes=[6000], patient_list=test_pat, reannotate=True)
    print(f'done val = {test_pat}')
    UVAF_W_db.parse_raw_data(window_sizes=[6000], patient_list=val_pat)
    print(f'done test = {val_pat}')
    UVAF_W_db.parse_raw_data(window_sizes=[6000], patient_list=train_pat)
    print(f'done train = {train_pat}')

    print('Done')

    ## From Shany example how to use the reannotation
    # ann_ids = next(os.walk(consts.REANNOTATION_DIR / (UVAF_W_db.name + '-annotated')))[1]
    # pat_list = pat_list[np.isin(pat_list, ann_ids)]
    # for pat in pat_list:
    #     print(pat)
    #     db.parse_elem_data(pat=pat, reannotated=True)
    #     db._win_lab(id=pat, win=window_size)
    #     db._af_win_lab(id=pat, win=window_size)
    #     db._af_pat_lab(id=pat)
    #     test_anns = np.setdiff1d(db.annotation_types, db.sqi_ref_ann)
    #     for ann_type in test_anns:  # Computing SQI
    #         db._sqi(pat, window_size, test_ann=ann_type)
