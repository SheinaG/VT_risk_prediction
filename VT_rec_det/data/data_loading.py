# General imports
import pickle

import h5py
import numpy as np
from data.data_loader import Generator
from sklearn.model_selection import train_test_split

# Relative imports
import utils.consts as consts


def stratified_split(db, pat_set, test_part):  # Precise that we removed the "only reann in test set" criteria !!
    """ Perform a multi-criteria stra§tified train-test split on a set of patients ids.
    It is stratified with respect to the AF severity label, the age and the gender.
    Because of how the sklearn `train_test_split` function is implemented, we need to handle the multi-criteria bins in which there is only one patient separately.

    :param db: the database on which we perform the split
    :param pat_set: a list of patients ids related to the database
    :param test_part: the float percentage of patients in the test set. Must be between 0 and 1.
    :return train_pat, test_pat: the resulting lists of patients ids, after split
    """
    # Extract stratification criteria
    AFsev_strat = np.array(list(db.af_pat_lab_dict[id] for id in pat_set))
    pat_set_age = np.array(list(db.features_dict[id][db.window_size]['Age'] for id in pat_set))
    age_strat = (pat_set_age > np.median(pat_set_age))
    gender_strat = np.array(list(db.features_dict[id][db.window_size]['Gender'] for id in pat_set))

    # Remove patients isolated in their multi-criteria bin, and add them to a separate list (sklearn train_test_split doesn't handle 1-sized bins)
    isolated_patients = []
    for AFsev in range(5):  # 5 AF severity categories : NonAF, AFmild, AFmoderate, AFsevere, otherCVD
        for age in range(2):  # 2 age categories : below and above the median age
            for gender in range(2):  # 2 gender categories : woman and man
                in_multi_bin = np.logical_and(np.logical_and(AFsev_strat == AFsev, age_strat == age),
                                              gender_strat == gender)
                if np.sum(in_multi_bin) == 1:  # if only one patient in the multi-criteria bin
                    pat_idx = np.where(in_multi_bin)[0][0]  # get its idx in the baseline list
                    isolated_patients.append(pat_set[pat_idx])  # add its id to isolated patients
                    AFsev_strat = np.delete(AFsev_strat,
                                            pat_idx)  # remove it from AFsev, age, gender and baseline before stratification
                    age_strat = np.delete(age_strat, pat_idx)
                    gender_strat = np.delete(gender_strat, pat_idx)
                    pat_set = np.delete(pat_set, pat_idx)
    isolated_patients = np.array(isolated_patients)

    # Split patients ids into train and test sets, with multi-criteria stratification
    train_pat, test_pat = train_test_split(pat_set, test_size=test_part,
                                           stratify=np.vstack((AFsev_strat, age_strat, gender_strat)).T,
                                           random_state=consts.SEED)

    # Randomly add isolated patients to train and test sets, weighted by the size of the sets
    if len(isolated_patients) > 0:
        print("Adding isolated patients")
        rd = np.random.random(len(isolated_patients))
        print("Patients added in first set : ", isolated_patients[np.where(rd <= 1 - test_part)[0]])
        train_pat = np.append(train_pat, isolated_patients[np.where(rd <= 1 - test_part)[0]])
        print("Patients added in second set : ", isolated_patients[np.where(rd > 1 - test_part)[0]])
        test_pat = np.append(test_pat, isolated_patients[np.where(rd > 1 - test_part)[0]])

    else:
        print("No isolated patient")

    # Plot the train-test distribution
    plot_train_test_distribution(db, train_pat, test_pat, np.median(pat_set_age))

    return train_pat, test_pat


def load_uvafdb(train_part=0.6, val_part=0.2, reann_pat_only=False, high_sqi_pat_only=True, over_18_pat_only=True,
                window_size=60, load_ectopics=False):
    """ Load UVAF database and perform a train-val-test split with multi-criteria stratification.

    :param train_part: the float percentage of patients in the training set. Must be between 0 and 1.
    :param val_part: the float percentage of patients in the val set. Must be between 0 and 1.
    :param reann_pat_only: a boolean indicating whether we want to exclude the non-reannotated patients
    :param high_sqi_pat_only: a boolean indicating whether we want to exclude the low sqi patients
    :param over_18_pat_only: a boolean indicating whether we want to exclude the <18 years old patients
    :param window_size: the size of the RR interval windows
    :param load_ectopics: a boolean indicating whether we want to load the number of ectopic beats for each patient
    :return UVAF_db: the parsed UVAF database
    :return train_pat, val_pat, test_pat: the lists of patients ids for each set, after split
    """

    # Extract UVAF DB reannotated and not-reannotated patients
    UVAF_db = UVAFDB_Parser(window_size=window_size, load_ectopics=load_ectopics)

    # Exclude low sqi and under 18 patients
    baseline = UVAF_db.non_corrupted_ecg_patients()
    if high_sqi_pat_only:
        baseline = np.intersect1d(baseline, UVAF_db.high_sqi_patients())
    if over_18_pat_only:
        baseline = np.intersect1d(baseline, UVAF_db.over_18_patients)

    # Keep re-annotated patients only
    if reann_pat_only:
        UVAF_db.extract_reann_pat()
        reann_pat = UVAF_db.reann_pat
        baseline = np.intersect1d(baseline, reann_pat)

    print("Train-Test split")
    train_val_pat, test_pat = stratified_split(UVAF_db, baseline,
                                               test_part=int(len(baseline) * (1 - (train_part + val_part))))
    print("Train-Val split")
    train_pat, val_pat = stratified_split(UVAF_db, train_val_pat, test_part=int(
        len(train_val_pat) * (1 - (train_part / (train_part + val_part)))))

    return UVAF_db, train_pat, val_pat, test_pat


def return_data(db, pat_list):
    """ Load the patients features and concatenate them for the model input.

    :param db: the parsed database
    :param pat_list: the list of training patients ids
    :return: the rr data and xgb features
    """

    # Get training variables
    ids = db.return_patient_ids(pat_list=pat_list)
    rr, y, glob_lab = db.return_rr(pat_list=pat_list, return_binary=True, return_global_label=True)
    prec = db.return_preceeding_windows(pat_list=pat_list)
    feats, _ = db.return_features(pat_list=pat_list, feats_list=np.append(consts.SELECTED_FEATURES, 'sqi'))

    # Fill NA
    feats, mean = dp.fillna(feats)

    # Concatenate the features
    X = np.concatenate((rr, prec.reshape(-1, 1), glob_lab.reshape(-1, 1), ids.reshape(-1, 1)), axis=1)

    return X, y, feats


def load_data(algo):
    """ A general script to load the different datasets used for training and evaluation

    :param algo: a string representing the model's name
    :return data : a tuple of datasets, each dataset being a tuple (X, y), with additionally timestamps for the RR windows
    """
    db = UVAFDB_Parser(window_size=60, load_ectopics=False)
    train_pat = np.load(db.main_path / "train_pat.npy", allow_pickle=True)
    val_pat = np.load(db.main_path / "val_pat.npy", allow_pickle=True)
    test_pat = np.load(db.main_path / "test_pat_reann.npy", allow_pickle=True)
    X_train, y_train, feats_train = return_data(db, train_pat)
    X_val, y_val, feats_val = return_data(db, val_pat)
    X_test, y_test, feats_test = return_data(db, test_pat)
    ltaf_db = LTAFDB_Parser(window_size=60)
    ltaf_pat = np.intersect1d(ltaf_db.non_corrupted_ecg_patients(), ltaf_db.high_sqi_patients())
    X_ltaf, y_ltaf, feats_ltaf = return_data(ltaf_db, ltaf_pat)
    X_jpaf, y_jpaf, t_s_jpaf = pickle.load(open("/MLdata/AIMLab/Tom/JPAFDB_input.pickle", "rb"))
    feats_jpaf = pickle.load(open("/MLdata/AIMLab/Tom/JPAFDB_xgboost_input.pickle", "rb"))
    X_afdb, y_afdb, t_s_afdb = pickle.load(open("/MLdata/AIMLab/Tom/AFDB_input.pickle", "rb"))
    feats_afdb = pickle.load(open("/MLdata/AIMLab/Tom/AFDB_xgboost_input.pickle", "rb"))
    if algo == "XGB":
        data = (np.concatenate((feats_train, X_train[:, -3:]), axis=1), y_train), \
               (np.concatenate((feats_val, X_val[:, -3:]), axis=1), y_val), \
               (np.concatenate((feats_test, X_test[:, -3:]), axis=1), y_test), \
               (np.concatenate((feats_ltaf, X_ltaf[:, -3:]), axis=1), y_ltaf), \
               (np.concatenate((feats_jpaf, X_jpaf[:, -3:]), axis=1), y_jpaf, t_s_jpaf), \
               (np.concatenate((feats_afdb, X_afdb[:, -3:]), axis=1), y_afdb,
                t_s_afdb)  # Rk : we add the 3 "patient features" for homogeneity of notations
    else:
        data = (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_ltaf, y_ltaf), (X_jpaf, y_jpaf, t_s_jpaf), (
        X_afdb, y_afdb, t_s_afdb)  # (index by [11500:11500+10*1024] to get a sped up version for debugging)

    return data


def return_data_wave(db, pat_list, hdf5file=None):
    """ Load the patients features and concatenate them for the model input.

    :param db: the parsed database
    :param pat_list: the list of training patients ids
    :return: the ecg raw data
    """

    # Get training variables
    ids = db.return_patient_ids(pat_list=pat_list)
    ecg, y, glob_lab = db.return_ecg_data(hdf5file=hdf5file, pat_list=pat_list, return_binary=True,
                                          return_global_label=True)
    prec = db.return_preceeding_windows(pat_list=pat_list)
    # feats, _ = db.return_features(pat_list=pat_list, feats_list=np.append(consts.SELECTED_FEATURES, 'sqi')) #TODO

    # Fill NA
    # feats, mean = dp.fillna(feats)

    # Concatenate the features
    X = np.concatenate((ecg, prec.reshape(-1, 1), glob_lab.reshape(-1, 1), ids.reshape(-1, 1)), axis=1)

    return X, y, None  # feats #TODO add features


def correct_patient_list(hdf5file, pat_list):
    f = h5py.File(hdf5file, "r")
    dataset_patients_list = f[consts.HDF5_DATASET][:, -3:-1].astype(int)

    correct_pat_list = []
    for elem in pat_list:
        if int(elem) in dataset_patients_list[:, 0]:
            correct_pat_list.append(elem)
    return correct_pat_list


def load_data_wave(number_of_samples, DEBUG=False):
    db = UVAFDB_Wave_Parser(window_size=number_of_samples, load_on_start=True, load_ectopics=False)
    train_pat = np.load(db.main_path / "UVAFDB_train_pat.npy", allow_pickle=True)
    val_pat = np.load(db.main_path / "UVAFDB_val_pat.npy", allow_pickle=True)
    test_pat = np.load(db.main_path / "UVAFDB_test_pat.npy", allow_pickle=True)

    val_pat = correct_patient_list(consts.UVAF_HDF5_VAL, val_pat)
    train_pat = correct_patient_list(consts.UVAF_HDF5, train_pat)
    test_pat = correct_patient_list(consts.UVAF_HDF5_TEST, test_pat)

    X_test, y_test, feats_test = return_data_wave(db, test_pat, hdf5file=consts.UVAF_HDF5_TEST)
    print('got test')
    X_train, y_train, feats_train = return_data_wave(db, train_pat, hdf5file=consts.UVAF_HDF5)
    print('got train')
    X_val, y_val, feats_val = return_data_wave(db, val_pat, hdf5file=consts.UVAF_HDF5_VAL)
    print('got val')

    data = (X_train, y_train), (X_val, y_val), (X_test, y_test)
    return data


def plot_train_test_distribution(UVAF_db, train_pat, test_pat, median_age):
    import matplotlib.pyplot as plt
    AFsev_train = list(UVAF_db.af_pat_lab_dict[id] for id in train_pat)
    AFsev_test = list(UVAF_db.af_pat_lab_dict[id] for id in test_pat)
    plt.hist(AFsev_train, label='train')
    plt.hist(AFsev_test, label='test')
    plt.legend()
    plt.title(
        f'AFsev train-test distribution : {100 * AFsev_train.count(0) / (AFsev_train.count(0) + AFsev_test.count(0)):.1f} - {100 * AFsev_train.count(1) / (AFsev_train.count(1) + AFsev_test.count(1)): .1f} - {100 * AFsev_train.count(2) / (AFsev_train.count(2) + AFsev_test.count(2)):.1f} - {100 * AFsev_train.count(3) / (AFsev_train.count(3) + AFsev_test.count(3)):.1f} - {100 * AFsev_train.count(4) / (AFsev_train.count(4) + AFsev_test.count(4)):.1f}')
    plt.show()
    age_train = list(
        np.array(list(UVAF_db.features_dict[id][UVAF_db.window_size]['Age'] for id in train_pat)) > median_age)
    age_test = list(
        np.array(list(UVAF_db.features_dict[id][UVAF_db.window_size]['Age'] for id in test_pat)) > median_age)
    plt.hist(list(UVAF_db.features_dict[id][UVAF_db.window_size]['Age'] for id in train_pat), label='train')
    plt.hist(list(UVAF_db.features_dict[id][UVAF_db.window_size]['Age'] for id in test_pat), label='test')
    plt.vlines(median_age, 0, 500, colors='red', linestyles='--', label='median')
    plt.legend()
    plt.title(
        f'Age train-test distribution : {100 * age_train.count(0) / (age_train.count(0) + age_test.count(0)):.1f} - {100 * age_train.count(1) / (age_train.count(1) + age_test.count(1)): .1f}')
    plt.show()
    gender_train = list(UVAF_db.features_dict[id][UVAF_db.window_size]['Gender'] for id in train_pat)
    gender_test = list(UVAF_db.features_dict[id][UVAF_db.window_size]['Gender'] for id in test_pat)
    plt.hist(gender_train, label='train')
    plt.hist(gender_test, label='test')
    plt.legend()
    plt.title(
        f'Gender train-test distribution : {100 * gender_train.count(0) / (gender_train.count(0) + gender_test.count(0)):.1f} - {100 * gender_train.count(1) / (gender_train.count(1) + gender_test.count(1)): .1f}')
    plt.show()


def create_generator(db, hdf5file, window_size, patient_list, batch_size, return_binary, return_global_label,
                     shuffle=True):
    f = h5py.File(hdf5file, "r")
    database = f[consts.HDF5_DATASET]
    # database_patient_list = database.attrs[consts.HDF5_PATIENT_IDS]

    train_loader = Generator(db_parser=db,
                             window_size=window_size,
                             dataset=database,
                             patients=patient_list,
                             # dataset_patients_list=database_patient_list,
                             batch_size=batch_size,
                             return_binary=return_binary,
                             return_global_label=return_global_label,
                             shuffle=shuffle)
    print(f'Debug train_loader.len = {len(train_loader)}')
    return train_loader


def create_data_loader(number_of_samples, bs, return_binary, return_global_label=True, shuffle=True, debug=False):
    db = UVAFDB_Wave_Parser(window_size=number_of_samples, load_on_start=True, load_ectopics=False)
    train_pat = np.load(db.main_path / "UVAFDB_train_pat.npy", allow_pickle=True)
    val_pat = np.load(db.main_path / "UVAFDB_val_pat.npy", allow_pickle=True)
    test_pat = np.load(db.main_path / "UVAFDB_test_pat.npy", allow_pickle=True)

    if consts.UVAF_HDF5 == consts.PREPROCESSED_DATA_DIR / "uvaf_train_indexes_normalized.hdf5":
        train_pat = train_pat[:1000]

    val_pat = correct_patient_list(consts.UVAF_HDF5_VAL, val_pat)
    train_pat = correct_patient_list(consts.UVAF_HDF5, train_pat)

    test_pat = correct_patient_list(consts.UVAF_HDF5_TEST, test_pat)
    if debug:
        test_pat = test_pat[:2]
    # if debug:
    #     print(f' Debug = {debug}')
    #     n_of_pat = 100
    #     healthy_patients = []
    #     af_patients = []
    #     for p in train_pat:
    #         if len(healthy_patients) >= n_of_pat/2 and len(af_patients) >= n_of_pat/2:
    #             break
    #         if db.af_burden_dict[p] < 0.9:
    #              healthy_patients.append(p)
    #         else:
    #             af_patients.append(p)
    #     af_patients_used = af_patients[:int(n_of_pat/2)] if len(af_patients) >= int(n_of_pat/2) else af_patients
    #     train_pat = healthy_patients[:n_of_pat-len(af_patients_used)] + af_patients_used
    # train_pat = healthy_patients[:40] + af_patients[:40]
    # val_pat = healthy_patients[40:] +af_patients[40:]
    # train_pat = ['2144', '0397'] #train_pat[:2] 0197 0397
    # val_pat = ['2144', '0397']
    # test_pat = ['2144', '0397']
    # print(f'test_pat = {test_pat}')
    # print(f'val_pat = {val_pat}')
    # print(f'train_pat = {train_pat}')

    train_loader = create_generator(db, consts.UVAF_HDF5,
                                    window_size=number_of_samples,
                                    patient_list=train_pat,
                                    batch_size=bs,
                                    return_binary=return_binary,
                                    return_global_label=return_global_label,
                                    shuffle=shuffle)

    val_loader = create_generator(db, consts.UVAF_HDF5_VAL,
                                  window_size=number_of_samples,
                                  patient_list=val_pat,
                                  batch_size=bs,
                                  return_binary=return_binary,
                                  return_global_label=return_global_label,
                                  shuffle=False)

    test_loader = create_generator(db, consts.UVAF_HDF5_TEST,
                                   window_size=number_of_samples,
                                   patient_list=test_pat,
                                   batch_size=bs,
                                   return_binary=return_binary,
                                   return_global_label=return_global_label,
                                   shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # check the split files
    db = UVAFDB_Parser(window_size=60, load_ectopics=False)
    train_pat = np.load(db.main_path / "train_pat.npy", allow_pickle=True)
    val_pat = np.load(db.main_path / "val_pat.npy", allow_pickle=True)
    test_pat = np.load(db.main_path / "test_pat_reann.npy", allow_pickle=True)
    print("Done")
    # #creat files with split train,validation,test patients
    # db, train_pat_list, val_pat_list, test_pat_list = load_uvafdb(reann_pat_only=True)
    # np.save(db.main_path / "train_pat.npy",train_pat_list,allow_pickle=True)
    # np.save(db.main_path / "val_pat.npy",val_pat_list,allow_pickle=True)
    # np.save(db.main_path / "test_pat_reann.npy",test_pat_list,allow_pickle=True)
