import sys
import h5py
import numpy as np
import utils.consts as cts
from parsing.base_VT_parser import VtParser


def find_relenant_indexes(indexes, p_list):
    first_index = min(np.squeeze(np.where(indexes[:, 0] == p_list[0])))
    last_index = max(np.squeeze(np.where(indexes[:, 0] == p_list[-1])))
    r_indexes = indexes[first_index:last_index + 1, :]
    return r_indexes


def read_hdf5_file(filename, attr_name=None):
    with h5py.File(filename, 'r') as f:
        data = f[cts.HDF5_DATASET]
        # patients = data.attrs[attr_name]
        print('Done')


#######################################################################################################################
# Create dataset ordered by (patient id, window index)
#######################################################################################################################

def calc_index_tuples(db_parser, patient_list):
    masks, lens = db_parser.return_masks_dict(patient_list)
    num_duplicates = {pat: np.sum(masks[pat]) for pat in masks.keys()}
    # window_indexes = np.concatenate(tuple(np.tile(range(60),num_duplicates[elem]) for elem in patient_list ), axis=0)
    bsqi_window_indexes = np.concatenate(tuple(range(num_duplicates[elem] * 180) for elem in patient_list), axis=0)
    patients_duplicate = np.concatenate(
        tuple(elem * np.ones(num_duplicates[elem] * 180, dtype=object) for elem in patient_list), axis=0)
    index_tuples = np.concatenate(
        (np.expand_dims(patients_duplicate, axis=1), np.expand_dims(bsqi_window_indexes, axis=1)), axis=1)
    return index_tuples


def create_data_by_indexes(db_parser, patient_list, part_pat):
    '''
    :param db_parser: database parser that has all the information about each window, for example if it needs to be masked or not
    :param patient_list: list of patients to add to the database
    :param window_size: the size of each window
    :return: This function will create data that is organized as num_of_windows X (window_size + patient id +window  index)
    '''

    masks, lens = db_parser.return_masks_dict(patient_list)
    ecg_data = None

    for idx, p in enumerate(patient_list):
        print(f'{idx}: {p}')
        ecg = db_parser.parse_pp_rec(p)
        bsqi_window_size = db_parser.bsqi_window_size
        if len(ecg) < bsqi_window_size:
            print(p)
            continue

        window_size = db_parser.window_size
        ecg = ecg[:lens[p] * bsqi_window_size]
        ecg = ecg[:(len(ecg) // bsqi_window_size) * bsqi_window_size].reshape(-1, bsqi_window_size)[masks[p]]
        if part_pat:
            ecg_split = ecg[0:, :].reshape(-1, window_size)
        else:
            ecg_split = ecg.reshape(-1, window_size)
        if ecg_data is None:
            ecg_data = ecg_split
        else:
            ecg_data = np.concatenate((ecg_data, ecg_split), axis=0)
    return ecg_data


def create_hdf5_win_size(filename, patient_list, db_parser, indexes, part_pat):
    '''
    creates a database file o
    :param filename: name of hdf5 file to create
    :param patient_list: patients to add to data
    :param db_parser: database parser that has all the information about each window, for example if it needs to be masked or not
    '''
    # assert  os.path.isfile(filename)==False, f"file already exists need to delete it"
    ecg_data = create_data_by_indexes(db_parser, patient_list, part_pat)
    r_indexes = find_relenant_indexes(indexes, patient_list)
    assert len(r_indexes) == len(
        ecg_data), f'len(indexes)  = {len(r_indexes)} len(ecg_data) = len(ecg_data) indexes and data should be the same length otherwise there is a winodw we don\'t know where it belongs '
    ecg_data = ecg_data.astype('float64')

    with h5py.File(filename, 'a') as hdf5_file:
        hdf5_file.create_dataset(name=cts.HDF5_DATASET, data=ecg_data, maxshape=(None, None))
        hdf5_file.flush()
        hdf5_file.close()


def check_no_duplicates(data, patient_list):
    """
    :param data: database that is already in the file, executes is to be organized as num_of_windows X (window_size + patient id +window  index)
    :param patient_list:
    :return:
    """
    for p in patient_list:
        exists = np.where(data[:, -3].astype(int) == int(p))
        assert len(exists[0]) == 0, f' patient {p} already exists in database'


def get_labels(db_parser, patient_list):
    masks, lens = db_parser.return_masks_dict(patient_list)
    num_duplicates = {pat: np.sum(masks[pat]) for pat in masks.keys()}
    labels = db_parser.get_labels(patient_list)
    labels = np.concatenate(tuple(np.ones(num_duplicates[elem] * 180) * labels[elem] for elem in patient_list), axis=0)
    return labels


def create_files(data_filename, idx_filename, label_filename, list_ids, db, max_ids_to_add, part_pat=False):
    indexes = calc_index_tuples(db, list_ids).astype(str)
    labels = get_labels(db, list_ids)
    if len(list_ids) > max_ids_to_add:
        create_hdf5_win_size(data_filename, list_ids[:max_ids_to_add], db, indexes, part_pat)
    else:
        create_hdf5_win_size(data_filename, list_ids, db, indexes, part_pat)
    np.save(idx_filename, indexes)
    np.save(label_filename, labels)


def add_data(db, data_filename, list_ids, range, part_pat):
    patient_list = list_ids[range[0]:range[1]]
    ecg_data = create_data_by_indexes(db, patient_list, part_pat=False)
    with h5py.File(data_filename, 'a') as hdf5_file:
        hdf5_file[cts.HDF5_DATASET].resize((hdf5_file[cts.HDF5_DATASET].shape[0] + len(ecg_data)), axis=0)
        hdf5_file[cts.HDF5_DATASET][-len(ecg_data):] = ecg_data


if __name__ == '__main__':
    task = 'train_part'  # 'test'
    data_filename = '/MLAIM/AIMLab/Sheina/databases/VTdb/DL/train/' + str("data_" + task + ".hdf5")
    idx_filename = '/MLAIM/AIMLab/Sheina/databases/VTdb/DL/train/' + str("idx_" + task + ".npy")
    label_filename = '/MLAIM/AIMLab/Sheina/databases/VTdb/DL/train/' + str("labels_" + task + ".npy")
    # assert os.path.isfile(filename) == False, f"file already exists need to delete it"
    db = VtParser()
    df = db.parse_ids(task=task)
    max_ids_to_add = 150
    if len(df['holter_id']) > max_ids_to_add:
        create_files(data_filename, idx_filename, label_filename, list(df['holter_id']), db, max_ids_to_add,
                     part_pat=False)
        add_n_times = len(df['holter_id']) // max_ids_to_add
        for j in range(0, add_n_times):
            start_id = max_ids_to_add * (j + 1)
            stop_id = max_ids_to_add * (j + 2)
            add_data(db, data_filename, list(df['holter_id']), range=[start_id, stop_id], part_pat=False)
        print(f' done  adding data to {data_filename}')
    else:
        create_files(data_filename, idx_filename, label_filename, list(df['holter_id']), db, max_ids_to_add)
        print(f' done  adding data to {data_filename}')
