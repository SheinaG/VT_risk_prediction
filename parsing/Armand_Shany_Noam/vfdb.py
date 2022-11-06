import numpy as np
import wfdb

from vfdb_parser import VFDB_Parser

R_VT = 14
dl_dir = '/MLdata/AIMLab/Sheina/databases/vfdb/'
db = VFDB_Parser(load_on_start=False)
ids_vt = []
ids = db.parse_available_ids()
num_leads = np.ndarray(shape=[len(ids), 1])

for i, id in enumerate(ids):
    ann, rhythm = db.parse_reference_annotation(str(id))
    print(set(rhythm))

    record = wfdb.rdrecord(dl_dir + str(id))
    num_leads[i] = np.shape(record.p_signal)[1]

a = 5
