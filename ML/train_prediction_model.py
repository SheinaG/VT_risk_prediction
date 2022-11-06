import sys

sys.path.append("/home/sheina/armand_repo/")
sys.path.append("/home/sheina/armand_repo/sheina")
import sheina.bayesiansearch as bs
import train_model
import os
import sheina.consts as cts
import pathlib

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.metrics import confusion_matrix

typ = 4


def train_prediction_model(DATA_PATH, results_dir, model_type, dataset):
    print("loading data")
    algo = 'RF'
    tr_uv_p = 7
    tr_uv_n = 42
    ids_train, ids_test, y_train, y_test = train_model.split_ids(tr_uv_p, tr_uv_n)

    features_train, y_train, train_ids_groups = train_model.create_dataset(ids_train, y_train, path=DATA_PATH, model=0)
    features_train = train_model.model_features(features_train, model_type)
    train_groups = bs.split_to_group(train_ids_groups, cts.ids_VT[:tr_uv_p] + cts.ids_rbdb_VT,
                                     cts.ids_no_VT[:tr_uv_n] + cts.ids_rbdb_no_VT)
    opt = bs.vt_bayesianCV(features_train, y_train, algo, normalize=cts.normalize, groups=train_groups,
                           weighting=cts.weighting, n_jobs=10, typ=model_type, results_dir=results_dir, dataset=dataset)

    features_test, y_test, test_ids_groups = train_model.create_dataset(ids_test, y_test, path=DATA_PATH, model=0)
    features_test = train_model.model_features(features_test, model_type)
    print(confusion_matrix(y_test, opt.predict(features_test)))
    print(opt.score(features_test, y_test))


if __name__ == "__main__":
    DATA_PATH = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')
    results_dir = cts.RESULTS_DIR
    train_prediction_model(DATA_PATH, results_dir, model_type=1, dataset='bsqi')
