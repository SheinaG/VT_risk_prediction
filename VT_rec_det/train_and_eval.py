import sys

sys.path.append('/home/b.noam/')
from Noam_Repo.AF_DETECT.utils.helper_functions import install

install(["pandas"], quietly=True)

# General imports
import numpy as np
import pandas as pd
import sklearn.utils.class_weight as skl_cw
import tensorflow as tf
import os

# Relative imports
import sys

sys.path.append('/home/b.noam/')
import Noam_Repo.AF_DETECT.data.data_loading as data_loading
# import data.data_augmentation as da
import Noam_Repo.AF_DETECT.utils.consts as consts
import Noam_Repo.AF_DETECT.models.metrics as metrics
import Noam_Repo.AF_DETECT.models.utils as model_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train_with_generators(train_loader, val_loader, hypercomb, algo, n_epochs, checkpoint_filename, workers,
                          class_weights, path_feature_extractor=None) -> object:
    tf.compat.v1.keras.backend.clear_session()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # Model instantiation
    if algo == "ArNet2":
        model = model_utils.class_funcs[algo](**hypercomb, path_feature_extractor=path_feature_extractor)
    else:
        model = model_utils.class_funcs[algo](**hypercomb)

    # Model training
    if model_utils.use_history[algo]:
        model.fit_gen(train_data=train_loader, validation_data=val_loader)
        # if task == 'inprogress':
        #     model.fit_gen(train_data= train_loader, validation_data=val_loader)
        # else:
        #     X,y =train_loader
        #     model.fit(X,y, validation_data = val_loader,  n_epochs=n_epochs)
    else:
        model.fit_gen(checkpoint_filename=checkpoint_filename,
                      n_epochs=n_epochs,
                      train_data=train_loader,
                      validation_data=val_loader,
                      verbose=1,
                      workers=workers,
                      class_weights=class_weights)
    return model


def train(data_train, hypercomb, algo="1D-CNN", n_epochs=5, validation_data=None, path_feature_extractor=None):
    """ Instantiate and train the chosen model on training data.

    :param data_train: the training data, being a tuple (X_train, y_train)
    :param hypercomb: the dictionary of model's hyperparameters
    :param algo: a string representing the model's name
    :param n_epochs: the number of epochs for training
    :param validation_data: the optional validation data, for the learning curves, being a tuple (X_val, y_val)
    :param path_feature_extractor: if the algo is "ArNet2", the full path of the feature extractor, being either a 1D-CNN or a ResNet
    :return model: the trained ML model
    """

    # Model instantiation
    if algo == "XGB":
        model = model_utils.class_funcs[algo](**hypercomb, n_jobs=-1)
    elif algo == "ArNet2":
        model = model_utils.class_funcs[algo](**hypercomb, path_feature_extractor=path_feature_extractor)
    else:
        model = model_utils.class_funcs[algo](**hypercomb)

    # Model training
    X_train, y_train = data_train
    if model_utils.use_history[algo]:
        model.fit(X_train, y_train, validation_data=validation_data, n_epochs=n_epochs)
    else:
        rr_train = X_train[:, :-3].astype('float32')
        if validation_data is not None:
            validation_data = (validation_data[0][:, :-3].astype('float32'), validation_data[1])
        if algo == "XGB":
            mean_train, std_train = np.mean(rr_train, axis=0), np.std(rr_train, axis=0)
            rr_train = (rr_train - mean_train) / std_train
            eval_set = [(rr_train, y_train)]
            if validation_data is not None:
                validation_data = ((validation_data[0] - mean_train) / std_train, validation_data[1])
                eval_set.append(validation_data)
            model.fit(rr_train, y_train, sample_weight=skl_cw.compute_sample_weight("balanced", y_train),
                      eval_set=eval_set, eval_metric='logloss', verbose=True)
        else:
            model.fit(rr_train, y_train, validation_data=validation_data, n_epochs=n_epochs)

    return model


def eval(model, algo="1D-CNN", decision_th=None, save=False, path_models=None, hypercomb=None, error_analysis=False,
         **eval_sets):  # Rk : train must be in eval sets, and the first one
    """ Evaluate the trained model on all the input evaluation sets, and optionally save them and perform error analysis.

    :param model: the trained ML model
    :param algo: a string representing the model's name
    :param decision_th: the float decision threshold chosen for the classifier. If None, it is computed to maximize the F1 score on the training set.
    :param save: a boolean to indicate whether we want to save the model
    :param path_models: the path of the folder in which we want to save the model
    :param hypercomb: the dictionary of model's hyperparameters (for saving)
    :param error_analysis: a boolean to indicate whether we want to perform error analysis
    :param eval_sets: a dictionary of datasets of the form (X, y), and possibly a third element t_s representing the timestamps of the windows
    :return metrics_dict: a dictionary of performance metrics, for each dataset
    :return best_th_dict: a dictionary of optimal decision thresholds, for each dataset
    :return mean_abs_afb_error_dict:  a dictionary of mean AFB error, for each dataset
    """

    # Initialize the results dictionaries
    metrics_dict, best_th_dict, mean_abs_afb_error_dict = {}, {}, {}
    model_dict = {'hyperparameters': hypercomb, 'best_th': decision_th}

    # Loop over the different sets to evaluate, and add their results to output dictionaries
    for set_name, set_data in eval_sets.items():

        # Predict probas
        X, y = set_data[:2]
        if model_utils.use_history[algo]:
            probas = model.predict_proba(X)[:, 1]
        else:
            rr = X[:, :-3].astype('float32')
            if algo == "XGB":
                if "train" in set_name:
                    mean_train, std_train = np.mean(rr, axis=0), np.std(rr, axis=0)
                try:
                    rr = (rr - mean_train) / std_train
                except NameError:
                    raise NameError(
                        "mean_train undefined. You have to put the training set as the first element of the eval_sets dict")
            probas = model.predict_proba(rr)[:, 1]

        # Compute metrics
        best_th = metrics.maximize_f_beta(probas, y)
        best_th_dict[set_name] = best_th
        if (decision_th is None) and (
                "train" in set_name):  # if the decision threshold is not given, set it to the optimal decision threshold on train
            decision_th = best_th_dict['train']
            model_dict['best_th'] = decision_th
        metrics_dict[set_name] = metrics.model_metrics(probas, y, probas > decision_th, print_metrics=True)
        mean_abs_afb_error_dict[set_name] = metrics.mean_abs_afb_error(X, y, probas,
                                                                       decision_th)  # takes a lot of time, so maybe we should comment it for now before making it faster

        # Update model_dict
        model_dict[f'metrics_{set_name}'] = dict(zip(consts.METRICS, metrics_dict[set_name]))
        model_dict[f'metrics_{set_name}']['mean_abs_afb_error'] = mean_abs_afb_error_dict[set_name]

        # Perform error analysis and save the related dataframe
        if error_analysis:
            df = pd.DataFrame(columns=['id', 'proba', 'decision_th', 'pred', 'lab', 'start_time', 'end_time'])
            df['id'] = X[:, -1]
            df['proba'] = probas
            df['decision_th'] = decision_th
            df['pred'] = (probas > decision_th)
            df['lab'] = y
            if len(set_data) == 3:
                ts = set_data[2]
                df['start_time'] = ts[:, 0]
                df['end_time'] = ts[:, -1]
            model_utils.save_df(df, consts.NOAM_DIR + f"/{set_name}_pred.csv")

    # Save model and results
    if save:
        model_utils.save_model(model_dict, model, path_models / f"{algo}.pkl", algo)

    return metrics_dict, best_th_dict, mean_abs_afb_error_dict


def eval_wave(model, batch_size, algo="1D-CNN", decision_th=None, save=False, path_models=None, hypercomb=None,
              error_analysis=False, **eval_sets):  # Rk : train must be in eval sets, and the first one
    """ Evaluate the trained model on all the input evaluation sets, and optionally save them and perform error analysis.

    :param model: the trained ML model
    :param algo: a string representing the model's name
    :param decision_th: the float decision threshold chosen for the classifier. If None, it is computed to maximize the F1 score on the training set.
    :param save: a boolean to indicate whether we want to save the model
    :param path_models: the path of the folder in which we want to save the model
    :param hypercomb: the dictionary of model's hyperparameters (for saving)
    :param error_analysis: a boolean to indicate whether we want to perform error analysis
    :param eval_sets: a dictionary of datasets of the form (X, y), and possibly a third element t_s representing the timestamps of the windows
    :return metrics_dict: a dictionary of performance metrics, for each dataset
    :return best_th_dict: a dictionary of optimal decision thresholds, for each dataset
    :return mean_abs_afb_error_dict:  a dictionary of mean AFB error, for each dataset
    """

    # Initialize the results dictionaries
    metrics_dict, best_th_dict, mean_abs_afb_error_dict = {}, {}, {}
    model_dict = {'hyperparameters': hypercomb, 'best_th': decision_th}

    if save:  # Save model before other things work, once all the evalutaion flow works then you can remove this line
        model_utils.save_model(model_dict, model, path_models / f"{algo}.pkl", algo)

    # Loop over the different sets to evaluate, and add their results to output dictionaries
    for set_name, data_loader in eval_sets.items():
        print(f'%%%%%%%%%%%%%%%%%%%%% set name = {set_name}%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        if set_name == 'test' or set_name == 'val' or set_name == 'train':
            # Predict probas
            if algo == 'ResNet':
                probas = model.model.predict(data_loader, batch_size=batch_size, verbose=1, workers=8,
                                             use_multiprocessing=True)  # steps = len(data_loader))
                assert data_loader.shuffle == False, f'data_loader.shuffle ={data_loader.shuffle} and for labels to be true the shuffle needs to be false'
                y = data_loader.subset_labels
            if algo == 'ArNet2':
                # probas = model.predict_proba(data_loader[0])[:, 1]
                # y =data_loader[1]

                probas = model.predict_gen(data_loader)[:, 1]
                assert data_loader.shuffle == False, f'data_loader.shuffle ={data_loader.shuffle} and for labels to be true the shuffle needs to be false'
                y = data_loader.subset_labels
            X = data_loader.dataset[:, :-3]
            X = X[:len(data_loader.subset_labels)]
            ids = data_loader.ids
            start_time = data_loader.start_time
        else:
            X = data_loader[0].astype('float32')  # np.asarray(data_loader[0]).astype('float32')
            y = data_loader[1]
            ids = X[:, -1].astype(int)
            X = X[:, :-3]
            if algo == 'ResNet':
                probas = model.model.predict(X, batch_size=batch_size, verbose=1, workers=8,
                                             use_multiprocessing=True)  # steps = len(data_loader))
            elif algo == 'ArNet2':
                probas = model.predict_gen(data_loader)[:, 1]

        # Compute metrics
        best_th = metrics.maximize_f_beta(probas, y)
        print(f' best_th = {best_th}')
        best_th_dict[set_name] = best_th
        if (decision_th is None) and (
                "train" in set_name):  # if the decision threshold is not given, set it to the optimal decision threshold on train
            decision_th = best_th_dict['train']
            model_dict['best_th'] = decision_th
        elif (decision_th is None):
            decision_th = 0.5
            print(f'used hardcoded decision_th = {decision_th}')

        metrics_dict[set_name] = metrics.model_metrics(probas, y, probas > decision_th, print_metrics=True)
        mean_abs_afb_error_dict[set_name] = metrics.mean_abs_afb_error(X, y, probas, decision_th,
                                                                       ids=ids)  # takes a lot of time, so maybe we should comment it for now before making it faster

        # Update model_dict
        model_dict[f'metrics_{set_name}'] = dict(zip(consts.METRICS, metrics_dict[set_name]))
        model_dict[f'metrics_{set_name}']['mean_abs_afb_error'] = mean_abs_afb_error_dict[set_name]
        print(f'mean_abs_afb_error  = {mean_abs_afb_error_dict[set_name]}')
        # Perform error analysis and save the related dataframe
        if error_analysis:
            df = pd.DataFrame(columns=['id', 'proba', 'decision_th', 'pred', 'lab', 'start_time', 'end_time'])
            df['id'] = ids
            df['proba'] = probas
            df['decision_th'] = decision_th
            df['pred'] = (probas > decision_th)
            df['lab'] = y
            df['start_time'] = start_time
            df['end_time'] = start_time + 30  # todo change this to a consts or calcuate it
            model_utils.save_df(df, path_models / f"{set_name}_pred.csv")

    # Save model and results
    if save:
        model_utils.save_model(model_dict, model, path_models / f"{algo}.pkl", algo)

    return metrics_dict, best_th_dict, mean_abs_afb_error_dict


if __name__ == '__main__':
    data_type = "wave"
    algo = "ResNet"
    task = "train"
    folder_date = np.datetime64(
        'now')  # '2022-03-02T21:46:57_ArNet2' #ArNet '2022-02-20T07:23:26' #ResNet '2022-02-28T12:52:56' #'2021-12-29T09:57:30' #np.datetime64('now') #/
    path_models = cts.NOAM_DIR / "models" / str(folder_date)  # "models" / "hyperparameter_tuning"

    VT_w = 4
    DEBUG = False
    win_size = 6000
    batch_size = 128
    shuffle = False if task == 'eval' else True

    train_loader, val_loader, test_loader = data_loading.create_data_loader(number_of_samples=win_size, bs=batch_size,
                                                                            return_binary=True,
                                                                            return_global_label=False, shuffle=shuffle,
                                                                            debug=DEBUG)
    class_weights = {0: 1, 1: VT_w}
    model = train_with_generators(train_loader, val_loader, hypercomb[algo], algo, n_epochs=4 if DEBUG else 2,
                                  checkpoint_filename=path_models / f"{algo}_best_model.pkl", workers=num_of_workers,
                                  class_weights=class_weights)
