# General imports
import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# Relative imports
from Noam_Repo.AF_DETECT.parsing.uvafdb_wave_parser import UVAFDB_Wave_Parser
from matplotlib.cm import get_cmap

sys.path.append('/home/b.noam/')
import Noam_Repo.AF_DETECT.utils.consts as consts
import Noam_Repo.AF_DETECT.models.utils as model_utils

font = {'weight': 'normal',
        # 'family' : 'normal',
        'size': 22}
matplotlib.rc('font', **font)
cmap = get_cmap("tab10")
colors = cmap.colors


def plot_performance_bars(models, db_metrics, db_list, score, width=0.1, savefig=False, savedir=None):
    """
    plot performace distriubtion.
    :param models: models to plot
    :param db_metrics: list of dataset metrics to plot
    :param db_list: list of x-labels
    :param score: performance score to use (example: Fb-score)
    :param width: the width of the bars
    :param savefig: True/False
    :param savedir: full path to save location
    :return: None
    """
    x = np.arange(len(models))  # the label locations
    fig, ax = plt.subplots(figsize=(10, 8))
    for k, model in enumerate(models):
        ([ax.bar(k + db_set * width, 100 * round(metric_dict[model][db_set][score], 3), width, label=set_lab,
                 color=colors[db_set]) for
          (db_set, set_lab) in zip(np.arange(len(db_metrics)), db_list)])
        # ax.bar_label(rects, padding=3)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(score)
    ax.set_title('{} by architecture'.format(score))
    ax.set_xticks(x - (width / 2) + (width * len(models)) / 2)
    ax.set_xticklabels(models)
    ax.legend(db_list, loc='upper left')
    score_values = []
    for container in ax.containers:
        ax.bar_label(container, padding=3)
        score_values.append(container.datavalues[0])
    fig.tight_layout()
    plt.ylim(min(score_values) - 1, max(score_values) + 2)
    if savefig:
        plt.tight_layout()
        lab = score + 'distribution.png'
        plt.savefig(str(savedir) + "/" + lab, dpi=400, transparent=True)
    return


def plot_performance_dot(models, db_metrics, db_list, db_list_plot, score, savefig=False, savedir=None):
    x = np.tile(np.arange(len(db_metrics)) + 1, (len(models), 1))  # the label locations
    y = np.empty(shape=[len(models), len(db_metrics)])
    for k, model in enumerate(models):
        y[k] = np.array([100 * round(metric_dict[model][db_set][score], 3) for
                         (db_set, set_lab) in zip(np.arange(len(db_metrics)), db_list)])
    idx = np.argwhere(np.isin(db_list, db_list_plot)).flatten()
    fig, ax = plt.subplots(figsize=(12, 12))
    [ax.plot(np.arange(len(idx)) + 1, y[i][idx], color=colors[i], linewidth=8, alpha=0.3, label=model[i], zorder=0) for
     i in range(len(models))]
    [ax.scatter(np.arange(len(idx)) + 1, y[i][idx], s=55, label=model[i], zorder=1) for i in range(len(models))]
    ax.legend(models, loc='upper right')
    ax.set_ylabel(score)
    ax.set_title('{} by architecture'.format(score))
    ax.set_xticks(np.arange(len(idx)) + 1)
    ax.set_xticklabels(db_list_plot, rotation=45)
    fig.tight_layout()
    if savefig:
        plt.tight_layout()
        lab = score + 'points.png'
        plt.savefig(str(savedir) + "/" + lab, dpi=400, transparent=True)
    return


def simple_plot_from_list(score_values, score, savefig=False, savedir=None):
    # x = np.tile(np.arange(len(db_metrics)) + 1,(len(models),1))  # the label locations
    # y = np.empty(shape=[len(models), len(db_metrics)])
    # for k, model in enumerate(models):
    #     y[k] = np.array([100 * round(metric_dict[model][db_set][score], 3) for
    #                  (db_set, set_lab) in zip(np.arange(len(db_metrics)), db_list)])
    # idx = np.argwhere(np.isin(db_list, db_list_plot)).flatten()
    fig, ax = plt.subplots(figsize=(12, 12))
    # [ax.plot(np.arange(len(idx))+1, y[i][idx], color=colors[i], linewidth=8, alpha=0.3, label=model[i], zorder=0) for i in range(len(models))]
    # [ax.scatter(np.arange(len(idx))+1, y[i][idx], s=55, label=model[i], zorder=1) for i in range(len(models))]
    ax.plot(score_values, linewidth=8, alpha=0.3, label='ResNet', zorder=0)
    ax.legend(loc='upper right')
    ax.set_ylabel(score)
    ax.set_title('{} by architecture'.format(score))
    # ax.set_xticks(np.arange(len(idx))+1)
    # ax.set_xticklabels(db_list_plot, rotation=45)
    fig.tight_layout()
    if savefig:
        plt.tight_layout()
        lab = score + 'points.png'
        plt.savefig(str(savedir) + "/" + lab, dpi=400, transparent=True)
    return


def find_af_lables_time(db, pat_id, wind_idx):
    masks = db.return_masks_dict([pat_id])
    find_where_mask_true_idx = np.where(masks[pat_id] == True)
    rrt_relevant_idx = np.where(np.logical_and(db.rrt_dict[pat_id] >= find_where_mask_true_idx[0][wind_idx] * 30,
                                               db.rrt_dict[pat_id] < (find_where_mask_true_idx[0][wind_idx] + 1) * 30))
    rrt_time_idx = db.rrt_dict[pat_id][rrt_relevant_idx[0]] - find_where_mask_true_idx[0][wind_idx] * 30

    rlab = db.rlab_dict[pat_id]
    relevant_rlab = rlab[rrt_relevant_idx]
    rrt_time_af = rrt_time_idx[relevant_rlab == 1.0]
    rrt_sample_af = rrt_time_af * 200
    rrt_sample_af = rrt_sample_af.astype(int)
    return rrt_time_af, rrt_sample_af


def plot_input(save_fig_path: str = None):
    f = h5py.File(consts.PREPROCESSED_DATA_DIR / "uvaf_test_indexes_normalized_99_1.hdf5", "r")
    database = f[consts.HDF5_DATASET]
    ind = np.where(database[:, -1] == 1.0)
    af_idx = ind[0][0]
    non_af_idx = 1415
    ecg_af = database[af_idx][:-3]
    ecg_n = database[non_af_idx][:-3]

    # AF
    pat_id = str(int(database[af_idx][-3])).zfill(4)
    wind_idx = int(database[af_idx][-2])

    # None-AF
    pat_id_non = str(int(database[non_af_idx][-3])).zfill(4)
    wind_idx_non = int(database[non_af_idx][-2])

    db = UVAFDB_Wave_Parser(window_size=6000, load_on_start=True, load_ectopics=False)
    rrt_time_af, rrt_sample_af = find_af_lables_time(db, pat_id, wind_idx)
    rrt_time_non_af, rrt_sample_non_af = find_af_lables_time(db, pat_id_non, wind_idx_non)
    t = np.linspace(0, len(ecg_af) / 200, len(ecg_af))
    simple_fig, aa = plt.subplots(1, figsize=(22, 8))

    fig, ax = plt.subplots(2, figsize=(22, 15))
    fig.tight_layout(h_pad=5)
    ax[0].plot(t, ecg_af)
    ax[0].scatter(rrt_time_af, ecg_af[rrt_sample_af], color='hotpink', label='AF')
    ax[0].set_ylabel('Amplitude [mV]')
    ax[0].set_xlabel('Seconds')
    ax[0].set_title('Input Example - AF')
    ax[0].legend()
    ax[1].set_title('Input Example - None - AF')
    ax[1].plot(t, ecg_n)
    ax[1].scatter(rrt_time_non_af, ecg_n[rrt_sample_non_af], color='hotpink', label='AF')
    ax[1].set_ylabel('Amplitude [mV]')
    ax[1].set_xlabel('Seconds')
    ax[1].legend()
    # plt.show()
    if save_fig_path:
        plt.tight_layout()
        fig_name = str(np.datetime64('now')) + 'input.png'
        plt.savefig(str(save_fig_path) + "/" + fig_name, dpi=400, transparent=True)
        print("done")


def simple_plot(l, lab, title=None, save_name=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    assert len(l) == len(lab), f"no label for each fig"
    for i, y in enumerate(l):
        ax.plot(y, label=lab[i])
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if save_name is not None:
        fig.savefig(save_name + ".png")
    plt.show()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Visualize model results for AF detection')
    # parser.add_argument('--global_path', default=consts.REPO_DIR / "saved_models",
    #                     help='where the models are saved')
    # parser.add_argument('--fig_path', default=cts.REPO_DIR / 'figs',
    #                     help= 'where to save figures')
    # args, unk = parser.parse_known_args()
    # if unk:
    #         warnings.warn("Unknown arguments:" + str(unk) + ".")

    folder_date = '2021-12-29T09:57:30'  # '2021-12-29T09:57:30' #np.datetime64('now') #/
    path_model = consts.BASE_DIR / "Noam" / "models" / str(folder_date)
    path_save_figs = consts.FIGS_DIR
    path_models = {'ResNet': path_model / f"ResNet.pkl",
                   }

    plot_input(consts.FIGS_DIR)
    exit()
    score = 'Fb-Score'
    db_list = ['UVAF train', 'UVAF test']
    models = ['ResNet']  # , 'CRNN(noHT)', "CRNN+DA(noHT)"]  # Not sure about CRNN results, be wary
    db_metrics = ['metrics_train', 'metrics_test_UVAFDB']
    data = dict(zip(db_metrics, db_list))
    metric_dict = {}
    for model in models:
        path = path_models[model]
        model_dict = model_utils.load_model(path, algo=model)
        metric_dict[model] = list(map(model_dict.get, db_metrics))
    # plot_performance_bars(models, path_models, db_metrics, db_list, score, savefig=True, savedir=args.fig_path/'performance/')
    # plot_performance_dot(models, db_metrics=db_metrics, db_list=db_list, db_list_plot=['UVAF train', 'UVAF test'], score=score, savefig=True, savedir=args.fig_path/'performance/')
    simple_plot_from_list(score_values, score, savefig=True, savedir=path_save_figs)
