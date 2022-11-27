import pathlib

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from ML.ML_utils import *


def handel_strings(in_string):
    # in_string = 'cosEn'
    sp_str = in_string.split('_')

    if len(sp_str) == 3:
        unit = find_units(sp_str[1] + '_' + sp_str[2])
        out_string = r'${0}{1}_{{{2}}}[{3}]$'.format(sp_str[0], sp_str[1], sp_str[2], unit)
    if len(sp_str) == 2:
        unit = find_units(sp_str[1])
        out_string = r'${0}{1}[{2}]$'.format(sp_str[0], sp_str[1], unit)
    if len(sp_str) == 1:
        unit = find_units(sp_str[0])
        out_string = r'${0}[{1}]$'.format(sp_str[0], unit)

    return out_string
    # if try_figure:
    #     plt.figure()
    #     x = np.arange(200)/20
    #     y = np.sin(x)
    #     plt.plot(x,y)
    #     plt.title(out_string)
    #     plt.show()


def find_units(name):
    unit = 'nu'
    if name in cts.INTS:
        unit_idx = cts.INTS.index(name)
        unit = cts.pebm_units[unit_idx]
    if name in cts.WAVES:
        unit_idx = cts.WAVES.index(name)
        unit = cts.pebm_units[unit_idx + 14]
    if name in cts.IMPLEMENTED_FEATURES.tolist():
        unit_idx = cts.IMPLEMENTED_FEATURES.tolist().index(name)
        unit = cts.HRV_units[unit_idx]
    return unit


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


def plot_violin_stst(data, n_arr, title, save_path):
    n_arr = np.round(n_arr, 2) * 100
    plt.style.use('bmh')
    plt.rcParams.update({'font.size': 16})
    color = 'steelblue'
    colors = ['dodgerblue', 'blue']
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    violins = axe.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    for i, vp in enumerate(violins['bodies']):
        vp.set_facecolor(colors[i])
        vp.set_edgecolor(color)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = violins[partname]
        vp.set_edgecolor(color)
    axe.set_xlabel(handel_strings(title))
    quartile11, medians1, quartile31 = np.percentile(data[0], [25, 50, 75])
    quartile12, medians2, quartile32 = np.percentile(data[1], [25, 50, 75])
    inds = range(1, 3)
    axe.scatter(inds, [medians1, medians2], marker='o', color='white', s=30, zorder=3)
    axe.vlines(inds, [quartile11, quartile12], [quartile31, quartile32], color=color, linestyle='-', lw=5)
    axe.set_ylabel(r'$probability\ density$')
    axe.legend(
        ['%p = ' + str(n_arr[0]) + ', %w = ' + str(n_arr[1]), '%p = ' + str(n_arr[2]) + ', %w = ' + str(n_arr[3])],
        loc=9, fontsize=10)
    labels = [r'$C_0$', r'$C_1$']
    set_axis_style(axe, labels)
    plt.tight_layout()
    plt.savefig(save_path + '.png')


def perform_statistical_test(features_path, stat_path, exmp_file):
    y_vt = np.ones([1, len(cts.ids_tp + cts.ids_sp)]).squeeze()
    y_no_vt = np.ones([1, len(cts.ids_tn + cts.ids_sn + cts.ids_vn)]).squeeze()

    # create dataset ( one VT each grop)
    data_vt, _, _ = create_dataset(cts.ids_tp + cts.ids_sp, y_vt, features_path, model=0)
    data_no_vt, _, _ = create_dataset(cts.ids_tn + cts.ids_sn + cts.ids_vn, y_no_vt, features_path, model=0)

    sample_features_xl = pd.read_excel(exmp_file, engine='openpyxl')
    features_arr = np.asarray(sample_features_xl.columns[1:])
    features_list = choose_right_features(np.expand_dims(features_arr, axis=0))
    statistical_test_print = pd.DataFrame(columns=['mannwhitneyu', 'VT', 'Non-VT'], index=features_list[0])
    statistical_test_alz = pd.DataFrame(columns=['mannwhitneyu'], index=features_list[0])

    for i, feature in enumerate(features_list[0]):
        data1 = data_vt[:, i]
        data2 = data_no_vt[:, i]
        median1 = np.around(np.median(data1), 2)
        median2 = np.around(np.median(data2), 2)
        iqr1 = np.around(np.percentile(data1, 75) - np.percentile(data1, 25), 2)
        iqr2 = np.around(np.percentile(data1, 75) - np.percentile(data2, 25), 2)
        statistical_test_print['VT'][feature] = str(median1) + '(' + str(iqr1) + ')'
        statistical_test_print['Non-VT'][feature] = str(median2) + '(' + str(iqr2) + ')'
        try:
            stat, p = mannwhitneyu(data1, data2)
            if p < 0.001:
                statistical_test_print['mannwhitneyu'][feature] = '<0.001'
            else:
                statistical_test_print['mannwhitneyu'][feature] = np.around(p, 3)
            statistical_test_alz['mannwhitneyu'][feature] = p
        except:
            statistical_test_print['mannwhitneyu'][feature] = 1
            statistical_test_alz['mannwhitneyu'][feature] = 1

    statistical_test_alz.to_excel(stat_path / 'statistical_test_alz.xlsx')
    statistical_test_print.to_excel(stat_path / 'statistical_test_print.xlsx')


def analyze_statistical_test(features_path, stat_path, exmp_file):
    stat_test_df = pd.read_excel(stat_path / 'statistical_test_alz.xlsx', engine='openpyxl')
    p_values = np.asarray(stat_test_df['mannwhitneyu'])
    features_arrey = np.asarray(stat_test_df['Unnamed: 0'])
    idx = np.argsort(p_values).squeeze()
    best_features = features_arrey[idx]

    y_vt = np.ones([1, len(cts.ids_tp + cts.ids_sp)]).squeeze()
    y_no_vt = np.ones([1, len(cts.ids_tn + cts.ids_sn + cts.ids_vn)]).squeeze()

    # create dataset ( one VT each grop)
    data_vt, _, ids_group_p = create_dataset(cts.ids_tp + cts.ids_sp, y_vt, features_path, model=0)
    data_no_vt, _, ids_group_n = create_dataset(cts.ids_tn + cts.ids_sn + cts.ids_vn, y_no_vt, features_path, model=0)

    for i in range(20):
        data1 = data_vt[:, idx[i]].astype(float)

        q1 = np.quantile(data1, 0.01)
        q99 = np.quantile(data1, 0.99)
        data1_clean = data1[(data1 <= q99) & (data1 >= q1)]
        ids_group_p_clean = np.asarray(ids_group_p)[(data1 <= q99) & (data1 >= q1)]
        np1 = len(np.unique(ids_group_p_clean)) / 56
        nw1 = len(ids_group_p_clean) / len(ids_group_p)

        data2 = data_no_vt[:, idx[i]].astype(float)

        q1 = np.quantile(data2, 0.01)
        q99 = np.quantile(data2, 0.99)
        data2_clean = data2[(data2 <= q99) & (data2 >= q1)]
        ids_group_n_clean = np.asarray(ids_group_n)[(data2 <= q99) & (data2 >= q1)]
        np2 = len(np.unique(ids_group_n_clean)) / len(np.unique(ids_group_n))
        nw2 = len(ids_group_n_clean) / len(ids_group_n)
        title = best_features[i]
        print(title)
        plot_violin_stst([data2_clean, data1_clean], [np2, nw2, np1, nw1], title,
                         cts.VTdb_path + 'stat_test_norm/' + title)

    stat_df = pd.Dataframe(index=features_arrey, columns=['p-values', 'VT - median', 'Non VT median'])


if __name__ == '__main__':
    features_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')
    stat_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/stat_test/')
    exmp_file = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/C720Dc84/features_nd.xlsx')
    perform_statistical_test(features_path, stat_path, exmp_file)
    analyze_statistical_test(features_path, stat_path, exmp_file)
