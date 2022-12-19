from ML.ML_utils import *


def bsqi_stataictics(ids, bsqi_path, win_len):
    if win_len == 30:
        wl = 0
    else:
        wl = win_len
    not_exit_ids = []
    excluded_recordings = []
    initial_windows = 0
    bad_bsqi_windows = 0
    bsqi_all = []
    bsqi_per_patient = pd.DataFrame()
    # all_ids = list(set(cts.ids_sn + cts.ids_vn + cts.ids_tp + cts.ids_sp + cts.ids_tn + cts.bad_bsqi))
    for id_ in ids:
        try:
            bsqi = np.load(bsqi_path / id_ / str('bsqi_' + str(wl) + '.npy'))
            if len(bsqi[bsqi < 0.8]) == len(bsqi):
                excluded_recordings.append(id_)

            if len(bsqi) == 0:
                print(id_)
            else:
                median_bsqi = np.median(bsqi)
                q75, q25 = np.percentile(bsqi, [99, 1])
                iqr_bsqi = q75 - q25
                initial_windows += len(bsqi)
                bad_bsqi_windows += len(bsqi[bsqi < 0.8])
                bsqi_all = bsqi_all + list(bsqi)
                bsqi_per_patient = bsqi_per_patient.append(
                    pd.DataFrame([id_, len(bsqi), len(bsqi), median_bsqi, iqr_bsqi]).transpose(), ignore_index=True)
        except FileNotFoundError:
            not_exit_ids.append(id_)
    bsqi_per_patient = bsqi_per_patient.set_axis(
        labels=['holter id', 'windows num', 'bad bsqi windows num', 'median_bsqi', 'iqr_bsqi'], axis=1)
    bsqi_per_patient.to_excel(bsqi_path / 'bsqi_stat.xlsx')
    np.save('/MLAIM/AIMLab/Sheina/databases/VTdb/IDS/bad_bsqi/bad_bsqi_' + str(win_len) + '.npy', excluded_recordings)

    print(str(initial_windows))
    print(str(bad_bsqi_windows))


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


def plot_hist_stst(data, n_arr, title, save_path):
    plt.figure(figsize=[5, 5])
    axis = np.linspace(min(np.concatenate([data[0], data[1]])), max(np.concatenate([data[0], data[1]])), 10)
    colors = ['dodgerblue', 'blue']
    n, bins, patches = plt.hist(data, label=[r'$C_0$', r'$C_1$'], color=colors, bins=10)
    ratio = len(data[1]) / len(data[0])
    for rec in patches[1]:
        rec.set_height(rec.get_height() / ratio)
    max_rec = 0
    for patch in patches:
        for rec in patch:
            if rec.get_height() > max_rec:
                max_rec = rec.get_height()
    plt.ylim(0, max_rec)
    y_vals = plt.yticks()
    plt.yticks(y_vals[0], ['{:0.2f}'.format(x / len(data[0])) for x in y_vals[0]])
    print(str(sum(np.concatenate([data[0], data[1]])) / len(np.concatenate([data[0], data[1]]))))
    plt.legend(loc='upper center')
    plt.xlim([min(np.concatenate([data[0], data[1]])), max(np.concatenate([data[0], data[1]]))])
    plt.xticks([float('{:0.2f}'.format(x)) for x in axis], [float('{:0.2f}'.format(x)) for x in axis], fontsize=8)
    # plt.title('Sex among ' + dataset.upper() + ' recordings', fontsize=14)
    plt.xlabel(handel_strings(title), fontsize=14)
    plt.ylabel(r'$Density$', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(save_path) + '.png')
    plt.show()


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
    axe.set_ylabel(handel_strings(title))
    quartile11, medians1, quartile31 = np.percentile(data[0], [25, 50, 75])
    quartile12, medians2, quartile32 = np.percentile(data[1], [25, 50, 75])
    inds = range(1, 3)
    axe.scatter(inds, [medians1, medians2], marker='o', color='white', s=30, zorder=3)
    axe.vlines(inds, [quartile11, quartile12], [quartile31, quartile32], color=color, linestyle='-', lw=5)
    axe.set_xlabel(r'$probability\ density$')
    labels = [r'$C_0$', r'$C_1$']
    set_axis_style(axe, labels)
    plt.tight_layout()
    plt.savefig(str(save_path) + '.png')


def perform_statistical_test(features_path, stat_path, exmp_file):
    y_vt = np.ones([1, len(cts.ids_tp + cts.ids_sp)]).squeeze()
    y_no_vt = np.ones([1, len(cts.ids_tn + cts.ids_sn + cts.ids_vn)]).squeeze()

    # create dataset ( one VT each grop)
    data_vt, _, ids_group_p = create_dataset(cts.ids_tp + cts.ids_sp, y_vt, features_path, model=0)
    data_no_vt, _, ids_group_n = create_dataset(cts.ids_tn + cts.ids_sn + cts.ids_vn, y_no_vt, features_path, model=0)

    sample_features_xl = pd.read_excel(exmp_file, engine='openpyxl')
    features_arr = np.asarray(sample_features_xl.columns[1:])
    features_list = choose_right_features(np.expand_dims(features_arr, axis=0))
    statistical_test_print = pd.DataFrame(columns=['mannwhitneyu', 'VT', 'Non-VT', '% pos patients', '% pos windows',
                                                   '% neg patients', '% neg windows'], index=features_list[0])
    stat_summery_positive = pd.DataFrame(columns=['min', 'Q1', 'median', 'Q3', 'max', '% patients', '% windows'],
                                         index=features_list[0])
    stat_summery_negative = pd.DataFrame(columns=['min', 'Q1', 'median', 'Q3', 'max', '% patients', '% windows'],
                                         index=features_list[0])
    statistical_test_alz = pd.DataFrame(columns=['mannwhitneyu'], index=features_list[0])

    for i, feature in enumerate(features_list[0]):
        data1 = data_vt[:, i].astype(np.float)
        data2 = data_no_vt[:, i].astype(np.float)
        median1 = np.around(np.median(data1), 2)
        median2 = np.around(np.median(data2), 2)
        Q31 = np.around(np.percentile(data1, 75), 2)
        Q11 = np.around(np.percentile(data1, 25), 2)
        Q32 = np.around(np.percentile(data2, 75), 2)
        Q12 = np.around(np.percentile(data2, 25), 2)
        min1 = np.around(min(data1), 2)
        max1 = np.around(max(data1), 2)
        min2 = np.around(min(data2), 2)
        max2 = np.around(max(data2), 2)
        iqr1 = np.around(np.percentile(data1, 75) - np.percentile(data1, 25), 2)
        iqr2 = np.around(np.percentile(data2, 75) - np.percentile(data2, 25), 2)
        statistical_test_print['VT'][feature] = str(median1) + '(' + str(iqr1) + ')'
        statistical_test_print['Non-VT'][feature] = str(median2) + '(' + str(iqr2) + ')'
        q1 = np.quantile(data1, 0.01)
        q99 = np.quantile(data1, 0.99)
        ids_group_p_clean = np.asarray(ids_group_p)[(data1 <= q99) & (data1 >= q1)]
        np1 = np.around(len(np.unique(ids_group_p_clean)) / 56, 2)
        nw1 = np.around(len(ids_group_p_clean) / len(ids_group_p), 2)
        q1 = np.quantile(data2, 0.01)
        q99 = np.quantile(data2, 0.99)
        ids_group_n_clean = np.asarray(ids_group_n)[(data2 <= q99) & (data2 >= q1)]
        np2 = np.around(len(np.unique(ids_group_n_clean)) / len(np.unique(ids_group_n)), 2)
        nw2 = np.around(len(ids_group_n_clean) / len(ids_group_n), 2)
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
        stat_summery_positive['min'][feature] = min1
        stat_summery_positive['Q1'][feature] = Q11
        stat_summery_positive['median'][feature] = median1
        stat_summery_positive['Q3'][feature] = Q31
        stat_summery_positive['max'][feature] = max1
        stat_summery_positive['% patients'][feature] = np1
        stat_summery_positive['% windows'][feature] = nw1
        stat_summery_negative['min'][feature] = min2
        stat_summery_negative['Q1'][feature] = Q12
        stat_summery_negative['median'][feature] = median2
        stat_summery_negative['Q3'][feature] = Q32
        stat_summery_negative['max'][feature] = max2
        stat_summery_negative['% patients'][feature] = np2
        stat_summery_negative['% windows'][feature] = nw2

    statistical_test_alz.to_excel(stat_path / 'statistical_test_alz.xlsx')
    statistical_test_print.to_excel(stat_path / 'statistical_test_print.xlsx')
    stat_summery_positive.to_excel(stat_path / 'stat_summery_positive.xlsx')
    stat_summery_negative.to_excel(stat_path / 'stat_summery_negative.xlsx')


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
        plot_hist_stst([data2_clean, data1_clean], [np2, nw2, np1, nw1], title,
                       stat_path / title)

    stat_df = pd.Dataframe(index=features_arrey, columns=['p-values', 'VT - median', 'Non VT median'])


if __name__ == '__main__':
    win_len_n = 'uvafdb'
    win_len = 30
    features_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')
    stat_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/stat_test/')
    exmp_file = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/C720Dc84/features_nd.xlsx')
    bsqi_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/preprocessed_data/') / win_len_n
    perform_statistical_test(features_path, stat_path, exmp_file)
    # analyze_statistical_test(features_path, stat_path, exmp_file)
    # ids = cts.ext_test_no_vt
    # bsqi_stataictics(ids, bsqi_path, win_len)

    # mean_VT = np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/stat_test/mean_VT.npy')
    # mean_non_VT = np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/stat_test/mean_no_VT.npy')
    # std_VT = np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/stat_test/std_VT.npy')
    # std_non_VT = np.load('/MLAIM/AIMLab/Sheina/databases/VTdb/stat_test/std_no_VT.npy')
    # plot_hist_stst([mean_non_VT, mean_VT], [], 'mean', stat_path)
    # plot_hist_stst([std_non_VT, std_VT], [], 'std', stat_path)

    a = 5
