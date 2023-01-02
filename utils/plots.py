from utils.base_packages import *
import utils.consts as cts
from ML.ML_utils import *


def roc_plot_envelope(y_preds, y_tests, K_test, augmentation, typ, title='', algo='RF', majority_vote=False,
                      soft_lines=False):
    fprs = []
    tprs = []
    fprs_major = []
    tprs_major = []

    for i in range(K_test):
        # Plot of a ROC curve:
        y_true = y_tests[i]
        y_score = y_preds[i]
        fpr, tpr, THs = roc_curve(y_true, y_score)
        fprs.append(fpr)
        tprs.append(tpr)
        if majority_vote:
            fpr_major, tpr_major = majority_roc(THs, augmentation, y_score, y_true)
            fprs_major.append(fpr_major)
            tprs_major.append(tpr_major)

        if majority_vote and soft_lines:
            plt.plot(fpr_major, tpr_major, '--', lw=1, color='lightgray')
        elif soft_lines:
            plt.plot(fpr, tpr, '--', lw=1, color='lightgray')
    color = cts.colors[typ - 1]
    fill_color = cts.light_colors[typ - 1]
    if majority_vote:
        fpr_all, mean_tpr, max_tpr, min_tpr = get_K_rocs(K_test, fprs_major, tprs_major)
    else:
        fpr_all, mean_tpr, max_tpr, min_tpr = get_K_rocs(K_test, fprs, tprs)

    roc_auc = auc(fpr_all, mean_tpr)
    low_auroc = auc(fpr_all, min_tpr)
    high_auroc = auc(fpr_all, max_tpr)
    legend_i = algo + ' model ' + str(typ) + ' ' + str(np.around(roc_auc, 2)) + '(' + str(
        np.round(low_auroc, 3)) + ',' + str(
        np.round(high_auroc, 3)) + ')'
    plt.plot(fpr_all, mean_tpr, lw=2, color=color, label=legend_i)
    # label = f'K={K_test}; (AUROC = {round(roc_auc, 3)})'
    plt.fill_between(fpr_all, min_tpr, max_tpr, color=fill_color)

    # Add coin flipping line:
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Sp')
    plt.ylabel('Se')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    return low_auroc, high_auroc


def majority_roc(THs, augmentation, y_score, y_true):
    # Reshape to work as an array - every different patient is a row
    y_score_shaped = np.reshape(y_score, (-1, augmentation))
    y_true_singles = np.reshape(y_true, (-1, augmentation))[:, 0]
    tpr = []
    fpr = []

    # compute Se and 1-Sp for each threshold
    for i, TH in enumerate(THs):
        thresholded = y_score_shaped > TH
        thresholded_sum = np.sum(thresholded, axis=1)
        y_pred_singles = (thresholded_sum > (augmentation / 2)) * 1  # preds by majority
        tn, fp, fn, tp = confusion_matrix(y_true_singles, y_pred_singles).ravel()
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    # Set limits between 0 and 1:
    tpr[0] = 0
    tpr[-1] = 1
    fpr[0] = 0
    fpr[-1] = 1

    return np.array(tpr), np.array(fpr)


def get_K_rocs(K_test, fprs, tprs):
    """
    Get the roc parameter suits for the roc mean and envelope
    """
    lengths = []
    for i in range(K_test):
        lengths.append(fprs[i].shape)

    max_l = np.max(lengths)
    max_index = np.argmax(lengths)

    tprs_interp = np.zeros((K_test, max_l))
    fpr_all = fprs[max_index]

    for i in range(K_test):
        f = interp1d(fprs[i], tprs[i])
        tprs_interp[i, :] = f(fpr_all)

    mean_tpr = np.mean(tprs_interp, axis=0)
    high_tpr = np.zeros([len(fpr_all), ])
    low_tpr = np.zeros([len(fpr_all), ])
    for i in range(len(fpr_all)):
        low_tpr[i], high_tpr[i] = calc_confidence_interval_of_array(tprs_interp[:, i], confidence=0.95)

    return fpr_all, mean_tpr, low_tpr, high_tpr


def plot_BS(opt_d, path_d, algo):
    for i in range(1, cts.NM + 1):
        _ = plot_objective(opt_d[i].optimizer_results_[0],
                           dimensions=cts.hyp_list[algo],
                           n_minimum_search=int(1e8))
        plt.savefig(path_d[i] / str(str(i) + algo + '.png'))
        plt.show()


def train_val(opt_d, model_path, algo):
    columns = ['AUROC train', 'AUROC validation']
    train_val = pd.DataFrame(columns=columns, index=range(1, len(opt_d) + 1))
    for j in opt_d:
        opt = opt_d[j]
        be = opt.best_index_
        mean_test = np.around(opt.cv_results_['mean_test_score'][be], 2)
        std_test = np.around(opt.cv_results_['std_test_score'][be], 2)
        mean_train = np.around(opt.cv_results_['mean_train_score'][be], 2)
        std_train = np.around(opt.cv_results_['std_train_score'][be], 2)
        train_ms = str(mean_train) + '±' + str(std_train)
        val_ms = str(mean_test) + '±' + str(std_test)
        train_val['AUROC train'][j] = train_ms
        train_val['AUROC validation'][j] = val_ms

    train_val.to_excel(model_path / str('all_model_train_val_' + algo + '.xlsx'))


def hyper_model(opt_d, path, algo):
    hyp_pd = pd.DataFrame(columns=opt_d[1].best_params_.keys())
    for i in range(1, cts.NM + 1):
        best_hyp = pd.DataFrame(opt_d[i].best_params_, columns=opt_d[i].best_params_.keys(), index=[i])
        hyp_pd = hyp_pd.append(best_hyp)
    if algo == 'XGB':
        hyp_pd = hyp_pd.apply(f2)
    hyp_pd.to_excel(path / str('hyperparameters_' + algo + '.xlsx'))


def hit_map_bsqi(bsqi_path, win_len=30):
    if win_len == 30:
        wl = 0
    else:
        wl = win_len
    all_ids = list(set(cts.ids_sn + cts.ids_vn + cts.ids_tp + cts.ids_sp + cts.ids_tn + cts.bad_bsqi))
    for id_ in all_ids:
        bsqi = np.load(bsqi_path / id_ / str('bsqi_' + str(win_len) + '.npy'))


def plot_demographics(dataset, features_path):
    if dataset == ('uvaf' or 'uvafdb'):
        datase_load = 'uvafdb'
    else:
        datase_load = 'rbdb'
    demographic_no_VT = pd.read_excel(features_path / 'VTn' / str('demographic_features_' + datase_load + '.xlsx'),
                                      engine='openpyxl')
    demographic_VT = pd.read_excel(features_path / 'VTp' / str('demographic_features_' + datase_load + '.xlsx'),
                                   engine='openpyxl')

    plt.style.use('bmh')

    Age_VT = demographic_VT['age']
    Age_no_VT = demographic_no_VT['age']
    ratio = len(Age_no_VT) / len(Age_VT)
    Age_all = np.concatenate([Age_VT, Age_no_VT], axis=0)
    age_start = (np.min(Age_all) // 10) * 10
    age_stop = ((np.max(Age_all) // 10) + 1) * 10
    bins = np.linspace(int(age_start), int(age_stop), int((age_stop - age_start) // 10 + 1))
    print(str(np.median(Age_all)))
    print(str(np.percentile(Age_all, 75)) + '-' + str(np.percentile(Age_all, 25)))
    plt.figure()
    n, bins, patches = plt.hist([Age_VT, Age_no_VT], bins, label=[r'$C_1$', r'$C_0$'], color=cts.colors[:2])
    for rec in patches[1]:
        rec.set_height(rec.get_height() / ratio)
    max_rec = 0
    for patch in patches:
        for rec in patch:
            if rec.get_height() > max_rec:
                max_rec = rec.get_height()
    plt.ylim(0, max_rec)
    y_vals = plt.yticks()
    plt.yticks(y_vals[0], ['{:0.2f}'.format(x / len(Age_VT)) for x in y_vals[0]])
    plt.legend(loc='upper right')
    # plt.title('Age among '+dataset.upper()+' recordings', fontsize=14)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tight_layout()
    plt.savefig(features_path / 'stat_test' / str('age_' + dataset + '.png'))
    plt.show()

    gender_VT = list(demographic_VT['gender'])
    gender_no_VT = list(demographic_no_VT['gender'])
    plt.figure(figsize=[5, 5])
    n, bins, patches = plt.hist([gender_VT, gender_no_VT], label=[r'$C_1$', r'$C_0$'], color=cts.colors[:2], bins=3)
    for rec in patches[1]:
        rec.set_height(rec.get_height() / ratio)
    max_rec = 0
    for patch in patches:
        for rec in patch:
            if rec.get_height() > max_rec:
                max_rec = rec.get_height()
    plt.ylim(0, max_rec)
    y_vals = plt.yticks()
    plt.yticks(y_vals[0], ['{:0.2f}'.format(x / len(gender_VT)) for x in y_vals[0]])
    print(str(sum(np.asarray(gender_VT + gender_no_VT)) / len(gender_VT + gender_no_VT)))
    plt.legend(loc='upper center')
    plt.xlim([0, 1])
    plt.xticks(range(2), ['Female', 'Male'], fontsize=14)
    # plt.title('Sex among ' + dataset.upper() + ' recordings', fontsize=14)
    plt.xlabel('Sex', fontsize=14)
    plt.ylabel('Density', fontsize=14)

    plt.tight_layout()
    plt.savefig(features_path / 'stat_test' / str('gender_' + dataset + '.png'))
    plt.show()


def plot_norm_hist(data_list, legend_list, title, axis, xlabel, bin_num, save_path):
    plt.figure(figsize=[5, 5])
    n, bins, patches = plt.hist(data_list, label=legend_list, color=cts.colors[:2], bins=bin_num)
    ratio = len(data_list[1]) / len(data_list[0])
    for rec in patches[1]:
        rec.set_height(rec.get_height() / ratio)
    max_rec = 0
    for patch in patches:
        for rec in patch:
            if rec.get_height() > max_rec:
                max_rec = rec.get_height()
    plt.ylim(0, max_rec)
    y_vals = plt.yticks()
    plt.yticks(y_vals[0], ['{:0.2f}'.format(x / len(data_list[0])) for x in y_vals[0]])
    print(str(sum(np.asarray(data_list[0] + data_list[1])) / len(data_list[0] + data_list[1])))
    plt.legend(loc='upper center')
    plt.xlim([axis[0], axis[-1]])
    plt.xticks([float('{:0.2f}'.format(x)) for x in axis], [float('{:0.2f}'.format(x)) for x in axis], fontsize=10)
    # plt.title('Sex among ' + dataset.upper() + ' recordings', fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Density', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path / str(title + '.png'))
    plt.show()


def plot_V_ratio():
    features_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/')
    exmp_file = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/ML_model/C720Dc84/features_nd.xlsx')
    y_vt = np.ones([1, len(cts.ids_tp + cts.ids_sp)]).squeeze()
    y_no_vt = np.ones([1, len(cts.ids_tn + cts.ids_sn + cts.ids_vn)]).squeeze()
    data_vt, _, _ = create_dataset(cts.ids_tp + cts.ids_sp, y_vt, features_path, model=0,
                                   features_name='features_nd.xlsx')
    data_no_vt, _, _ = create_dataset(cts.ids_tn + cts.ids_sn + cts.ids_vn, y_no_vt, features_path, model=0,
                                      features_name='features_nd.xlsx')

    sample_features_xl = pd.read_excel(exmp_file, engine='openpyxl')
    features_arr = np.asarray(sample_features_xl.columns[1:])
    features_list = choose_right_features(np.expand_dims(features_arr, axis=0))

    data1 = data_vt[:, -9].astype(float)
    q1 = np.quantile(data1, 0.01)
    q99 = np.quantile(data1, 0.99)
    data1_clean = data1[(data1 <= q99) & (data1 >= q1)]

    data2 = data_no_vt[:, -9].astype(float)
    q1 = np.quantile(data2, 0.01)
    q99 = np.quantile(data2, 0.99)
    data2_clean = data2[(data2 <= q99) & (data2 >= q1)]
    data_list = [data1_clean.tolist(), data2_clean.tolist()]
    legend_list = ['C1', 'C0']
    title = "PVC's ratio histogram"
    axis = np.linspace(0, 0.45, 10)
    xlabel = '% of PVC beats'
    bin_num = 10
    save_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/figures/')
    plot_norm_hist(data_list, legend_list, title, axis, xlabel, bin_num, save_path)


if __name__ == '__main__':
    features_path = pathlib.PurePath('/MLAIM/AIMLab/Sheina/databases/VTdb/')
    dataset = 'rbdb'
    plot_demographics(dataset, features_path)
