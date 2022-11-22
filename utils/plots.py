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
        fpr, tpr, THs = roc_curve(y_true.ravel(), y_score.ravel())
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


def train_val(opt_d, model_path):
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

    train_val.to_excel(model_path / 'all_model_train_val.xlsx')


def hyper_model(opt_d, path):
    hyp_pd = pd.DataFrame(columns=opt_d[1].best_params_.keys())
    for i in range(1, cts.NM + 1):
        best_hyp = pd.DataFrame(opt_d[i].best_params_, columns=opt_d[i].best_params_.keys(), index=[i])
        hyp_pd = hyp_pd.append(best_hyp)
    hyp_pd.to_excel(path / 'hyperparameters.xlsx')


if __name__ == '__main__':
    a = 5
