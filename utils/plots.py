from utils.base_packages import *
from utils.consts import *

font = {'weight': 'normal',
        # 'family' : 'normal',
        'size': 22}
matplotlib.rc('font', **font)


def roc_plot_envelope(y_preds, y_tests, K_test, augmentation, typ, title='', majority_vote=False, soft_lines=False):
    fprs = []
    tprs = []
    fprs_major = []
    tprs_major = []
    for i in range(K_test):
        # Plot of a ROC curve:
        y_true = y_tests[i]
        y_score = y_preds[i][:, 1]
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
    plt.plot(fpr_all, mean_tpr, lw=2, color=color, label='model ' + str(typ))
    # label = f'K={K_test}; (AUROC = {round(roc_auc, 3)})'
    plt.fill_between(fpr_all, min_tpr, max_tpr, color=fill_color)

    # Add coin flipping line:
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Sp')
    plt.ylabel('Se')
    plt.title(title)
    plt.legend(loc="lower right")


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
    max_tpr = np.max(tprs_interp, axis=0)
    min_tpr = np.min(tprs_interp, axis=0)

    return fpr_all, mean_tpr, max_tpr, min_tpr


if __name__ == '__main__':

    plot_input()
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
