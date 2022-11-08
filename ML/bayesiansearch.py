from ML.ML_utils import *
from utils import consts as cts
from utils.base_packages import *


def bayesianCV(train_pat_features, train_pat_labels, algo, groups, normalize=False,
               weighting=True, n_jobs=20, typ=1, dataset='both', results_dir=cts.RESULTS_DIR):
    if weighting:
        clf = cts.class_funcs[algo](class_weight='balanced', n_jobs=n_jobs)
    else:
        clf = cts.class_funcs[algo](n_jobs=n_jobs)
    if normalize:
        standartization = StandardScaler()
    pipe = Pipeline([
        ('normalize', standartization), ('model', clf)
    ])
    logo = LeaveOneGroupOut()
    if algo == 'RF':
        search_space = cts.search_spaces_RF
    elif algo == 'XGB':
        search_space = cts.search_spaces_XGB
    opt = BayesSearchCV(
        pipe,
        search_spaces=search_space,

        # search_spaces={
        #     'model__C': Real(0.1, 1000),
        # },
        scoring=rc_scorer,
        n_iter=600,

        cv=logo.split(train_pat_features, train_pat_labels, groups=groups),
        return_train_score=True, verbose=1, n_jobs=n_jobs)

    # callback handler
    def on_step(res):
        path = set_path(algo, dataset, typ, results_dir)
        if res.func_vals.shape[0] == 1:
            results_c = pd.DataFrame(
                columns=['combination', 'score'])
        else:
            results_c = pd.read_csv(path / 'results.csv')
            results_c = results_c.set_axis(results_c['Unnamed: 0'], axis='index')
            results_c = results_c.drop(columns=['Unnamed: 0'])
        # c_iter = res.x_iters[-1]
        # score =res.func_vals[-1]
        # c_iter = c_iter.append(score)
        results_c.loc[results_c.shape[0]] = [res.x_iters[-1], res.func_vals[-1]]
        results_c.to_csv(path / 'results.csv')

    opt.fit(train_pat_features, train_pat_labels, eval_metric='auc')  # callback=on_step
    delattr(opt, 'cv')
    # save your model or results
    path = set_path(algo, dataset, typ, results_dir)

    plt.show()

    return opt
