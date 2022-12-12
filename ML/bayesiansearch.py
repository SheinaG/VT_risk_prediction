from ML.ML_utils import *
from utils import consts as cts
from utils.base_packages import *


def bayesianCV(train_pat_features, train_pat_labels, algo, groups, normalize=False,
               weighting=True, n_jobs=20, typ=1, dataset='both', results_dir=cts.ML_RESULTS_DIR):
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
        cv=stratified_group_shuffle_split(train_pat_features, train_pat_labels, groups, train_size=0.75, n_splits=600),
        return_train_score=True, verbose=1, n_jobs=n_jobs)

    opt.fit(train_pat_features, train_pat_labels)  # callback=on_step
    delattr(opt, 'cv')
    # save your model or results
    path = set_path(algo, dataset, typ, results_dir)

    plt.show()

    return opt
