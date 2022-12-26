from ML.ML_utils import *
from utils import consts as cts
from utils.base_packages import *
from ML.cv_methods import *


def bayesianCV(train_pat_features, train_pat_labels, algo, groups, normalize=False,
               weighting=True, n_jobs=20, typ=1, dataset='both', results_dir=cts.ML_RESULTS_DIR):
    if normalize:
        standartization = StandardScaler()
    logo = LeaveOneGroupOut()
    gss = GroupShuffleSplit(n_splits=4, test_size=0.25)
    Ocv = OneCrossValidation()
    Ccv = ConfCrossValidation()
    if algo == 'RF':
        if weighting:
            clf = cts.class_funcs[algo](class_weight='balanced', n_jobs=n_jobs)
        else:
            clf = cts.class_funcs[algo](n_jobs=n_jobs)
        search_space = cts.search_spaces_RF
    elif algo == 'XGB':
        clf = cts.class_funcs[algo](n_jobs=n_jobs)
        search_space = cts.search_spaces_XGB
    pipe = Pipeline([
        ('normalize', standartization), ('model', clf)
    ])
    opt = BayesSearchCV(
        pipe,
        search_spaces=search_space,

        # search_spaces={
        #     'model__C': Real(0.1, 1000),
        # },
        scoring=rc_scorer,
        n_iter=50,
        cv=Ocv.split(train_pat_features, train_pat_labels, groups=groups),
        return_train_score=True, verbose=1, n_jobs=n_jobs)

    opt.fit(train_pat_features, train_pat_labels)  # callback=on_step
    delattr(opt, 'cv')
    # save your model or results
    path = set_path(algo, dataset, typ, results_dir)

    plt.show()

    return opt
