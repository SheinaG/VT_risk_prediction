import random

from utils import consts as cts
from utils.base_packages import *


def stratified_group_shuffle_split(X, y, groups, train_size=0.8, random_state=None, n_splits=10):
    if random_state:
        random.seed(random_state)

    # Create a list of the unique groups and their corresponding labels
    # unique_groups = [(g, y[i]) for i, g in enumerate(groups)]
    neg_unique_groups = list(set([(g, y[i]) for i, g in enumerate(np.asarray(groups)[y == 0])]))
    pos_unique_groups = list(set([(g, y[i]) for i, g in enumerate(np.asarray(groups)[y == 1])]))

    # Create empty lists for the training and test sets
    train, test = [], []

    random.shuffle(neg_unique_groups)
    random.shuffle(pos_unique_groups)
    # Split the list of groups into two lists: the training set and the test set
    train_groups = neg_unique_groups[:int(train_size * len(neg_unique_groups))] + pos_unique_groups[
                                                                                  int(train_size * len(
                                                                                      pos_unique_groups)):]
    train_idx, test_idx = [], []
    for g, label in neg_unique_groups + pos_unique_groups:
        if (g, label) in train_groups:
            train_idx += list(np.where(np.asarray(groups) == g)[0])
        else:
            test_idx += list(np.where(np.asarray(groups) == g)[0])

    train.append(train_idx)
    test.append(test_idx)

    yield train, test


class CustomCrossValidation:

    @classmethod
    def split(cls, x, y, groups: np.ndarray = None):
        """Returns to a grouped time series split generator."""
        random.shuffle(cts.ids_conf)
        # The min max index must be sorted in the range
        for group_idx in range(groups.min(), groups.max()):

            training_group = group_idx
            # Gets the next group right after
            # the training as test
            test_group = group_idx + 1
            training_indices = np.where(
                groups == training_group)[0]
            test_indices = np.where(groups == test_group)[0]
            if len(test_indices) > 0:
                # Yielding to training and testing indices
                # for cross-validation generator
                yield training_indices, test_indices


class ConfCrossValidation:

    @classmethod
    def split(cls, x, y, groups: np.ndarray = None):
        """Returns to a grouped time series split generator."""
        # The min max index must be sorted in the range
        train_n = cts.ids_tn + cts.ids_vn

        random.shuffle(train_n)
        random.shuffle(cts.ids_conf)

        test_indices, train_indices = [], []
        for i, group in enumerate(groups):
            if group in train_n[:200] + cts.ids_conf[:10]:
                test_indices.append(i)
            else:
                train_indices.append(i)

        yield train_indices, test_indices


class OneCrossValidation:

    @classmethod
    def split(cls, x, y, groups: np.ndarray = None):
        """Returns to a grouped time series split generator."""
        # The min max index must be sorted in the range
        test_indices, train_indices = [], []
        for i, group in enumerate(groups):

            if group in cts.ids_vn_2 + cts.ids_vp_2:
                test_indices.append(i)
            else:
                train_indices.append(i)

        yield train_indices, test_indices
