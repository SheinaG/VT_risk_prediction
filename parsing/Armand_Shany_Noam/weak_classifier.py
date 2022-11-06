from sklearn.metrics import precision_recall_curve

import utils.consts as cts


class WeakClassifier:
    """ This class implements a weak classifier, based on one feature. It has the same API as the scikit-learn classifiers."""

    def __init__(self, opt_metric='f1'):

        self.opt_metric = opt_metric
        self.thresh = None
        self.sign = None
        self.trained = False

    def fit(self, X, y):
        # Returning threshold and sign for the best weak classifier

        # First we check the SUP possibility, i.e. data > threshold
        best_f1_sup, thresh_sup = self.optimize_f1(X, y)

        # First we check the INF possibility, i.e. data < threshold
        best_f1_inf, thresh_inf = self.optimize_f1(-X, y)

        if best_f1_inf >= best_f1_sup:
            self.sign = cts.INF
            self.thresh = -thresh_inf
        else:
            self.sign = cts.SUP
            self.thresh = thresh_sup

        self.trained = True

    def predict(self, X_new):

        predicted = self.sign * X_new >= self.sign * self.thresh
        return predicted

    def predict_proba(self, X_new):
        return X_new

    def optimize_f1(self, X, y):
        precision, recall, thresholds = precision_recall_curve(y, X)
        f1 = 2 * precision * recall / (precision + recall)
        f1[np.isnan(f1)] = 0.0
        best_f1_idx = np.argmax(f1)
        best_f1 = f1[best_f1_idx]
        thresh = thresholds[best_f1_idx]
        return best_f1, thresh
