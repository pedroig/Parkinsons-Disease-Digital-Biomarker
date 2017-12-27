import numpy as np
import pandas as pd
from sklearn import metrics
from imblearn.over_sampling import SMOTE


def metricsAccumulate(X, y, clf, metrics_total):
    """
    Accumulates metrics results from one random forest in the undersampling ensemble.
    """
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)
    y_prob = y_prob[:, 1]  # positiveClass
    metrics_total["Accuracy"] += metrics.accuracy_score(y, y_pred)
    metrics_total["Precision"] += metrics.precision_score(y, y_pred)
    metrics_total["Recall"] += metrics.recall_score(y, y_pred)
    metrics_total["F1 Score"] += metrics.f1_score(y, y_pred)
    metrics_total["ROC score"] += metrics.roc_auc_score(y, y_prob)


def metricsShowAccumulate(metrics_total, ensemble_size):
    """
    Prints metrics results for the undersampling ensemble in the training set.

    Input:
    - metrics_total: dict
        Dictionary with the accumulated metrics results from the ensemble.
    - ensemble_size: int
    """
    print("\nMetrics on Training Set")
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC score"]:
        print("{}: {}".format(metric, metrics_total[metric] / ensemble_size))


def metricsPrint(y_test, y_pred, y_prob):
    """
    Input:
    - y_test: numpy.ndarray
        Ground truth (correct) labels.
    - y_pred: numpy.ndarray
        Predicted labels, as returned by a classifier.
    - y_prob: numpy.ndarray
        Probability estimates of the positive class.
    """
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1 Score:", metrics.f1_score(y_test, y_pred))
    print("ROC score:", metrics.roc_auc_score(y_test, y_prob))


def metricsShowEnsemble(y_test, y_pred_total, setName, ensemble_size, threshold=0.5):
    """
    Input:
    - y_test: numpy.ndarray
        Ground truth (correct) labels.
    - y_pred_total: numpy.ndarray
        Sum of the votes of all the random forests in the undersampling ensemble.
    - setName: string
        Name of the development set to be printed as the title.
    - ensemble_size: int
        The number of random forests in the undersampling ensemble.
    - threshold: float
        0 < threshold < 1
    """
    print("\nMetrics on {} Set".format(setName))
    y_prob = y_pred_total / ensemble_size
    y_prob = y_prob[:, 1]  # positiveClass
    y_pred = y_prob > threshold
    metricsPrint(y_test, y_pred, y_prob)


def load_data(featuresSplitName, selectOldAge=False, dropAge=False,
              balance_undersampling=False, balance_oversampling=False):
    """
    Loads table with the features and applies the selected preprocessing.

    Input:
    - featuresSplitName: string
        Name of the CSV table to be loaded.
    - selectOldAge: bool (default=False):
        Whether to select only people older 56 years in the set.
    - dropAge: bool (default=False)
        Whether to use age as a feature.
    - balance_undersampling: bool (default=False)
        Whether to undersample the majority class in the set.
    - balance_oversampling: bool (default=False)
        Whether to oversample the minority class in the set.
    """

    X = pd.read_csv("../data/{}.csv".format(featuresSplitName), index_col=0)

    if selectOldAge:
        X = X[X.age > 56]
    if dropAge:
        X = X.drop(["age"], axis=1)

    y = X.Target
    X = X.drop("Target", axis=1)

    feature_names = X.axes[1]

    if balance_undersampling:
        pd_indices = X[y].index
        healthy_indices = X[~y].index
        if len(pd_indices) > len(healthy_indices):
            random_pd_indices = np.random.choice(pd_indices, len(healthy_indices), replace=False)
            balanced_indices = np.append(random_pd_indices, healthy_indices)
        else:
            random_healthy_indices = np.random.choice(healthy_indices, len(pd_indices), replace=False)
            balanced_indices = np.append(random_healthy_indices, pd_indices)
        X = X.loc[balanced_indices, :]
        y = y[balanced_indices]
        y = np.asarray(y.values, dtype=np.int8)
    elif balance_oversampling:
        sm = SMOTE(ratio='minority')
        X, y = sm.fit_sample(X, y)
    else:
        y = np.asarray(y.values, dtype=np.int8)

    return X, y, feature_names
