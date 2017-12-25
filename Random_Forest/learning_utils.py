import numpy as np
import pandas as pd
from sklearn import metrics
from imblearn.over_sampling import SMOTE


def metricsAccumulate(X, y, clf, metrics_total):
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)
    y_prob = y_prob[:, 1]  # positiveClass
    metrics_total["Accuracy"] += metrics.accuracy_score(y, y_pred)
    metrics_total["Precision"] += metrics.precision_score(y, y_pred)
    metrics_total["Recall"] += metrics.recall_score(y, y_pred)
    metrics_total["F1 Score"] += metrics.f1_score(y, y_pred)
    metrics_total["ROC score"] += metrics.roc_auc_score(y, y_prob)


def metricsShowAcumulate(metrics_total, setName, ensemble_size):
    print("\nMetrics on {} Set".format(setName))
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC score"]:
        print("{}: {}".format(metric, metrics_total[metric] / ensemble_size))


def metricsPrint(y_test, y_pred, y_prob):
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1 Score:", metrics.f1_score(y_test, y_pred))
    print("ROC score:", metrics.roc_auc_score(y_test, y_prob))


def metricsShowEnsemble(y_test, y_pred_total, setName, ensemble_size, threshold=0.5):
    print("\nMetrics on {} Set".format(setName))
    y_prob = y_pred_total / ensemble_size
    y_prob = y_prob[:, 1]  # positiveClass
    y_pred = y_prob > threshold
    metricsPrint(y_test, y_pred, y_prob)


def load_data(featuresSplitName, selectOldAge=False, dropAge=False,
              balance_undersampling=False, balance_oversampling=False):

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
