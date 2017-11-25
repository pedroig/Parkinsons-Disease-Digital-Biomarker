import numpy as np
import pandas as pd
import os
from sklearn import tree
from sklearn import metrics
from imblearn.over_sampling import SMOTE


def metricsShow(X_test, y_test, clf, setName):
    print("\nMetrics on {} Set".format(setName))
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    y_prob_positiveClass = y_prob[:, 1]
    metricsPrint(y_test, y_pred, y_prob_positiveClass)


def metricsPrint(y_test, y_pred, y_prob):
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1 Score:", metrics.f1_score(y_test, y_pred))
    print("ROC score:", metrics.roc_auc_score(y_test, y_prob))


def exportTreeGraphs(folder, trees, names):
    for i_tree, tree_clf in enumerate(trees):
        path = '{}/{}/tree_{}.dot'.format(os.getcwd(), folder, str(i_tree))
        with open(path, 'w') as my_file:
            my_file = tree.export_graphviz(tree_clf,
                                           out_file=my_file,
                                           feature_names=names,
                                           class_names=['Normal', 'Parkinsons'],
                                           rounded=True,
                                           filled=True)


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
            random_healthy_indices = np.random.choice(pd_indices, len(pd_indices), replace=False)
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
