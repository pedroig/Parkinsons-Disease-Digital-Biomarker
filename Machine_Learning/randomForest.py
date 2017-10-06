import numpy as np
import learning_utils as lu
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def randomForestModel(graphs=False, showTest=False, balance_train=False, balance_test=False, balance_val=False):
    X_train, y_train = lu.load_data("train", balance_samples=balance_train)

    rnd_clf = RandomForestClassifier(n_estimators=20, max_depth=12, min_samples_split=4, n_jobs=-1)
    rnd_clf.fit(X_train, y_train)
    lu.metricsPrint(X_train, y_train, rnd_clf, "training")

    X_val, y_val = lu.load_data("val", balance_samples=balance_val)
    lu.metricsPrint(X_val, y_val, rnd_clf, "validation")

    if showTest:
        X_test, y_test = lu.load_data("test", balance_samples=balance_test)
        lu.metricsPrint(X_test, y_test, rnd_clf, "test")

    print('\nRanking feature importances')
    importances = rnd_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.bar(range(X_train.shape[1]), importances[indices])
    plt.xticks(range(X_train.shape[1]), indices)
    plt.show()
    for i in range(X_train.shape[1]):
        print("%d . feature %s (%f)" % (i + 1, X_train.axes[1][indices[i]], importances[indices[i]]))

    if graphs:
        lu.exportTreeGraphs('Forest_Graphs', rnd_clf.estimators_, X_train.axes[1])
