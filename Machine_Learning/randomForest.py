import numpy as np
import learning_utils as lu
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate


def randomForestModel(graphs=False, showTest=False, random_state_classifier=None, balance_train=False, balance_test=False):
    X_train, y_train = lu.load_data("train", balance_samples=balance_train)

    rnd_clf = RandomForestClassifier(n_estimators=20, max_depth=8, min_samples_split=4, n_jobs=-1, random_state=random_state_classifier)

    print("\nMetrics on 10-fold Cross-validation")
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores = cross_validate(rnd_clf, X_train, y_train, scoring=scoring, cv=10, return_train_score=False)
    for scoreType in scores.keys():
        print("{}: {}".format(scoreType, scores[scoreType].mean()))

    rnd_clf.fit(X_train, y_train)

    print('\nRanking feature importances')
    importances = rnd_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.bar(range(X_train.shape[1]), importances[indices])
    plt.xticks(range(X_train.shape[1]), indices)
    plt.show()
    for i in range(X_train.shape[1]):
        print("%d . feature %s (%f)" % (i + 1, X_train.axes[1][indices[i]], importances[indices[i]]))

    if showTest:
        X_test, y_test = lu.load_data("test", balance_samples=balance_test)
        lu.metricsTestSet(X_test, y_test, rnd_clf)

    if graphs:
        lu.exportTreeGraphs('Forest_Graphs', rnd_clf.estimators_, X_train.axes[1])
