import numpy as np
import learning_utils as lu
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def randomForestModel(undersampling_train=False, oversampling_train=False,
                      oldAgeTrain=False, oldAgeVal=False, oldAgeTest=False,
                      showTest=False, dropAge=False, graphs=False,
                      criterion='gini'):
    """
        Input:
        - undersamplig_train: bool (default=False)
            Whether to undersample the majority class in the training set.
        - oversamplig_train: bool (default=False)
            Whether to oversample the minority class in the training set.
        - oldAgeTrain: bool (default=False):
            Whether to select only people older 56 years in the training set.
        - oldAgeVal: bool (default=False)
            Whether to select only people older 56 years in the validation set.
        - oldAgeTest: bool (default=False)
            Whether to select only people older 56 years in the test set.
        - showTest: bool (default=False)
            Whether to show the metrics of predictions on the test set.
        - dropAge: bool (default=False)
            Whether to use age as a feature.
        - graphs: bool (default=False)
            Whether to export image representations of the decision trees in the forest.
        -criterion: string (default='gini')
            The function to measure the quality of a split: 'gini' or 'entropy'
    """
    X_train, y_train, feature_names = lu.load_data("train", selectOldAge=oldAgeTrain, dropAge=dropAge,
                                                   balance_undersampling=undersampling_train,
                                                   balance_oversampling=oversampling_train)

    # rnd_clf = RandomForestClassifier(n_estimators=13, criterion=criterion, max_depth=2, min_samples_split=120, n_jobs=-1)  # overregularization
    rnd_clf = RandomForestClassifier(n_estimators=13, criterion=criterion, max_depth=7, min_samples_split=12, n_jobs=-1)  # normal
    # rnd_clf = RandomForestClassifier(n_estimators=20, criterion=criterion, n_jobs=-1)  # overfitting

    rnd_clf.fit(X_train, y_train)
    lu.metricsShow(X_train, y_train, rnd_clf, "training")

    X_val, y_val, _ = lu.load_data("val", selectOldAge=oldAgeVal, dropAge=dropAge)
    lu.metricsShow(X_val, y_val, rnd_clf, "validation")

    if showTest:
        X_test, y_test, _ = lu.load_data("test", selectOldAge=oldAgeTest, dropAge=dropAge)
        lu.metricsShow(X_test, y_test, rnd_clf, "test")

    print('\nRanking feature importances')
    importances = rnd_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.bar(range(X_train.shape[1]), importances[indices])
    plt.xticks(range(X_train.shape[1]), indices)
    plt.show()
    for i in range(X_train.shape[1]):
        print("%d . feature %s (%f)" % (i + 1, feature_names[indices[i]], importances[indices[i]]))

    if graphs:
        lu.exportTreeGraphs('Forest_Graphs', rnd_clf.estimators_, feature_names)
