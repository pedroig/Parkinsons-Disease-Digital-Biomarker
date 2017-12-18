import numpy as np
import learning_utils as lu
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def randomForestModel(undersampling_train=False, oversampling_train=False,
                      oldAgeTrain=False, oldAgeVal=False, oldAgeTest=False,
                      showTest=False, dropAge=False,
                      criterion='gini', ensemble_size=1):
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
        -criterion: string (default='gini')
            The function to measure the quality of a split: 'gini' or 'entropy'
        -ensemble_size: int
            Number of classifiers trained on different training sets when undersampling is applied. This number must be odd.
    """
    if undersampling_train is False:
        ensemble_size = 1

    X_train = {}
    y_train = {}
    rnd_clf = {}
    y_pred_total = {
        "val": 0,
        "test": 0,
        "val_test": 0
    }
    metrics_train_total = {
        "Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1 Score": 0,
        "ROC score": 0
    }
    importances = 0
    X = {}
    X["val"], y_val, feature_names = lu.load_data("val", selectOldAge=oldAgeVal, dropAge=dropAge)
    X["test"], y_test, _ = lu.load_data("test", selectOldAge=oldAgeTest, dropAge=dropAge)
    X["val_test"] = np.concatenate((X["val"], X["test"]))
    y_val_test = np.concatenate((y_val, y_test))

    for i in range(ensemble_size):
        X_train[i], y_train[i], _ = lu.load_data("train", selectOldAge=oldAgeTrain, dropAge=dropAge,
                                                 balance_undersampling=undersampling_train,
                                                 balance_oversampling=oversampling_train)

        # rnd_clf[i] = RandomForestClassifier(n_estimators=13, criterion=criterion, max_depth=2, min_samples_split=120, n_jobs=-1)  # overregularization
        rnd_clf[i] = RandomForestClassifier(n_estimators=13, criterion=criterion, max_depth=5, min_samples_split=12, n_jobs=-1)  # normal
        # rnd_clf[i] = RandomForestClassifier(n_estimators=20, criterion=criterion, n_jobs=-1)  # overfitting

        rnd_clf[i].fit(X_train[i], y_train[i])
        importances += rnd_clf[i].feature_importances_
        lu.metricsAcumulate(X_train[i], y_train[i], rnd_clf[i], metrics_train_total)
        for setName in ["val", "test", "val_test"]:
            y_pred_total[setName] += rnd_clf[i].predict_proba(X[setName]) > 0.5  # threshold

    lu.metricsShowAcumulate(metrics_train_total, "Training", ensemble_size)
    lu.metricsShowEnsemble(y_val, y_pred_total["val"], "Validation", ensemble_size, threshold=0.5)

    if showTest:
        lu.metricsShowEnsemble(y_test, y_pred_total["test"], "Test", ensemble_size, threshold=0.5)
        lu.metricsShowEnsemble(y_val_test, y_pred_total["val_test"], "Validation + Test", ensemble_size, threshold=0.5)

    print('\nRanking feature importances')
    importances /= ensemble_size
    indices = np.argsort(importances)[::-1]
    plt.figure()
    number_of_features = X["val"].shape[1]
    plt.bar(range(number_of_features), importances[indices])
    plt.xticks(range(number_of_features), indices)
    plt.show()
    for i in range(number_of_features):
        print("%d . feature %s (%f)" % (i + 1, feature_names[indices[i]], importances[indices[i]]))
