import matplotlib.pyplot as plt
import learning_utils as lu
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate


def randomForestTesting(undersampling_train=False, oversampling_train=False,
                        oldAgeTrain=False, oldAgeVal=False,
                        dropAge=False, criterion='gini'):

    X_train, y_train, _ = lu.load_data("train", selectOldAge=oldAgeTrain, dropAge=dropAge,
                                       balance_undersampling=undersampling_train,
                                       balance_oversampling=oversampling_train)
    X_val, y_val, _ = lu.load_data("val", selectOldAge=oldAgeVal, dropAge=dropAge)

    roc_auc_cross = []
    roc_auc_val = []

    std_values = {
        "max_depth": 7,
        "n_estimators": 13,
        "min_samples_split": 12
    }

    seq = {
        "max_depth": range(2, 20, 1),
        "n_estimators": range(1, 30, 1),
        "min_samples_split": range(2, 20, 1)
    }

    hyperparameterOptions = ["max_depth",
                             "n_estimators",
                             "min_samples_split"]
    print("Choose the hyperparameter to test")
    for index, hyperparameter in enumerate(hyperparameterOptions):
        print(index, hyperparameter)
    hyperparameter = hyperparameterOptions[int(input("Select the corresponding number: "))]

    for nTest in seq[hyperparameter]:
        std_values[hyperparameter] = nTest

        rnd_clf = RandomForestClassifier(n_estimators=std_values["n_estimators"],
                                         max_depth=std_values["max_depth"],
                                         min_samples_split=std_values["min_samples_split"],
                                         criterion=criterion,
                                         n_jobs=-1)

        scores = cross_validate(rnd_clf, X_train, y_train, scoring="roc_auc", cv=10, return_train_score=False)
        roc_auc_cross.append(scores["test_score"].mean())

        rnd_clf.fit(X_train, y_train)
        y_prob = rnd_clf.predict_proba(X_val)
        roc_auc_val.append(metrics.roc_auc_score(y_val, y_prob[:, 1]))

    plt.plot(seq[hyperparameter], roc_auc_cross, color='red', label='Cross-validation (Training)', marker='o')
    plt.plot(seq[hyperparameter], roc_auc_val, color='blue', label='Validation set', marker='o')
    plt.xlabel(hyperparameter)
    plt.ylabel("ROC score")
    plt.legend()
    fileName = 'roc_score_{}_{}'.format(hyperparameter, criterion)

    if undersampling_train:
        fileName += '_undersampling'
    elif oversampling_train:
        fileName += '_oversampling'

    if oldAgeTrain:
        fileName += '_TrainAbove56years'
    if oldAgeVal:
        fileName += '_ValAbove56years'
    if dropAge:
        fileName += '_withoutAgeFeature'

    plt.savefig('Forest_Graphs/{}.png'.format(fileName))
    plt.show()
