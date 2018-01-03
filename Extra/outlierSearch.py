import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.insert(0, '../Features')
sys.path.insert(0, '../Random_Forest')
import splitSets
import learning_utils as lu


def outlierRemoval():

    outliers1 = ["e31788d0-7834-477a-a718-fef116c04816",
                 "9a41dd95-337d-4f23-8b3e-f0f0dd40fc4d",
                 "64aedea6-b1f9-49da-8b10-3f02d8ed04b6",
                 "bae1bf32-94bf-42a7-96d0-ee23fd98245e",
                 "7fb7afc9-b006-4a44-99dc-409ba90d3fe8"]

    outliers2 = ["080274a4-cddf-47b7-9b8e-679153859229",
                 "6ed887bb-394b-40dc-a8d5-96e836468a8b"]

    outliers = outliers1 + outliers2

    walking_activity_features = pd.read_csv("../data/walking_activity_features.csv", index_col=0)
    dropRows = walking_activity_features[walking_activity_features.healthCode.isin(outliers)].index
    walking_activity_features.drop(dropRows, inplace=True)
    walking_activity_features.to_csv("../data/walking_activity_features.csv")


def outlierSearch(iterations):
    outlierRemoval()
    demographics = pd.read_csv("../data/demographics.csv", index_col=0)
    demographics.loc[:, "outlierCounter"] = 0
    demographics.loc[:, "valTestCounter"] = 0
    demographics.loc[:, "valTestCounterBad"] = 0
    demographics = demographics.loc[:, ["healthCode", "outlierCounter", "valTestCounter", "valTestCounterBad"]]

    total_auc = 0
    for i in range(iterations):
        print("\nIteration {}".format(i))
        splitSets.generateSetTables(quickSplit=True)
        val_auc, test_auc = randomForestModel(dropAge=True, ensemble_size=11)
        total_auc += (val_auc + test_auc)

        possibleOutliers = pd.read_csv("../data/val_extra_columns.csv", index_col=0).healthCode.unique()
        rowsToAdd = demographics['healthCode'].isin(possibleOutliers)
        demographics.loc[rowsToAdd, "valTestCounter"] += 1
        demographics.loc[rowsToAdd, "outlierCounter"] += 1 - val_auc
        if val_auc < 0.6:
            demographics.loc[rowsToAdd, "valTestCounterBad"] += 1

        possibleOutliers = pd.read_csv("../data/test_extra_columns.csv", index_col=0).healthCode.unique()
        rowsToAdd = demographics['healthCode'].isin(possibleOutliers)
        demographics.loc[rowsToAdd, "valTestCounter"] += 1
        demographics.loc[rowsToAdd, "outlierCounter"] += 1 - test_auc
        if test_auc < 0.6:
            demographics.loc[rowsToAdd, "valTestCounterBad"] += 1

    demographics.loc[:, "valTestBadProportion"] = demographics.valTestCounterBad / demographics.valTestCounter
    demographics.loc[:, "avgOutlierCounter"] = demographics.outlierCounter / demographics.valTestCounter
    demographics.sort_values(by=['avgOutlierCounter'], ascending=False, inplace=True)
    demographics.to_csv("outlierSort.csv")
    print("\nAvg AUC score:", total_auc / (2 * iterations))


def randomForestModel(oldAgeTrain=False, criterion='gini', ensemble_size=1, dropAge=False):

    X_train = {}
    y_train = {}
    rnd_clf = {}
    y_pred_total = {
        "val": 0,
        "test": 0,
    }
    metrics_train_total = {
        "Accuracy": 0,
        "Precision": 0,
        "Recall": 0,
        "F1 Score": 0,
        "ROC score": 0
    }
    X = {}
    X["val"], y_val, feature_names = lu.load_dataStandart("val", selectOldAge=True, dropAge=dropAge)
    X["test"], y_test, _ = lu.load_dataStandart("test", selectOldAge=True, dropAge=dropAge)

    for i in range(ensemble_size):
        X_train[i], y_train[i], _ = lu.load_dataStandart("train", selectOldAge=oldAgeTrain, dropAge=dropAge,
                                                         balance_undersampling=True)

        rnd_clf[i] = RandomForestClassifier(n_estimators=13, criterion=criterion, max_depth=5, min_samples_split=12, n_jobs=-1)

        rnd_clf[i].fit(X_train[i], y_train[i])
        lu.metricsAccumulate(X_train[i], y_train[i], rnd_clf[i], metrics_train_total)
        for setName in ["val", "test"]:
            y_pred_total[setName] += rnd_clf[i].predict_proba(X[setName]) > 0.5  # threshold

    lu.metricsShowAccumulate(metrics_train_total, ensemble_size)
    val_auc = lu.metricsShowEnsemble(y_val, y_pred_total["val"], "Validation", ensemble_size, threshold=0.5)
    test_auc = lu.metricsShowEnsemble(y_test, y_pred_total["test"], "Test", ensemble_size, threshold=0.5)

    return val_auc, test_auc
