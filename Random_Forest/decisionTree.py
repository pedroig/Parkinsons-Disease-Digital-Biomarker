import learning_utils as lu
from sklearn import tree
from sklearn.model_selection import cross_validate


def decisionTreeModel(graphs=False, showTest=False, balance_train=False, balance_test=False):
    X_train, y_train = lu.load_data("train", balance_samples=balance_train)

    tree_clf = tree.DecisionTreeClassifier(max_depth=12)

    print("\nMetrics on 10-fold Cross-validation")
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores = cross_validate(tree_clf, X_train, y_train, scoring=scoring, cv=10, return_train_score=False)
    for scoreType in scores.keys():
        print("{}: {}".format(scoreType, scores[scoreType].mean()))

    if showTest:
        tree_clf.fit(X_train, y_train)
        X_test, y_test = lu.load_data("test", balance_samples=balance_test)
        lu.metricsShow(X_test, y_test, tree_clf, "test")

    if graphs:
        lu.exportTreeGraphs('DecisionTreeGraph', [tree_clf], X_train.axes[1])
