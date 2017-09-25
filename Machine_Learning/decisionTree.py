import pandas as pd
import numpy as np
import metrics_utils as mu
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

def decisionTreeModel(graphs=False, showTest=False, random_state_split=None, balance_samples=False):
	X = pd.read_csv("../data/features.csv", index_col=0)
	
	if balance_samples:
		X = mu.balanceSamples(X)

	y = X.loc[:, "Target"]
	y = np.asarray(y.values, dtype=np.int8)
	X = X.drop("Target", axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state_split)

	tree_clf = tree.DecisionTreeClassifier(max_depth=25)

	print("\nMetrics on 10-fold Cross-validation")
	scoring = ["accuracy", "precision", "recall", "f1"]
	scores = cross_validate(tree_clf, X_train, y_train, scoring=scoring, cv=10, return_train_score=False)
	
	for scoreType in scores.keys():
		print("{}: {}".format(scoreType, scores[scoreType].mean()))
	
	if showTest:
		tree_clf.fit(X_train, y_train)
		mu.metricsTestSet(X_test, y_test, tree_clf)

	if graphs:
		mu.exportTreeGraphs('DecisionTreeGraph', [tree_clf], X.axes[1])