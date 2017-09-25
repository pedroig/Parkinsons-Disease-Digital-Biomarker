import pandas as pd
import numpy as np
import metrics_utils as mu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def randomForestModel(graphs=False, showTest=False, random_state_split=None, random_state_classifier=None, balance_samples=False):
	X = pd.read_csv("../data/features.csv", index_col=0)

	if balance_samples:
		X = mu.balanceSamples(X)

	y = X.loc[:, "Target"]
	y = np.asarray(y.values, dtype=np.int8)
	X = X.drop("Target", axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state_split)

	rnd_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True, max_depth=25, random_state=random_state_classifier)
	rnd_clf.fit(X_train, y_train)

	print("Score of the training dataset obtained using an out-of-bag estimate:", rnd_clf.oob_score_)
	
	if showTest:
		mu.metricsTestSet(X_test, y_test, rnd_clf)

	if graphs:
		mu.exportTreeGraphs('Forest_Graphs', rnd_clf.estimators_, X.axes[1])