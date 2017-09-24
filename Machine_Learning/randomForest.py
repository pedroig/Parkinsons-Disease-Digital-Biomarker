import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import os

def randomForestModel(graphs=False, showTest=False, random_state_split=None, random_state_classifier=None):
	X = pd.read_csv("../data/features.csv", index_col=0)
	y = X.loc[:, "Target"]
	X = X.drop("Target", axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state_split)

	rnd_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True, verbose=True, max_depth=10, random_state=random_state_classifier)
	rnd_clf.fit(X_train, np.asarray(y_train.values, dtype="|S6"))

	print("Score of the training dataset obtained using an out-of-bag estimate: ", 100*rnd_clf.oob_score_)
	
	if showTest:
		y_pred = rnd_clf.predict(X_test)
		diffy = (y_pred == np.asarray(y_test.values, dtype="|S6"))
		print("Test Set accuracy: ", 100*diffy.sum()/len(diffy))

	if graphs:
		for i_tree, tree_in_forest in enumerate(rnd_clf.estimators_):
			path = '{}/Forest_Graphs/tree_{}.dot'.format(os.getcwd(), str(i_tree))
			with open(path, 'w') as my_file:
				my_file = tree.export_graphviz(	tree_in_forest, 
												out_file = my_file, 
												feature_names = X.axes[1], 
												class_names = ['Parkinsons', 'Normal'], 
												rounded = True, 
												filled = True)