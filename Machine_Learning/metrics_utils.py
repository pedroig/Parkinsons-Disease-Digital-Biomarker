from sklearn import metrics
import numpy as np
from sklearn import tree
import os

def metricsTestSet(X_test, y_test, clf):
	print("\nMetrics on Test Set")
	y_pred = clf.predict(X_test)
	print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
	print("Precision:", metrics.precision_score(y_test, y_pred))
	print("Recall:", metrics.recall_score(y_test, y_pred))
	print("F1 Score:", metrics.f1_score(y_test, y_pred))

def balanceSamples(X):
	X = X[X.loc[:, "age"] > 60]
	pd_indices = X[X.Target == 1].index
	healthy_indices = X[X.Target == 0].index
	random_pd_indices = np.random.choice(pd_indices, len(healthy_indices), replace=False)
	balanced_indices = np.append(random_pd_indices, healthy_indices)
	X = X.loc[balanced_indices,:]
	return X

def exportTreeGraphs(folder, trees, names):
	for i_tree, tree_clf in enumerate(trees):
		path = '{}/{}/tree_{}.dot'.format(os.getcwd(), folder, str(i_tree))
		with open(path, 'w') as my_file:
			my_file = tree.export_graphviz(	tree_clf, 
											out_file = my_file, 
											feature_names = names, 
											class_names = ['Normal', 'Parkinsons'], 
											rounded = True, 
											filled = True)