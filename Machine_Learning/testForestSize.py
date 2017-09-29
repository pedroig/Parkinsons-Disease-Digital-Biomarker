import matplotlib.pyplot as plt
import learning_utils as lu
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

def randomForestTesting(balance_train=False, balance_test=False):
	X_train, y_train = lu.load_data("train", balance_samples=balance_train)
	X_test, y_test = lu.load_data("test", balance_samples=balance_test)

	num_estimators_seq = range(1,40,2)
	roc_auc_cross = []
	roc_auc_test = []

	for n_estimators in num_estimators_seq:

		rnd_clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, max_depth=12)

		scores = cross_validate(rnd_clf, X_train, y_train, scoring="roc_auc", cv=10, return_train_score=False)
		roc_auc_cross.append(scores["test_score"].mean())

		rnd_clf.fit(X_train, y_train)
		y_pred = rnd_clf.predict(X_test)
		roc_auc_test.append(metrics.roc_auc_score(y_test, y_pred))

	plt.plot(num_estimators_seq, roc_auc_cross, color='red', label='Cross-validation', marker='o')
	plt.plot(num_estimators_seq, roc_auc_test, color='blue', label='Test set', marker='o')
	plt.xlabel("Number of trees")
	plt.ylabel("ROC score")
	plt.legend()
	fileName = 'roc_score_ForestSize'
	if balance_train:
		fileName += '_balancedTrain'
	if balance_test:
		fileName += '_balancedTest'
	plt.savefig('Forest_Graphs/{}.png'.format(fileName))
	plt.show()