import matplotlib.pyplot as plt
import learning_utils as lu
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

def randomForestTesting(balance_train=False, balance_test=False):
	X_train, y_train = lu.load_data("train", balance_samples=balance_train)
	X_test, y_test = lu.load_data("test", balance_samples=balance_test)

	num_estimators_seq = range(1,70,2)
	roc_auc_cross = []
	roc_auc_test = []

	std_values = {
					"max_depth" : 8,
					"n_estimators" : 20,
					"min_samples_split" : 4
	}

	seq = {
			"max_depth" : range(2, 20, 2),
			"n_estimators" : range(1, 70, 2),
			"min_samples_split" : range(2, 20, 2)
	}

	hyperparameterOptions = [	"max_depth",
								"n_estimators",
								"min_samples_split"]
	print("Choose the hyperparameter to test")
	for index, hyperparameter in enumerate(hyperparameterOptions):
		print(index, hyperparameter)
	hyperparameter = hyperparameterOptions[int(input("Select the corresponding number: "))]

	for nTest in seq[hyperparameter]:
		std_values[hyperparameter] = nTest

		rnd_clf = RandomForestClassifier(	n_estimators=std_values["n_estimators"],
											max_depth=std_values["max_depth"],
											min_samples_split=std_values["min_samples_split"],
											n_jobs=-1)

		scores = cross_validate(rnd_clf, X_train, y_train, scoring="roc_auc", cv=10, return_train_score=False)
		roc_auc_cross.append(scores["test_score"].mean())

		rnd_clf.fit(X_train, y_train)
		y_pred = rnd_clf.predict(X_test)
		roc_auc_test.append(metrics.roc_auc_score(y_test, y_pred))

	plt.plot(seq[hyperparameter], roc_auc_cross, color='red', label='Cross-validation', marker='o')
	plt.plot(seq[hyperparameter], roc_auc_test, color='blue', label='Test set', marker='o')
	plt.xlabel(hyperparameter)
	plt.ylabel("ROC score")
	plt.legend()
	fileName = 'roc_score_' + hyperparameter
	if balance_train:
		fileName += '_balancedTrain'
	if balance_test:
		fileName += '_balancedTest'
	plt.savefig('Forest_Graphs/{}.png'.format(fileName))
	plt.show()