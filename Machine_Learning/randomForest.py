import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = pd.read_csv("../data/features.csv", index_col=0)
X = X.iloc[:500, :]		#For testing
y = X.loc[:, "Target"]
X = X.drop("Target", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rnd_clf = RandomForestClassifier(n_estimators=10, max_leaf_nodes=20, n_jobs=-1, verbose=True)
rnd_clf.fit(X_train, np.asarray(y_train.values, dtype="|S6"))

y_pred = rnd_clf.predict(X_test)

diffy = (y_pred == np.asarray(y_test.values, dtype="|S6"))

print("Test performance: ", 100*diffy.sum()/len(diffy))