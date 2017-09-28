import pandas as pd
from sklearn import preprocessing

def getDataNorm():
	X = pd.read_csv("../../data/features.csv", index_col=0)
	y = X.loc[:, "Target"]
	X = X.drop("Target", axis=1)
	X_norm = preprocessing.scale(X)
	return X_norm, y