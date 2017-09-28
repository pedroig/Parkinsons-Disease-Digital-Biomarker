import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import visualization_utils as vu

X, y = vu.getDataNorm()

pca = PCA(n_components = 3)
X3 = pca.fit_transform(X)
print("\nPCA variance per axis for 3 components:")
print(pca.explained_variance_ratio_)

pca = PCA(n_components = 0.95)
Xvar95 = pca.fit_transform(X)
print("\nNumber of components to keep 95% variance using PCA:", Xvar95.shape[1])