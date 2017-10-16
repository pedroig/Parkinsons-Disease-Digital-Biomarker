import pandas as pd
import numpy as np
import sys
import time
sys.path.insert(0, '../Features')
import features_utils as fu


def euclideanTimeSeries(data1, data2):
    if len(data1) < 2950 or len(data2) < 2950:
        return np.inf
    data1 = data1.iloc[:2950, :]
    data2 = data2.iloc[:2950, :]
    dist2 = 0
    for axis in ['x', 'y', 'z']:
        dist2 += ((data1.loc[:, axis] - data2.loc[:, axis])**2).sum()
    return np.sqrt(dist2)


X = pd.read_csv("../data/features_extra_columns.csv", index_col=0)
# Renaming to use the column name to access a named tuple
X.rename(columns={'accel_walking_rest.json.items': 'accel_walking_rest'}, inplace=True)
# Only PD samples
X = X[X.Target]

for row in X.itertuples():
    data = fu.readJSON_data(row.accel_walking_rest, 'accel_walking_rest')
    if len(data) < 2950:
        X.drop(row.Index, inplace=True)
X.reset_index(inplace=True)
X.to_csv("Time-series_Distance_Matrices/features_extra_columns2950.csv")

dim = len(X)
distances = np.full((dim, dim), np.inf)

for row1 in X.itertuples():
    startTime = time.time()
    ind1 = row1.Index
    print(100 * ind1 / dim, "%")
    distances[ind1][ind1] = 0
    pointer1 = row1.accel_walking_rest
    data1 = fu.readJSON_data(pointer1, 'accel_walking_rest')
    for row2 in X[ind1 + 1:].itertuples():
        ind2 = row2.Index
        pointer2 = row2.accel_walking_rest
        data2 = fu.readJSON_data(pointer2, 'accel_walking_rest')
        distances[ind1][ind2] = euclideanTimeSeries(data1, data2)
        distances[ind2][ind1] = distances[ind1][ind2]
    print("Time for {}: {}".format(ind1, time.time() - startTime))

np.save("Time-series_Distance_Matrices/distances_accel_walking_rest.npy", distances)
