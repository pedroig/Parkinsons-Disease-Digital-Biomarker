import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../Features')
import utils
import features_utils as fu


wavelet = ""
level = 4
timeSeries = 'rest'


def dtwDistanceRadial(data1, data2, windowSize=300, reduceResolution=False, resolutionLoss=2):
    data1 = fu.cart2sphRadialDist(data1, raw=True)
    data2 = fu.cart2sphRadialDist(data2, raw=True)
    if reduceResolution:
        data1 = data1.groupby(data1.index // resolutionLoss).mean()
        data2 = data2.groupby(data2.index // resolutionLoss).mean()
    data1.index = np.arange(1, len(data1) + 1)
    data2.index = np.arange(1, len(data2) + 1)
    n1, n2 = len(data1), len(data2)
    DTW = np.full((n1 + 1, n2 + 1), np.inf)
    DTW[0, 0] = 0
    w = max(windowSize, abs(n1 - n2))

    for i in range(1, n1 + 1):
        for j in range(max(1, i - w), min(n2 + 1, i + w + 1)):
            dist = (data1[i] - data2[j])**2
            DTW[i, j] = dist + min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])

    return np.sqrt(DTW[n1, n2])


def dtwDistanceXYZ(data1, data2, windowSize=300):
    n1 = len(data1)
    n2 = len(data2)
    data1.index = np.arange(1, n1 + 1)
    data2.index = np.arange(1, n2 + 1)
    DTW = np.full((n1 + 1, n2 + 1), np.inf)
    DTW[0, 0] = 0
    w = max(windowSize, abs(n1 - n2))

    for i in range(1, n1 + 1):
        for j in range(max(1, i - w), min(n2 + 1, i + w + 1)):
            dist = 0
            for axis in ['x', 'y', 'z']:
                dist += (data1.loc[i, axis] - data2.loc[j, axis])**2
            DTW[i, j] = dist + min(DTW[i - 1, j], DTW[i, j - 1], DTW[i - 1, j - 1])

    return np.sqrt(DTW[n1, n2])


def euclideanTimeSeries(data1, data2):
    finalLen = min(len(data1), len(data2))
    data1 = data1.iloc[:finalLen, :]
    data2 = data2.iloc[:finalLen, :]
    dist2 = 0
    for axis in ['x', 'y', 'z']:
        dist2 += ((data1.loc[:, axis] - data2.loc[:, axis])**2).sum()
    return np.sqrt(dist2)


if __name__ == '__main__':
    X = pd.read_csv("../data/features_extra_columns.csv", index_col=0)
    # Renaming to use the column name to access a named tuple
    for timeSeriesName in ['outbound', 'rest']:  # , 'return']:
        X.rename(columns={'deviceMotion_walking_{}.json.items'.format(timeSeriesName): 'deviceMotion_walking_' + timeSeriesName}, inplace=True)

    healthCodes = X.healthCode.unique()
    fileNameRotRate = utils.genFileName('RotRate', wavelet, level)
    axes = ['x', 'y', 'z']
    timeSeriesName = 'deviceMotion_walking_' + timeSeries

    for healthCodeInd, healthCode in enumerate(healthCodes):
        print(100 * healthCodeInd / len(healthCodes), "%")
        codeSelected = X[X.healthCode == healthCode]
        codeSelected.reset_index(inplace=True)
        dim = len(codeSelected)
        if dim <= 10:
            continue

        data = []
        for row in codeSelected.itertuples():
            dataTemp = utils.readJSON_data(getattr(row, timeSeriesName), timeSeriesName, fileNameRotRate)
            data.append(dataTemp.loc[:, axes])

        # Number of clusters
        k = 10
        # healthCode indexes of random centroids
        centroidInitialIndexes = np.random.randint(low=0, high=dim, size=k)
        centroids = data[centroidInitialIndexes]

        # To store the value of centroids when it updates
        centroidsOld = np.zeros(centroids.shape)

        clusters = np.zeros(dim)

        while centroids != centroidsOld:
            # Assigning each value to its closest cluster
            for i in range(dim):
                distances = []
                for j in range(k):
                    distances.append(euclideanTimeSeries(data[i], centroids[j, :]))
                cluster = centroids[np.argmin(distances)]
                clusters[i] = cluster

            # Storing the old centroid values
            centroidsOld = centroids

            # Finding the new centroids by taking the average value

            # !!!!!!!!!!!!
            # TO BE CONTINUED !!!!!!!!!!!!!!!!!!!!!!!
            # !!!!!!!!!!!!
            # for i in range(k):
            #     points = [X[j] for j in range(dim) if clusters[j] == i]
            #     C[i] = np.mean(points, axis=0)
            # error = dist(C, C_old, None)
