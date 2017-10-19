import numpy as np
import sys
sys.path.insert(0, '../Features')
import features_utils as fu


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
