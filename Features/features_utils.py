import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import mutual_info_score
from scipy import stats

# Mixed dimensions


def cart2sphRadialDist(data, raw=False):
    hxy = np.hypot(data['x'], data['y'])
    hxyz = np.hypot(hxy, data['z'])
    if not raw:
        hxyz = hxyz.mean()
    return hxyz


def cart2sphPolarAngle(data, raw=False):
    pAngle = np.arctan2(data['y'], data['x'])
    if not raw:
        pAngle = pAngle.mean()
    return pAngle


def cart2sphAzimuthAngle(data, raw=False):
    hxy = np.hypot(data['x'], data['y'])
    azAngle = np.arctan2(data['z'], hxy)
    if not raw:
        azAngle = azAngle.mean()
    return azAngle


def sma(data):
    ans = 0
    for axis in ['x', 'y', 'z']:
        ans += np.abs(data.loc[:, axis]).sum()
    ans /= len(data)
    return ans

# 1 Dimension


def dataRange(data, axis):
    return data[axis].max() - data[axis].min()


def interquartile(data, axis):
    return data[axis].quantile(0.75) - data[axis].quantile(0.25)


def rms(data, axis):
    return np.sqrt((data[axis]**2).sum() / len(data))


def zeroCrossingRate(data, axis):
    data1 = np.array(data[axis].iloc[:-1])
    data2 = np.array(data[axis].iloc[1:])
    return (data1 * data2 < 0).sum()


def entropy(data, axis, bins=10):
    hist = np.histogram(data.loc[:, axis], bins)[0]
    return stats.entropy(hist)


def dominantFreqComp(data, axis):
    sp = np.fft.fft(data[axis])
    freq = np.fft.fftfreq(len(data), d=0.01)
    freqIndex = np.argmax(np.abs(sp))
    dominantFreq = freq[freqIndex]
    return np.abs(dominantFreq)

# 2 Dimensions


def crossCorrelation(data, axis1, axis2, lag=0):
    diffs = pd.DataFrame(columns=[axis1, axis2])
    diffs.loc[:, axis1] = data[axis1] - data.loc[:, axis1].mean()
    diffs.loc[:, axis2] = data[axis2] - data.loc[:, axis2].mean()
    num = (diffs.iloc[:len(diffs) - lag, 0] * diffs.iloc[lag:, 1]).sum()

    diffs.loc[:, axis1] = diffs.loc[:, axis1]**2
    diffs.loc[:, axis2] = diffs.loc[:, axis2]**2
    den = np.sqrt(diffs.loc[:, axis1].sum() * diffs.loc[:, axis2].sum())

    return num / den


def mutualInfo(data, axis1, axis2, bins=10):
    contingency = np.histogram2d(data.loc[:, axis1], data.loc[:, axis2], bins)[0]
    mi = mutual_info_score(None, None, contingency=contingency)
    return mi


def crossEntropy(data, axis1, axis2, bins=10):
    low = min(data.loc[:, axis1].min(), data.loc[:, axis2].min())
    high = max(data.loc[:, axis1].max(), data.loc[:, axis2].max())
    hist1 = np.histogram(data.loc[:, axis1], bins=bins, range=(low, high))[0]
    hist2 = np.histogram(data.loc[:, axis2], bins=bins, range=(low, high))[0]
    return stats.entropy(hist1, hist2)

# Extra for the pedometer data


def avgSpeed(data):
    start = datetime.strptime(data.loc[data.index[-1], 'startDate'], '%Y-%m-%dT%H:%M:%S%z')
    end = datetime.strptime(data.loc[data.index[-1], 'endDate'], '%Y-%m-%dT%H:%M:%S%z')
    delta = end - start
    dist = data.loc[data.index[-1], 'distance']
    if delta.seconds == 0:
        return np.nan
    else:
        return dist / delta.seconds


def avgStep(data):
    stepNum = data.loc[data.index[-1], 'numberOfSteps']
    if stepNum == 0:
        return np.nan
    else:
        return data.loc[data.index[-1], 'distance'] / stepNum

# Extra for loading data


def readJSON_data(pointer, timeSeriesName):
    pointer = int(pointer)
    path = '../data/{}/{}/{}/'
    path = path.format(timeSeriesName, str(pointer % 1000), str(pointer))
    try:
        for fileName in os.listdir(path):
            if fileName.startswith(timeSeriesName):
                path += fileName
                break
        json_df = pd.read_json(path)
    except IOError:
        json_df = None
    return json_df


def loadUserInput():
    timeSeriesOptions = ['accel_walking_outbound',
                         'accel_walking_return',
                         'accel_walking_rest']
    print("Choose the time series")
    for index, timeSeriesName in enumerate(timeSeriesOptions):
        print(index, timeSeriesName)
    timeSeriesSelected = int(input("Select the corresponding number: "))

    target = int(input('\n0 for normal or 1 for PD: '))
    features = pd.read_csv('../data/features_extra_columns.csv', index_col=0)
    features = features[features.Target == target]

    timeSeriesName = timeSeriesOptions[timeSeriesSelected] + '.json.items'
    pointer = features[timeSeriesName].sample().iloc[0]
    data = readJSON_data(pointer, timeSeriesOptions[timeSeriesSelected])
    return data
