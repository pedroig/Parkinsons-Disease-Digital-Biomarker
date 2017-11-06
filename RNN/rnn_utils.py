import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../Features')
import features_utils as fu


def generateSetFromTable(featuresTable, n_steps, n_inputs, waveletFileName=''):
    axes = ['x', 'y', 'z']
    y = featuresTable.Target
    y = np.array(y)
    X = {}
    seq_length = {}
    for timeSeries in ['outbound', 'rest', 'return']:
        timeSeriesName = 'accel_walking_' + timeSeries
        X[timeSeriesName] = pd.DataFrame(columns=axes)
        seq_length[timeSeriesName] = np.array([])
        for row in featuresTable.itertuples():
            data = fu.readJSON_data(getattr(row, timeSeriesName), timeSeriesName, waveletFileName)
            seq_length[timeSeriesName] = np.append(seq_length[timeSeriesName], len(data))
            XElement = data.loc[:, axes]
            zeros = pd.DataFrame(0, index=np.arange(n_steps[timeSeriesName] - len(data)), columns=axes)
            X[timeSeriesName] = pd.concat([X[timeSeriesName], XElement, zeros])
        X[timeSeriesName] = np.asarray(X[timeSeriesName])
        X[timeSeriesName] = X.reshape((-1, n_steps[timeSeriesName], n_inputs))
    return X, y, seq_length


def generateSetFromTable1TimeSeries(featuresTable, n_steps, n_inputs, timeSeriesName, waveletFileName=''):
    timeSeriesName = 'accel_walking_' + timeSeriesName
    axes = ['x', 'y', 'z']
    y = featuresTable.Target
    y = np.array(y)
    X = pd.DataFrame(columns=axes)
    seq_length = np.array([])
    for row in featuresTable.itertuples():
        data = fu.readJSON_data(getattr(row, timeSeriesName), timeSeriesName, waveletFileName)
        seq_length = np.append(seq_length, len(data))
        XElement = data.loc[:, axes]
        zeros = pd.DataFrame(0, index=np.arange(n_steps - len(data)), columns=axes)
        X = pd.concat([X, XElement, zeros])
    X = np.asarray(X)
    X = X.reshape((-1, n_steps, n_inputs))
    return X, y, seq_length


def readPreprocessTable(name, timeSeriesName, chooseWaveletTable):
    waveletFiltering = ''
    if chooseWaveletTable:
        waveletFiltering = '_waveletFiltering'
    featuresTable = pd.read_csv("../data/{}{}_extra_columns.csv".format(name, waveletFiltering), index_col=0)
    # Renaming to use the column name to access a named tuple
    featuresTable.rename(columns={'accel_walking_{}.json.items'.format(timeSeriesName): 'accel_walking_' + timeSeriesName}, inplace=True)
    featuresTable.reset_index(inplace=True)
    return featuresTable


def findMaximumLength(timeSeriesName):
    """
        Input:
        - timeSeriesName: 'rest', 'outbound', 'return'

        Outputs:
        maximum length for the time-series stage selected
    """
    timeSeriesName = 'accel_walking_' + timeSeriesName
    maximumLength = -1
    for tableType in ['train', 'val', 'test']:
        print("Working on {} set.".format(tableType))
        featuresTable = pd.read_csv("../data/{}_extra_columns.csv".format(tableType), index_col=0)
        # Renaming to use the column name to access a named tuple
        featuresTable.rename(columns={timeSeriesName + '.json.items': timeSeriesName}, inplace=True)

        for row in featuresTable.itertuples():
            data = fu.readJSON_data(getattr(row, timeSeriesName), timeSeriesName)
            maximumLength = max(maximumLength, len(data))

    return maximumLength
