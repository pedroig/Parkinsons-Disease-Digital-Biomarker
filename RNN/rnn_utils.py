import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../Features')
import features_utils as fu


def generateSetFromTable(featuresTable, n_steps, n_inputs, timeSeriesName, middle60=False, waveletFileName=''):
    timeSeriesName = 'accel_walking_' + timeSeriesName
    axes = ['x', 'y', 'z']
    y = featuresTable.Target
    y = np.array(y)
    X = pd.DataFrame(columns=axes)
    seq_length = np.array([])
    for row in featuresTable.itertuples():
        data = fu.readJSON_data(getattr(row, timeSeriesName), timeSeriesName, waveletFileName)
        if middle60:
            lower_bound = int(np.ceil(len(data) * 0.2))
            upper_bound = int(np.ceil(len(data) * 0.8))
            data = data.iloc[lower_bound: upper_bound]
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
    featuresTable = pd.read_csv("../data/{}_extra_columns{}.csv".format(name, waveletFiltering), index_col=0)
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
