import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../Features')
import utils


def generateSetFromTable(featuresTable, wavelet, level, timeSeries, channels_input, timeSeriesPaddedLength):
    '''
    Input:
        -timeSeries: string
            'outbound' or 'rest'
    '''
    fileNameRotRate = utils.genFileName('RotRate', wavelet, level)
    axes = ['x', 'y', 'z']
    y = featuresTable.Target
    y = np.array(y)
    X = {}
    timeSeriesName = 'deviceMotion_walking_' + timeSeries
    X = pd.DataFrame(columns=axes)
    for row in featuresTable.itertuples():
        data = utils.readJSON_data(getattr(row, timeSeriesName), timeSeriesName, fileNameRotRate)
        XElement = data.loc[:, axes]
        zeros = pd.DataFrame(0, index=np.arange(timeSeriesPaddedLength - len(data)), columns=axes)
        X = pd.concat([X, XElement, zeros])
    X = np.asarray(X)
    X = X.reshape((-1, timeSeriesPaddedLength, channels_input))
    return X, y


def readPreprocessTable(name):
    featuresTable = pd.read_csv("../data/{}_extra_columns.csv".format(name), index_col=0)
    # Renaming to use the column name to access a named tuple
    for timeSeriesName in ['outbound', 'rest']:  # , 'return']:
        featuresTable.rename(columns={'deviceMotion_walking_{}.json.items'.format(timeSeriesName): 'deviceMotion_walking_' + timeSeriesName}, inplace=True)
    featuresTable.reset_index(inplace=True)
    return featuresTable
