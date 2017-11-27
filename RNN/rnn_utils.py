import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../Features')
import utils


def generateSetFromTable(featuresTable, n_steps, n_inputs, wavelet, level):
    fileNameRotRate = utils.genFileName('RotRate', wavelet, level)
    fileNameAccel = utils.genFileName('Accel', wavelet, level)
    axes = ['x', 'y', 'z']
    y = featuresTable.Target
    y = np.array(y)
    X = {}
    columns = []
    for feature in ['Accel_', 'RotRate_']:
        for axis in axes:
            columns.append(feature + axis)
    seq_length = {}
    for timeSeries in ['outbound', 'rest']:  # , 'return']:
        timeSeriesName = 'deviceMotion_walking_' + timeSeries
        X[timeSeries] = pd.DataFrame(columns=columns)
        seq_length[timeSeries] = np.array([])
        for row in featuresTable.itertuples():
            dataAccel = utils.readJSON_data(getattr(row, timeSeriesName), timeSeriesName, fileNameAccel)
            dataAccel.rename(inplace=True,
                             columns={
                                 'x': 'Accel_x',
                                 'y': 'Accel_y',
                                 'z': 'Accel_z'
                             })
            dataRotRate = utils.readJSON_data(getattr(row, timeSeriesName), timeSeriesName, fileNameRotRate)
            dataRotRate.rename(inplace=True,
                               columns={
                                   'x': 'RotRate_x',
                                   'y': 'RotRate_y',
                                   'z': 'RotRate_z'
                               })
            data = pd.merge(dataAccel, dataRotRate, on="timestamp")
            seq_length[timeSeries] = np.append(seq_length[timeSeries], len(data))
            XElement = data.loc[:, columns]
            zeros = pd.DataFrame(0, index=np.arange(n_steps - len(data)), columns=columns)
            X[timeSeries] = pd.concat([X[timeSeries], XElement, zeros])
        X[timeSeries] = np.asarray(X[timeSeries])
        X[timeSeries] = X[timeSeries].reshape((-1, n_steps, n_inputs))
    return X, y, seq_length


def readPreprocessTable(name):
    featuresTable = pd.read_csv("../data/{}_extra_columns.csv".format(name), index_col=0)
    # Renaming to use the column name to access a named tuple
    for timeSeriesName in ['outbound', 'rest']:  # , 'return']:
        featuresTable.rename(columns={'deviceMotion_walking_{}.json.items'.format(timeSeriesName): 'deviceMotion_walking_' + timeSeriesName}, inplace=True)
    featuresTable.reset_index(inplace=True)
    return featuresTable


def findMaximumLength(timeSeriesName):
    """
        Input:
        - timeSeriesName: 'rest', 'outbound', 'return'

        Outputs:
        maximum length for the time-series stage selected
    """
    timeSeriesName = 'deviceMotion_walking_' + timeSeriesName
    maximumLength = -1
    featuresTable = pd.read_csv("../data/features_extra_columns.csv", index_col=0)
    # Renaming to use the column name to access a named tuple
    featuresTable.rename(columns={timeSeriesName + '.json.items': timeSeriesName}, inplace=True)

    for row in featuresTable.itertuples():
        data = utils.readJSON_data(getattr(row, timeSeriesName), timeSeriesName)
        maximumLength = max(maximumLength, len(data))

    return maximumLength
