import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../Features')
import features_utils as fu


def generateSetFromTable(featuresTable, n_steps, n_inputs):
    axes = ['x', 'y', 'z']
    y = featuresTable.Target
    y = np.array(y)
    X = pd.DataFrame(columns=axes)
    seq_length = np.array([])
    for row in featuresTable.itertuples():
        data = fu.readJSON_data(row.accel_walking_rest, 'accel_walking_rest')
        seq_length = np.append(seq_length, len(data))
        XElement = data.loc[:, axes]
        zeros = pd.DataFrame(0, index=np.arange(n_steps - len(data)), columns=axes)
        X = pd.concat([X, XElement, zeros])
    X = np.asarray(X)
    X = X.reshape((-1, n_steps, n_inputs))
    return X, y, seq_length


def readPreprocessTable(name):
    featuresTable = pd.read_csv("../data/{}_extra_columns.csv".format(name), index_col=0)
    # Renaming to use the column name to access a named tuple
    featuresTable.rename(columns={'accel_walking_rest.json.items': 'accel_walking_rest'}, inplace=True)
    featuresTable.reset_index(inplace=True)
    return featuresTable


def findMaximumLength():
    maximumLength = -1
    for tableType in ['train', 'val', 'test']:
        print("Working on {} set.".format(tableType))
        featuresTable = pd.read_csv("../data/{}_extra_columns.csv".format(tableType), index_col=0)
        # Renaming to use the column name to access a named tuple
        featuresTable.rename(columns={'accel_walking_rest.json.items': 'accel_walking_rest'}, inplace=True)

        for row in featuresTable.itertuples():
            data = fu.readJSON_data(row.accel_walking_rest, 'accel_walking_rest')
            maximumLength = max(maximumLength, len(data))

    return maximumLength
