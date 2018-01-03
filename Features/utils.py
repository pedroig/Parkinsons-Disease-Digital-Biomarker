import pandas as pd
import numpy as np
import multiprocessing
import os
import pywt
import json
import quaternion
from pandas.io.json import json_normalize

# File manipulation


def generatePath(pointer, timeSeriesName):
    """
    Outputs the path to find the specified JSON file.

    Input:
    - pointer: float
        Folder number to find the JSON file.
    - timeSeriesName: string
        'deviceMotion_walking_outbound', 'deviceMotion_walking_rest' or 'pedometer_walking_outbound'
    """
    pointer = int(pointer)
    path = '../data/{}/{}/{}/'
    path = path.format(timeSeriesName, str(pointer % 1000), str(pointer))
    return path


def readJSON_data(pointer, timeSeriesName, fileName=''):
    """
    Outputs a pandas DataFrame generated from the specified JSON file or None if the file could not be loaded.

    Input:
    - pointer: float
        Folder number to find the JSON file.
    - timeSeriesName: string
        'deviceMotion_walking_outbound', 'deviceMotion_walking_rest' or 'pedometer_walking_outbound'
    - fileName: string (default='')
        Name of the JSON file if it is a preprocessed version or an empty string if it is the original.
    """

    path = generatePath(pointer, timeSeriesName)
    try:
        if len(fileName) > 0:
            path += fileName
            json_df = pd.read_json(path, orient='split')
        else:
            for fileName in os.listdir(path):
                if fileName.startswith(timeSeriesName):
                    path += fileName
                    break
            if timeSeriesName.startswith('deviceMotion'):
                data = json.load(open(path))
                json_df = json_normalize(data)
                json_df.drop([
                    'magneticField.accuracy',
                    'magneticField.x',
                    'magneticField.y',
                    'magneticField.z',
                    'gravity.x',
                    'gravity.y',
                    'gravity.z'],
                    axis=1, inplace=True)
                for feature in ['attitude', 'rotationRate', 'userAcceleration']:
                    for axis in ['x', 'y', 'z']:
                        json_df.rename(columns={'{}.{}'.format(feature, axis): '{}{}'.format(feature, axis.upper())}, inplace=True)
                json_df.rename(columns={'attitude.w': 'attitudeW'}, inplace=True)
            else:
                json_df = pd.read_json(path)
    except:
        json_df = None
    return json_df


def loadUserInput():
    """
    Outputs two pandas DataFrames, one with time-series data of the acceleration or rotation rate and the second with pedometer data.
    The DataFrames are from a random sample but have properties as specified by the user. The user has the following options:
        * Choose between one of the three stages of the experiment (outbound, rest or return);
        * If the data is from a person with or without Parkinson's disease;
        * Choose between time-series data of the rotation rate or the acceleration.
    If the user selects the rest stage, the pedometer DataFrame returned is None.
    """
    timeSeriesOptions = ['walking_outbound',
                         'walking_return',
                         'walking_rest']
    print("Choose the time series")
    for index, timeSeriesName in enumerate(timeSeriesOptions):
        print(index, timeSeriesName)
    timeSeriesSelected = int(input("Select the corresponding number: "))

    target = int(input('\n0 for normal or 1 for PD: '))
    features = pd.read_csv('../data/features_extra_columns.csv', index_col=0)
    features = features[features.Target == target]

    sampled = features.sample().iloc[0]

    timeSeriesName = "deviceMotion_" + timeSeriesOptions[timeSeriesSelected]
    pointerDeviceMotion = sampled[timeSeriesName + '.json.items']
    dataDeviceMotion = readJSON_data(pointerDeviceMotion, timeSeriesName)
    dataAcc, dataRot = preprocessDeviceMotion(dataDeviceMotion)

    if timeSeriesSelected == 2:
        dataPedo = None
    else:
        timeSeriesName = "pedometer_" + timeSeriesOptions[timeSeriesSelected]
        pointerPedo = sampled[timeSeriesName + '.json.items']
        dataPedo = readJSON_data(pointerPedo, timeSeriesName)

    mainData = int(input('\n0 for Rotation Rate or 1 for Acceleration: '))
    if mainData == 0:
        return dataRot, dataPedo
    else:
        return dataAcc, dataPedo


def saveTimeSeries(data, pointer, timeSeriesName, sampleType):
    """
    Converts the time-series table to a JSON string.

    Input:
        - data: pandas DataFrame
            Time-series table.
        - pointer: float
            Folder number to find the JSON file.
        - timeSeriesName: string
            'deviceMotion_walking_outbound', 'deviceMotion_walking_rest' or 'pedometer_walking_outbound'
        - sampleType: string
            'Accel' or 'RotRate'
    """
    path = generatePath(pointer, timeSeriesName)
    filePath = "{}{}.json".format(path, sampleType)
    data.to_json(filePath, orient='split')

# Augmentation and data preprocessing


def waveletFiltering(data, wavelet, level):
    """
    Applies the specified smoothing to the time-series.

    Input:
        - data: pandas DataFrame
            Time-series table.
        - wavelet: string
            Wavelet to use.
            example: 'db9'
        - level: integer
            Decomposition level for the wavelet.
    """
    for axis in ['x', 'y', 'z']:
        coeffs = pywt.wavedec(data[axis], wavelet, level=level)
        data[axis] = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(data))


def generateUnitQuaternion(theta, ux, uy, uz):
    """
    Input:
    - theta: float
        Angle
    - ux, uy, uz: float
        Vector components
    """
    uNorm = np.sqrt(ux**2 + uy**2 + uz**2)
    ux /= uNorm
    uy /= uNorm
    uz /= uNorm

    # Unit quaternion
    q = quaternion.Quaternion(np.cos(theta / 2), ux * np.sin(theta / 2), uy * np.sin(theta / 2), uz * np.sin(theta / 2))
    return q


def rotation3D(q, dataX, dataY, dataZ):
    """
    Input:
    - theta: float
        Unit quaternion of the corresponding rotation
    - dataX, dataY, dataZ: float
        Vector components of the data that is going to be rotated

    Returns:
    - out : ndarray
        Array with the three components of the rotated data
    """
    dataQ = quaternion.Quaternion(0, dataX, dataY, dataZ)
    out = (q * dataQ * q.inverse()).im
    return out


def preprocessDeviceMotion(data):
    """
    Outputs two DataFrames with time-series data in the world frame coordinate system, one of the
    acceleration and a second of the rotation rate.

    Input:
        data: pandas DataFrame
            DataFrame with time-series data of the acceleration, the rotation rate and the attitude.
    """
    dataAcc = data['timestamp'].to_frame()
    dataRot = data['timestamp'].to_frame()
    for index, row in enumerate(data.itertuples()):
        q = quaternion.Quaternion(row.attitudeW, row.attitudeX, row.attitudeY, row.attitudeZ)
        rotRate = rotation3D(q, row.rotationRateX, row.rotationRateY, row.rotationRateZ)
        userAccel = rotation3D(q, row.userAccelerationX, row.userAccelerationY, row.userAccelerationZ)
        for axisNum, axis in enumerate(['x', 'y', 'z']):
            dataRot.loc[index, axis] = rotRate[axisNum]
            dataAcc.loc[index, axis] = userAccel[axisNum]
    return dataAcc, dataRot


def augmentRow(rowFeaturesTable):
    """
    Generates JSON augmented versions for the sample in one row of the table.

    Input:
        - rowFeaturesTable: pandas Series
            The row of the features table that is going to be processed.
    """
    axes = ['x', 'y', 'z']

    for timeSeriesName in ['deviceMotion_walking_outbound', 'deviceMotion_walking_rest']:
        columnName = timeSeriesName + '.json.items'
        pointer = rowFeaturesTable[columnName]
        data = readJSON_data(pointer, timeSeriesName, 'RotRate.json')

        max_scale = 1.2
        min_scale = 0.8
        random_scale = np.random.random_sample() * (max_scale - min_scale) + min_scale
        data[axes] = data[axes] * random_scale
        theta = np.random.random_sample() * np.pi * 2
        q = generateUnitQuaternion(theta, 0, 0, 1)
        for index, row in enumerate(data.itertuples()):
            rotRate = rotation3D(q, row.x, row.y, row.z)
            for axisNum, axis in enumerate(axes):
                data.loc[index, axis] = rotRate[axisNum]

        path = generatePath(pointer, timeSeriesName)
        fileName = 'RotRate_augmented.json'
        data.to_json(path + fileName, orient='split')


def generateAugmentedTable(tableName, augmentFraction=0.5):
    """
    Generates an augmented version of a given table.

    Warning: the features in the table are not updated, the purpose of this table is to
    have additional rows with pointers to the augmented JSON files.

    Input:
    - tableName: string
    - augmentFraction: float
        0 < augmentFraction <=1
    """
    table = pd.read_csv("../data/{}_extra_columns.csv".format(tableName))
    table.loc[:, "augmented"] = False
    tableSelected = table.sample(frac=augmentFraction)
    tableSelected.loc[:, "augmented"] = True

    tableAugmented = pd.concat([table, tableSelected])
    tableAugmented.to_csv("../data/{}_augmented_extra_columns.csv".format(tableName))

# Parallelization for the augmentation (augmentRow)


def apply_df_augmentation(args):
    """
        Input:
            args: tuple
                A tuple with the DataFrame and the function to be applied.
    """
    df, func = args
    return df.apply(func, axis=1)


def apply_by_multiprocessing_augmentation(df, func, workers):
    """
    Applies a function to all the rows of a DataFrame with the use of multiple processes.

    Input:
        df: pandas DataFrame
            DataFrame that is going to be operated by multiple processes.
        func: function
            The function applied to the DataFrame.
        workers: int
            The number of processes.
    """
    pool = multiprocessing.Pool(processes=workers)
    pool.map(apply_df_augmentation,
             [(df_fraction, func) for df_fraction in np.array_split(df, workers)])
    pool.close()

# Parallelization for data cleaning and extraction of features (rowFeaturise)


def apply_df_featurise(args):
    """
        Input:
            args: tuple
                A tuple with the DataFrame, the function, and the function parameters.
    """
    df, func, kwargs = args
    timeSeriesName, wavelet, level = kwargs["args"]
    args = df, timeSeriesName, wavelet, level
    return df.apply(func, args=args, axis=1)


def apply_by_multiprocessing_featurise(df, func, **kwargs):
    """
    Applies a function to all the rows of a DataFrame with the use of multiple processes.

    Input:
        df: pandas DataFrame
            DataFrame that is going to be operated by multiple processes.
        func: function
            The function applied to the DataFrame.
        kwargs: dict
            Dictionary with the parameters that are going to be used by func and the number
            of processes under the keyword 'workers'.
    """
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(apply_df_featurise, [(d, func, kwargs)
                                           for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))

# Treating outliers in the dataset


def outlierRemovalSaving():
    """
    Generates new 'extra_columns' tables that do not contain healthCodes from a specific list
    of possible outliers.
    """

    for dataSplitName in ["train", "train_augmented", "test", "val", "features"]:
        table = pd.read_csv("../data/{}_extra_columns.csv".format(dataSplitName), index_col=0)
        table = outlierRemoval(table)
        table.to_csv("../data/{}_noOutliers_extra_columns.csv".format(dataSplitName))


def outlierRemoval(table):
    """
    Outputs a new table without the rows from the input table that contain healthCodes
    from a specific list of possible outliers.

    Input:
        table: pandas DataFrame
    """
    outliers1 = ["e31788d0-7834-477a-a718-fef116c04816",
                 "9a41dd95-337d-4f23-8b3e-f0f0dd40fc4d",
                 "64aedea6-b1f9-49da-8b10-3f02d8ed04b6",
                 "bae1bf32-94bf-42a7-96d0-ee23fd98245e",
                 "7fb7afc9-b006-4a44-99dc-409ba90d3fe8"]

    outliers2 = ["080274a4-cddf-47b7-9b8e-679153859229",
                 "6ed887bb-394b-40dc-a8d5-96e836468a8b"]

    outliers = outliers1 + outliers2

    dropRows = table[table.healthCode.isin(outliers)].index
    return table.drop(dropRows)
