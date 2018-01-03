import pandas as pd
import numpy as np
import createFeatures as cf
import utils
import time


def rowFeaturise(row, features, timeSeriesName, wavelet, level):
    """
    Input:
        - row: pandas Series
            The row of the features table that is going to be processed.
        - features: pandas DataFrame
            Table with pointers to JSON files that is going to have new columns with the extracted features.
        - timeSeriesName: string
            'deviceMotion_walking_outbound', 'deviceMotion_walking_rest' or 'pedometer_walking_outbound'
        - wavelet: string
            Wavelet to use, empty string if no wavelet is used for smoothing.
            example: 'db9'
        - level: integer
            Decomposition level for the wavelet. This parameter is not considered if no wavelet is used.
    """
    pointer = features.loc[row.name, timeSeriesName + '.json.items']
    if ~np.isnan(pointer):
        data = utils.readJSON_data(pointer, timeSeriesName)
        if (data is None) or data.empty:  # No file matching the pointer or data file Null
            features.loc[row.name, "Error"] = True
        else:
            if timeSeriesName.startswith("deviceMotion"):
                if len(data) < 300:     # Data too short
                    features.loc[row.name, "Error"] = True
                else:
                    dataAcc, dataRot = utils.preprocessDeviceMotion(data)
                    if wavelet is not '' and level is not None:
                        utils.waveletFiltering(dataAcc, wavelet, level)
                        utils.waveletFiltering(dataRot, wavelet, level)
                    utils.saveTimeSeries(dataAcc, pointer, timeSeriesName, 'Accel')
                    utils.saveTimeSeries(dataRot, pointer, timeSeriesName, 'RotRate')
                    cf.createFeatures(features, row.name, dataRot, 'RotRate_' + timeSeriesName)
                    cf.createFeatures(features, row.name, dataAcc, 'Accel_' + timeSeriesName)
            else:
                cf.createFeaturePedo(features, row.name, data, timeSeriesName)
    else:
        features.loc[row.name, "Error"] = True
    return features.loc[row.name]


def generateFeatures(num_cores=1, wavelet='', level=None):
    """
    Accesses the walking activity CSV table, executes the first stages of data cleaning,
    performs the feature generation and saves the results in a new table.

    Input:
        - num_cores: int (default=1)
            The number of worker processes to use.
        - wavelet: string (default='')
            Wavelet to use, empty string if no wavelet is used for smoothing.
            example: 'db9'
        - level: integer (default=None)
            Decomposition level for the wavelet. This parameter is not considered if no wavelet is used.
    """
    startTime = time.time()
    walking_activity = pd.read_csv("../data/walking_activity.csv", index_col=0)
    columns_to_keep_walking = [
        # 'ROW_VERSION',
        # 'recordId',
        'healthCode',
        # 'createdOn',
        # 'appVersion',
        # 'phoneInfo',
        # 'accel_walking_outbound.json.items',
        'deviceMotion_walking_outbound.json.items',
        'pedometer_walking_outbound.json.items',
        # 'accel_walking_return.json.items',
        # 'deviceMotion_walking_return.json.items',
        # 'pedometer_walking_return.json.items',
        # 'accel_walking_rest.json.items',
        'deviceMotion_walking_rest.json.items',
        'medTimepoint'
    ]
    walking_activity_features = walking_activity[columns_to_keep_walking]

    walking_activity_features.index.name = 'ROW_ID'
    walking_activity_features = walking_activity_features.sample(frac=1)

    walking_activity_features["Error"] = False
    for namePrefix in ['deviceMotion_walking_', 'pedometer_walking_']:
        for phase in ["outbound", "rest"]:  # , "return"]:
            timeSeriesName = namePrefix + phase
            if timeSeriesName == 'pedometer_walking_rest':
                continue
            print("Working on {}.".format(timeSeriesName))
            args = (timeSeriesName, wavelet, level)
            walking_activity_features = utils.apply_by_multiprocessing_featurise(walking_activity_features, rowFeaturise,
                                                                                 args=args, workers=num_cores)

            # Dropping rows with errors
            walking_activity_features = walking_activity_features[walking_activity_features.loc[:, "Error"] == False]

    walking_activity_features.drop('Error', axis=1, inplace=True)

    # Dropping rows with invalid values
    walking_activity_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    walking_activity_features.dropna(axis=0, how='any', inplace=True)

    fileName = 'walking_activity_features'
    walking_activity_features.to_csv("../data/{}.csv".format(fileName))

    print(len(walking_activity) - len(walking_activity_features), "rows dropped")
    print("Total time:", time.time() - startTime)
