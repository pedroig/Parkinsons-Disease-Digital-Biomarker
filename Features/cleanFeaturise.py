import pandas as pd
import numpy as np
import createFeatures as cf
import utils
import time
import multiprocessing


def apply_df(args):
    df, func, kwargs = args
    timeSeriesName, wavelet, level = kwargs["args"]
    args = df, timeSeriesName, wavelet, level
    return df.apply(func, args=args, axis=1)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(apply_df, [(d, func, kwargs)
                                 for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))


def rowFeaturise(row, features, timeSeriesName, wavelet, level):
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
                    utils.saveTimeSeries(dataAcc, pointer, timeSeriesName, 'Accel', wavelet, level)
                    utils.saveTimeSeries(dataRot, pointer, timeSeriesName, 'RotRate', wavelet, level)
                    cf.createFeatures(features, row.name, dataRot, 'RotRate_' + timeSeriesName)
                    cf.createFeatures(features, row.name, dataAcc, 'Accel_' + timeSeriesName)
            else:
                cf.createFeaturePedo(features, row.name, data, timeSeriesName)
    else:
        features.loc[row.name, "Error"] = True
    return features.loc[row.name]


def generateFeatures(num_cores=1, dataFraction=1, wavelet='', level=None):
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
    walking_activity_features = walking_activity_features.sample(frac=dataFraction)

    walking_activity_features["Error"] = False
    for namePrefix in ['deviceMotion_walking_', 'pedometer_walking_']:
        for phase in ["outbound", "rest"]:  # , "return"]:
            timeSeriesName = namePrefix + phase
            if timeSeriesName == 'pedometer_walking_rest':
                continue
            print("Working on {}.".format(timeSeriesName))
            args = (timeSeriesName, wavelet, level)
            walking_activity_features = apply_by_multiprocessing(walking_activity_features, rowFeaturise, args=args, workers=num_cores)

            # Dropping rows with errors
            walking_activity_features = walking_activity_features[walking_activity_features.loc[:, "Error"] == False]

    walking_activity_features.drop('Error', axis=1, inplace=True)

    # Dropping rows with invalid values
    walking_activity_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    walking_activity_features.dropna(axis=0, how='any', inplace=True)

    fileName = 'walking_activity_features'
    if wavelet is not "":
        fileName += utils.waveletName(wavelet, level)
    walking_activity_features.to_csv("../data/{}.csv".format(fileName))

    print(len(walking_activity) - len(walking_activity_features), "rows dropped")
    print("Total time:", time.time() - startTime)
