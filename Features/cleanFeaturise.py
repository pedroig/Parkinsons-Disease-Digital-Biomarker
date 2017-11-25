import pandas as pd
import numpy as np
import createFeatures as cf
import utils
import time
import multiprocessing
from sklearn.model_selection import train_test_split


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


def dropExtraColumns(features):
    features.drop(['healthCode',
                   # 'accel_walking_outbound.json.items',
                   'deviceMotion_walking_outbound.json.items',
                   'pedometer_walking_outbound.json.items',
                   # 'accel_walking_return.json.items',
                   # 'deviceMotion_walking_return.json.items',
                   # 'pedometer_walking_return.json.items',
                   # 'accel_walking_rest.json.items',
                   'deviceMotion_walking_rest.json.items',
                   'medTimepoint'
                   ], axis=1, inplace=True)


def generateFeatures(num_cores=1, dataFraction=1, wavelet='', level=None):
    startTime = time.time()
    demographics = pd.read_csv("../data/demographics.csv", index_col=0)
    # Dropping rows without answer for gender
    demographics[(demographics.gender == "Male") | (demographics.gender == "Female")]
    demographics = demographics.join(pd.get_dummies(demographics["gender"]).Male)
    columns_to_keep_demographics = [
        # 'ROW_VERSION',
        # 'recordId',
        'healthCode',
        # 'appVersion',
        # 'phoneInfo',
        'age',
        # 'are-caretaker',
        # 'deep-brain-stimulation',
        # 'diagnosis-year',
        # 'education',
        # 'employment',
        # 'health-history',
        # 'healthcare-provider',
        # 'home-usage',
        # 'last-smoked',
        # 'maritalStatus',
        # 'medical-usage',
        # 'medical-usage-yesterday',
        # 'medication-start-year',
        # 'onset-year',
        # 'packs-per-day',
        # 'past-participation',
        # 'phone-usage',
        'professional-diagnosis',
        # 'race',
        # 'smartphone',
        # 'smoked',
        # 'surgery',
        # 'video-usage',
        # 'years-smoking'
        # 'gender',
        'Male'
    ]
    demographics = demographics[columns_to_keep_demographics]

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
    walking_activity = walking_activity[columns_to_keep_walking]

    demographics_train, demographics_test_val = train_test_split(demographics, test_size=0.2)
    demographics_test, demographics_val = train_test_split(demographics_test_val, test_size=0.5)
    train = pd.merge(walking_activity, demographics_train, on="healthCode")
    test = pd.merge(walking_activity, demographics_test, on="healthCode")
    val = pd.merge(walking_activity, demographics_val, on="healthCode")
    listFeatures = [(train, 'train'), (test, 'test'), (val, 'val')]

    noSplitFeatures = pd.DataFrame()

    for features, featuresSplitName in listFeatures:
        if wavelet is not "":
            featuresSplitName += utils.waveletName(wavelet, level)

        features.index.name = 'ROW_ID'
        features = features.sample(frac=dataFraction)

        features["Error"] = False
        for namePrefix in ['deviceMotion_walking_', 'pedometer_walking_']:
            for phase in ["outbound", "rest"]:  # , "return"]:
                timeSeriesName = namePrefix + phase
                if timeSeriesName == 'pedometer_walking_rest':
                    continue
                print("Working on {} from {}.".format(timeSeriesName, featuresSplitName))
                args = (timeSeriesName, wavelet, level)
                features = apply_by_multiprocessing(features, rowFeaturise, args=args, workers=num_cores)

                # Dropping rows with errors
                features = features[features.loc[:, "Error"] == False]

        features = features.drop('Error', axis=1)

        features.rename(columns={'professional-diagnosis': 'Target'}, inplace=True)

        # Dropping rows with invalid values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna(axis=0, how='any')

        noSplitFeatures = pd.concat([features, noSplitFeatures])
        features.to_csv("../data/{}_extra_columns.csv".format(featuresSplitName))
        dropExtraColumns(features)
        features.to_csv("../data/{}.csv".format(featuresSplitName))

    featuresName = 'features'
    if wavelet is not "":
        featuresName += utils.waveletName(wavelet, level)
    noSplitFeatures.to_csv("../data/{}_extra_columns.csv".format(featuresName))
    dropExtraColumns(noSplitFeatures)
    noSplitFeatures.to_csv("../data/{}.csv".format(featuresName))

    print(len(walking_activity) - len(noSplitFeatures), "rows dropped")
    print("Total time:", time.time() - startTime)
