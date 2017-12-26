import pandas as pd
import numpy as np


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


def generateSetTables():

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
        # 'age',
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
        # 'Male'
    ]
    demographics = demographics[columns_to_keep_demographics]

    demographics.rename(columns={'professional-diagnosis': 'Target'}, inplace=True)

    # Dropping rows with invalid values
    demographics.replace([np.inf, -np.inf], np.nan, inplace=True)
    demographics.dropna(axis=0, how='any', inplace=True)

    fileName = 'walking_activity_features'
    walking_activity_features = pd.read_csv("../data/{}.csv".format(fileName), index_col=0)

    maxHealthCode = walking_activity_features.groupby("healthCode").size().sort_values(ascending=False).index[2]

    demographics_val = demographics[demographics.healthCode == maxHealthCode]
    demographics_train = demographics[demographics.healthCode != maxHealthCode]
    train = pd.merge(walking_activity_features, demographics_train, on="healthCode")
    val = pd.merge(walking_activity_features, demographics_val, on="healthCode")
    listFeatures = [(train, 'train'), (val, 'val')]

    for features, featuresSplitName in listFeatures:
        dropExtraColumns(features)
        features.to_csv("../data/{}_outlier.csv".format(featuresSplitName))
