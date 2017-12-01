import pandas as pd
import utils
from sklearn.model_selection import train_test_split


def dropExtraColumns(features):
    features.drop(['healthCode',
                   'accel_walking_outbound.json.items',
                   'deviceMotion_walking_outbound.json.items',
                   'pedometer_walking_outbound.json.items',
                   'accel_walking_return.json.items',
                   'deviceMotion_walking_return.json.items',
                   'pedometer_walking_return.json.items',
                   'accel_walking_rest.json.items',
                   'deviceMotion_walking_rest.json.items',
                   'medTimepoint'
                   ], axis=1, inplace=True)


def generateSetTables(wavelet='', level=None):

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

    fileName = 'walking_activity_features'
    if wavelet is not "":
        fileName += utils.waveletName(wavelet, level)
    walking_activity_features = pd.read_csv("../data/{}.csv".format(fileName), index_col=0)

    demographics_train, demographics_test_val = train_test_split(demographics, test_size=0.2)
    demographics_test, demographics_val = train_test_split(demographics_test_val, test_size=0.5)
    train = pd.merge(walking_activity_features, demographics_train, on="healthCode")
    test = pd.merge(walking_activity_features, demographics_test, on="healthCode")
    val = pd.merge(walking_activity_features, demographics_val, on="healthCode")
    listFeatures = [(train, 'train'), (test, 'test'), (val, 'val')]

    noSplitFeatures = pd.DataFrame()

    for features, featuresSplitName in listFeatures:
        if wavelet is not "":
            featuresSplitName += utils.waveletName(wavelet, level)

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
