import pandas as pd
import numpy as np
import createFeatures as cf
import features_utils as fu
from sklearn.model_selection import train_test_split

def rowFeaturise(row, features, timeSeriesName):
	pointer = features.loc[row.name, timeSeriesName+'.json.items']
	if ~np.isnan(pointer) :
		data = fu.readJSON_data(pointer, timeSeriesName)
		if (data is None) or data.empty:		#No file matching the pointer or data file Null
			features.loc[row.name, "Error"] = True
		else:
			if timeSeriesName.startswith("accel"):
				cf.createFeatureAcc(features, row.name, data, timeSeriesName)
			elif timeSeriesName.startswith("pedometer"):
				cf.createFeaturePedo(features, row.name, data, timeSeriesName)
	else:
		features.loc[row.name, "Error"] = True

def generateFeatures(dataFraction=1, earlySplit=True, dropExtraCol=True):
	demographics = pd.read_csv("../data/demographics.csv", index_col=0)
	#Dropping rows without answer for gender
	demographics[(demographics.gender == "Male") | (demographics.gender == "Female")]
	demographics = demographics.join(pd.get_dummies(demographics["gender"]).Male)
	columns_to_keep_demographics = [#'ROW_VERSION',
									#'recordId',
									'healthCode',
									#'appVersion',
									#'phoneInfo',
									'age',
									#'are-caretaker',
									#'deep-brain-stimulation',
									#'diagnosis-year',
									#'education',
									#'employment',
									#'health-history',
									#'healthcare-provider',
									#'home-usage',
									#'last-smoked',
									#'maritalStatus',
									#'medical-usage',
									#'medical-usage-yesterday',
									#'medication-start-year',
									#'onset-year',
									#'packs-per-day',
									#'past-participation',
									#'phone-usage',
									'professional-diagnosis',
									#'race',
									#'smartphone',
									#'smoked',
									#'surgery',
									#'video-usage',
									#'years-smoking'
									#'gender',
									'Male'
									]
	demographics = demographics[columns_to_keep_demographics]

	walking_activity = pd.read_csv("../data/walking_activity.csv", index_col=0)
	columns_to_keep_walking = [	#'ROW_VERSION',
								#'recordId',
								'healthCode',
								#'createdOn',
								#'appVersion',
								#'phoneInfo',
								'accel_walking_outbound.json.items',
								'deviceMotion_walking_outbound.json.items',
								'pedometer_walking_outbound.json.items',
								'accel_walking_return.json.items',
								'deviceMotion_walking_return.json.items',
								'pedometer_walking_return.json.items',
								'accel_walking_rest.json.items',
								'deviceMotion_walking_rest.json.items',
								#'medTimepoint'
								]
	walking_activity = walking_activity[columns_to_keep_walking]

	if earlySplit:
		demographics_train, demographics_test = train_test_split(demographics, test_size=0.2)
		features_train = pd.merge(walking_activity, demographics_train, on="healthCode")
		features_test = pd.merge(walking_activity, demographics_test, on="healthCode")
		listFeatures = [(features_train, 'features_train'), (features_test, 'features_test')]
	else:
		features = pd.merge(walking_activity, demographics, on="healthCode")
		listFeatures = [(features, 'features')]

	for features, featuresSplitName in listFeatures:
		if dropExtraCol == False:
			featuresSplitName += '_extra_columns'
		print("\nGenerating", featuresSplitName)

		features.index.name='ROW_ID'
		features = features.sample(frac=dataFraction)

		errors = 0
		features.loc[:, "Error"] = False
		for namePrefix in ['accel_walking_', 'pedometer_walking_']:
			for phase in ["outbound", "return", "rest"]:
				timeSeriesName = namePrefix + phase
				if timeSeriesName == 'pedometer_walking_rest':
					continue
				features.apply(rowFeaturise, axis=1, args=(features, timeSeriesName))
				errors += features.loc[:, "Error"].sum()
				# Dropping rows with errors
				features = features[features.loc[:, "Error"] == False]
				print(timeSeriesName, "done.")
		
		print(errors, "rows dropped due to error during the reading")

		features.rename(columns={'professional-diagnosis' : 'Target'}, inplace=True)
		
		if dropExtraCol:
			features = features.drop([	'healthCode',
										'accel_walking_outbound.json.items',
										'deviceMotion_walking_outbound.json.items',
										'pedometer_walking_outbound.json.items',
										'accel_walking_return.json.items',
										'deviceMotion_walking_return.json.items',
										'pedometer_walking_return.json.items',
										'accel_walking_rest.json.items',
										'deviceMotion_walking_rest.json.items',
										'Error'
										], axis=1)

		#Dropping rows with invalid values
		errors = len(features)
		features = features.replace([np.inf, -np.inf], np.nan)
		features = features.dropna(axis=0, how='any')
		errors -= len(features)
		print(errors, "rows dropped due to invalid values")

		features.to_csv("../data/{}.csv".format(featuresSplitName))