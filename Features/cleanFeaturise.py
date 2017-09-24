import pandas as pd
import numpy as np
import createFeatures as cf
import os

def readJSON_data(pointer, timeSeriesName):
	pointer = int(pointer)
	path = '../data/{}/{}/{}/'
	path = path.format(timeSeriesName, str(pointer%1000), str(pointer))
	try:
		for fileName in os.listdir(path):
			if fileName.startswith(timeSeriesName):
				path += fileName
				break
		json_df = pd.read_json(path)
	except IOError:
		json_df = None
	return json_df

def rowFeaturise(row, features, timeSeriesName):
	pointer = features.loc[row.name, timeSeriesName+'.json.items']
	if ~np.isnan(pointer) :
		data = readJSON_data(pointer, timeSeriesName)
		if (data is None) or data.empty:		#No file matching the pointer or data file Null
			features.loc[row.name, "Error"] = True
		else:
			if timeSeriesName.startswith("accel"):
				cf.createFeatureAcc(features, row.name, data, timeSeriesName)
			elif timeSeriesName.startswith("pedometer"):
				cf.createFeaturePedo(features, row.name, data, timeSeriesName)
	else:
		features.loc[row.name, "Error"] = True

def generateFeatures(dataFraction=1):
	demographics = pd.read_csv("../data/demographics.csv", index_col=0)
	demographics = demographics.join(pd.get_dummies(demographics["gender"]))
	demographics.rename(columns={'Prefer not to answer' : 'Prefer not to answer gender'}, inplace=True)
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
									'Male',
									'Female',
									'Prefer not to answer gender'
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

	features = pd.merge(walking_activity, demographics, on="healthCode")
	features.index.name='ROW_ID'

	errors = 0
	rowLimit = int(dataFraction*len(features))
	features.loc[:, "Error"] = False
	for namePrefix in ['accel_walking_', 'pedometer_walking_']:
		for phase in ["outbound", "return", "rest"]:
			timeSeriesName = namePrefix + phase
			if timeSeriesName == 'pedometer_walking_rest':
				continue
			features.iloc[:rowLimit, :].apply(rowFeaturise, axis=1, args=(features, timeSeriesName))
			errors += features.loc[:, "Error"].sum()
			#Dropping rows with errors
			features = features[features.loc[:, "Error"] == False]
			print(timeSeriesName, "done.")
	
	print(errors, "rows dropped due to error during the reading")
	rowLimit -= errors

	features.rename(columns={'professional-diagnosis' : 'Target'}, inplace=True)
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
	errors = rowLimit
	features = features.replace([np.inf, -np.inf], np.nan)
	features = features.dropna(axis=0, how='any')
	errors -= len(features)
	print(errors, "rows dropped due to invalid values")

	features.to_csv("../data/features.csv")
	return features