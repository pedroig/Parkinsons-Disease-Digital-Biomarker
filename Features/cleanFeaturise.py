import pandas as pd
import numpy as np
import os
import createFeatures as cf

def readJSON_data(pointer, column):

	pointer = int(pointer)
	path = 'data/accel_walking_rest/{}/{}/{}/'
	path = path.format(column, str(pointer%1000), str(pointer))

	try:
		path += os.listdir(path)[1]
		json_df = pd.read_json(path)
	except IOError:
		json_df = None
	return json_df

def rowFeaturise(row, features, timeSeriesName):
	pointer = features.loc[row.name, timeSeriesName+'.json.items']
	if ~np.isnan(pointer) :
		data = readJSON_data(pointer, timeSeriesName)
		if data is None:		#No file matching the pointer
			features.loc[row.name, "Error"] = True
		else:
			cf.createFeatureXYZ(features, row.name, data, timeSeriesName)
	else:
		features.loc[row.name, "Error"] = True

def generateFeatures():

	demographics = pd.read_csv("demographics.csv", index_col=0)
	demographics = demographics.join(pd.get_dummies(demographics["gender"]))
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
									'Prefer not to answer'
									]
	demographics = demographics[columns_to_keep_demographics]

	walking_activity = pd.read_csv("walking_activity.csv", index_col=0)
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

	features.loc[:, "Error"] = False
	features.iloc[:1000, :].apply(rowFeaturise, axis=1, args=(features, 'accel_walking_rest'))
	print(features.loc[:, "Error"].sum(), "rows dropped due to error during the reading")
	#Dropping rows with errors
	features = features[features.loc[:, "Error"] == False]

	features.rename(columns={'professional-diagnosis' : 'Target'})
	features = features.drop([	'healthCode',
								'accel_walking_outbound.json.items',
								'deviceMotion_walking_outbound.json.items',
								'pedometer_walking_outbound.json.items',
								'accel_walking_return.json.items',
								'deviceMotion_walking_return.json.items',
								'pedometer_walking_return.json.items',
								'accel_walking_rest.json.items',
								'deviceMotion_walking_rest.json.items'
								], axis=1)

	features.to_csv('features.csv')
	return features