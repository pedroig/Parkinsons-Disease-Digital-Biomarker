from mutual_info import entropy
import features_utils as fu
import pandas as pd
import numpy as np

def createFeatureXYZ(features, index, data, timeSeriesName):

	for axis in ["x", "y", "z"]:
		nameSuffix = timeSeriesName + "_" + axis
		
		features.loc[index, "Mean_" + nameSuffix] = data[axis].mean()
		features.loc[index, "Std Deviation_" + nameSuffix] = data[axis].std()
		features.loc[index, "Q1_" + nameSuffix] = data[axis].quantile(0.25)
		features.loc[index, "Q3_" + nameSuffix] = data[axis].quantile(0.75)
		features.loc[index, "Interquartile_" + nameSuffix] = features.loc[index, "Q3_"+nameSuffix] - features.loc[index, "Q1_"+nameSuffix]
		features.loc[index, "Median_" + nameSuffix] = data[axis].median()
		#features.loc[index, "Mode_" + nameSuffix] = data[axis].mode().iloc[0]							#Does it make sense?!!
		features.loc[index, "Data Range_" + nameSuffix] = data[axis].max() - data[axis].min()
		features.loc[index, "Kurtosis_" + nameSuffix] = data[axis].kurtosis()
		features.loc[index, "Mean Squared Energy_" + nameSuffix] = np.sqrt( (data[axis]**2).sum()/len(data) )
		features.loc[index, "Entropy_" + nameSuffix] = entropy( data[axis].values.reshape((len(data),1)), k=100)
		features.loc[index, "Zero Crossing Rate_" + nameSuffix] = fu.zeroCrossingRate(data, axis)
		features.loc[index, "Radial Distance_" + nameSuffix] = fu.cart2sphRadialDist(data)
		features.loc[index, "Polar Angle_" + nameSuffix] = fu.cart2sphPolarAngle(data)
		features.loc[index, "Azimuth Angle_" + nameSuffix] = fu.cart2sphAzimuthAngle(data)