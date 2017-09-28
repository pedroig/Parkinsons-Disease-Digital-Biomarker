import features_utils as fu
import pandas as pd
import numpy as np

def createFeatureAcc(features, index, data, timeSeriesName):

	#Mixed dimensions
	nameSuffix = "_" + timeSeriesName
	features.loc[index, "Radial Distance Mean" + nameSuffix] = fu.cart2sphRadialDist(data)
	features.loc[index, "Polar Angle Mean" + nameSuffix] = fu.cart2sphPolarAngle(data)
	features.loc[index, "Azimuth Angle Mean" + nameSuffix] = fu.cart2sphAzimuthAngle(data)

	#1 dimension
	axes = ["x", "y", "z"]
	
	for axis in axes:
		nameSuffix = "_" + timeSeriesName + "_" + axis
		
		features.loc[index, "Mean" + nameSuffix] = data[axis].mean()
		features.loc[index, "Std Deviation" + nameSuffix] = data[axis].std()
		features.loc[index, "Q1" + nameSuffix] = data[axis].quantile(0.25)
		features.loc[index, "Q3" + nameSuffix] = data[axis].quantile(0.75)
		features.loc[index, "Interquartile" + nameSuffix] = features.loc[index, "Q3" + nameSuffix] - features.loc[index, "Q1" + nameSuffix]
		features.loc[index, "Median" + nameSuffix] = data[axis].median()
		#features.loc[index, "Mode" + nameSuffix] = data[axis].mode().iloc[0]											#Does it make sense?!!
		features.loc[index, "Data Range" + nameSuffix] = data[axis].max() - data[axis].min()
		features.loc[index, "Kurtosis" + nameSuffix] = data[axis].kurtosis()
		features.loc[index, "Mean Squared Energy" + nameSuffix] = np.sqrt( (data[axis]**2).sum()/len(data) )
		features.loc[index, "Entropy" + nameSuffix] = fu.entropy(data, axis, bins=100)									#How many bins?
		features.loc[index, "Zero Crossing Rate" + nameSuffix] = fu.zeroCrossingRate(data, axis)
		features.loc[index, "Dominant Frequency Component" + nameSuffix] = fu.dominantFreqComp(data, axis)				#Check?	

	#2 dimensions
	for axis1 in axes:
		for axis2 in axes:
			if axis1 < axis2:
				nameSuffix = "_" + timeSeriesName + "_" + axis1 + axis2

				features.loc[index, "Cross-correlation 0 lag" + nameSuffix] = fu.crossCorrelation(data, axis1, axis2, lag=0)#lag?
				features.loc[index, "Mutual Information" + nameSuffix] = fu.mutualInfo(data, axis1, axis2, bins=100)		#How many bins?
				#features.loc[index, "Cross Entropy" + nameSuffix] = fu.crossEntropy(data, axis1, axis2, bins=100)			#How many bins?  Inf? 	Which definition to use?	#XY and YZ?

def createFeaturePedo(features, index, data, timeSeriesName):

	nameSuffix = "_" + timeSeriesName
	features.loc[index, "Distance walked" + nameSuffix] = data.loc[data.index[-1], 'distance']
	features.loc[index, "Average Speed" + nameSuffix] = fu.avgSpeed(data)
	features.loc[index, "Average Size of the Step" + nameSuffix] = fu.avgStep(data)