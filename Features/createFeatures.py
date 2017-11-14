import features_utils as fu


def createFeatures(features, index, data, timeSeriesName):

    # Mixed dimensions
    nameSuffix = "_" + timeSeriesName
    features.loc[index, "Radial Distance Mean" + nameSuffix] = fu.cart2sphRadialDist(data)
    features.loc[index, "Polar Angle Mean" + nameSuffix] = fu.cart2sphPolarAngle(data)
    features.loc[index, "Azimuth Angle Mean" + nameSuffix] = fu.cart2sphAzimuthAngle(data)
    features.loc[index, "Signal Magnitude Area (SMA)" + nameSuffix] = fu.sma(data)

    # 1 dimension
    axes = ["x", "y", "z"]

    for axis in axes:
        nameSuffix = "_" + timeSeriesName + "_" + axis

        features.loc[index, "Maximum" + nameSuffix] = data[axis].max()
        features.loc[index, "Minimum" + nameSuffix] = data[axis].min()
        features.loc[index, "Mean" + nameSuffix] = data[axis].mean()
        features.loc[index, "Std Deviation" + nameSuffix] = data[axis].std()
        features.loc[index, "Q1" + nameSuffix] = data[axis].quantile(0.25)
        features.loc[index, "Q3" + nameSuffix] = data[axis].quantile(0.75)
        features.loc[index, "Interquartile" + nameSuffix] = fu.interquartile(data, axis)
        features.loc[index, "Median" + nameSuffix] = data[axis].median()
        features.loc[index, "Data Range" + nameSuffix] = fu.dataRange(data, axis)
        features.loc[index, "Skewness" + nameSuffix] = data[axis].skew()
        features.loc[index, "Kurtosis" + nameSuffix] = data[axis].kurtosis()
        features.loc[index, "Root Mean Square (RMS)" + nameSuffix] = fu.rms(data, axis)
        features.loc[index, "Entropy" + nameSuffix] = fu.entropy(data, axis, bins=4000)
        features.loc[index, "Zero Crossing Rate" + nameSuffix] = fu.zeroCrossingRate(data, axis)
        features.loc[index, "Dominant Frequency Component" + nameSuffix] = fu.dominantFreqComp(data, axis)  # Check?

    # 2 dimensions
    for axis1 in axes:
        for axis2 in axes:
            if axis1 < axis2:
                nameSuffix = "_" + timeSeriesName + "_" + axis1 + axis2

                features.loc[index, "Cross-correlation 0 lag" + nameSuffix] = fu.crossCorrelation(data, axis1, axis2, lag=0)
                features.loc[index, "Mutual Information" + nameSuffix] = fu.mutualInfo(data, axis1, axis2, bins=1000)
                # features.loc[index, "Cross Entropy" + nameSuffix] = fu.crossEntropy(data, axis1, axis2, bins=100)			#How many bins?  Inf? 	Which definition to use?	#XY and YZ?


def createFeaturePedo(features, index, data, timeSeriesName):

    nameSuffix = "_" + timeSeriesName
    features.loc[index, "Distance walked" + nameSuffix] = data.loc[data.index[-1], 'distance']
    features.loc[index, "Average Speed" + nameSuffix] = fu.avgSpeed(data)
    features.loc[index, "Average Size of the Step" + nameSuffix] = fu.avgStep(data)
