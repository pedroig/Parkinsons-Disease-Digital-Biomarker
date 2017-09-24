import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mutual_info_score
from scipy import stats

#Mixed dimensions

def cart2sphRadialDist(data):
	hxy = np.hypot(data['x'], data['y'])
	return np.hypot(hxy, data['z']).mean()

def cart2sphPolarAngle(data):
	return np.arctan2(data['y'], data['x']).mean()

def cart2sphAzimuthAngle(data):
	hxy = np.hypot(data['x'], data['y'])
	return np.arctan2(data['z'], hxy).mean()

#1 Dimension

def zeroCrossingRate(data, axis):
	data1 = np.array(data[axis].iloc[:-1])
	data2 = np.array(data[axis].iloc[1:])
	return (data1*data2 < 0).sum()

def entropy(data, axis, bins=10):
	hist = np.histogram(data.loc[:, axis], bins)[0]
	return stats.entropy(hist)

def dominantFreqComp(data, axis):
	sp = np.fft.fft(data[axis])
	freq = np.fft.fftfreq(len(data))
	# plt.plot(freq, sp.real, freq, sp.imag)
	# plt.plot(freq, np.abs(sp))
	# plt.show()
	freqIndex = np.argmax(np.abs(sp))
	dominantFreq = freq[freqIndex]
	return dominantFreq

#2 Dimensions

def crossCorrelation(data, axis1, axis2, lag=0):
	diffs = pd.DataFrame(columns=[axis1, axis2])
	diffs.loc[:, axis1] = data[axis1] - data.loc[:, axis1].mean()
	diffs.loc[:, axis2] = data[axis2] - data.loc[:, axis2].mean()
	num = (diffs.iloc[:len(diffs)-lag, 0] * diffs.iloc[lag:, 1]).sum()
	
	diffs.loc[:, axis1] = diffs.loc[:, axis1]**2
	diffs.loc[:, axis2] = diffs.loc[:, axis2]**2
	den = np.sqrt( diffs.loc[:, axis1].sum() * diffs.loc[:, axis2].sum() )
	
	return num/den

def mutualInfo(data, axis1, axis2, bins=10):
	c_xy = np.histogram2d(data.loc[:, "x"], data.loc[:, "y"], bins)[0]
	mi = mutual_info_score(None, None, contingency=c_xy)
	return mi

def crossEntropy(data, axis1, axis2, bins=10):
	low = min(data.loc[:, axis1].min(), data.loc[:, axis2].min())
	high = max(data.loc[:, axis1].max(), data.loc[:, axis2].max())
	hist1 = np.histogram(data.loc[:, axis1], bins=bins, range=(low, high))[0]
	hist2 = np.histogram(data.loc[:, axis2], bins=bins, range=(low, high))[0]
	return stats.entropy(hist1, hist2)

#Extra for the pedometer data

def avgSpeed(data):
	start = datetime.strptime(data.loc[data.index[-1], 'startDate'], '%Y-%m-%dT%H:%M:%S%z')
	end = datetime.strptime(data.loc[data.index[-1], 'endDate'], '%Y-%m-%dT%H:%M:%S%z')
	delta = end - start
	dist = data.loc[data.index[-1], 'distance']
	if delta.seconds == 0:
		return np.nan
	else:
		return dist/delta.seconds

def avgStep(data):
	stepNum = data.loc[data.index[-1], 'numberOfSteps']
	if stepNum == 0:
		return np.nan
	else:
		return data.loc[data.index[-1], 'distance']/stepNum