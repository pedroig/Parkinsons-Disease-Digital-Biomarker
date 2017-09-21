import pandas as pd
import numpy as np

def cart2sphRadialDist(data):
	hxy = np.hypot(data['x'], data['y'])
	return np.hypot(hxy, data['z']).mean()

def cart2sphPolarAngle(data):
	return np.arctan2(data['y'], data['x']).mean()

def cart2sphAzimuthAngle(data):
	hxy = np.hypot(data['x'], data['y'])
	return np.arctan2(data['z'], hxy).mean()

def zeroCrossingRate(data, axis):
	data1 = np.array(data[axis].iloc[:-1])
	data2 = np.array(data[axis].iloc[1:])
	return (data1*data2 < 0).sum()