import pandas as pd
import os
import pywt
import json
import quaternion
from pandas.io.json import json_normalize

# File manipulation


def generatePath(pointer, timeSeriesName):
    pointer = int(pointer)
    path = '../data/{}/{}/{}/'
    path = path.format(timeSeriesName, str(pointer % 1000), str(pointer))
    return path


def readJSON_data(pointer, timeSeriesName, fileName=''):
    path = generatePath(pointer, timeSeriesName)
    try:
        if len(fileName) > 0:
            path += fileName
            json_df = pd.read_json(path, orient='split')
        else:
            for fileName in os.listdir(path):
                if fileName.startswith(timeSeriesName):
                    path += fileName
                    break
            if timeSeriesName.startswith('deviceMotion'):
                data = json.load(open(path))
                json_df = json_normalize(data)
                json_df.drop([
                    'magneticField.accuracy',
                    'magneticField.x',
                    'magneticField.y',
                    'magneticField.z',
                    'gravity.x',
                    'gravity.y',
                    'gravity.z'],
                    axis=1, inplace=True)
                for feature in ['attitude', 'rotationRate', 'userAcceleration']:
                    for axis in ['x', 'y', 'z']:
                        json_df.rename(columns={'{}.{}'.format(feature, axis): '{}{}'.format(feature, axis.upper())}, inplace=True)
                json_df.rename(columns={'attitude.w': 'attitudeW'}, inplace=True)
            else:
                json_df = pd.read_json(path)
    except:
        json_df = None
    return json_df


def loadUserInput():
    timeSeriesOptions = ['walking_outbound',
                         'walking_return',
                         'walking_rest']
    print("Choose the time series")
    for index, timeSeriesName in enumerate(timeSeriesOptions):
        print(index, timeSeriesName)
    timeSeriesSelected = int(input("Select the corresponding number: "))

    target = int(input('\n0 for normal or 1 for PD: '))
    features = pd.read_csv('../data/features_extra_columns.csv', index_col=0)
    features = features[features.Target == target]

    sampled = features.sample().iloc[0]

    timeSeriesName = "deviceMotion_" + timeSeriesOptions[timeSeriesSelected]
    pointerDeviceMotion = sampled[timeSeriesName + '.json.items']
    dataDeviceMotion = readJSON_data(pointerDeviceMotion, timeSeriesName)
    dataAcc, dataRot = preprocessDeviceMotion(dataDeviceMotion)

    if timeSeriesSelected == 2:
        dataPedo = None
    else:
        timeSeriesName = "pedometer_" + timeSeriesOptions[timeSeriesSelected]
        pointerPedo = sampled[timeSeriesName + '.json.items']
        dataPedo = readJSON_data(pointerPedo, timeSeriesName)

    mainData = int(input('\n0 for Rotation Rate or 1 for Acceleration: '))
    if mainData == 0:
        return dataRot, dataPedo
    else:
        return dataAcc, dataPedo


def waveletName(wavelet, level):
    return '-{}-level{}'.format(wavelet, str(level))


def genFileName(sampleType, wavelet, level):
    if wavelet == "":
        return '{}.json'.format(sampleType)
    waveletFileName = '{}{}.json'.format(sampleType, waveletName(wavelet, level))
    return waveletFileName


def saveTimeSeries(data, pointer, timeSeriesName, sampleType, wavelet, level):
    path = generatePath(pointer, timeSeriesName)
    fileName = genFileName(sampleType, wavelet, level)
    data.to_json(path + fileName, orient='split')

# Preprocess Device Motion data


def preprocessDeviceMotion(data):
    dataAcc = data['timestamp'].to_frame()
    dataRot = data['timestamp'].to_frame()
    for index, row in enumerate(data.itertuples()):
        q = quaternion.Quaternion(row.attitudeW, row.attitudeX, row.attitudeY, row.attitudeZ)
        rotRate = quaternion.Quaternion(0, row.rotationRateX, row.rotationRateY, row.rotationRateZ)
        userAccel = quaternion.Quaternion(0, row.userAccelerationX, row.userAccelerationY, row.userAccelerationZ)
        rotRate = (q * rotRate * q.inverse()).im
        userAccel = (q * userAccel * q.inverse()).im
        for axisNum, axis in enumerate(['x', 'y', 'z']):
            dataRot.loc[index, axis] = rotRate[axisNum]
            dataAcc.loc[index, axis] = userAccel[axisNum]
    return dataAcc, dataRot


def waveletFiltering(data, wavelet, level):
    for axis in ['x', 'y', 'z']:
        coeffs = pywt.wavedec(data[axis], wavelet, level=level)
        data[axis] = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(data))
