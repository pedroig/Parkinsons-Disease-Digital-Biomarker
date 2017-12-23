import numpy as np
import pandas as pd
import os
import json
from pandas.io.json import json_normalize
import sys
sys.path.insert(0, '../Features')
import utils
import quaternion


def preprocessDeviceMotionG(data):
    dataG = data['timestamp'].to_frame()
    for index, row in enumerate(data.itertuples()):
        q = quaternion.Quaternion(row.attitudeW, row.attitudeX, row.attitudeY, row.attitudeZ)
        g = utils.rotation3D(q, row.gravityX, row.gravityY, row.gravityZ)
        q = utils.generateUnitQuaternion(np.pi / 2, 1, 0, 0)
        g = utils.rotation3D(q, g[0], g[1], g[2])
        for axisNum, axis in enumerate(['x', 'y', 'z']):
            dataG.loc[index, axis] = g[axisNum]
    return dataG


def readJSON_dataG(pointer, timeSeriesName, fileName=''):
    path = utils.generatePath(pointer, timeSeriesName)
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
                    'magneticField.z'],
                    axis=1, inplace=True)
                for feature in ['attitude', 'rotationRate', 'userAcceleration', 'gravity']:
                    for axis in ['x', 'y', 'z']:
                        json_df.rename(columns={'{}.{}'.format(feature, axis): '{}{}'.format(feature, axis.upper())}, inplace=True)
                json_df.rename(columns={'attitude.w': 'attitudeW'}, inplace=True)
            else:
                json_df = pd.read_json(path)
    except:
        json_df = None
    return json_df
