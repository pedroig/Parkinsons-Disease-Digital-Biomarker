import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../Features')
import utils
import quaternion

train = pd.read_csv("../data/train_extra_columns.csv")
train.loc[:, "augmented"] = False
trainSelected = train.sample(frac=0.3)
trainSelected.loc[:, "augmented"] = True

trainAugmented = pd.concat([train, trainSelected])
trainAugmented.to_csv("../data/train_augmented_extra_columns.csv")

axes = ['x', 'y', 'z']

for timeSeriesName in ['deviceMotion_walking_outbound', 'deviceMotion_walking_rest']:  # , 'deviceMotion_walking_return']:
    trainSelected.rename(columns={'{}.json.items'.format(timeSeriesName): timeSeriesName}, inplace=True)

    for row in trainSelected.itertuples():
        pointer = getattr(row, timeSeriesName)
        data = utils.readJSON_data(pointer, timeSeriesName, 'RotRate.json')

        max_scale = 1.2
        min_scale = 0.8
        random_scale = np.random.random_sample() * (max_scale - min_scale) + min_scale
        data[axes] = data[axes] * random_scale
        theta = np.random.random_sample() * np.pi * 2
        ux = np.random.random_sample() * 2 - 1
        uy = np.random.random_sample() * 2 - 1
        uz = np.random.random_sample() * 2 - 1
        q = quaternion.Quaternion(np.cos(theta / 2), ux * np.sin(theta / 2), uy * np.sin(theta / 2), uz * np.sin(theta / 2))
        for index, row in enumerate(data.itertuples()):
            rotRate = quaternion.Quaternion(0, row.x, row.y, row.z)
            rotRate = (q * rotRate * q.inverse()).im
            for axisNum, axis in enumerate(axes):
                data.loc[index, axis] = rotRate[axisNum]

        print(theta)
        path = utils.generatePath(pointer, timeSeriesName)
        fileName = 'RotRate_augmented.json'
        data.to_json(path + fileName, orient='split')
