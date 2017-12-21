import pandas as pd
import sys
sys.path.insert(0, '../Features')
import utils

train = pd.read_csv("../data/train_extra_columns.csv")

axes = ['x', 'y', 'z']

for timeSeriesName in ['deviceMotion_walking_outbound', 'deviceMotion_walking_rest']:  # , 'deviceMotion_walking_return']:
    train.rename(columns={'{}.json.items'.format(timeSeriesName): timeSeriesName}, inplace=True)

    for row in train.itertuples():
        pointer = getattr(row, timeSeriesName)
        data = utils.readJSON_data(pointer, timeSeriesName, 'RotRate.json')

        midPoint = len(data) // 2
        print(pointer, midPoint)
        data = data[midPoint - 150: midPoint + 150]
        path = utils.generatePath(pointer, timeSeriesName)
        fileName = 'RotRate_reduced.json'
        data.to_json(path + fileName, orient='split')
