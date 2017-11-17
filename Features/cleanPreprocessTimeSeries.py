import pandas as pd
import numpy as np
import utils


def rowCleanPreprocess(row, table, timeSeriesName, wavelet, level):
    pointer = table.loc[row.name, timeSeriesName + '.json.items']
    print(row.name / len(table), "%", "Pointer:", pointer)
    if ~np.isnan(pointer):
        data = utils.readJSON_data(pointer, timeSeriesName)
        if (data is None) or data.empty:  # No file matching the pointer or data file Null
            table.loc[row.name, "Error"] = True
        else:
            if len(data) < 300:     # Data too short
                table.loc[row.name, "Error"] = True
            else:
                dataAcc, dataRot = utils.preprocessDeviceMotion(data)
                if wavelet is not '' and level is not None:
                    utils.waveletFiltering(dataAcc, wavelet, level)
                    utils.waveletFiltering(dataRot, wavelet, level)
                utils.saveTimeSeries(dataAcc, pointer, timeSeriesName, 'Accel', wavelet, level)
                utils.saveTimeSeries(dataRot, pointer, timeSeriesName, 'RotRate', wavelet, level)
    else:
        table.loc[row.name, "Error"] = True


def preprocessTimeSeries(phaseNumber, wavelet='', level=None):
    walking_activity = pd.read_csv("../data/walking_activity.csv", index_col=0)
    columns_to_keep_walking = [
        # 'ROW_VERSION',
        # 'recordId',
        'healthCode',
        # 'createdOn',
        # 'appVersion',
        # 'phoneInfo',
        'accel_walking_outbound.json.items',
        'deviceMotion_walking_outbound.json.items',
        'pedometer_walking_outbound.json.items',
        'accel_walking_return.json.items',
        'deviceMotion_walking_return.json.items',
        'pedometer_walking_return.json.items',
        'accel_walking_rest.json.items',
        'deviceMotion_walking_rest.json.items',
        'medTimepoint'
    ]
    walking_activity = walking_activity[columns_to_keep_walking]
    initialLength = len(walking_activity)

    walking_activity.index.name = 'ROW_ID'

    walking_activity.loc[:, "Error"] = False

    phases = ["outbound", "rest", "return"]
    phase = phases[phaseNumber]

    timeSeriesName = 'deviceMotion_walking_' + phase
    walking_activity.apply(rowCleanPreprocess, axis=1, args=(walking_activity, timeSeriesName, wavelet, level))
    # Dropping rows with errors
    walking_activity = walking_activity[walking_activity.loc[:, "Error"] == False]

    walking_activity = walking_activity.drop('Error', axis=1)

    fileName = 'preprocessed_walking_activity'
    if wavelet is not "":
        fileName += utils.waveletName(wavelet, level)

    walking_activity.to_csv("../data/{}.csv".format(fileName))

    print(initialLength - len(walking_activity), "rows dropped")
