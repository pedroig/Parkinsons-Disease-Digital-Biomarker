import numpy as np
import dynamicTimeWarping as dtw
import sys
import heapq
import time
sys.path.insert(0, '../Features')
sys.path.insert(0, '../Random_Forest')
import learning_utils as lu
import features_utils as fu


def knn(k=3, balance_train=False, balance_val=False):
    startTime = time.time()
    X_train, y_train = lu.load_data("train_extra_columns", balance_samples=balance_train)
    X_val, y_val = lu.load_data("val_extra_columns", balance_samples=balance_val)
    timeSeriesName = 'accel_walking_outbound'
    y_pred = []

    # Limiting data due to performance issues
    X_train = X_train[:280]
    X_val = X_val[:15]
    y_val = y_val[:15]

    for ind_val, pointer_val in enumerate(X_val.loc[:, timeSeriesName + '.json.items']):
        print(100 * ind_val / len(X_val), "%")
        data_val = fu.readJSON_data(pointer_val, timeSeriesName)
        closest = []
        for i in range(k):
            closest.append((-np.inf, -1))
        heapq.heapify(closest)

        for ind_train, pointer_train in enumerate(X_train.loc[:, timeSeriesName + '.json.items']):
            data_train = fu.readJSON_data(pointer_train, timeSeriesName)
            dist = dtw.dtwDistanceRadial(data_train, data_val, reduceResolution=True)
            newTuple = max((-dist, y_train[ind_train]), heapq.heappop(closest))
            heapq.heappush(closest, newTuple)

        _, closest_y = zip(*closest)
        y_pred.append(np.mean(closest_y) >= 0.5)

    y_pred = np.asarray(y_pred, dtype=np.int8)
    print("y_val")
    print(y_val)
    print("y_pred")
    print(y_pred)
    lu.metricsPrint(y_val, y_pred)
    print("Time:", time.time() - startTime)
