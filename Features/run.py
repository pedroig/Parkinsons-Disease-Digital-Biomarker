import cleanFeaturise
import utils
import sys
import splitSets
import pandas as pd

if __name__ == '__main__':
    if sys.argv[1] == 'cleanFeaturise':
        print("Running cleanFeaturise")
        # testing
        # cleanFeaturise.generateFeatures(num_cores=16, dataFraction=0.0005)

        # full execution
        cleanFeaturise.generateFeatures(num_cores=16)

    elif sys.argv[1] == 'augmentation':
        print("Running augmentation")
        features_extra_columns = pd.read_csv("../data/features_extra_columns.csv")
        num_cores = 16
        utils.apply_by_multiprocessing_augmentation(features_extra_columns, num_cores)

    elif sys.argv[1] == 'splitSets':
        print("Running splitSets")
        splitSets.generateSetTables(augmentFraction=0.5)
