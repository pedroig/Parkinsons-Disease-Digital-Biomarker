import cleanFeaturise
import utils
import sys
import splitSets

if __name__ == '__main__':
    if sys.argv[1] == 'cleanFeaturise':
        print("Running cleanFeaturise")
        # testing
        # cleanFeaturise.generateFeatures(num_cores=16, dataFraction=0.0005)

        # full execution
        cleanFeaturise.generateFeatures(num_cores=16)
    elif sys.argv[1] == 'augmentation':
        print("Running augmentation")
        utils.augmentData()
    elif sys.argv[1] == 'splitSets':
        print("Running splitSets")
        splitSets.generateSetTables(naiveLimitHealthCode=False, augmentFraction=0.5)
