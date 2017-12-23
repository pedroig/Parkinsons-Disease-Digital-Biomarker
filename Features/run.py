import cleanFeaturise
import utils
import sys

if __name__ == '__main__':
    if sys.argv[1] == 'cleanFeaturise':
        # testing
        # cleanFeaturise.generateFeatures(num_cores=16, dataFraction=0.0005)

        # full execution
        cleanFeaturise.generateFeatures(num_cores=16)
    elif sys.argv[1] == 'augmentation':
        utils.augmentData(augmentFraction=0.5)
