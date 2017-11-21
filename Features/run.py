import cleanFeaturise

if __name__ == '__main__':
    # testing
    cleanFeaturise.generateFeatures(num_cores=16, dataFraction=0.0005)

    # full execution
    # cleanFeaturise.generateFeatures(num_cores=16)
