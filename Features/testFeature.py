import features_utils as fu
import matplotlib.pyplot as plt
import numpy as np

# Level 1


class TestFeature:
    def __init__(self):
        self.data = fu.loadUserInput()
        self.data.timestamp -= self.data.timestamp.iloc[0]
        self.x = None
        self.y = None
        self.xlabel = None
        self.ylabel = None

    def plot(self):
        plt.plot(self.x, self.y)
        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)
        plt.show()

# Level 2


class RawFeature(TestFeature):
    def __init__(self, axis):
        super(RawFeature, self).__init__()
        self.x = self.data.timestamp
        self.y = self.data.loc[:, axis]
        self.ylabel = 'Acceleration in axis ' + axis
        self.xlabel = 'time'


class Feature2dBins(TestFeature):
    def __init__(self, axis1, axis2):
        super(Feature2dBins, self).__init__()
        self.x = [i for i in range(1, 1000)]
        self.y = [self.mainCalc(axis1, axis2, xx) for xx in self.x]
        self.xlabel = '#bins'

    def mainCalc(self, axis1, axis2, bins):
        pass


class RawSpherical(TestFeature):
    def __init__(self):
        super(RawSpherical, self).__init__()
        self.x = self.data.timestamp
        self.xlabel = 'time'
        self.y = self.mainCalc()

    def mainCal(self):
        pass


class Entropy(TestFeature):
    def __init__(self, axis):
        super(Entropy, self).__init__()
        self.x = [i for i in range(1, 10000)]
        self.y = [fu.entropy(self.data, axis, xx) for xx in self.x]
        self.ylabel = 'Entropy ' + axis
        self.xlabel = '#bins'


class Frequency(TestFeature):
    def __init__(self, axis):
        super(Frequency, self).__init__()
        self.y = np.abs(np.fft.fft(self.data[axis]))
        self.x = np.fft.fftfreq(len(self.data))
        self.ylabel = 'Acceleration in frequency domain ' + axis
        self.xlabel = 'Frequency'

# Level 3


class MutualInfo(Feature2dBins):
    def __init__(self, axis1, axis2):
        super(MutualInfo, self).__init__(axis1, axis2)
        self.ylabel = 'Mutual Information ' + axis1 + axis2

    def mainCalc(self, axis1, axis2, bins):
        return fu.mutualInfo(self.data, axis1, axis2, bins)


class RadialDistance(RawSpherical):
    def __init__(self):
        super(RadialDistance, self).__init__()
        self.ylabel = 'Radial Distance of the acceleration'

    def mainCalc(self):
        return fu.cart2sphRadialDist(self.data, raw=True)


class PolarAngle(RawSpherical):
    def __init__(self):
        super(PolarAngle, self).__init__()
        self.ylabel = 'Polar Angle of the acceleration'

    def mainCalc(self):
        return fu.cart2sphPolarAngle(self.data, raw=True)


class AzimuthAngle(RawSpherical):
    def __init__(self):
        super(AzimuthAngle, self).__init__()
        self.ylabel = 'Azimuth Angle of the acceleration'

    def mainCalc(self):
        return fu.cart2sphAzimuthAngle(self.data, raw=True)
