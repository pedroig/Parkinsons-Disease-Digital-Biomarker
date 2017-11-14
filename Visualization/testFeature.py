import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import sys
import pywt
sys.path.insert(0, '../Features')
import features_utils as fu

# Level 1


class TestFeature:
    def __init__(self, data=None):
        if(data is None):
            self.data, _ = fu.loadUserInput()
        else:
            self.data = data
        self.data.timestamp -= self.data.timestamp.iloc[0]
        self.x = None
        self.y = None
        self.xlabel = None
        self.ylabel = None
        self.yWavelet = None

    def plot(self):
        plt.plot(self.x, self.y, label='raw')
        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)

    def show(self):
        self.plot()
        plt.show()

    def plotWavelet(self, wavelet, level):
        coeffs = pywt.wavedec(self.y, wavelet, level=level)
        self.yWavelet = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(self.y))
        name = '{}, level:{}'.format(wavelet, str(level))
        plt.plot(self.x, self.yWavelet, label=name)


# Level 2


class RawFeature(TestFeature):
    def __init__(self, axis, data=None):
        super(RawFeature, self).__init__(data)
        self.x = self.data.timestamp
        self.y = self.data.loc[:, axis]
        self.ylabel = 'Axis ' + axis
        self.xlabel = 'time'


class Feature2dBins(TestFeature):
    def __init__(self, axis1, axis2, data=None):
        super(Feature2dBins, self).__init__(data)
        self.x = [i for i in range(1, 1000)]
        self.y = [self.mainCalc(axis1, axis2, xx) for xx in self.x]
        self.xlabel = '#bins'

    def mainCalc(self, axis1, axis2, bins):
        pass


class RawSpherical(TestFeature):
    def __init__(self, data=None):
        super(RawSpherical, self).__init__(data)
        self.x = self.data.timestamp
        self.xlabel = 'time'
        self.y = self.mainCalc()

    def mainCal(self):
        pass


class Entropy(TestFeature):
    def __init__(self, axis, data=None):
        super(Entropy, self).__init__(data)
        self.x = [i for i in range(1, 10000)]
        self.y = [fu.entropy(self.data, axis, xx) for xx in self.x]
        self.ylabel = 'Entropy ' + axis
        self.xlabel = '#bins'


class Frequency(TestFeature):
    def __init__(self, axis, data=None):
        super(Frequency, self).__init__(data)
        self.y = np.abs(np.fft.fft(self.data[axis]))
        self.x = np.fft.fftfreq(len(self.data), d=0.01)
        self.ylabel = 'Frequency domain Amplitude' + axis
        self.xlabel = 'Frequency'

# Level 3


class MutualInfo(Feature2dBins):
    def __init__(self, axis1, axis2, data=None):
        super(MutualInfo, self).__init__(axis1, axis2, data)
        self.ylabel = 'Mutual Information ' + axis1 + axis2

    def mainCalc(self, axis1, axis2, bins):
        return fu.mutualInfo(self.data, axis1, axis2, bins)


class CrossEntropy(Feature2dBins):
    def __init__(self, axis1, axis2, data=None):
        super(CrossEntropy, self).__init__(axis1, axis2, data)
        self.ylabel = 'Cross Entropy ' + axis1 + axis2

    def mainCalc(self, axis1, axis2, bins):
        return fu.crossEntropy(self.data, axis1, axis2, bins)


class RadialDistance(RawSpherical):
    def __init__(self, data=None):
        super(RadialDistance, self).__init__(data)
        self.ylabel = 'Radial Distance'

    def mainCalc(self):
        return fu.cart2sphRadialDist(self.data, raw=True)


class PolarAngle(RawSpherical):
    def __init__(self, data=None):
        super(PolarAngle, self).__init__(data)
        self.ylabel = 'Polar Angle'

    def mainCalc(self):
        return fu.cart2sphPolarAngle(self.data, raw=True)


class AzimuthAngle(RawSpherical):
    def __init__(self, data=None):
        super(AzimuthAngle, self).__init__(data)
        self.ylabel = 'Azimuth Angle'

    def mainCalc(self):
        return fu.cart2sphAzimuthAngle(self.data, raw=True)


# Extra for overview

class Overview():
    def __init__(self, segments=False, totalAvgStep=False):
        self.data, self.dataPedo = fu.loadUserInput()
        self.segments = segments
        self.totalAvgStep = totalAvgStep
        self.waveletList = []

    def overiewPlotting(self, obj):
        obj.plot()
        for wavelet, level in self.waveletList:
            obj.plotWavelet(wavelet, level)
        if self.segments:
            self.plotSegments()
        plt.legend()

    def show(self):
        for index, axis in enumerate(['x', 'y', 'z']):
            plt.subplot(2, 2, index + 1)
            obj = RawFeature(axis, self.data)
            self.overiewPlotting(obj)
        plt.subplot(2, 2, 4)
        obj = RadialDistance(self.data)
        self.overiewPlotting(obj)
        plt.show()

    def addWavelet(self, wavelet, level):
        self.waveletList.append((wavelet, level))

    def removeAllWavelets(self):
        self.waveletList = []

    def removeWavelet(self, wavelet, level):
        self.waveletList.remove((wavelet, level))

    def plotSegments(self):
        start = datetime.strptime(self.dataPedo.loc[0, 'startDate'], '%Y-%m-%dT%H:%M:%S%z')
        if self.totalAvgStep:
            end = datetime.strptime(self.dataPedo.loc[:, 'endDate'].iloc[-1], '%Y-%m-%dT%H:%M:%S%z')
            delta = end - start
            steps = self.dataPedo.loc[:, 'numberOfSteps'].iloc[-1]
            segment = delta.seconds / steps
            for i in range(steps):
                pos = i * segment
                plt.axvline(x=pos, color='red')
        else:
            zero = start
            prevStep = 0
            for row in self.dataPedo.itertuples():
                end = datetime.strptime(row.endDate, '%Y-%m-%dT%H:%M:%S%z')
                steps = row.numberOfSteps - prevStep
                delta = end - start
                if steps > 0:
                    segment = delta.seconds / steps
                    delta = start - zero
                    for i in range(steps):
                        pos = delta.seconds + i * segment
                        plt.axvline(x=pos, color='red')
                prevStep += steps
                start = end
