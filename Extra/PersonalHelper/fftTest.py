import numpy as np
import matplotlib.pyplot as plt


def rect(x):
    x1 = x >= -0.5
    x2 = x <= 0.5
    return x1 & x2


x = np.linspace(-10, 10, 2001)
dt = x[1] - x[0]
y = rect(x)

yf = np.abs(np.fft.fft(y))
xf = np.fft.fftfreq(len(x), d=dt)

# plt.plot(xf, yf)
# plt.show()


from scipy.fftpack import fft
# Number of sample points
N = 2001
# sample spacing
T = 0.01
x = np.linspace(0.0, N * T, N)
y = rect(x)
yf = fft(y)
xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
plt.grid()
plt.show()
