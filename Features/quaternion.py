import numpy as np


class Quaternion:
    def __init__(self, re, *im):
        self.re = re
        if len(im) == 1:
            if type(im[0]) == np.ndarray:
                self.im = im[0]
            else:
                self.im = np.array(im[0])
        else:
            self.im = np.array(im)

    def __add__(self, q):
        return Quaternion(self.re + q.re, self.im + q.im)

    def __sub__(self, q):
        return Quaternion(self.re - q.re, self.im - q.im)

    def __eq__(self, q):
        return self.re == q.re and self.im == q.im

    def __mul__(self, q):
        re = self.re * q.re - self.im.dot(q.im)
        im = self.re * q.im + q.re * self.im + np.cross(self.im, q.im)
        return Quaternion(re, im)

    def __str__(self):
        return "[ {}, {}i, {}j, {}k ]".format(self.re, self.im[0], self.im[1], self.im[2])

    def mulScalar(self, x):
        return Quaternion(x * self.re, self.im * x)

    def conjugate(self):
        return Quaternion(self.re, -self.im)

    def norm2(self):
        return self.im.dot(self.im) + self.re * self.re

    def inverse(self):
        return self.conjugate().mulScalar(1 / self.norm2())
