import numpy as np
from numpy.linalg import eig, inv
import pandas as pd
from numba import jit

# input should be in the format
# [[x1 x2 x3 ... x8], 
#  [x1 x2 x3 ... x8], 
#  [x1 x2 x3 ... x8], 
# ]

@jit(nopython=True)
def fit_ellipse(x, y):
    x = x.reshape((x.shape[0], 1))
    y = y.reshape((y.shape[0], 1))
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros(shape=(6, 6))
    C[0, 2] = C[2, 0] = 2;
    C[1, 1] = -1
    E, V = eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a

def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return 0.5 * np.arctan(2 * b / (a - c))


def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])


def ellipse_angle_of_rotation2(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi / 2
    else:
        if a > c:
            return np.arctan(2 * b / (a - c)) / 2
        else:
            return np.pi / 2 + np.arctan(2 * b / (a - c)) / 2


def get_ellipse_attributes(x, y):
    a = fit_ellipse(x, y)

    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    a, b = axes
    R = np.linspace(0, 2 * np.pi, 100)

    theta = (phi / np.pi * 180 + 360) % 360

    if a < b and theta < 90 or a > b and 90 < theta < 180:
        phi = phi + np.pi / 2

    xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
    yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
    area = np.pi * a * b

    dictionary = {"center": center, "phi": phi, "axes": axes, "area": area, "x": xx, "y": yy}

    return dictionary