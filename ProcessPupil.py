import statsmodels.nonparametric.api as smnp
from scipy.signal import find_peaks
import cv2
import numpy as np
from numba import jit
from UtilFuncs import get_pupil_coordinates_time_series
from UtilFuncs import get_eyelid_coordinates_time_series
from FitEllipse import get_ellipse_attributes_time_series


# time series of the area of the eyelid using  shoelace formula on the assumption that all the points are connected
# in order
@jit(nopython=True, parallel=True)
def get_eyelid_area_time_series(x, y):
    area = np.zeros(shape=x.shape[0])

    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]

        area[i] = 0.5 * np.abs(np.dot(xi, np.roll(yi, 1)) - np.dot(yi, np.roll(xi, 1)))

    return area


# gets a time series of the area of a the rotated rectangle that is fitted to a set of points
def get_fitted_rect_area_time_series(lid_x, lid_y):
    num_frames = lid_x.shape[0]
    rect_area = np.zeros(shape=(num_frames, 1))

    for n in range(num_frames):
        pts = np.stack((lid_x[n], lid_y[n])).T
        rect = cv2.minAreaRect(pts)
        area = rect[1][0] * rect[1][1]
        rect_area[n] = area

    return rect_area


def get_time_elapsed(df, vid_file):
    video = cv2.VideoCapture(vid_file)
    fps = video.get(cv2.CAP_PROP_FPS)

    return df.index.values / fps


def get_threshold(rot_rect_area):
    dens = smnp.KDEUnivariate(rot_rect_area)
    dens.fit(gridsize=np.max(rot_rect_area).astype(int), bw=2000)
    x, y = dens.support, dens.density

    peaks = find_peaks(y)
    peaks = peaks[0]
    highest_peaks = peaks[y[peaks].argsort()[-2:][::-1]]  # we get the indices of the two highest peaks
    thresh = (x[highest_peaks[0]] - x[highest_peaks[1]]) / 4 + x[highest_peaks[1]]
    # we get the threshold. code works on  assumption that there is a small peak followed by a large peak
    # in the distribution of the rotated rectangle area

    return thresh


def is_blink(df, lid_x, lid_y):
    thresh = get_threshold(df['rotated rect area'])
    df['is_blink'] = df['rotated rect area'] < thresh

    x_mean = np.mean(lid_x, axis=1).reshape((lid_x.shape[0], 1))
    x_std = np.std(lid_x, axis=1).reshape((lid_x.shape[0], 1))
    x_z = (lid_x - x_mean) / x_std

    y_mean = np.mean(lid_y, axis=1).reshape((lid_y.shape[0], 1))
    y_std = np.std(lid_y, axis=1).reshape((lid_y.shape[0], 1))
    y_z = (lid_y - y_mean) / y_std

    df['is_blink'] = df['is_blink'] | (x_z > 2).any(axis=1) | (y_z > 2).any(axis=1) | (df['major minor ratio'] > 1.5) | (df[
        'major minor ratio'].isnull().values)

    df['x_z > 2'] = (x_z > 2).any(axis=1)
    df['y_z > 2'] = (y_z > 2).any(axis=1)

    df.loc[df['is_blink'], 'pupil'] = 0

    return df


def extract_features(df, pupil_x, pupil_y, lid_x, lid_y):
    df['eyelid area'] = get_eyelid_area_time_series(lid_x, lid_y)
    df['major minor ratio'] = df['major'] / df['minor']
    df['x standard deviation'] = np.std(pupil_x, axis=1)
    df['y standard deviation'] = np.std(pupil_y, axis=1)
    df['rotated rect area'] = get_fitted_rect_area_time_series(lid_x, lid_y)

    return df


def create_time_series(deep_cut, crop_x, crop_y):
    pupil_x, pupil_y, mask = get_pupil_coordinates_time_series(deep_cut)

    pupil_x += crop_x
    pupil_y += crop_y

    df = get_ellipse_attributes_time_series(pupil_x, pupil_y, mask)

    lid_x, lid_y, _ = get_eyelid_coordinates_time_series(deep_cut)
    lid_x = lid_x.astype(np.float32)
    lid_y = lid_y.astype(np.float32)

    lid_x += crop_x
    lid_y += crop_y

    df = extract_features(df, pupil_x, pupil_y, lid_x, lid_y)
    df = is_blink(df, lid_x, lid_y)

    return df
