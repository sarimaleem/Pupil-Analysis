import numpy as np
import pandas as pd
import pickle
import platform
import statsmodels.nonparametric.api as smnp
from scipy.signal import find_peaks
import cv2

def get_frame_rate(vid_file):
    video = cv2.VideoCapture(vid_file)
    fps = video.get(cv2.CAP_PROP_FPS)

    return fps

def get_frame(filename, frame_nr):
    # read frame:
    cap = cv2.VideoCapture(filename)
    cap.set(1, frame_nr)
    ret, frame = cap.read()

    # Just in case
    assert frame is not None, f'Frame {frame_nr} is NoneType in {filename}'

    # ensure that we're working integers:
    frame = frame.astype('uint8')

    # cast to gray scale:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame

def get_crop_adjust(filename):
    with open(filename.replace('.mp4', '.pickle'), 'rb') as handle:
        video_info = pickle.load(handle)

    x = video_info['crop_cords'][1][0]
    y = video_info['crop_cords'][0][0]

    return x, y

def get_pupil_coordinates(frame_num, df):
    cols = ['pupil1', 'pupil2', 'pupil3', 'pupil4', 'pupil5', 'pupil6', 'pupil7', 'pupil8']

    x = df.loc[frame_num, (cols, 'x')]
    y = df.loc[frame_num, (cols, 'y')]

    x = np.asarray(x)
    y = np.asarray(y)

    zx = (x - x.mean()) / x.std()
    zy = (y - y.mean()) / y.std()

    mask = np.invert((zx > 2) | (zy > 2))

    return x, y, mask


def get_eyelid_coordinates(frame_num, df):
    cols = ['edge1', 'edge8', 'edge3', 'edge5', 'edge2', 'edge6', 'edge4', 'edge7']

    x = df.loc[frame_num, (cols, 'x')]
    y = df.loc[frame_num, (cols, 'y')]

    x = x.droplevel('coords')
    x = pd.DataFrame(x)
    x = x.T

    y = y.droplevel('coords')
    y = pd.DataFrame(y)
    y = y.T

    x = x[['edge1', 'edge8', 'edge3', 'edge5', 'edge2', 'edge6', 'edge4', 'edge7']]
    y = y[['edge1', 'edge8', 'edge3', 'edge5', 'edge2', 'edge6', 'edge4', 'edge7']]

    x = np.asarray(x)
    y = np.asarray(y)

    x_z = (x - x.mean()) / x.std()
    y_z = (y - y.mean()) / y.std()

    mask = np.invert((x_z > 2) | (y_z > 2))

    x = x[mask]
    y = y[mask]

    return x, y


def get_likelihoods(df):
    cols = ['edge' + str(n + 1) for n in range(8)] + ['pupil' + str(n + 1) for n in range(8)]
    likelihoods = df.loc[:, (cols, 'likelihood')]
    likelihoods = np.asarray(likelihoods).tolist()

    return likelihoods


def get_eyelid_coordinates_time_series(df):
    cols = ['edge1', 'edge2', 'edge4', 'edge3', 'edge4', 'edge5', 'edge6', 'edge7', 'edge8']

    x = df.loc[:, (cols, 'x')]
    y = df.loc[:, (cols, 'y')]

    x = x[['edge1', 'edge8', 'edge3', 'edge5', 'edge2', 'edge6', 'edge4',
           'edge7']]  # order of the points connected in order. It is very important that they are returned like this
    y = y[['edge1', 'edge8', 'edge3', 'edge5', 'edge2', 'edge6', 'edge4', 'edge7']]

    x = np.asarray(x)
    x_mean = np.mean(x, axis=1).reshape((x.shape[0], 1))
    x_std = np.std(x, axis=1).reshape((x.shape[0], 1))
    x_z = abs((x - x_mean) / x_std)

    y = np.asarray(y)
    y_mean = np.mean(y, axis=1).reshape((y.shape[0], 1))
    y_std = np.std(y, axis=1).reshape((y.shape[0], 1))
    y_z = abs((y - y_mean) / y_std)

    mask = np.invert((x_z > 2) | (y_z > 2))

    return x.astype(np.float32), y.astype(np.float32), mask


def get_pupil_coordinates_time_series(df):

    cols = ['pupil1', 'pupil2', 'pupil3', 'pupil4', 'pupil5', 'pupil6', 'pupil7', 'pupil8']
    x = df.loc[:, (cols, 'x')]
    y = df.loc[:, (cols, 'y')]
    x = np.asarray(x)
    y = np.asarray(y)

    return x, y

def get_mask(x,y):

    x_mean = np.mean(x, axis=1).reshape((x.shape[0], 1))
    x_std = np.std(x, axis=1).reshape((x.shape[0], 1))
    x_z = abs((x - x_mean) / x_std)

    y_mean = np.mean(y, axis=1).reshape((y.shape[0], 1))
    y_std = np.std(y, axis=1).reshape((y.shape[0], 1))
    y_z = abs((y - y_mean) / y_std)

    mask = np.invert((x_z > 2) | (y_z > 2))

    return mask

def get_kde_threshold(array):
    dens = smnp.KDEUnivariate(array)
    dens.fit(gridsize=np.max(array).astype(int), bw=2000)
    x, y = dens.support, dens.density
    peaks = find_peaks(y)
    peaks = peaks[0]
    highest_peaks = peaks[y[peaks].argsort()[-2:][::-1]]  # we get the indices of the two highest peaks
    try:
        thresh = (x[highest_peaks[0]] - x[highest_peaks[1]]) / 4 + x[highest_peaks[1]]
    except:
        thresh = np.min(array)
    # we get the threshold. code works on  assumption that there is a small peak followed by a large peak
    # in the distribution of the rotated rectangle area

    return thresh