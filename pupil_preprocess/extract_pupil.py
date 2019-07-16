import os, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import jit
import cv2

from pupil_preprocess import fit_ellipse
from pupil_preprocess import plot_funcs
from pupil_preprocess import utils

@jit(nopython=True, parallel=True)  # numba for optimizing loops
def get_a_times_series(x, y, mask):

    assert x.shape[1] == 8, 'time series not proper shape'
    a_time_series = np.zeros(shape=(x.shape[0], 6))
    for i in range(x.shape[0]):
        xi = x[i][mask[i]]
        yi = y[i][mask[i]]
        a_time_series[i] = fit_ellipse.fit_ellipse(xi, yi)
        # print(i)

    return a_time_series

def get_ellipse_attributes_time_series(a_time_series):
    
    center = fit_ellipse.ellipse_center(a_time_series.T).T
    phi = fit_ellipse.ellipse_angle_of_rotation(a_time_series.T).T
    axes_length = fit_ellipse.ellipse_axis_length(a_time_series.T).T
    minor = np.sort(axes_length)[:, 0]
    major = np.sort(axes_length)[:, 1]
    a = axes_length[:, 0]
    b = axes_length[:, 1]
    area = np.pi * a * b
    df = pd.DataFrame(
        {'x': center[:,0], 'y': center[:,1], 'pupil': area, 
        'minor': minor, 'major': major, 'phi': phi})

    return df

@jit(nopython=True, parallel=True)
def get_eyelid_area_time_series(x, y):

    """
    time series of the area of the eyelid using shoelace formula on the assumption that all the points are connected in order
    """

    area = np.zeros(shape=x.shape[0])
    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]
        area[i] = 0.5 * np.abs(np.dot(xi, np.roll(yi, 1)) - np.dot(yi, np.roll(xi, 1)))

    return area

def get_fitted_rect_area_time_series(lid_x, lid_y):

    """
    gets a time series of the area of a the rotated rectangle that is fitted to a set of points
    """

    num_frames = lid_x.shape[0]
    rect_area = np.zeros(shape=(num_frames, 1))
    for i in range(num_frames):
        pts = np.stack((lid_x[i], lid_y[i])).T
        rect = cv2.minAreaRect(pts)
        area = rect[1][0] * rect[1][1]
        rect_area[i] = area

    return rect_area


def is_blink(df, lid_x, lid_y, max_z=2, max_ratio=1.75):
    
    # get kde threshold:
    thresh = utils.get_kde_threshold(np.array(df['eyelid']))

    # get coordinate z-scores:
    x_mean = np.mean(lid_x, axis=1).reshape((lid_x.shape[0], 1))
    x_std = np.std(lid_x, axis=1).reshape((lid_x.shape[0], 1))
    x_z = abs((lid_x - x_mean) / x_std)
    y_mean = np.mean(lid_y, axis=1).reshape((lid_y.shape[0], 1))
    y_std = np.std(lid_y, axis=1).reshape((lid_y.shape[0], 1))
    y_z = abs((lid_y - y_mean) / y_std)
    
    # compute blink:
    df['is_blink'] = ((df['eyelid'] < thresh) | (x_z > max_z).any(axis=1) | (y_z > max_z).any(axis=1) | 
                        (df['major_minor_ratio'] > max_ratio) | (df['major_minor_ratio'].isnull().values))
    
    return df

def extract_timeseries(hdf_file, mp4_file, crop_file, output_dir, subject_id):

    print('analyzing {}'.format(hdf_file))

    # load video attributes:
    fps = utils.get_frame_rate(mp4_file)

    # load dlc output:
    deep_cut = pd.read_hdf(hdf_file)
    deep_cut.columns = deep_cut.columns.droplevel()
    pupil_x, pupil_y = utils.get_pupil_coordinates_time_series(deep_cut)
    lid_x, lid_y, _ = utils.get_eyelid_coordinates_time_series(deep_cut)

    # get mask:
    mask = utils.get_mask(pupil_x, pupil_y)

    # add crop coordinates:
    if crop_file is None:
        crop_x, crop_y = 0, 0
    else:
        crop_x, crop_y = utils.get_crop_adjust(crop_file)
    lid_x += crop_x
    lid_y += crop_y
    pupil_x += crop_x
    pupil_y += crop_y
    
    # compute pupil area:
    print('computing pupil area...')
    a_time_series = get_a_times_series(pupil_x, pupil_y, mask)
    df = get_ellipse_attributes_time_series(a_time_series)

    # compute eyelid area:
    print('computing eyelid area...')
    # df['eyelid'] = get_eyelid_area_time_series(lid_x, lid_y)
    df['eyelid'] = get_fitted_rect_area_time_series(lid_x, lid_y)
    
    # add major / minor ratio:
    df['major_minor_ratio'] = df['major'] / df['minor']

    # compute blinks:
    df = is_blink(df, lid_x, lid_y)

    # set pupil to 0 for detected blinks:
    df.loc[df['is_blink'], 'pupil'] = 0
    
    # add time:
    df['time'] = df.index.values / fps

    # add dlc likelihoods:
    df['likelihoods'] = utils.get_likelihoods(deep_cut)

    # plots:
    plot_funcs.plot_kde(df['eyelid'], output_dir, subject_id)
    plot_funcs.plot_blinks(df, deep_cut, mp4_file, crop_x, crop_y, output_dir, subject_id)
    plot_funcs.plot_axes_ratio_dist(df['major_minor_ratio'], output_dir, subject_id)
    plot_funcs.plot_verify_grid(10, df, output_dir, deep_cut, crop_x, crop_y, mp4_file, subject_id)

    # save:
    df.to_hdf(os.path.join(output_dir, subject_id + '_df_pupil.hdf'), key='pupil')

    return df