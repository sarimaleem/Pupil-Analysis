import statsmodels.nonparametric.api as smnp
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from FitEllipse import get_ellipse_attributes
from ProcessPupil import get_threshold
from UtilFuncs import get_frame, get_eyelid_coordinates, get_pupil_coordinates
import cv2


def plot_rotated_rect_area(rect_area, path: str, animal_name):
    dens = smnp.KDEUnivariate(rect_area)
    dens.fit(gridsize=2000, bw=2000)
    x, y = dens.support, dens.density

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    thresh = get_threshold(rect_area)
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(10, 10))

    plt.plot(x, y)
    plt.axvline(thresh, linestyle='-.', color='red', label=f'threshold={thresh // 1}')
    plt.xlabel('Rotated Rectangle Area')
    plt.ylabel('Kernel Density')
    plt.title('Rotated Rectangle Area KDE')
    plt.legend()

    fig.savefig(path + '/' + animal_name + 'rotated_rect_area_density.pdf')


def plot_axes_ratio_dist(axes_ratio, path, animal_name):
    assert type(axes_ratio) == pd.core.series.Series, 'data type must be series'

    fig = plt.figure(figsize=(15, 15))

    minimum = str(round(np.min(axes_ratio.dropna()), 6))
    median = str(round(np.percentile(axes_ratio.dropna(), 50), 6))
    maximum = str(round(np.max(axes_ratio.dropna()), 6))
    num_na = axes_ratio.isnull().values.sum()
    thresh1 = (axes_ratio.dropna() > 1.5).sum()
    thresh2 = (axes_ratio.dropna() > 2).sum()
    label = f'min = {minimum}\nmax = {maximum}\nmedian={median}\nnum na = {num_na}\n num>1.5 = {thresh1}\n num>2 = {thresh2}'

    sns.distplot(axes_ratio[axes_ratio < 2.5].dropna(), bins=500, label=label)
    plt.title('major minor ratio distribution')
    plt.ylabel('density')

    plt.legend()
    fig.savefig(path + '/' + animal_name + 'axes_ratio.png', dpi=400)


def plot_blinks(df, deep_cut, vid, crop_x, crop_y, path, animal_name):
    indices = df[df['is_blink']].sample(25).index.values.reshape(5, 5)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25, 25))

    for i in range(5):
        for j in range(5):
            n = indices[i, j]
            fr = get_frame(vid, n)
            plot_eye(n, fr, deep_cut, axes[i, j], crop_x, crop_y)
            mmr = str(round(df.loc[n, 'major minor ratio'], 4))
            x_mask = df.loc[n, 'x_z > 2']
            y_mask = df.loc[n, 'y_z > 2']

            axes[i, j].set_title(f'mmr:{mmr} x_z > 2:{x_mask} y_z > 2{y_mask}')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Random Sample of 25 Blinks Based on KDE Threshold Method', fontsize=24)

    fig.savefig(path + '/' + animal_name + 'blinks.pdf')


def plot_verify_grid(quantiles, df, path, deep_cut, crop_x, crop_y, vid, animal_name):
    fig, axes = plt.subplots(4, quantiles, figsize=(30, 30))
    non_blinks = df[~df['is_blink']]

    plt.suptitle('Verification Grid')
    eyelid_area_sort = non_blinks.sort_values(by='eyelid area')
    pupil_area_sort = non_blinks.sort_values(by='pupil')

    axes[0, 0].set_ylabel('smallest pupil by lid area quantile', fontsize=6)
    for x in range(quantiles):
        start = eyelid_area_sort.shape[0] // quantiles * x
        end = eyelid_area_sort.shape[0] // quantiles * (x + 1)

        quantile_df = eyelid_area_sort.iloc[start:end]
        quantile_df = quantile_df.sort_values(by='pupil')
        index = quantile_df.iloc[0].name

        plot_eye(index, get_frame(vid, index), deep_cut, axes[0, x], crop_x, crop_y)
        axes[0, x].set_title(f'{100 / quantiles * x}% - {100 / quantiles * (x + 1)}%')

    axes[1, 0].set_ylabel('largest pupil by lid area quantile', fontsize=6)
    for x in range(quantiles):
        start = eyelid_area_sort.shape[0] // quantiles * x
        end = eyelid_area_sort.shape[0] // quantiles * (x + 1)

        quantile_df = eyelid_area_sort.iloc[start:end]
        quantile_df = quantile_df.sort_values(by='pupil')
        index = quantile_df.iloc[-1].name

        plot_eye(index, get_frame(vid, index), deep_cut, axes[1, x], crop_x, crop_y)

    axes[2, 0].set_ylabel('smallest eyelid by pupil area quantile', fontsize=6)
    for x in range(quantiles):
        start = pupil_area_sort.shape[0] // quantiles * x
        end = pupil_area_sort.shape[0] // quantiles * (x + 1)

        quantile_df = pupil_area_sort.iloc[start:end]
        quantile_df = quantile_df.sort_values(by='eyelid area')
        index = quantile_df.iloc[0].name

        plot_eye(index, get_frame(vid, index), deep_cut, axes[2, x], crop_x, crop_y)

    axes[3, 0].set_ylabel('largest eyelid by pupil area quantile', fontsize=6)
    for x in range(quantiles):
        start = pupil_area_sort.shape[0] // quantiles * x
        end = pupil_area_sort.shape[0] // quantiles * (x + 1)

        quantile_df = pupil_area_sort.iloc[start:end]
        quantile_df = quantile_df.sort_values(by='eyelid area')
        index = quantile_df.iloc[-1].name

        plot_eye(index, get_frame(vid, index), deep_cut, axes[3, x], crop_x, crop_y)

    fig.savefig(path + '/' + animal_name + 'verify_grid.pdf')


def plot_eyelid(x, y, ax):
    # ax.scatter(x, y)

    pts = np.stack((x, y)).T
    pts = pts.astype(int)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    ax.plot(box[:, 0], box[:, 1], color='magenta')
    ax.plot(np.roll(box[:, 0], shift=1), np.roll(box[:, 1], shift=1), color='magenta')

    ax.plot(x, y, color='yellow')
    ax.plot(np.roll(x, shift=1), np.roll(y, shift=1), color='yellow')


def plot_pupil(x, y, ax, mask):
    masked_x = x[mask]
    masked_y = y[mask]

    ax.scatter(x, y, color='cyan')
    ax.scatter(masked_x, masked_y, color='green')
    attr = get_ellipse_attributes(masked_x, masked_y)
    ax.plot(attr['x'], attr['y'], color='red')


def plot_eye(frame_num, frame, deep_cut, ax, adj_x, adj_y):
    # adjX and adjY is for cropping
    # assuming the top row in deepcut is already dropped

    lid_x, lid_y = get_eyelid_coordinates(frame_num, deep_cut)
    pupil_x, pupil_y, mask = get_pupil_coordinates(frame_num, deep_cut)

    lid_x += adj_x
    pupil_x += adj_x

    lid_y += adj_y
    pupil_y += adj_y

    plot_eyelid(lid_x, lid_y, ax)
    plot_pupil(pupil_x, pupil_y, ax, mask)

    ax.imshow(frame, cmap='gray', rasterized=True)
