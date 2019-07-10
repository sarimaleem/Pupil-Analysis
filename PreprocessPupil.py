import os, glob
import numpy as np
import scipy as sp
from scipy import stats
from scipy import signal
import pandas as pd
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns


def _find_nearest(array, value):
    idx = (np.abs(array - value)).idxmin()
    return idx


def _butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = sp.signal.butter(order, [high], btype='lowpass')
    return b, a


def _butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = _butter_lowpass(highcut, fs, order=order)
    y = sp.signal.filtfilt(b, a, data)
    return y


def _butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = sp.signal.butter(order, [low], btype='highpass')
    return b, a


def _butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = _butter_highpass(lowcut, fs, order=order)
    y = sp.signal.filtfilt(b, a, data)
    return y


def _detect_blinks(df, blink_detection_measure='pupil', cutoff=50, coalesce_period=1):
    pupil = (np.array(df[blink_detection_measure]) - np.median(df[blink_detection_measure])) / np.median(
        df[blink_detection_measure])
    pupil_diff = np.diff(pupil) / np.median(np.diff(df.time))  # % signal change / s
    blink_periods = np.array(df.loc[np.where((pupil_diff < -cutoff) | (pupil_diff > cutoff))[0], 'time'])
    blink_periods = blink_periods[blink_periods != df['time'].iloc[0]]
    blink_periods = blink_periods[blink_periods != df['time'].iloc[-1]]
    if len(blink_periods) > 0:
        blink_start_ind = np.where(np.concatenate((np.array([True]), np.diff(blink_periods) > coalesce_period)))[0]
        blink_end_ind = np.concatenate(((blink_start_ind - 1)[1:], np.array([blink_periods.shape[0] - 1])))
        return blink_periods[blink_start_ind], blink_periods[blink_end_ind]
    else:
        return None, None


def interpolate_blinks(df, measure='pupil', blink_detection_measure='pupil', cutoff=50, coalesce_period=1, buffer=0.2):
    blink_starts, blink_ends = _detect_blinks(df, blink_detection_measure, cutoff, coalesce_period)
    df['{}_int'.format(measure)] = df[measure].copy()
    df['is_blink'] = 0
    if blink_starts is not None:
        for start, end in zip(blink_starts, blink_ends):
            s = _find_nearest(df['time'], start - buffer)
            e = _find_nearest(df['time'], end + buffer)
            df.loc[s:e, '{}_int'.format(measure)] = np.linspace(df.loc[s, '{}_int'.format(measure)],
                                                                df.loc[e, '{}_int'.format(measure)], e - s + 1)
            s = _find_nearest(df['time'], start)
            e = _find_nearest(df['time'], end)
            df.loc[s:e, 'is_blink'] = 1


def filter_pupil(df, fs=15, hp=0.01, lp=6.0, order=3):
    df['pupil_int_lp'] = _butter_lowpass_filter(data=df['pupil_int'], highcut=lp, fs=fs, order=order)
    df['pupil_int_bp'] = _butter_highpass_filter(data=df['pupil_int'], lowcut=hp, fs=fs, order=order) - (
            df['pupil_int'] - df['pupil_int_lp'])


def psc_pupil(df):
    df['pupil_psc'] = (df['pupil_int_lp'] - df['pupil_int_lp'].median()) / df['pupil_int_lp'].median() * 100


def fraction_pupil(df):
    # min_pupil = np.percentile(df['pupil_int_lp'], 2.5)
    # max_pupil = np.percentile(df['pupil_int_lp'], 97.5)
    # range_ori = (max_pupil - min_pupil)
    # range_new = (0.975 - 0.025)
    # df['pupil_frac'] = (((df['pupil_int_lp'] - min_pupil) * range_new) / range_ori)

    max_pupil = np.percentile(df['pupil_int_lp'], 99)
    df['pupil_frac'] = df['pupil_int_lp'] / max_pupil


def slope_pupil(df, hp=2.0, fs=15, order=3):
    slope = np.concatenate((np.array([0]), np.diff(df['pupil_frac']))) * fs
    slope = _butter_lowpass_filter(slope, highcut=hp, fs=fs, order=order)
    # slope = np.array(pd.Series(slope).rolling(5, center=True, win_type=None).mean())
    df['pupil_slope'] = slope


def preprocess_pupil(params, df):


    # unpack:
    fs = params['fs']
    hp = params['hp']
    lp = params['lp']
    order = params['order']
    s_hp = params['s_hp']

    df['time'] = np.linspace(0, df.shape[0] / fs, df.shape[0])
    interpolate_blinks(df=df)
    filter_pupil(df=df, fs=fs, hp=hp, lp=lp, order=order)
    psc_pupil(df=df)
    fraction_pupil(df=df)
    slope_pupil(df=df, hp=s_hp, fs=fs, order=order)

    return df


def plot_preprocessing(df, path, animal_name):

    # plot:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(211)
    plt.plot(df['time'] / 60, df['pupil'], label='raw')
    plt.plot(df['time'] / 60, df['pupil_int_lp'], label='interpolated+filtered')
    plt.xlabel('Time (min)')
    plt.ylabel('Pupil size (a.u.)')

    # add blink timestamps:
    x = np.array(df['time']) / 60
    sig_indices = np.array(df['is_blink'] == 1, dtype=int)
    sig_indices[0] = 0
    sig_indices[-1] = 0
    s_bar = zip(np.where(np.diff(sig_indices) == 1)[0] + 1, np.where(np.diff(sig_indices) == -1)[0])
    for sig in s_bar:
        rect = patches.Rectangle((x[int(sig[0])], ax.get_ylim()[0]), x[int(sig[1])] - x[int(sig[0])], ax.get_ylim()[1],
                                 linewidth=0, facecolor='red', alpha=0.2)
        ax.add_patch(rect)

    ax = fig.add_subplot(212)
    plt.plot(df['time'] / 60, df['pupil_slope'], label='slope')
    plt.xlabel('Time (min)')
    plt.ylabel('Pupil slope (a.u.)')
    plt.tight_layout()

    fig.savefig(path + '/' + animal_name + 'time_series_graph.png')

