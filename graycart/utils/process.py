import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import splev, splrep

from graycart.utils import fit, plotting

import matplotlib.pyplot as plt


def find_single_peak(df, target_width, min_width, width_space, fit_func='parabola',
                     rel_height=0.95, prominence=1, plot_width_rel_target=1.05):
    """
    dfpk, peak_properties = process.find_single_peak(df, target_width, min_width, width_space=50, fit_func='parabola', rel_height=0.9375, prominence=1)

    :param df:
    :param width_space: 500
    :param rel_height:
    :param prominence:
    :return:
    """

    # 1. pre-process data
    x = df.x.to_numpy()
    z = df.z.to_numpy() * -1

    # 2. find peaks
    height = z.max() / 2  # minimum peak height
    distance = target_width
    peaks, peak_properties = find_peaks(z, height=height, width=min_width, distance=distance,
                                        prominence=prominence, rel_height=rel_height)

    # 3. fit function to peak
    peak_properties = fit.fit_single_peak(df, peak_properties, width_space, fit_func)

    # 4. get data
    dfpk = df[np.abs(df['x'] - peak_properties['pk_xc']) < target_width * plot_width_rel_target]

    return dfpk, peak_properties


def find_multi_peak(df, peak_labels, peak_ids, target_width, min_width, width_space, fit_func='parabola',
                    rel_height=0.95, prominence=1, plot_width_rel_target=1.05):
    """
    dfpk, peak_details = process.find_multi_peak(df, peak_labels, peak_ids, target_width, min_width, width_space=50, fit_func='parabola', rel_height=0.9375, prominence=1)

    :param df:
    :param width_space: 500
    :param rel_height:
    :param prominence:
    :return:
    """

    # 1. pre-process data
    x = df.x.to_numpy()
    z = df.z.to_numpy() * -1

    # 2. find peaks
    height = z.max() / 1.25  # 2  # minimum peak height
    distance = min_width * 0.5
    # min_width = min_width * 0.5
    # rel_height = 0.9
    # prominence = 0.9

    peaks, peak_properties = find_peaks(z, height=height, width=min_width, distance=distance, prominence=prominence,
                                        rel_height=rel_height)

    # 3. iterate through each peak
    peak_details = {}
    dfpks = []
    peak_idx = np.arange(len(peaks))
    for pidx, plbl, pid, pk in zip(peak_idx, peak_labels, peak_ids, peaks):

        # 4. fit function to peak
        pk_properties = fit.fit_single_peak(df, peak_properties, pidx, width_space, fit_func)

        # 5. get data
        dfpk = df[np.abs(df['x'] - pk_properties['pk_xc']) < target_width * plot_width_rel_target]
        dfpk['r'] = dfpk['x'] - pk_properties['pk_xc']
        dfpk['fid'] = pid

        # 6. store data
        dfpks.append(dfpk)
        peak_details.update({plbl: {'df': dfpk, 'peak_properties': pk_properties}})

    if len(dfpks) == 0:
        plt.plot(df.x, df.z)
        plt.show()

    dfpk = pd.concat(dfpks)

    return dfpk, peak_details


def downsample_dataframe(df, xcol, ycol, num_points, sampling_rate):
    """
    df, input_sampling_rate, output_sampling_rate = process.downsample_array(df, xcol, ycol, num_points, sampling_rate)

    :param df:
    :param xcol:
    :param ycol:
    :param num_points:
    :param sampling_rate:
    :return:
    """

    # down-sample to reduce computation and file size
    x, y = downsample_array(x=df[xcol].to_numpy(),
                            y=df[ycol].to_numpy(),
                            num_points=num_points,
                            sampling_rate=sampling_rate,
                            )

    data_ds = np.vstack([x, y]).T
    df = pd.DataFrame(data_ds, columns=[xcol, ycol])

    return df


def tilt_correct_array(x, y, num_points):
    """
    y_corr, fit_func, popt, rmse, r_squared = process.tilt_correct_array(x, y, num_points)
    """
    # tilt_y = np.mean(y[-num_points:]) - np.mean(y[:num_points])
    # tilt_x = np.mean(x[-num_points:]) - np.mean(x[:num_points])
    # tilt_deg = np.rad2deg(np.arcsin(tilt_y / tilt_x))

    x_edges = np.hstack([x[:num_points], x[-num_points:]])
    y_edges = np.hstack([y[:num_points], y[-num_points:]])

    popt, fit_func = fit.fit_curve('line', x_edges, y_edges)
    y_corr = y - fit_func(x, *popt)

    rmse, r_squared = fit.calculate_fit_error(fit_results=fit_func(x_edges, *popt), data_fit_to=y_edges)

    return y_corr, fit_func, popt, rmse, r_squared


def downsample_array(x, y, num_points, sampling_rate):
    """
    x, y = process.downsample_array(x, y, num_points, sampling_rate)

    :param x:
    :param y:
    :param sampling_rate: microns per sample (e.g., 5 microns per sample)
    :return:
    """

    if sampling_rate is not None:
        x_span = x.max() - x.min()
        num_points = x_span / sampling_rate
    elif num_points is None:
        raise ValueError("Must provide either 'num_points' or 'sampling_rate'.")

    num_points = int(num_points)

    sp1 = splrep(x, y)
    x2 = np.linspace(x.min(), x.max(), num_points)
    y2 = splev(x2, sp1)

    return x2, y2