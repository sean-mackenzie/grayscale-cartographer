from os.path import join, isdir
from os import makedirs
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from graycart.GraycartFeature import ProcessFeature


sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'
scipurple = '#845B97'
sciblack = '#474747'
scigray = '#9e9e9e'
sci_color_list = [sciblue, scigreen, scired, sciorange, scipurple, sciblack, scigray]

plt.style.use(['science', 'ieee', 'std-colors'])  # , 'std-colors'
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------
# PLOTTING FUNCTIONS - GraycartWafer

def plot_features_by_process(gcw, px, py, did=None, normalize=False, save_fig=False, save_type='.png'):

    # get dataframe
    df = gcw.dfps

    # filter on 'did': Design ID
    if isinstance(did, (int, float)):
        df = df[df['did'] == did]
    elif isinstance(did, (list, np.ndarray)):
        df = df[df['did'].isin(did)]
    else:
        did = 'all'

    if len(df) < 5:
        raise ValueError("Design ID {} not in: {}".format(did, df.did.unique()))

    # get steps and Feature IDs
    df = df.sort_values(by=['step', 'fid'])
    steps = df.step.unique()
    fids = df.fid.unique()

    # plot
    fig, axs = plt.subplots(nrows=len(steps), sharex=True,
                            figsize=(size_x_inches * 1.25, size_y_inches * len(steps) / 1.875),
                            facecolor='white')

    for step, ax in zip(steps, axs):
        for fid in fids:
            # get slice
            dfds = df[(df['fid'] == fid) & (df['step'] == step)].reset_index()
            if len(dfds) < 5:
                continue

            # get params
            design_id = int(dfds.iloc[0].did)
            dose = int(dfds.iloc[0].dose)
            focus = int(dfds.iloc[0].focus)

            # normalize
            if normalize:
                ax.plot(dfds[px], dfds[py] / dfds[py].min(), linewidth=0.5, label="({}, {}, {})".format(design_id, dose, focus))
                ax.set_ylabel(r'$z/H$')
            else:
                ax.plot(dfds[px], dfds[py], linewidth=0.5, label="({}, {}, {})".format(design_id, dose, focus))
                ax.set_ylabel(r'$z \: (\mu m)$')

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1), title=r'$(d_{ID}, I_{o}, f)$')
            ax.set_title(gcw.processes[step].descriptor)

    axs[-1].set_xlabel(r'$r \: (\mu m)$')

    plt.tight_layout()
    if save_fig:
        plt.savefig(join(gcw.path_results, 'figs',
                         'features_by_process_px-{}_py-{}_did-{}_norm-{}'.format(px, py, did, normalize) + save_type))
    else:
        plt.show()
    plt.close()


def plot_processes_by_feature(gcw, px, py, did=None, normalize=False, save_fig=False, save_type='.png'):

    # get dataframe
    df = gcw.dfps

    # filter on 'did': Design ID
    if isinstance(did, (int, float)):
        df = df[df['did'] == did]
    elif isinstance(did, (list, np.ndarray)):
        df = df[df['did'].isin(did)]
    else:
        did = 'all'

    if len(df) < 5:
        raise ValueError("Design ID {} not in: {}".format(did, df.did.unique()))

    # get Steps and Feature IDs
    df = df.sort_values(by=['step', 'fid'])
    steps = df.step.unique()
    fids = df.fid.unique()

    # plot
    fig, axs = plt.subplots(nrows=len(fids), sharex=True,
                            figsize=(size_x_inches * 1.125, size_y_inches * len(fids) / 1.5),
                            facecolor='white')

    for fid, ax in zip(fids, axs):
        for step in steps:
            # get slice
            dfds = df[(df['fid'] == fid) & (df['step'] == step)].reset_index()
            if len(dfds) < 5:
                continue

            # get params
            design_id = int(dfds.iloc[0].did)
            dose = int(dfds.iloc[0].dose)
            focus = int(dfds.iloc[0].focus)

            # normalize
            if normalize:
                ax.plot(dfds[px], dfds[py] / dfds[py].min(), linewidth=0.5, label="{}".format(step))
                ax.set_ylabel(r'$z/H$')
            else:
                ax.plot(dfds[px], dfds[py], linewidth=0.5, label="{}".format(step))
                ax.set_ylabel(r'$z \: (\mu m)$')

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1), title='Step')
            ax.set_title(r'$(f_{ID}, d_{ID}, I_{o}, f)=$' + '({}, {}, {}, {})'.format(fid, design_id, dose, focus), fontsize=6)

    axs[-1].set_xlabel(r'$r \: (\mu m)$')

    plt.tight_layout()
    if save_fig:
        plt.savefig(join(gcw.path_results, 'figs',
                         'processes_by_feature_px-{}_py-{}_did-{}_norm-{}'.format(px, py, did, normalize) + save_type))
    else:
        plt.show()
    plt.close()


def compare_target_to_features_by_process(gcw, px, py, did=None, normalize=False, save_fig=False, save_type='.png'):

    # get dataframe
    df = gcw.dfps

    # filter on 'did': Design ID
    if isinstance(did, (int, float)):
        df = df[df['did'] == did]
    elif isinstance(did, (list, np.ndarray)):
        df = df[df['did'].isin(did)]
    else:
        did = 'all'

    if len(df) < 5:
        raise ValueError("Design ID {} not in: {}".format(did, df.did.unique()))

    # get steps and Feature IDs
    df = df.sort_values(by=['step', 'fid'])
    steps = df.step.unique()
    fids = df.fid.unique()

    # plot
    fig, axs = plt.subplots(nrows=len(steps), sharex=True,
                            figsize=(size_x_inches * 1.25, size_y_inches * len(steps) / 1.875),
                            facecolor='white')

    for step, ax in zip(steps, axs):
        z_amplitude = []
        for fid in fids:
            # get slice
            dfds = df[(df['fid'] == fid) & (df['step'] == step)].reset_index()
            if len(dfds) < 5:
                continue

            # get params
            design_id = int(dfds.iloc[0].did)
            dose = int(dfds.iloc[0].dose)
            focus = int(dfds.iloc[0].focus)

            # normalize
            if normalize:
                ax.plot(dfds[px], dfds[py] / dfds[py].min(), linewidth=0.5, label="({}, {}, {})".format(design_id, dose, focus))
                ax.set_ylabel(r'$z/H$')
            else:
                ax.plot(dfds[px], dfds[py], linewidth=0.5, label="({}, {}, {})".format(design_id, dose, focus))
                ax.set_ylabel(r'$z \: (\mu m)$')

            z_amplitude.append(dfds[py].min())

        # --------------------------------------------------------------------------------------------------------
        # UNDER CONSTRUCTION

        # get target profile for this design
        gcwf = gcw.features['a1']

        target_did = gcwf.did
        target_radius = gcwf.dr
        mean_amplitude = np.abs(np.mean(z_amplitude))  # np.abs(np.mean([df[(df['fid'] == fid_) & (df['step'] == step)].z.min() for fid_ in fids]))
        mdft = gcwf.mdft.copy()

        mdft['r'] = mdft['r'] * target_radius / 2
        mdft['z'] = mdft['z'] * mean_amplitude

        if normalize:
            ax.plot(mdft['r'], mdft['z'] / mdft['z'].min(), linewidth=0.5, linestyle='dotted', color='k',
                    label="Target({})".format(target_did))
        else:
            ax.plot(mdft['r'], mdft['z'], linewidth=0.5, linestyle='dotted', color='k',
                    label="Target({})".format(target_did))

        # --------------------------------------------------------------------------------------------------------

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1), title=r'$(d_{ID}, I_{o}, f)$')
        ax.set_title(gcw.processes[step].descriptor)

    axs[-1].set_xlabel(r'$r \: (\mu m)$')

    plt.tight_layout()
    if save_fig:
        plt.savefig(join(gcw.path_results, 'figs',
                         'compare_target-to-features_by_process_px-{}_py-{}_did-{}_norm-{}'.format(px, py, did, normalize) + save_type))
    else:
        plt.show()
    plt.close()


# ------------------------------------------------------------------------------------------------------------------
# PLOTTING FUNCTIONS - GraycartProcess


def plot_process_feature_profilometry(graycart_process, save_fig=False, save_type='.png'):
    """
    Plot all ProcessFeatures on the same figure.

    :param graycart_process:
    :return:
    """

    fig, ax = plt.subplots(
                           figsize=(size_x_inches * 1.5, size_y_inches * 1),
                           facecolor='white',
                           )

    for f_lbl, pf in graycart_process.features.items():
        if not isinstance(pf, ProcessFeature):
            continue

        ax.plot(pf.dfpk.r, pf.dfpk.z, '.', ms=1, label='{}, {}'.format(pf.dose, pf.focus))

    ax.set_xlabel(r'$r \: (\mu m)$')
    ax.set_ylabel(r'$z \: (\mu m)$')

    ax.set_title(graycart_process.descriptor)
    ax.legend(title='Dose, Focus', loc='upper left', bbox_to_anchor=(1, 1), markerscale=3)

    plt.tight_layout()
    if save_fig:
        save_path = join(graycart_process.ppath, 'results')
        if not isdir(save_path):
            makedirs(save_path)
        plt.savefig(join(save_path, graycart_process.subpath + '_multi' + save_type))
    else:
        plt.show()
    plt.close()


def plot_process_feature_fit_profilometry(graycart_process, include_vhlines=True, save_fig=False, save_type='.png'):
    """
    Plot a figure for each ProcessFigure, including:
        * scan profile
        * function fitted to peak

    :param save_type:
    :param save_fig:
    :param include_vhlines:
    :param graycart_process:
    :return:
    """

    num_figures = graycart_process.num_process_features

    fig, ax = plt.subplots(nrows=num_figures, sharex=True,
                           figsize=(size_x_inches, size_y_inches * num_figures * 0.5),
                           facecolor='white',
                           )

    if not isinstance(ax, (list, np.ndarray)):
        ax = [ax]

    if graycart_process.num_process_features == graycart_process.num_features:
        ax_id = [pf.fid for pf in graycart_process.features.values()]
    else:
        ax_id = np.arange(len(ax))

    j = 0
    for f_lbl, pf in graycart_process.features.items():
        if not isinstance(pf, ProcessFeature):
            continue
        i = ax_id[j]

        # data
        ax[i].plot(pf.dfpk.index - pf.peak_properties['pk_idx'], -pf.dfpk['z'],
                        '.', ms=1, color='gray', alpha=0.35,
                        )

        # fit
        ax[i].plot(pf.peak_properties['res_x'] - pf.peak_properties['pk_idx'], pf.peak_properties['res_z'],
                        linewidth=0.5, color='b')

        # vertical and horizontal lines from scipy.find_peaks()
        if include_vhlines:
            ax[i].plot(0, pf.peak_properties['fit_z'][pf.peak_properties['fit_pk_idx']],
                            'o', ms=1.5, color='k')

            ax[i].vlines(x=0,
                              ymin=pf.peak_properties['fit_z'][pf.peak_properties['fit_pk_idx']] - pf.peak_properties[
                                  "prominences"],
                              ymax=pf.peak_properties['fit_z'][pf.peak_properties['fit_pk_idx']],
                              color="r", linewidth=0.5,
                              label='{}, {}'.format(np.round(pf.peak_properties['pk_h'], 1),
                                                    int(np.round(pf.peak_properties['pk_r'], -1))),
                              )

            ax[i].hlines(y=pf.peak_properties["width_heights"],
                              xmin=pf.peak_properties["left_ips"] - pf.peak_properties['pk_idx'],
                              xmax=pf.peak_properties["right_ips"] - pf.peak_properties['pk_idx'],
                              color="r", linewidth=0.5)

        ax[i].set_ylabel(r'$z \: (\mu m)$')
        ax[i].set_title(f_lbl + r'$(x_{c}=$' + str(int(np.round(pf.peak_properties["pk_xc"], -2))) + r'$ \: \mu m)$')
        ax[i].legend(title=r'$H, r$', loc='upper left', bbox_to_anchor=(1, 1))

        j += 1

    ax[-1].set_xlabel(r'$i \: (\#)$')

    plt.tight_layout()
    if save_fig:
        save_path = join(graycart_process.ppath, 'results')
        if not isdir(save_path):
            makedirs(save_path)
        plt.savefig(join(save_path, graycart_process.subpath + '_fit-peaks' + save_type))
    else:
        plt.show()
    plt.close()


def plot_tilt_correct_array(x, y, num_points, fit_func, popt, rmse, r_squared, save_id,
                            save_path=None, save_type='.png'):

    # x-y displacements between measurement range
    tilt_y = np.mean(y[-num_points:]) - np.mean(y[:num_points])
    tilt_x = np.mean(x[-num_points:]) - np.mean(x[:num_points])
    tilt_deg = np.rad2deg(np.arcsin(tilt_y / tilt_x))

    # get measurement range array
    x_edges = np.hstack([x[:num_points], x[-num_points:]])
    y_edges = np.hstack([y[:num_points], y[-num_points:]])

    # fit a function
    y_fit = fit_func(x, *popt)
    y_corr = y - fit_func(x, *popt)

    # plot
    fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches * 0.875))

    ax.plot(x, y, color='b', alpha=0.5, label=r'$\Delta y=$' + '{}'.format(np.round(tilt_y, 2)))
    ax.scatter(x_edges, y_edges, s=3, color='yellow', alpha=0.125, label='Fit(x, y, n={})'.format(num_points))
    ax.plot(x, y_fit, color='k', linewidth=0.5, label=r'$y_{fit}$')
    ax.plot(x, y_corr, color='r', linewidth=0.5, alpha=0.5, label=r'$y_{corr}$')

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(alpha=0.125)

    ax.set_title('{}: rmse={}, r sq={}'.format(save_id, np.round(rmse, 3), np.round(r_squared, 4)))

    plt.tight_layout()
    if save_path:
        save_path = join(save_path, 'results')
        if not isdir(save_path):
            makedirs(save_path)
        plt.savefig(join(save_path, save_id + '_tilt-correction' + save_type))

    if r_squared < 0.9 or rmse > 0.25:
        plt.show()

    plt.close()