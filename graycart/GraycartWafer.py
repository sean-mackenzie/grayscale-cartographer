from os.path import join, isdir
from os import makedirs
from collections import OrderedDict

import pandas as pd

from .GraycartProcess import GraycartProcess
from .GraycartFeature import ProcessFeature, initialize_designs, initialize_design_features
from .utils import plotting, io


class GraycartWafer(object):

    def __init__(self, wid, path, features, process_flow, processes, measurement_methods, path_results='results'):
        """

        :param folder:
        """

        super(GraycartWafer, self).__init__()

        # mandatory
        self._wid = wid
        self.path = path

        # file paths
        self.path_results = join(self.path, path_results)
        self.make_dirs()

        # add features
        self.features = features

        # add measurement methods
        self.measurement_methods = measurement_methods

        # add processes
        self.process_flow = process_flow
        self.processes = processes
        self.add_process()

        # initialize variables
        self.dfps = None

    # ------------------------------------------------------------------------------------------------------------------
    # DATA INPUT FUNCTIONS

    def add_process(self):

        # iterate through process flow
        processes = OrderedDict()
        for step, prcs in self.process_flow.items():

            # add measurement data to each process
            data_paths = {}
            for meas_meth, data in self.measurement_methods.items():

                # only add if data is present
                if isdir(join(self.path, data['header'], str(prcs['path']))):
                    meas_meth_data_path = join(self.path, data['header'], str(prcs['path']))
                else:
                    meas_meth_data_path = None

                if not 'x_units_read' in data.keys():
                    data.update({'x_units_read': 1, 'x_units_write': 1,
                                 'y_units_read': 1, 'y_units_write': 1})

                data_paths.update({meas_meth: {'tool': data['header'],
                                               'path': meas_meth_data_path,
                                               'filetype_read': data['filetype_read'],
                                               'x_units_read': data['x_units_read'],
                                               'y_units_read': data['y_units_read'],
                                               'filetype_write': data['filetype_write'],
                                               'x_units_write': data['x_units_write'],
                                               'y_units_write': data['y_units_write'],
                                               },
                                   })

            # instantiate process
            process = GraycartProcess(step=prcs['step'],
                                      process_type=prcs['process_type'],
                                      recipe=prcs['recipe'],
                                      time=prcs['time'],
                                      details=prcs['details'],
                                      basepath=self.path,
                                      subpath=prcs['path'],
                                      features=self.features.copy(),
                                      data=data_paths,
                                      )
            processes.update({process.step: process})

        self.processes = processes

    # ------------------------------------------------------------------------------------------------------------------
    # DATA PROCESSING FUNCTIONS

    def evaluate_process_profilometry(self,
                                      plot_fits=True,
                                      perform_rolling_on=False,
                                      downsample=3.75,
                                      width_rel_radius=0.01,
                                      peak_rel_height=None,
                                      fit_func='parabola',
                                      prominence=1,
                                      plot_width_rel_target=1.1,
                                      ):
        """
        For each process:
            *  add available profilometry data to each feature
                * plots tilt correction
            * plot peak_finding algorithm
            * plot all profiles on the same figure for comparison

        :param plot_fits:
        :param perform_rolling_on:
        :param downsample:
        :param width_rel_radius:
        :param peak_rel_height:
        :param fit_func:
        :param prominence:
        :param plot_width_rel_target:
        :return:
        """

        if peak_rel_height is not None:
            raise ValueError("peak_rel_height is hard-coded.")

        for step, gcprocess in self.processes.items():
            if gcprocess.ppath is not None:
                peak_rel_height = min([0.93 + step / 100, 0.97])

                gcprocess.add_profilometry_to_features(plot_fits=plot_fits,
                                                       perform_rolling_on=perform_rolling_on,
                                                       downsample=downsample,
                                                       width_rel_radius=width_rel_radius,
                                                       peak_rel_height=peak_rel_height,
                                                       fit_func=fit_func,
                                                       prominence=prominence,
                                                       plot_width_rel_target=plot_width_rel_target,
                                                       )

                if plot_fits:
                    gcprocess.plot_profilometry_feature_fits(save_fig=plot_fits)
                    gcprocess.plot_profilometry_features(save_fig=plot_fits)

    def merge_processes_profilometry(self, export=False):

        dfs = []
        for step, gcp in self.processes.items():
            for f_lbl, gcf in gcp.features.items():
                if isinstance(gcf, ProcessFeature):
                    df = gcf.dfpk
                    df['did'] = gcf.did
                    df['step'] = step
                    df['dose'] = gcf.dose
                    df['focus'] = gcf.focus
                    dfs.append(df)

        dfs = pd.concat(dfs)
        self.dfps = dfs

        if export:
            dfs.to_excel(join(self.path, 'results',
                              'w{}_merged_process_profiles'.format(self._wid) +
                              self.measurement_methods['Profilometry']['filetype_write']),
                         index=False,
                         )

    # ------------------------------------------------------------------------------------------------------------------
    # PLOTTING FUNCTIONS

    def plot_feature_evolution(self, px='r', py='z', save_fig=True):
        dids = self.dids
        dids.append(None)

        for did in dids:
            for norm in [False, True]:
                self.compare_target_to_features_by_process(px=px, py=py, did=did, normalize=norm, save_fig=save_fig,
                                                           save_type='.png')

                self.plot_features_by_process(px=px, py=py, did=did, normalize=norm, save_fig=save_fig,
                                              save_type='.png')

                self.plot_processes_by_feature(px=px, py=py, did=did, normalize=norm, save_fig=save_fig,
                                               save_type='.png')

    def plot_features_by_process(self, px, py, did=None, normalize=False, save_fig=False, save_type='.png'):
        plotting.plot_features_by_process(self, px, py, did=did, normalize=normalize, save_fig=save_fig,
                                          save_type=save_type)

    def plot_processes_by_feature(self, px, py, did=None, normalize=False, save_fig=False, save_type='.png'):
        plotting.plot_processes_by_feature(self, px, py, did=did, normalize=normalize, save_fig=save_fig,
                                           save_type=save_type)

    def compare_target_to_features_by_process(self, px, py, did=None, normalize=False, save_fig=False,
                                              save_type='.png'):
        plotting.compare_target_to_features_by_process(self, px, py, did=did, normalize=normalize, save_fig=save_fig,
                                                       save_type=save_type)

    # ------------------------------------------------------------------------------------------------------------------
    # UTILITY FUNCTIONS

    def make_dirs(self):

        if not isdir(self.path_results):
            makedirs(self.path_results)

        if not isdir(join(self.path_results, 'figs')):
            makedirs(join(self.path_results, 'figs'))

    # ------------------------------------------------------------------------------------------------------------------
    # PROPERTIES

    @property
    def dids(self):
        if self.dfps is not None:
            dids = list(self.dfps.did.unique())
        else:
            dids = None
        return dids

    @property
    def fids(self):
        if self.dfps is not None:
            fids = self.dfps.fid.unique()
        else:
            fids = None
        return fids


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# WRAPPER FUNCTION


def evaluate_wafer_flow(wid, base_path, fn_pflow, path_results, profilometry_tool,
                        design_ids, design_lbls, design_locs,
                        design_spacing, dose_lbls, focus_lbls, dose, dose_step, focus, focus_step, fem_dxdy,
                        save_all_results,
                        perform_rolling_on=False,
                        ):
    """

    :param wid:
    :param base_path:
    :param fn_pflow:
    :param path_results:
    :param profilometry_tool:
    :param design_ids:
    :param design_lbls:
    :param design_locs:
    :param design_spacing:
    :param dose_lbls:
    :param focus_lbls:
    :param dose:
    :param dose_step:
    :param focus:
    :param focus_step:
    :param fem_dxdy:
    :param save_all_results:
    :param perform_rolling_on:
    :return:
    """
    # ------------------------------------------------------------------------------------------------------------------
    # SET UP THE DATA HIERARCHY

    # 1. initialize 'designs'
    designs = initialize_designs(wid, base_path, design_ids, design_lbls, design_locs)

    # 2. 'designs' on a wafer form 'features'
    features = initialize_design_features(designs,
                                          design_spacing,
                                          dose_lbls,
                                          focus_lbls,
                                          dose,
                                          dose_step,
                                          focus,
                                          focus_step,
                                          fem_dxdy,
                                          )

    # 3. 'features' undergo 'processes'
    process_flow = io.read_process_flow(fp=join(base_path, fn_pflow))

    # 4. 'measurements' record the effect of 'processes' on 'features'
    measurement_methods = io.read_measurement_methods(profilometry=profilometry_tool)

    # 5. the 'wafer' structures all of this data as a historical record of 'cause' and 'effect'
    wfr = GraycartWafer(wid=wid,
                        path=base_path,
                        path_results=path_results,
                        features=features,
                        process_flow=process_flow,
                        processes=None,
                        measurement_methods=measurement_methods,
                        )

    # ------------------------------------------------------------------------------------------------------------------
    # ANALYZE THE PROCESS DATA

    wfr.evaluate_process_profilometry(plot_fits=save_all_results, perform_rolling_on=perform_rolling_on)
    wfr.merge_processes_profilometry(export=save_all_results)

    if save_all_results:
        wfr.plot_feature_evolution(px='r', py='z', save_fig=save_all_results)

    return wfr