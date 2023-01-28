
from os.path import join, isdir, exists
from os import listdir

from pandas import read_excel

import pandas as pd
import numpy as np
from scipy.special import erf


class GraycartFeature(object):

    def __init__(self, did, dlbl, path_design, dxc=0, dyc=0, path_target=None):
        super(GraycartFeature, self).__init__()

        self.did = did
        self.dlbl = dlbl

        self.path_design = path_design
        self.dxc = dxc
        self.dyc = dyc

        self._dfd = None
        self.dr = None
        self.read_design_file()

        self.path_target = path_target
        self.dft = None
        self.read_target_file()


    def __repr__(self):
        class_ = 'GraycartFeature'
        repr_dict = {'Design ID': self.did,
                     'Design Label': self.dlbl,
                     'Design(xc, yc, r)': '({}, {}, {})'.format(self.dxc, self.dyc, self.dr)
                     }
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def read_design_file(self):
        df = read_excel(self.path_design)
        df = df.sort_values('r')
        self._dfd = df

        if 'ro' in df.columns:
            self.dr = df.ro.max()
        else:
            self.dr = df.r.max()

    def read_target_file(self):

        if exists(self.path_target):
            self.dft = pd.read_excel(self.path_target)

        else:
            print("No target profile found at {}. Using standard erf(r) function instead.".format(self.path_target))

            x = np.linspace(-2, 2, 256)
            px = (x + 2) / 2
            py = erf(x) / 2 - 0.5
            dft = pd.DataFrame(np.vstack([px, py]).T, columns=['r', 'z'])
            self.dft = dft

    def resize_target_profile(self, radius=1, amplitude=1):
        self.dft['r'] = self.dft['r'] * radius / 2
        self.dft['z'] = self.dft['z'] * amplitude

    @property
    def dfd(self):
        return self._dfd

    @property
    def mdft(self):
        dft_mirrored = self.dft.copy()
        dft_mirrored['r'] = dft_mirrored['r'] * -1
        mdft = pd.concat([dft_mirrored, self.dft])
        mdft = mdft.sort_values('r')
        return mdft


class WaferFeature(GraycartFeature):

    def __init__(self, graycart_feature, fid, label, fxc, fyc, dose, focus, feature_extents=None, feature_spacing=None,
                 target_radius=None, target_depth=None, target_profile=None):

        super().__init__(graycart_feature.did, graycart_feature.dlbl, graycart_feature.path_design,
                         dxc=graycart_feature.dxc, dyc=graycart_feature.dyc, path_target=graycart_feature.path_target)

        self.fid = fid  # Feature ID: unique identifier across the wafer
        self.label = label  # Feature Label: unique identifying label for each 'feature' across the wafer
        self.fxc = fxc
        self.fyc = fyc
        self.dose = dose
        self.focus = focus

        self.xc = self.fxc + self.dxc
        self.yc = self.fyc + self.dyc

        if feature_extents is None:
            feature_extents = self.dr * 1.15
        self.feature_extents = feature_extents
        self.feature_spacing = feature_spacing

        if target_radius is None:
            target_radius = self.dr
        self.target_radius = target_radius
        self.target_depth = target_depth
        self.target_profile = target_profile

    def __repr__(self):
        class_ = 'GraycartFeature'
        repr_dict = {'Label': self.label,
                     'Feature ID': self.fid,
                     'Design ID': self.did,
                     'Feature(xc, yc, r)': '({}, {}, {})'.format(self.xc, self.yc, self.dr),
                     'Dose, Focus': '{}, {}'.format(self.dose, self.focus),

                     }
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str


class ProcessFeature(WaferFeature):

    def __init__(self, graycart_wafer_feature, step, process_type, subpath, dfpk, peak_properties):

        # super().__init__(did, dlbl, path_design, fid, label, fxc, fyc, dose, focus)
        # graycart_feature, fid, label, fxc, fyc, dose, focus
        super().__init__(graycart_wafer_feature,
                         graycart_wafer_feature.fid, graycart_wafer_feature.label, graycart_wafer_feature.fxc,
                         graycart_wafer_feature.fyc, graycart_wafer_feature.dose, graycart_wafer_feature.focus)

        self.step = step
        self.process_type = process_type
        self.subpath = subpath

        self.dfpk = dfpk
        self.peak_properties = peak_properties

    def __repr__(self):
        class_ = 'ProcessFeature'
        repr_dict = {'Label': self.label,
                     'Feature ID': self.fid,
                     'Design ID': self.did,
                     'Step': self.step,
                     'Process Type': self.process_type,
                     }
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            out_str += '{}: {} \n'.format(key, str(val))
        return out_str


def initialize_designs(wid, base_path, design_ids, design_lbls, design_locs):
    """

    :param wid: wafer ID
    :param base_path: top-level directory for wafer
    :param design_ids: numeric identifier for each design (e.g., 0, 1, 2,...)
    :param design_lbls: string identifier, must match data files for each feature
    :param design_locs: the location of each feature in the mask design file
    :return:
    """

    designs = {}
    for k, design_lbl, design_loc in zip(design_ids, design_lbls, design_locs):
        # for design_loc in design_locs:
        path_design = join(base_path, 'mask', 'w{}_{}.xlsx'.format(wid, design_lbl))
        path_target = join(base_path, 'mask', 'target-profile_{}.xlsx'.format(design_lbl))

        designs.update({k: GraycartFeature(did=k,
                                           dlbl=design_lbl,
                                           path_design=path_design,
                                           dxc=design_loc[0],
                                           dyc=design_loc[1],
                                           path_target=path_target,
                                           )
                        }
                       )

    return designs


def initialize_design_features(designs, design_spacing, dose_lbls, focus_lbls, dose, dose_step, focus, focus_step, fem_dxdy):
    """

    :param designs:
    :param design_spacing:
    :param dose_lbls:
    :param focus_lbls:
    :param dose:
    :param dose_step:
    :param focus:
    :param focus_step:
    :param fem_dxdy:
    :return:
    """

    features = {}
    ij = 0
    for i, dose_lbl in enumerate(dose_lbls):
        for j, focus_lbl in enumerate(focus_lbls):
            for k, gcf in designs.items():

                if len(designs) > 1:
                    feature_label = "{}{}_{}".format(dose_lbl, focus_lbl, gcf.dlbl)
                else:
                    feature_label = "{}{}".format(dose_lbl, focus_lbl)

                feature_xc = fem_dxdy[0] * j
                feature_yc = fem_dxdy[1] * i
                feature_dose = dose + i * dose_step
                feature_focus = focus + j * focus_step

                features.update({feature_label: WaferFeature(graycart_feature=gcf,
                                                             fid=ij,
                                                             label=feature_label,
                                                             fxc=feature_xc,
                                                             fyc=feature_yc,
                                                             dose=feature_dose,
                                                             focus=feature_focus,
                                                             feature_spacing=design_spacing,
                                                             )
                                 }
                                )
                ij += 1

    return features