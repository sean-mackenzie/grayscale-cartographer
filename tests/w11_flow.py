from graycart import GraycartWafer, GraycartFeature
import numpy as np


# wafer
wid = 8
base_path = '/Users/mackenzie/Desktop/Zipper/Fabrication/Wafer8'
path_results = 'results'

# processes

# 1. Expose on MLA150
details_p1 = {'Dose': 350, 'Dose Step': 50, 'Focus': -25, 'Focus-Step': 0}
process_params_p1 = {'step': 1,
                     'process_type': 'Expose',
                     'recipe': 'erf3_Lr',
                     'time': 1200,
                     'details': details_p1,
                     'path': None,
                     }

# 2. Develop in CD-26A for 60 s, single-puddle
details_p2 = {'Developer': 'CD-26A', 'Method': 'Single Puddle'}
process_params_p2 = {'step': 2,
                     'process_type': 'Develop',
                     'recipe': 'CD-26A_SNGL_60s',
                     'time': 60,
                     'details': details_p2,
                     'path': 'step2_Dev-60s',
                     }

# 3. Thermal reflow @ 140C for 10 min
details_p3 = {'Temperature': '142.5 C', 'Ramp down': '90 C, 30 s', 'Ramp up': None}
process_params_p3 = {'step': 3,
                     'process_type': 'Thermal Reflow',
                     'recipe': '140C_10min',
                     'time': 600,
                     'details': details_p3,
                     'path': 'step3_Reflow-142.5C-10min',
                     }

# 4. O2 etch, 60 s
details_p4 = {'Pressure': '10 mTorr'}
process_params_p4 = {'step': 4,
                     'process_type': 'Etch',
                     'recipe': 'smOOth.V1',
                     'time': 60,
                     'details': details_p4,
                     'path': 'step4_O2etch-60s',
                     }

# 5. SF6 + O2 etch, 10 min
details_p5 = {'SF6-to-O2 ratio': 1}
process_params_p5 = {'step': 5,
                     'process_type': 'Etch',
                     'recipe': 'SF6+O2.V6',
                     'time': 600,
                     'details': details_p5,
                     'path': 'step5_SF6+O2etch-10min',
                     }

# 6. NMP strip, 24 hour
process_params_p6 = {'step': 6,
                     'process_type': 'Strip',
                     'recipe': 'NMP',
                     'time': 6000,
                     'details': None,
                     'path': 'step6_Strip',
                     }

process_flow = [process_params_p1, process_params_p2, process_params_p3, process_params_p4, process_params_p5]

# ---

# measurements
data_profile = {'header': 'Dektak',
                'filetype_read': '.csv', 'x_units_read': 1e-6, 'y_units_read': 1e-10,
                'filetype_write': '.xlsx', 'x_units_write': 1e-6, 'y_units_write': 1e-6,
                }
data_etch_monitor = {'header': 'DSEiii-LaserMon', 'filetype_read': '.csv', 'filetype_write': '.xlsx'}
data_optical = {'header': 'FluorScope', 'filetype_read': '.jpg', 'filetype_write': '.png'}
data_misc = {'header': 'MLA150', 'filetype_read': '.png', 'filetype_write': '.png'}
measurement_methods = {'Profilometry': data_profile,
                       'Etch Monitor': data_etch_monitor,
                       'Optical': data_optical,
                       'Misc': data_misc,
                       }

# ---

# DESIGN

# features
feature_lables = ['a1', 'b1', 'c1', 'd1', 'e1']

dose, dose_step = 350, 25
focus, focus_step = -25, 0

feature_radius = 7.5 * 255  # mask design radius = 1.92e3 (units: microns)
feature_extents = 2.25e3  # radius to extract from scan data (units: microns)
feature_spacing = 5e3  # distance between features

target_radius = feature_radius
target_depth = 50  # (units: microns)
target_profile = None

features = {}
for i, f in enumerate(feature_lables):
    gcf = GraycartFeature(fid=i,
                          label=f,
                          dose=dose + i * dose_step,
                          focus=focus + i * focus_step,
                          feature_radius=feature_radius,
                          feature_extents=feature_extents,
                          feature_spacing=feature_spacing,
                          target_radius=feature_radius,
                          target_depth=target_depth,
                          target_profile=target_profile,
                          )
    features.update({gcf.label: gcf})


# ---

wfr = GraycartWafer(wid=wid,
                    path=base_path,
                    path_results=path_results,
                    features=features,
                    process_flow=process_flow,
                    processes=None,
                    measurement_methods=measurement_methods,
                    )

# ---

for step, gcprocess in wfr.processes.items():
    if gcprocess._ppath is not None:
        peak_rel_height = 0.93 + step / 100
        gcprocess.add_profilometry_to_features(downsample=3.75,
                                               width_rel_radius=0.01,
                                               peak_rel_height=peak_rel_height,
                                               fit_func='parabola',
                                               prominence=1,
                                               plot_width_rel_target=1.1,
                                               )

        # gcprocess.plot_profilometry_feature_fits(save_fig=True)
        # gcprocess.plot_profilometry_features(save_fig=True)

wfr.merge_processes_profilometry(export=False)
# wfr.plot_features_by_process(px='r', py='z', save_fig=True, save_type='.png')
wfr.plot_processes_by_feature(px='r', py='z', save_fig=True, save_type='.png')


j = 1

# ---

print("test_flow.py completed without errors.")