from os.path import join
import numpy as np

from graycart.GraycartWafer import evaluate_wafer_flow
from graycart.utils import plotting


# ----------------------------------------------------------------------------------------------------------------------
# INPUTS

"""
Some important notes:
    
    1. On "Design Labels" or 'design_lbls':
        * if a wafer only has a single design, then features will be identified by 'a1', 'c2', etc...
        * if a wafer has multiple designs, the 'design_lbls' get attached to the feature labels, 'a1_LrO' or 'a1_erf5'
    2. On "Target Labels" or 'target_lbls':
        * the string 'target-profile_' is inserted before the design label, or 'design_lbl'. The target label doesn't
        really serve a purpose at this point. The 'design file' (x, y, r, l) and 'target profile' (r, z) should be
        combined. 
    3. On design spacing and design locations:
        * 'design_spacing' is primarily used to filter multiple peaks in a single scan. 
        * 'design_locs' isn't really used at this point. 
        * 'design_spacing' should be removed as an input and calculated using 'design_locs'. 

"""

"""
Etch Rates:

smOOth.V2: 
    SPR220-7:   (no hard bake)      vertical etch rate = 600 nm/min; lateral etch rate = 0 nm/s (calculated using w17)
                (Laser Monitor)     3.5 wavelengths --> 1 wavelength ~= 190 nm 
    Si:                             vertical etch rate = 0 nm/min; 

SF6+O2.V6:
    SPR220-7:   (no hard bake)      vertical etch rate = 260 nm/min                             (calculated using w16)
    Si:         (no hard bake)      vertical etch rate = 1950 nm/min 
    
    wafer 16:
        1.  post-dev PR thickness (b1, silicon-to-PR-surface)   =   6.5 microns
        2.  post-etch PR thickness (b1, silicon-to-PR-surface)  =   2.25 microns    -->     260 nm/min
            post-etch Si depth (b1, trench-to-PR-interface)     =   32 microns      -->     1.95 um/min

SF6:
    SPR220-7:   (no hard bake)      vertical etch rate = 10 nm/s; lateral etch rate = 0 nm/s (calculated using w17)
    Si:                             vertical etch rate = 0 nm/s; 
"""

# SHARED

# target feature
target_radius = 1920  # microns
plot_width_rel_target_radius = 1.2  # plot radius = target_radius * plot_width_rel_target_radius
target_depth_profile = 50

# data processing
evaluate_signal_processing = False  # True
lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
z_standoff_measure = -0.125
z_standoff_design = 1

thickness_PR = 7.5
thickness_PR_budget = 1.5

# WAFER
for wid in [19]:  # np.arange(12, 18):  # 15, 14, 13, 12
    if wid == 19:
        # DESIGN
        design_lbls = ['erf5']
        target_lbls = ['erf5']
        design_ids = [0]
        design_spacing = 5e3
        design_locs = [[0, n * design_spacing] for n in np.arange(-1, 2)]

        # field exposure matrix
        dose_lbls = ['a', 'b', 'c']  # , 'c', 'd', 'e'
        focus_lbls = [1]
        fem_dxdy = [0e3, 0e3]

        # data processing
        perform_rolling_on = False  # [[3, 'b1', 25]]
        features_of_interest = ['a1', 'b1', 'c1']
        target_radius = 2050  # microns

    elif wid == 18:
        # DESIGN
        design_lbls = ['erf5']
        target_lbls = ['erf5']
        design_ids = [0]
        design_spacing = 5e3
        design_locs = [[0, n * design_spacing] for n in np.arange(-1, 2)]

        # field exposure matrix
        dose_lbls = ['a', 'b', 'c']  # , 'c', 'd', 'e'
        focus_lbls = [1]
        fem_dxdy = [0e3, 0e3]

        # data processing
        perform_rolling_on = False  # [[3, 'b1', 25]]
        features_of_interest = ['a1', 'b1', 'c1']
        target_radius = 2050  # microns

    elif wid == 17:
        # DESIGN
        design_lbls = ['erf5_LrO']
        target_lbls = ['erf5_LrO']
        design_ids = [0]
        design_spacing = 5e3
        design_locs = [[0, n * design_spacing] for n in np.arange(-2, 3)]

        # field exposure matrix
        dose_lbls = ['a', 'b', 'c', 'd', 'e']  # , 'c', 'd', 'e'
        focus_lbls = [1]
        fem_dxdy = [0e3, 0e3]

        # data processing
        perform_rolling_on = False  # [[3, 'b1', 25]]  # False
        features_of_interest = ['d1']  # ['b1', 'b2', 'c1', 'c2', 'd1', 'd2']
        lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
        target_radius = 2100  # microns
        plot_width_rel_target_radius = 1.25  # plot radius = target_radius * plot_width_rel_target_radius

    elif wid == 16:
        # DESIGN
        design_lbls = ['erf5_LrO']
        target_lbls = ['erf5_LrO']
        design_ids = [0]
        design_spacing = 5e3
        design_locs = [[0, n * design_spacing] for n in np.arange(-2, 3)]

        # field exposure matrix
        dose_lbls = ['a', 'b', 'c']  # , 'c', 'd', 'e'
        focus_lbls = [1]
        fem_dxdy = [0e3, 0e3]

        # data processing
        perform_rolling_on = False  # [[3, 'b1', 25]]  # False
        features_of_interest = ['a1', 'b1', 'c1']  # ['b1', 'b2', 'c1', 'c2', 'd1', 'd2']

        # misc
        thickness_PR_budget = 1.65

    elif wid == 15:
        # DESIGN
        design_lbls = ['erf5_LrO']
        target_lbls = ['erf5_LrO']
        design_ids = [0]
        design_spacing = 5e3
        design_locs = [[0, n * design_spacing] for n in np.arange(-2, 3)]

        # field exposure matrix
        dose_lbls = ['c']  # , 'c', 'd', 'e'
        focus_lbls = [1, 2]
        fem_dxdy = [25e3, 0e3]

        # data processing
        perform_rolling_on = False  # [[3, 'b1', 25]]  # False
        features_of_interest = ['c1', 'c2']  # ['b1', 'b2', 'c1', 'c2', 'd1', 'd2']

    elif wid == 14:
        # DESIGN
        design_lbls = ['erf5_LrO']
        target_lbls = ['erf5_LrO']
        design_ids = [0]
        design_spacing = 5e3
        design_locs = [[0, n * design_spacing] for n in np.arange(-2, 3)]

        # field exposure matrix
        dose_lbls = ['a', 'b']  # , 'c', 'd', 'e'
        focus_lbls = [1]
        fem_dxdy = [0e3, 35e3]

        # data processing
        perform_rolling_on = [[3, 'b1', 25]]  # False
        features_of_interest = ['a1', 'b1']  # ['b1', 'b2', 'c1', 'c2', 'd1', 'd2']

    elif wid == 13:
        # DESIGN
        design_lbls = ['erf5_LrO']
        target_lbls = ['erf5_LrO']
        design_ids = [0]
        design_spacing = 5e3
        design_locs = [[0, n * design_spacing] for n in np.arange(-2, 3)]

        # field exposure matrix
        dose_lbls = ['a', 'b', 'c', 'd', 'e']  # , 'c', 'd', 'e'
        focus_lbls = [1, 2]
        fem_dxdy = [5e3, 5e3]

        # data processing
        perform_rolling_on = False  # [[3, 'b1', 25]]  # False
        features_of_interest = ['c1', 'c2']

    elif wid == 12:
        # DESIGN
        design_lbls = ['erf5_LrO']
        target_lbls = ['erf5_LrO']
        design_ids = [0]
        design_spacing = 5e3
        design_locs = [[0, n * design_spacing] for n in np.arange(-2, 3)]

        # field exposure matrix
        dose_lbls = ['a', 'b', 'c', 'd', 'e']
        focus_lbls = [1, 2, 3]
        dose, dose_step = 350, 0
        focus, focus_step = -25, 25
        fem_dxdy = [15e3, 5e3]

        # data processing
        perform_rolling_on = [[3, 'c1', 25], [3, 'c2', 15], [3, 'c3', 15]]
        features_of_interest = ['c{}'.format(i + 1) for i in range(3)]

    elif wid == 11:
        # DESIGN
        design_lbls = ['erf3', 'Lr']
        target_lbls = [None, None]
        design_ids = [0, 1]
        design_spacing = 5e3
        design_locs = [[0, n * design_spacing] for n in [-0.5, 0.5]]

        # field exposure matrix
        dose_lbls = ['a', 'b']  # 'a', 'b', 'c', 'd', 'e'
        focus_lbls = [1]
        fem_dxdy = [10e3, 10e3]

        # data processing
        perform_rolling_on = False  # [[3, 'b1', 25]]  # False
        features_of_interest = ['a1_erf3', 'b1_erf3', 'a1_Lr', 'b1_Lr']  # ['b1', 'b2', 'c1', 'c2', 'd1', 'd2']

    elif wid == 10:
        # design
        design_lbls = ['erf3', 'Lr']
        target_lbls = [None, None]
        design_ids = [0, 1]
        design_spacing = 5e3
        design_locs = [[0, n * design_spacing] for n in [-0.5, 0.5]]

        # field exposure matrix
        dose_lbls = ['a', 'b']  # 'a', 'b', 'c', 'd', 'e'
        focus_lbls = [1]
        fem_dxdy = [10e3, 10e3]

        # data processing
        perform_rolling_on = [[3, 'b1_erf3', 50]]  # False
        features_of_interest = ['a1_erf3', 'b1_erf3', 'a1_Lr', 'b1_Lr']  # ['b1', 'b2', 'c1', 'c2', 'd1', 'd2']

    else:
        continue
        # raise ValueError()

    # SHARED

    base_path = '/Users/mackenzie/Desktop/Zipper/Fabrication/Wafer{}'.format(wid)
    fn_pflow = 'process-flow_w{}.xlsx'.format(wid)
    path_results = 'results'
    profilometry_tool = 'KLATencor-P7'

    # results
    save_type = '.png'
    step_develop = 3
    save_all_results = False

    # ------------------------------------------------------------------------------------------------------------------

    wfr = evaluate_wafer_flow(wid, base_path, fn_pflow, path_results, profilometry_tool,
                              design_lbls, target_lbls, design_locs, design_ids,
                              design_spacing, dose_lbls, focus_lbls, fem_dxdy,
                              target_radius=target_radius,
                              plot_width_rel_target_radius=plot_width_rel_target_radius,
                              peak_rel_height=lambda_peak_rel_height,
                              save_all_results=save_all_results,
                              perform_rolling_on=perform_rolling_on,
                              evaluate_signal_processing=evaluate_signal_processing,
                              )

    wfr.backout_process_to_achieve_target(target_radius=target_radius,
                                          target_depth=target_depth_profile,
                                          thickness_PR=thickness_PR,
                                          thickness_PR_budget=thickness_PR_budget,
                                          save_fig=True)

    wfr.compare_target_to_feature_evolution(px='r', py='z', save_fig=True)

    wfr.characterize_exposure_dose_depth_relationship(z_standoff=z_standoff_measure,
                                                      plot_figs=save_all_results,
                                                      save_type=save_type,
                                                      )
    
    wfr.merge_exposure_doses_to_process_depths(export=save_all_results)
    wfr.plot_all_exposure_dose_to_depth(step=step_develop)
    wfr.compare_exposure_functions()
    wfr.correct_grayscale_design_profile(z_standoff=z_standoff_design,
                                         plot_figs=save_all_results,
                                         save_type=save_type,
                                         )

    # ---
    # ---

    # plots
    if save_all_results:
        wfr.plot_feature_evolution(px='r', py='z', save_fig=save_all_results)

        # compare target to features
        wfr.compare_target_to_feature_evolution(px='r', py='z', save_fig=save_all_results)

        # grade target accuracy
        # wfr.grade_profile_accuracy(step=max(wfr.list_steps), target_radius=target_radius, target_depth=target_depth_profile)

        # plot exposure profile
        for foi in features_of_interest:
            gpf = wfr.features[foi]
            plotting.plot_exposure_profile(gcf=gpf, path_save=join(wfr.path_results, 'figs'), save_type=save_type)

        # plot feature profile overlaid with exposure profile
        step = max(wfr.list_steps)
        for did in wfr.dids:
            plotting.plot_overlay_feature_and_exposure_profiles(gcw=wfr, step=step, did=did,
                                                                path_save=join(wfr.path_results, 'figs'),
                                                                save_type=save_type,
                                                                )

    # ---

    print("test_flow.py completed without errors.")