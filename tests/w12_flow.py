from graycart.GraycartWafer import evaluate_wafer_flow
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# INPUTS

# WAFER
wid = 12
base_path = '/Users/mackenzie/Desktop/Zipper/Fabrication/Wafer{}'.format(wid)
fn_pflow = 'process-flow_w{}.xlsx'.format(wid)
path_results = 'results'
profilometry_tool = 'KLATencor-P7'

# DESIGN
design_ids = [0]
design_lbls = ['erf5_LrO' for i in np.arange(len(design_ids))]
design_spacing = 5e3
design_locs = [[0, n * design_spacing] for n in np.arange(-2, 3)]

# field exposure matrix
dose_lbls = ['a', 'b', 'c', 'd', 'e']
focus_lbls = [1, 2, 3]
dose, dose_step = 350, 0
focus, focus_step = -25, 25
fem_dxdy = [15e3, 5e3]

# data processing
perform_rolling_on = [[3, 'c1']]


# ----------------------------------------------------------------------------------------------------------------------

wfr = evaluate_wafer_flow(wid, base_path, fn_pflow, path_results, profilometry_tool,
                          design_ids, design_lbls, design_locs,
                          design_spacing, dose_lbls, focus_lbls, dose, dose_step, focus, focus_step, fem_dxdy,
                          save_all_results=True,
                          perform_rolling_on=perform_rolling_on,
                          )

j = 1


# ---

print("test_flow.py completed without errors.")