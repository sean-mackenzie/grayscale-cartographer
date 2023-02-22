from os.path import join
import tkinter as tk  # tkinter for GUI

import numpy as np

from graycart.GraycartWafer import GraycartWafer
from graycart.GraycartFeature import initialize_designs, initialize_design_features
from graycart.utils import io, plotting
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

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

# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------
# SET UP THE DATA HIERARCHY

"""
1. Read the 'Process Flow'

Inputs:
    wid: Wafer ID
    base_path: the directory where all of our files are held.
    fn_pflow: the filename of our process flow.
    path_results: name of sub-directory within base_path to store all of the results.
    save_type: file type of output figures (e.g., .png, .svg, etc)

Functions:
    io.read_process_flow: read process_flow.xlsx and parse information.
"""
# inputs
# wid = 11
# base_path = '/Users\simon\Documents\Simels_daten\Epfl\sem_13_2022_Master_theis_USA\grayscale-cartographer\example\Wafer{}'.format(wid)
# fn_pflow = 'process-flow_w{}.xlsx'.format(wid)
# path_results = 'results'
# save_type = '.png'
process_flow = False
designs = False
features = False
#
# functions
def read_processFlow():
    global wid
    wid_err = ''
    global process_flow
    global base_path
    base_path_err = ''
    global path_results
    global save_type
    global fn_pflow
    fn_pflow_err = ''
    try:
        wid = int(wid_Txt.get())
        base_path = base_path_Txt.get() + '{}'.format(wid)
        fn_pflow = fn_pflow_Txt.get() + '{}.xlsx'.format(wid)
    except:
        wid_err = 'Wafer ID is not int'
    path_results = path_results_Txt.get()
    save_type = '.'+ save_type_Txt.get()
    try:
        process_flow = io.read_process_flow(fp=join(base_path, fn_pflow))
        Output.delete("1.0", "end")
        Output.insert("1.0", "Proces flow succesfully initialized" + '\n'
                      +'Wafer ID: ' + str(wid) + '\n'
                      +'Data path: ' + str(base_path) + '\n'
                      +'Process flow name: ' + str(fn_pflow) + '\n'
                      +'Results path: ' + str(path_results) + '\n'
                      +'Save type: ' + str(save_type) + '\n\n'
                      )
    except:
        base_path_err = 'Ceck your File directroy name'
        fn_pflow_err = 'Or your File name of the process flow'
        Output.delete("1.0", "end")
        Output.insert("1.0", "Step 1 errors:" + '\n'
                      +"your scrwed up" + '\n'
                      +'1) '+ base_path_err + '\n'
                      +'2) '+ fn_pflow_err + '\n'
                      +'3) '+ wid_err +'\n'
                      )
        process_flow = False
    return process_flow


# -

"""
2. Initialize 'Designs' and 'Features'

Inputs:
    design_lbls: A string that identifies which design this data (profilometry, image, etc) belongs to.
    design_ids: A unique number assigned to each design_lbl (in order to perform numerical operations)
    dose_lbls: A string that identifies which exposure setting (dose) this data belongs to.
    focus_lbls: A number that identifies which exposure setting (focus) this data belongs to.
    target_radius: (units: microns) defines the radial distance of our target profile.
    target_depth_profile: (units: microns) defines the axial distance (height) of our target profile.

    The following parameters aren't important (right now, at least):
    target_lbls:
    design_spacing:
    design_locs:
    fem_dxdy:

Functions:
    initialize_designs: creates a GraycartFeature instance for each feature (e.g., design 'erf3')
    initialize_design_features: initializes 'features' (GraycartFeature.WaferFeature), which are 'designs' patterned
    onto a wafer (as defined by the 'expose' step in the process flow).

"""
# inputs

# designs
design_lbls = ['erf3', 'Lr']
design_ids = [0, 1]

# field exposure matrix
dose_lbls = ['a', 'b']  # 'a', 'b', 'c', 'd', 'e'
focus_lbls = [1]

# target feature
target_radius = 1920
target_depth_profile = 50

# not important
target_lbls = [None, None]
design_spacing = 5e3
design_locs = [[0, n * design_spacing] for n in [-0.5, 0.5]]
fem_dxdy = [10e3, 10e3]

# functions

# initialize 'designs'
def init_featuresDesigns():
    global designs
    global features

    designs_err = ''
    global design_spacing
    design_spacing_err = ''
    global design_lbls
    global target_lbls
    global design_locs
    design_locs_err = ''
    global design_ids
    design_ids_err=''
    global target_depth_profile
    global dose_lbls
    global focus_lbls
    focus_lbls_err = ''
    global fem_dxdy
    fem_dxdy_err = ''
    global target_radius
    target_radius_err = ''

    design_spacing = design_spacing_Txt.get().replace(' ', '')
    if design_spacing.replace('.', '', 1).replace('e', '', 1).isdigit():
        design_spacing = float(design_spacing)
    else:
        design_spacing_err = 'Your desing spaicing is not a digital'
    design_lbls = design_lbls_Txt.get().replace(' ', '').split(',')
    target_lbls = [None, None]#target_lbls_Txt.get().replace(' ', '').split(',')
    design_locs = design_locs_Txt.get().replace(' ', '').split(',')
    for i in range(len(design_locs)):
        try:
            if design_locs[i].replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
                design_locs[i] = float(design_locs[i])
            else:
                design_locs_err =design_locs[i]+ ' is not convertable to float \n'
        except:
            design_locs_err = design_locs_err+ "fem_dxdy is wrongly initatilzed \n"
    try:
        # print(design_locs)
        design_locs = [[0, n * design_spacing] for n in design_locs]
        # print(design_locs)
    except:
        design_locs_err = design_locs_err + 'Desing location vector is worong'
    design_ids = design_ids_Txt.get().replace(' ', '').split(',')
    for i in range(len(design_ids)):
        try:
            design_ids[i] = int(design_ids[i])
        except:
            design_ids_err = "Design indices are wrongly initatilzed"
    target_depth_profile = target_depth_profile_Txt.get()

    dose_lbls = dose_lbls_Txt.get().replace(' ', '').split(',')
    focus_lbls = focus_lbls_Txt.get().replace(' ', '').split(',')
    for i in range(len(focus_lbls)):
        try:
            focus_lbls[i] = int(focus_lbls[i])
        except:
            focus_lbls_err = "Focus label is wrongly initatilzed"

    fem_dxdy = fem_dxdy_Txt.get().replace(' ', '').split(',')
    for i in range(len(fem_dxdy)):
        try:
            if fem_dxdy[i].replace('.', '', 1).replace('e', '', 1).isdigit():
                fem_dxdy[i] = float(fem_dxdy[i])
            else:
                fem_dxdy_err = fem_dxdy[i] + ' is not convertible to float \n'
            # print(fem_dxdy)
        except:
            fem_dxdy_err = fem_dxdy_err + "fem_dxdy is wrongly initatilzed"
    target_radius = target_radius_Txt.get()
    if target_radius.replace('.', '', 1).replace('e', '', 1).isdigit():
        target_radius = float(target_radius)
    else:
        target_radius_err = 'Target radius is wrong'

    if process_flow !=False:
        try:
            designs = initialize_designs(base_path, design_lbls, target_lbls, design_locs, design_ids)
        except:
            designs_err = 'There is a problem with either the: design_lbls, target_lbls, design_locs, design_ids'
        try:
            features = initialize_design_features(designs,
                                                  design_spacing,
                                                  dose_lbls,
                                                  focus_lbls,
                                                  process_flow,
                                                  fem_dxdy,
                                                  target_radius=target_radius,
                                                  )

            Output.delete("1.0", "end")
            Output.insert("1.0", 'Step 2 Features:' + '\n'
                          + 'Designs and Features succesfully initialized to:' + '\n'
                          + 'Target radius: ' + target_radius_Txt.get() + '\n'
                          + 'Target labels: ' + target_lbls_Txt.get() + '\n'
                          + 'Target depth: ' + target_depth_profile_Txt.get() + '\n'
                          + 'Dose labels: '+dose_lbls_Txt.get()+'\n'
                          + 'Focus labels: ' + focus_lbls_Txt.get() + '\n'
                          + 'Design location: ' + design_locs_Txt.get() + '\n'
                          + 'Desing spacing: '+design_spacing_Txt.get()+'\n'
                          + 'Design indices '+design_ids_Txt.get()+'\n'
                          + 'Design labels '+ design_lbls_Txt.get()+'\n'
                          + 'Fem_dxdy is: '+fem_dxdy_Txt.get()+'\n'
                          )
            plot1.clear()
            plot1.grid()
            canvas.draw()
            root.update()
            for i in design_locs:
                Features = cicleXY(i,target_radius)
                plotFeatures(Features.xcir,Features.ycir)
        except:
            features = False
            Output.delete("1.0", "end")
            Output.insert("1.0", "Step 2 Feature errors:" + '\n'
                          + "You scrwed up" + '\n'
                          + '1)- '+designs_err + '\n'
                          + '2)- '+design_ids_err + '\n'
                          + '3)- '+design_spacing_err + '\n'
                          + '4)- '+design_locs_err + '\n'
                          + '5)- '+focus_lbls_err + '\n'
                          + '6)- '+target_radius_err + '\n'
                          + '7)- '+fem_dxdy_err + '\n\n')
    else:
        Output.delete("1.0", "end")
        Output.insert("1.0", "Step 2 Desing errors:" + '\n'
                      +"Do step 1 first" + '\n\n')

class cicleXY:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.xcir = [(self.radius * np.cos(x)+self.center[0])/1000 for x in np.linspace(0, 2 * np.pi, 500)]
        self.ycir = [(self.radius * np.sin(x)+self.center[1])/1000 for x in np.linspace(0, 2 * np.pi, 500)]


def plotFeatures(x,y):
    plot1.legend('',frameon=False)
    plot1.plot(x, y, 'r')
    wafer_plot=cicleXY([0,0],50.8*1000)
    # plot1.set_title('Serial Data')
    plot1.plot(wafer_plot.xcir, wafer_plot.ycir,'b')
    plot1.grid()
    canvas.draw()
    root.update()
"""
3. Initialize the 'Wafer'

Now that we have parsed our 'process flow' and initialized our 'designs' as 'features' on a wafer, we create an instance
of GraycartWafer to structure all of this information.

Inputs:
    profilometry_tool: A string which identifies the directory to search for profilometry data. *Note: the other
    measurement methods are hard-coded in because I always use the same tools but they should be brought out to the GUI.
    See the io.read_measurement_methods() function for more details.

    path_results: the directory to save all of the results.
    processes: GraycartWafer has a built-in method which parses the process_flow into 'processes' so this isn't necessary.

Functions:
    io.read_measurement_methods: assigns file paths, file types, and units of measurement for each data source.
    GraycartWafer: create an instance of GraycartWafer

"""

# # inputs
# profilometry_tool = 'KLATencor-P7'
#
# # 4. 'measurements' record the effect of 'processes' on 'features'
# measurement_methods = io.read_measurement_methods(profilometry=profilometry_tool)
#
# # 5. the 'wafer' structures all of this data as a historical record of 'cause' and 'effect'
# wfr = GraycartWafer(wid=wid,
#                     path=base_path,
#                     path_results=path_results,
#                     designs=designs,
#                     features=features,
#                     process_flow=process_flow,
#                     processes=None,
#                     measurement_methods=measurement_methods,
#                     )
#
# """
# 4. Read, process, and evaluate the 'measurement_methods' data.
#
#     # the following inputs modify (turn on or off) functions to process profilometry data.
#     evaluate_signal_processing: (True or False) Optionally, plot various signal processing methods to smooth profilometry data.
#     perform_rolling_on: (True or False) Optionally, perform a rolling operation to smooth data; [[3, 'b1', 25]]
#
#     # the following parameters are passed into smoothing and peak_finding algorithms:
#     lambda_peak_rel_height: function to modify the scipy.find_peaks function
#     peak_rel_height: if 'lambda_peak_rel_height' is defined, then this variable isn't used.
#     downsample: downsample the data to reduce computation.
#     width_rel_radius: radial distance, beyond the target radius, to evaluate.
#     prominence: the relative height of a peak compared to adjacent peaks.
#     fit_func: function that is fit to the profilometry profile to find its peak (center)
#
#     # the following inputs modify plotting functions:
#     plot_width_rel_target_radius: plot radius = target_radius * plot_width_rel_target_radius
#     save_profilometry_processing_figures = True
#     save_merged_profilometry_data = True
#
# Functions:
#     evaluate_process_profilometry: iterate through each process and each feature and evaluate profilometry data.
#     merge_processes_profilometry: merge all of the profilometry data into a single database.
#     merge_exposure_doses_to_process_depths: merge exposure data (dose, focus) with profilometry data (r, z)
# """
#
# # inputs
# evaluate_signal_processing = False
# lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
# perform_rolling_on = False  # [[3, 'b1', 25]]
# plot_width_rel_target_radius = 1.2
# peak_rel_height = 0.975
# downsample = 5
# width_rel_radius = 0.01
# fit_func = 'parabola'
# prominence = 1
# save_profilometry_processing_figures = True
# save_merged_profilometry_data = True
#
# # functions
# wfr.evaluate_process_profilometry(plot_fits=save_profilometry_processing_figures,
#                                   perform_rolling_on=perform_rolling_on,
#                                   evaluate_signal_processing=evaluate_signal_processing,
#                                   plot_width_rel_target=plot_width_rel_target_radius,
#                                   peak_rel_height=peak_rel_height,
#                                   downsample=downsample,
#                                   width_rel_radius=width_rel_radius,
#                                   fit_func=fit_func,
#                                   prominence=prominence,
#                                   )
#
# wfr.merge_processes_profilometry(export=save_merged_profilometry_data)
#
#
# # ----------------------------------------------------------------------------------------------------------------------
# # END PRIMARY DATA PROCESSING FUNCTIONS
# # ----------------------------------------------------------------------------------------------------------------------
#
# """
# At this point, all of the data has been parsed and structured. The following functions can be optionally called to
# interpret the data.
# """
#
# # ---
#
# """
# 'plot_exposure_profile': plot figure showing your design profile.
# """
# features_of_interest = ['a1_erf3', 'b1_erf3', 'a1_Lr', 'b1_Lr']
# for foi in features_of_interest:
#     gpf = wfr.features[foi]
#     plotting.plot_exposure_profile(gcf=gpf, path_save=join(wfr.path_results, 'figs'), save_type=save_type)
#
# # ---
#
# """
# 'backout_process_to_achieve_target' starts with your 'target_profile' and reverse engineers what processes you should
# run and how you should pattern your photoresist.
# """
# thickness_PR = 7.5  # photoresist thickness, this variable could be interpreted from process_flow or inputted here.
# thickness_PR_budget = 1.5  # the thickness of photoresist protecting your wafer outside of your target profile.
# r_target = 20  # the radial distance (microns) over which the mean peak height (depth) is calculated.
# wfr.backout_process_to_achieve_target(target_radius=target_radius,
#                                       target_depth=target_depth_profile,
#                                       thickness_PR=thickness_PR,
#                                       thickness_PR_budget=thickness_PR_budget,
#                                       r_target=r_target,
#                                       save_fig=True)
#
# # ---
#
# """
# 'compare_target_to_feature_evolution' plots your profilometry data (i.e., 'features') on top of your target profile for
# each step in your process flow. The variables 'px' and 'py' are which coordinates you want to plot.
# """
# wfr.compare_target_to_feature_evolution(px='r', py='z', save_fig=True)
#
# # ---
#
# """
# 'characterize_exposure_dose_depth_relationship' calculates the relationship between exposure intensity (mJ) and
# photoresist depth (microns). 'z_standoff_measure
# """
# wfr.characterize_exposure_dose_depth_relationship(plot_figs=True,
#                                                   save_type=save_type,
#                                                   )
# wfr.merge_exposure_doses_to_process_depths(export=True)
#
# # ---
#
# """
# This function can only be run after running 'characterize_exposure_dose_depth_relationship' first.
#
# 'correct_grayscale_design_profile' redraws your grayscale map (r-coordinate and layer), according to the characterized
# exposure_dose-to-depth relationship, to achieve your target profile.
# """
# z_standoff_design = 1
# wfr.correct_grayscale_design_profile(z_standoff=z_standoff_design,
#                                      plot_figs=True,
#                                      save_type=save_type,
#                                      )
#
# # ---
#
# """
# 'plot_all_exposure_dose_to_depth': plot figure showing exposure dose to depth relationship. 'step_develop' indicates
# which step in the process flow is the 'Develop' step.. this variable should be deprecated.
# """
# step_develop = 3
# wfr.plot_all_exposure_dose_to_depth(step=step_develop)
#
# # ---
#
# """
# 'compare_exposure_functions': plot figures comparing exposure functions.
# """
# wfr.compare_exposure_functions()
#
# # ---
#
# """
# 'plot_feature_evolution': plot features at each process.
# """
# wfr.plot_feature_evolution(px='r', py='z', save_fig=True)
#
# # ---
#
# """
# 'compare_target_to_feature_evolution': plot figures comparing target profile to features at each process.
# """
# wfr.compare_target_to_feature_evolution(px='r', py='z', save_fig=True)
#
# # ---
#
# """
# 'plot_overlay_feature_and_exposure_profiles': plot figure showing your design profile and feature profile for a
# specified step. Here, I define 'step' to be the last step in the process flow, although it can be any step.
#
# 'did' is short for Design ID.
# """
# step = max(wfr.list_steps)
# for did in wfr.dids:
#     plotting.plot_overlay_feature_and_exposure_profiles(gcw=wfr, step=step, did=did,
#                                                         path_save=join(wfr.path_results, 'figs'),
#                                                         save_type=save_type,
#                                                         )
#
# # ---
#
# """
# 'grade_profile_accuracy': Grade the accuracy of your feature profile against your target profile. *Note, this function
# doesn't really give any meaningful information right now. It needs to be upgraded.
# """
# wfr.grade_profile_accuracy(step=max(wfr.list_steps), target_radius=target_radius, target_depth=target_depth_profile)
#
#
# # ---
#
# # ----------------------------------------------------------------------------------------------------------------------
# # END DATA ANALYSIS FUNCTIONS
# # ----------------------------------------------------------------------------------------------------------------------
#
# print("example_flow.py completed without errors.")
title_size = 10
letter_size = 8
if __name__ == '__main__':
    Path = str(Path.cwd())
    # Creat buttons of little user interface to control the plot starts etc
    root = tk.Tk()
    root.title('Real time plot')
    root.config(background='light blue')
    root.geometry('1600x800')  # set the window size

    # define fgures
    on = tk.PhotoImage(file=Path + "/on.png")
    off = tk.PhotoImage(file=Path + "/off.png")

    ############  Step 1 initalize procesfloww ##################################
    step1Label = tk.Label(root, text="Step 1: read process flow", font=('Helvetica', title_size))
    step1Label.place(x=10, y=10)
    root.update()
    # startSerial = tk.Button(root, image=off, font= ('calbiri',12), command= lambda: comActive())
    # startSerial.place(x= serialLabel.winfo_x()+serialLabel.winfo_width()+20, y=10)

    # Wafer ID input + label wid
    wid_label = tk.Label(root, text="Wafer ID (int)", font=("Helvetica", letter_size))
    wid_label.place(x=10, y=step1Label.winfo_y() + step1Label.winfo_height())
    root.update()
    wid_Txt = tk.Entry(root, width=15, font=("Helvetica", letter_size))
    wid_Txt.place(x=wid_label.winfo_x() + wid_label.winfo_width() + 20, y=wid_label.winfo_y())
    root.update()

    # file directry input + label base_path
    base_path_Label = tk.Label(root, text="Directory name (str)", font=("Helvetica", letter_size))
    base_path_Label.place(x=10, y=wid_Txt.winfo_y() + wid_Txt.winfo_height())  # Apply volt button data button
    root.update()
    base_path_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    base_path_Txt.place(x=base_path_Label.winfo_x() + base_path_Label.winfo_width() + 20, y=base_path_Label.winfo_y())
    root.update()

    # File name of proces flow fn_pflow
    fn_pflow_Label = tk.Label(root, text="Filename of our process flow (str)", font=('Helvetica', letter_size))
    fn_pflow_Label.place(x=10, y=base_path_Txt.winfo_y() + base_path_Txt.winfo_height())
    root.update()
    fn_pflow_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    fn_pflow_Txt.place(x=fn_pflow_Label.winfo_x() + fn_pflow_Label.winfo_width() + 20, y=fn_pflow_Label.winfo_y())
    root.update()

    # Name of sub-directory within base_path to store all of the results path_results
    path_results_Label = tk.Label(root, text="Sub-directory to store the results (str)", font=('Helvetica', letter_size))
    path_results_Label.place(x=10, y=fn_pflow_Label.winfo_y() + fn_pflow_Label.winfo_height())
    root.update()
    path_results_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    path_results_Txt.place(x=path_results_Label.winfo_x() + path_results_Label.winfo_width() + 20, y=path_results_Label.winfo_y())
    root.update()

    # File type of the ouput figures save_type
    save_type_Label = tk.Label(root, text="File type of ouput Figures (str)", font=('Helvetica', letter_size))
    save_type_Label.place(x=10, y=path_results_Txt.winfo_y() + path_results_Txt.winfo_height())
    root.update()
    save_type_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    save_type_Txt.place(x=save_type_Label.winfo_x() + save_type_Label.winfo_width() + 20, y=save_type_Label.winfo_y())
    root.update()

    # Read processflow buton lauches the functionio.read_processflow
    readProces = tk.Button(root, text="Read process flow", font=('Helvetica', 8), command=lambda: read_processFlow())
    readProces.place(x=10, y=save_type_Txt.winfo_y() + save_type_Txt.winfo_height() + 5)
    root.update()

    ################## Step 2 ############################################
    step2Label = tk.Label(root, text="Step 2: Initialize 'Designs' and 'Features'", font=('Helvetica', title_size))
    step2Label.place(x=10+path_results_Txt.winfo_x()+path_results_Txt.winfo_width(), y=10)
    root.update()

    # WA string that identifies which design this data (profilometry, image, etc) belongs to.  design_lbls
    design_lbls_label = tk.Label(root, text="Design Label (str)", font=("Helvetica", letter_size))
    design_lbls_label.place(x=step2Label.winfo_x(), y=step2Label.winfo_y() + step2Label.winfo_height())
    root.update()
    design_lbls_Txt = tk.Entry(root, width=15, font=("Helvetica", letter_size))
    design_lbls_Txt.place(x=design_lbls_label.winfo_x() + design_lbls_label.winfo_width() + 10, y=design_lbls_label.winfo_y())
    root.update()

    # target_lbs:not important yet
    target_lbls_Label = tk.Label(root, text="Target labels (str) not important yet", font=("Helvetica", letter_size))
    target_lbls_Label.place(x=step2Label.winfo_x(),
                            y=design_lbls_Txt.winfo_y() + design_lbls_Txt.winfo_height())  # Apply volt button data button
    root.update()
    target_lbls_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    target_lbls_Txt.place(x=target_lbls_Label.winfo_x() + target_lbls_Label.winfo_width() + 10,
                          y=target_lbls_Label.winfo_y())
    root.update()

    # design_locs: not important yet impletnt x,y coordiantes of the features
    design_locs_Label = tk.Label(root, text="Design locations [x,y] (str) not important yet", font=("Helvetica", letter_size))
    design_locs_Label.place(x=step2Label.winfo_x(),
                            y=target_lbls_Txt.winfo_y() + target_lbls_Txt.winfo_height())  # Apply volt button data button
    root.update()
    design_locs_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    design_locs_Txt.place(x=design_locs_Label.winfo_x() + design_locs_Label.winfo_width() + 10,
                         y=design_locs_Label.winfo_y())
    root.update()

    # design_ids: A unique number assigned to each design_lbl (in order to perform numerical operations)
    design_ids_Label = tk.Label(root, text="Design identity (int)", font=("Helvetica", letter_size))
    design_ids_Label.place(x=step2Label.winfo_x(), y=design_locs_Txt.winfo_y() + design_locs_Txt.winfo_height())  # Apply volt button data button
    root.update()
    design_ids_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    design_ids_Txt.place(x=design_ids_Label.winfo_x() + design_ids_Label.winfo_width() + 10, y=design_ids_Label.winfo_y())
    root.update()

    # target_depth_profile: (units: microns) defines the axial distance (height) of our target profile.
    target_depth_profile_Label = tk.Label(root, text="Hight of target profile (um int)", font=('Helvetica', letter_size))
    target_depth_profile_Label.place(x=step2Label.winfo_x(),
                                     y=design_ids_Txt.winfo_y() + design_ids_Txt.winfo_height())
    root.update()
    target_depth_profile_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    target_depth_profile_Txt.place(
        x=target_depth_profile_Label.winfo_x() + target_depth_profile_Label.winfo_width() + 20,
        y=target_depth_profile_Label.winfo_y())
    root.update()
    # design_spacing: not important yet.
    design_spacing_Label = tk.Label(root, text="Design spacing (str) not important yet", font=("Helvetica", letter_size))
    design_spacing_Label.place(x=step2Label.winfo_x(),
                          y=target_depth_profile_Txt.winfo_y() + target_depth_profile_Txt.winfo_height())  # Apply volt button data button
    root.update()
    design_spacing_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    design_spacing_Txt.place(x=design_spacing_Label.winfo_x() + design_spacing_Label.winfo_width() + 10,
                        y=design_spacing_Label.winfo_y())
    root.update()

    # dose_lbls: A string that identifies which exposure setting (dose) this data belongs to.
    dose_lbls_Label = tk.Label(root, text="Exposure dose setting (str)", font=("Helvetica", letter_size))
    dose_lbls_Label.place(x=step2Label.winfo_x(),
                           y=design_spacing_Txt.winfo_y() + design_spacing_Txt.winfo_height())  # Apply volt button data button
    root.update()
    dose_lbls_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    dose_lbls_Txt.place(x=dose_lbls_Label.winfo_x() + dose_lbls_Label.winfo_width() + 10,
                         y=dose_lbls_Label.winfo_y())
    root.update()
    #
    # focus_lbls: A number that identifies which exposure setting (focus) this data belongs to.
    focus_lbls_Label = tk.Label(root, text="Exposure focus setting (int)", font=("Helvetica", letter_size))
    focus_lbls_Label.place(x=step2Label.winfo_x(),
                          y=dose_lbls_Txt.winfo_y() + dose_lbls_Txt.winfo_height())  # Apply volt button data button
    root.update()
    focus_lbls_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    focus_lbls_Txt.place(x=focus_lbls_Label.winfo_x() + focus_lbls_Label.winfo_width() + 10,
                        y=focus_lbls_Label.winfo_y())
    root.update()

    # fem_dxdy: not important yet.
    fem_dxdy_Label = tk.Label(root, text="fem_dxdy (int)", font=("Helvetica", letter_size))
    fem_dxdy_Label.place(x=step2Label.winfo_x(),
                         y=focus_lbls_Txt.winfo_y() + focus_lbls_Txt.winfo_height())  # Apply volt button data button
    root.update()
    fem_dxdy_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    fem_dxdy_Txt.place(x=fem_dxdy_Label.winfo_x() + fem_dxdy_Label.winfo_width() + 10,
                       y=fem_dxdy_Label.winfo_y())
    root.update()

    # target_radius: (units: microns) defines the radial distance of our target profile.
    target_radius_Label = tk.Label(root, text="Radial distance of target profile (um int)", font=('Helvetica', letter_size))
    target_radius_Label.place(x=step2Label.winfo_x(), y=fem_dxdy_Txt.winfo_y() + fem_dxdy_Txt.winfo_height())
    root.update()
    target_radius_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    target_radius_Txt.place(x=target_radius_Label.winfo_x() + target_radius_Label.winfo_width() + 20, y=target_radius_Label.winfo_y())
    root.update()

    # Initailze Features
    initFeaturesDesigns = tk.Button(root, text="Initialize Features", font=('Helvetica', letter_size), command=lambda: init_featuresDesigns())
    initFeaturesDesigns.place(x=step2Label.winfo_x(),
                              y=target_radius_Txt.winfo_y() + target_radius_Txt.winfo_height() + 5)
    root.update()
    ############## Step 3 initalize the wafer ##########################

    step3Label = tk.Label(root, text="Step 3: Initialize the wafer", font=('Helvetica', title_size))
    step3Label.place(x=target_radius_Label.winfo_x()+target_radius_Label.winfo_width()+10, y=10)
    root.update()





    # # Read processflow buton lauches the functionio.read_processflow
    # readProces = tk.Button(root, text="Read process flow", font=('calbiri', 11), command=lambda: read_processFlow())
    # readProces.place(x=10, y=save_type_Txt.winfo_y() + save_type_Txt.winfo_height() + 5)
    # root.update()
    # # Apply volt button data button
    # root.update()
    # apply = tk.Button(root, text="Apply", font=('calbiri', 15), command=lambda: apply_V())
    # apply.place(x=10, y=saveTypTxt.winfo_y() + 50)
    #
    # # Aplied voltage to droptles on CCS device
    # root.update()
    # appVolt = tk.Entry(root, width=20, borderwidth=2, font=("Helvetica", 15))
    # appVolt.place(x=apply.winfo_x() + apply.winfo_width() + 20, y=apply.winfo_y() + 5)
    # root.update()
    #
    # puttig the text/error log field
    Output = tk.Text(root, height=20,width=35,bg="light cyan",font=("Helvetica", letter_size))
    Output.place(x=10, y=readProces.winfo_y() + readProces.winfo_height() + 10)
    root.update()

    # #The figure on the user interface window#
    fig = plt.figure(figsize=(3.5,3.5))
    plot1 = fig.add_subplot(111)
    xCirc = [50.8*np.cos(x) for x in np.linspace(0, 2*np.pi, 500)]
    yCirc = [50.8*np.sin(x) for x in np.linspace(0, 2*np.pi, 500)]
    # plot1.set_title('Serial Data')
    plot1.plot(xCirc,yCirc)
    # plot1.set_xlabel('$mm$')
    # plot1.set_ylabel('$mm$')
    plot1.set_title('Features in mm')
    plot1.grid()
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,master=root)
    canvas.get_tk_widget().place(x=step2Label.winfo_x(), y=initFeaturesDesigns.winfo_y() + initFeaturesDesigns.winfo_height() + 5)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas,root)
    toolbar.update()
    root.update()

    root.mainloop()