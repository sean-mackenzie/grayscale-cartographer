from os.path import join
import tkinter as tk #tkinter for GUI
from graycart.GraycartWafer import GraycartWafer
from graycart.GraycartFeature import initialize_designs, initialize_design_features
from graycart.utils import io, plotting
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)

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
wid = 11
base_path = '/Users\simon\Documents\Simels_daten\Epfl\sem_13_2022_Master_theis_USA\grayscale-cartographer\example\Wafer{}'.format(wid)
fn_pflow = 'process-flow_w{}.xlsx'.format(wid)
path_results = 'results'
save_type = '.png'
#
# functions
def read_processFlow():
    global process_flow
    process_flow = io.read_process_flow(fp=join(base_path, fn_pflow))
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
# designs = initialize_designs(base_path, design_lbls, target_lbls, design_locs, design_ids)
#
# # 'designs' on a wafer form 'features'
# features = initialize_design_features(designs,
#                                       design_spacing,
#                                       dose_lbls,
#                                       focus_lbls,
#                                       process_flow,
#                                       fem_dxdy,
#                                       target_radius=target_radius,
#                                       )

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
if __name__ == '__main__':
    Path = str(Path.cwd())
    #Creat buttons of little user interface to control the plot starts etc
    root = tk.Tk()
    root.title('Real time plot')
    root.config(background= 'light blue')
    root.geometry('1600x800') #set the window size

    #define fgures
    on = tk.PhotoImage(file=Path + "/on.png")
    off = tk.PhotoImage(file=Path + "/off.png")

    # Step 1 title
    step1Label = tk.Label(root, text="Step 1 read process flow", font= ('calbiri', 15))
    step1Label.place(x = 10, y = 20)
    root.update()
    # startSerial = tk.Button(root, image=off, font= ('calbiri',12), command= lambda: comActive())
    # startSerial.place(x= serialLabel.winfo_x()+serialLabel.winfo_width()+20, y=10)

    # Wafer ID input + label
    waferIDLabel = tk.Label(root, text="Wafer ID", font=("Helvetica", 11))
    waferIDLabel.place(x = 10, y =step1Label.winfo_y() + step1Label.winfo_height())
    root.update()
    waferIDTxt = tk.Entry(root, width=15, font=("Helvetica", 11))
    waferIDTxt.place(x=waferIDLabel.winfo_x() + waferIDLabel.winfo_width() + 20, y=waferIDLabel.winfo_y())
    root.update()

    # file directry input + label
    fileDirLabel = tk.Label(root, text="File directory name", font=("Helvetica", 11))
    fileDirLabel.place(x=10, y=waferIDTxt.winfo_y() + waferIDTxt.winfo_height()) # Apply volt button data button
    root.update()
    fileDirTxt = tk.Entry(root, width=10, font=("Helvetica", 11))
    fileDirTxt.place(x=fileDirLabel.winfo_x() + fileDirLabel.winfo_width() + 20, y=fileDirLabel.winfo_y() )
    root.update()

    # File name of proces flow
    fileNameLabel = tk.Label(root, text="Filename of our process flow", font= ('Helvetica', 11))
    fileNameLabel.place(x = 10, y =fileDirTxt.winfo_y() + fileDirTxt.winfo_height())
    root.update()
    fileNameTxt = tk.Entry(root, width=10, font=("Helvetica", 11))
    fileNameTxt.place(x=fileNameLabel.winfo_x() + fileNameLabel.winfo_width() + 20, y=fileNameLabel.winfo_y() )
    root.update()

    # Name of sub-directory within base_path to store all of the results
    pathResLabel = tk.Label(root, text="Name of sub-directory to store the results", font= ('calbiri', 11))
    pathResLabel.place(x = 10, y =fileNameLabel.winfo_y() + fileNameLabel.winfo_height())
    root.update()
    pathResTxt = tk.Entry(root, width=10, font=("Helvetica", 11))
    pathResTxt.place(x=pathResLabel.winfo_x() + pathResLabel.winfo_width() + 20, y=pathResLabel.winfo_y())
    root.update()

    # File type of the ouput figures
    saveTypLabel = tk.Label(root, text="File type of ouput Figures", font= ('Helvetica', 11))
    saveTypLabel.place(x=10, y=pathResTxt.winfo_y() + pathResTxt.winfo_height())
    root.update()
    saveTypTxt = tk.Entry(root, width=10, font=("Helvetica", 11))
    saveTypTxt.place(x=saveTypLabel.winfo_x() + saveTypLabel.winfo_width() + 20, y=saveTypLabel.winfo_y())
    root.update()

    # Read processflow buton lauches the functionio.read_processflow
    readProces = tk.Button(root,text= "Save", font= ('calbiri',11), command= lambda: read_processFlow())
    readProces.place(x=10, y=saveTypTxt.winfo_y() + saveTypTxt.winfo_height())
    root.update()

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
    # #puttig the text log field
    # Output = tk.Text(root, height=20,width=35,bg="light cyan",font=("Helvetica", 15))
    # Output.place(x=10, y=appVolt.winfo_y() + appVolt.winfo_height() + 10)
    # root.update()

    # #The figure on the user interface window#
    # fig = plt.figure()
    # plot1 = fig.add_subplot(111)
    # plot1.set_title('Serial Data')
    # plot1.set_xlabel('t')
    # plot1.set_ylabel('data')
    # plot1.grid()
    # # creating the Tkinter canvas
    # # containing the Matplotlib figure
    # canvas = FigureCanvasTkAgg(fig,master=root)
    # canvas.get_tk_widget().place(x=waferIDTxt.winfo_x() + waferIDTxt.winfo_width() + 50, y=step1Label.winfo_y())
    # canvas.draw()
    # toolbar = NavigationToolbar2Tk(canvas,root)
    # toolbar.update()
    # root.update()

    root.mainloop()