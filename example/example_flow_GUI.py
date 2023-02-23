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
                      +"You scrwed up" + '\n'
                      +'Wafer ID error:- '+wid_err+ '\n'
                      +'Data path error:- '+base_path_err+ '\n'
                      +'Process flow name error:- '+fn_pflow_err+'\n'
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
    designs_err = ''
    global features

    global target_radius
    target_radius_err = ''
    global target_lbls
    target_lbls_err = ''
    global target_depth_profile
    target_depth_profile_err=''
    global design_locs
    design_locs_err = ''
    global design_spacing
    design_spacing_err = ''
    global design_ids
    design_ids_err=''
    global design_lbls
    design_lbls_err = ''
    global dose_lbls
    dose_lbls_err=''
    global focus_lbls
    focus_lbls_err = ''
    global fem_dxdy
    fem_dxdy_err = ''


    design_spacing = design_spacing_Txt.get().replace(' ', '')
    if design_spacing.replace('.', '', 1).replace('e', '', 1).isdigit():
        design_spacing = float(design_spacing)
    else:
        design_spacing_err = 'Your desing spaicing is not a digital'
    design_lbls = design_lbls_Txt.get().replace(' ', '').split(',')
    target_lbls = target_lbls_Txt.get().replace(' ', '').split(',')
    if target_lbls[0]=='None':
        for i in range(len(target_lbls)):
            target_lbls[i]=None
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
                          + 'Design locations: ' + design_locs_Txt.get() + '\n'
                          + 'Design spacing: ' + design_spacing_Txt.get() + '\n'
                          + 'Design indices ' + design_ids_Txt.get() + '\n'
                          + 'Design labels ' + design_lbls_Txt.get() + '\n'
                          + 'Dose labels: '+dose_lbls_Txt.get()+'\n'
                          + 'Focus labels: ' + focus_lbls_Txt.get() + '\n'
                          + 'Fem_dxdy is: '+fem_dxdy_Txt.get()+'\n'
                          )
            plot1.clear()
            plot1.grid()
            canvas.draw()
            root.update()  #
            for i in design_locs:
                Features = cicleXY(i,target_radius)
                plotFeatures(Features.xcir,Features.ycir)
        except:
            features = False
            Output.delete("1.0", "end")
            Output.insert("1.0", "Step 2 Feature errors:" + '\n'
                          + "You scrwed up" + '\n'
                          + 'Target radius err:- ' + target_radius_err + '\n'
                          + 'Target labels err:- ' + target_lbls_err + '\n'
                          + 'Target depth err:- ' + target_depth_profile_err + '\n'
                          + 'Design err:- '+designs_err + '\n'
                          + 'Design locations err:- ' + design_locs_err + '\n'
                          + 'Design spacing err:- ' + design_spacing_err + '\n'
                          + 'Design indices err:- '+design_ids_err + '\n'
                          + 'Design labels err:- ' + design_lbls_err + '\n'
                          + 'Dose labels err:- ' + dose_lbls_err + '\n'
                          + 'Focus label err:- '+focus_lbls_err + '\n'
                          + 'Fem_dxdy err:- '+fem_dxdy_err + '\n\n')
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

# inputs
profilometry_tool = 'KLATencor-P7'


measurement_methods = False
measurement_methods_err =''
def slect_profilometry(profilometry_tool):
    profilometry_tool = profilometry_tool_Txt.get()
    profilo_window = tk.Toplevel()
    profilo_window.title('Select specifications of '+ profilometry_tool)
    profilo_window.config(background='light green')
    profilo_window.geometry('900x250')  # set the window size

    step1Label = tk.Label(profilo_window, text="Specify parameters of "+profilometry_tool, font=('Helvetica', title_size))
    step1Label.place(x=10, y=10)
    profilo_window.update()
    # startSerial = tk.Button(root, image=off, font= ('calbiri',12), command= lambda: comActive())
    # startSerial.place(x= serialLabel.winfo_x()+serialLabel.winfo_width()+20, y=10)
    # File type to read
    filetype_read_Label = tk.Label(profilo_window, text="Filetype read (str)", font=("Helvetica", letter_size))
    filetype_read_Label.place(x=10, y=step1Label.winfo_y() + step1Label.winfo_height())
    profilo_window.update()
    filetype_read_Txt = tk.Entry(profilo_window, width=15, font=("Helvetica", letter_size))
    filetype_read_Txt.place(x=filetype_read_Label.winfo_x() + filetype_read_Label.winfo_width() + 5, y=filetype_read_Label.winfo_y())
    profilo_window.update()

    # x units
    x_units_read_Label = tk.Label(profilo_window, text="X units read (float)", font=("Helvetica", letter_size))
    x_units_read_Label.place(x=filetype_read_Txt.winfo_x() + filetype_read_Txt.winfo_width() + 10, y=filetype_read_Label.winfo_y())  # Apply volt button data button
    profilo_window.update()
    x_units_read_Txt = tk.Entry(profilo_window, width=10, font=("Helvetica", letter_size))
    x_units_read_Txt.place(x=x_units_read_Label.winfo_x() + x_units_read_Label.winfo_width() + 5, y=filetype_read_Label.winfo_y())
    profilo_window.update()

    # y units
    y_units_read_Label = tk.Label(profilo_window, text="Y units read (float)", font=('Helvetica', letter_size))
    y_units_read_Label.place(x=x_units_read_Txt.winfo_x() + x_units_read_Txt.winfo_width() + 5, y=filetype_read_Label.winfo_y())
    profilo_window.update()
    y_units_read_Txt = tk.Entry(profilo_window, width=10, font=("Helvetica", letter_size))
    y_units_read_Txt.place(x=y_units_read_Label.winfo_x() + y_units_read_Label.winfo_width() + 5, y=filetype_read_Label.winfo_y())
    profilo_window.update()

    #file name write
    filetype_write_Label = tk.Label(profilo_window, text="Filetype write (str)", font=("Helvetica", letter_size))
    filetype_write_Label.place(x=10, y=filetype_read_Label.winfo_y()+filetype_read_Label.winfo_height()+5)
    profilo_window.update()
    filetype_write_Txt = tk.Entry(profilo_window, width=15, font=("Helvetica", letter_size))
    filetype_write_Txt.place(x=filetype_read_Label.winfo_x() + filetype_read_Label.winfo_width() + 5,
                            y=filetype_write_Label.winfo_y())
    profilo_window.update()

    # x units
    x_units_write_Label = tk.Label(profilo_window, text="X units read (float)", font=("Helvetica", letter_size))
    x_units_write_Label.place(x=filetype_read_Txt.winfo_x() + filetype_read_Txt.winfo_width() + 10,
                             y=filetype_read_Label.winfo_y()+filetype_read_Label.winfo_height()+5)  # Apply volt button data button
    profilo_window.update()
    x_units_write_Txt = tk.Entry(profilo_window, width=10, font=("Helvetica", letter_size))
    x_units_write_Txt.place(x=x_units_read_Label.winfo_x() + x_units_read_Label.winfo_width() + 5,
                           y=filetype_write_Txt.winfo_y()+5)
    profilo_window.update()

    # y units
    y_units_write_Label = tk.Label(profilo_window, text="Y units read (float)", font=('Helvetica', letter_size))
    y_units_write_Label.place(x=x_units_read_Txt.winfo_x() + x_units_read_Txt.winfo_width() + 5,
                             y=filetype_write_Txt.winfo_y()+5)
    profilo_window.update()
    y_units_write_Txt = tk.Entry(profilo_window, width=10, font=("Helvetica", letter_size))
    y_units_write_Txt.place(x=y_units_read_Label.winfo_x() + y_units_read_Label.winfo_width() + 5,
                           y=filetype_write_Txt.winfo_y()+5)
    profilo_window.update()
    ###### Data etch monitor ################
    headr_etch_moitor_Label = tk.Label(profilo_window, text="Data Etch monitor: Header (str)", font=("Helvetica", letter_size))
    headr_etch_moitor_Label.place(x=10, y=y_units_write_Label.winfo_y() + y_units_write_Label.winfo_height() + 5)
    profilo_window.update()
    header_etch_moitor_Txt = tk.Entry(profilo_window, width=15, font=("Helvetica", letter_size))
    header_etch_moitor_Txt.place(x=headr_etch_moitor_Label.winfo_x() + headr_etch_moitor_Label.winfo_width() + 5,
                                y=headr_etch_moitor_Label.winfo_y())
    profilo_window.update()

    # File name type read ethcer
    filetype_read_etcher_Label = tk.Label(profilo_window, text="File type read (float)", font=("Helvetica", letter_size))
    filetype_read_etcher_Label.place(x=header_etch_moitor_Txt.winfo_x() + header_etch_moitor_Txt.winfo_width() + 10,
                              y=headr_etch_moitor_Label.winfo_y() )  # Apply volt button data button
    profilo_window.update()
    filetype_read_etcher_Txt = tk.Entry(profilo_window, width=10, font=("Helvetica", letter_size))
    filetype_read_etcher_Txt.place(x=filetype_read_etcher_Label.winfo_x() + filetype_read_etcher_Label.winfo_width() + 5,
                            y=headr_etch_moitor_Label.winfo_y() )
    profilo_window.update()

    # File name type write ethcer
    filetype_write_etcher_Label = tk.Label(profilo_window, text="File type write (float)", font=('Helvetica', letter_size))
    filetype_write_etcher_Label.place(x=filetype_read_etcher_Txt.winfo_x() + filetype_read_etcher_Txt.winfo_width() + 5,
                              y=headr_etch_moitor_Label.winfo_y() )
    profilo_window.update()
    filetype_write_etcher_Txt = tk.Entry(profilo_window, width=10, font=("Helvetica", letter_size))
    filetype_write_etcher_Txt.place(x=filetype_write_etcher_Label.winfo_x() + filetype_write_etcher_Label.winfo_width() + 5,
                            y=headr_etch_moitor_Label.winfo_y() )
    profilo_window.update()
    ###### Data optical ################
    headr_optical_Label = tk.Label(profilo_window, text="Data Optical: Header (str)", font=("Helvetica", letter_size))
    headr_optical_Label.place(x=10, y=filetype_write_etcher_Txt.winfo_y() + filetype_write_etcher_Txt.winfo_height() + 5)
    profilo_window.update()
    headr_optical_Txt = tk.Entry(profilo_window, width=15, font=("Helvetica", letter_size))
    headr_optical_Txt.place(x=headr_optical_Label.winfo_x() + headr_optical_Label.winfo_width() + 5,
                                y=headr_optical_Label.winfo_y())
    profilo_window.update()

    # File name type read optical
    filetype_read_optical_Label = tk.Label(profilo_window, text="File type read (float)", font=("Helvetica", letter_size))
    filetype_read_optical_Label.place(x=headr_optical_Txt.winfo_x() + headr_optical_Txt.winfo_width() + 10,
                              y=headr_optical_Label.winfo_y() )  # Apply volt button data button
    profilo_window.update()
    filetype_read_optical_Txt = tk.Entry(profilo_window, width=10, font=("Helvetica", letter_size))
    filetype_read_optical_Txt.place(x=filetype_read_optical_Label.winfo_x() + filetype_read_optical_Label.winfo_width() + 5,
                            y=headr_optical_Label.winfo_y() )
    profilo_window.update()

    # File name type write optical
    filetype_write_optical_Label = tk.Label(profilo_window, text="File type write (float)", font=('Helvetica', letter_size))
    filetype_write_optical_Label.place(x=filetype_read_optical_Txt.winfo_x() + filetype_read_optical_Txt.winfo_width() + 5,
                              y=headr_optical_Label.winfo_y() )
    profilo_window.update()
    filetype_write_optical_Txt = tk.Entry(profilo_window, width=10, font=("Helvetica", letter_size))
    filetype_write_optical_Txt.place(x=filetype_write_optical_Label.winfo_x() + filetype_write_optical_Label.winfo_width() + 5,
                            y=headr_optical_Label.winfo_y() )
    profilo_window.update()

    ###### Data miscellaneous ################
    headr_misc_Label = tk.Label(profilo_window, text="Data Miscellaneous: Header (str)", font=("Helvetica", letter_size))
    headr_misc_Label.place(x=10,
                              y=filetype_read_optical_Txt.winfo_y() + filetype_read_optical_Txt.winfo_height() + 5)
    profilo_window.update()
    headr_misc_Txt = tk.Entry(profilo_window, width=15, font=("Helvetica", letter_size))
    headr_misc_Txt.place(x=headr_misc_Label.winfo_x() + headr_misc_Label.winfo_width() + 5,
                            y=headr_misc_Label.winfo_y())
    profilo_window.update()

    # File name type read miscellaneous
    filetype_read_misc_Label = tk.Label(profilo_window, text="File type read (float)", font=("Helvetica", letter_size))
    filetype_read_misc_Label.place(x=headr_misc_Txt.winfo_x() + headr_misc_Txt.winfo_width() + 10,
                                 y=headr_misc_Label.winfo_y())  # Apply volt button data button
    profilo_window.update()
    filetype_read_misc_Txt = tk.Entry(profilo_window, width=10, font=("Helvetica", letter_size))
    filetype_read_misc_Txt.place(x=filetype_read_misc_Label.winfo_x() + filetype_read_misc_Label.winfo_width() + 5,
                               y=headr_misc_Label.winfo_y())
    profilo_window.update()

    # File name type write miscellaneous
    filetype_write_misc_Label = tk.Label(profilo_window, text="File type write (float)", font=('Helvetica', letter_size))
    filetype_write_misc_Label.place(x=filetype_read_misc_Txt.winfo_x() + filetype_read_misc_Txt.winfo_width() + 5,
                                 y=headr_misc_Label.winfo_y())
    profilo_window.update()
    filetype_write_misc_Txt = tk.Entry(profilo_window, width=10, font=("Helvetica", letter_size))
    filetype_write_misc_Txt.place(x=filetype_write_misc_Label.winfo_x() + filetype_write_misc_Label.winfo_width() + 5,
                               y=headr_misc_Label.winfo_y())
    profilo_window.update()
    def quit():
        global measurement_methods
        global measurement_methods_err
        ### getting the data and convert intot the talbe that we want to have
        filetype_read = filetype_read_Txt.get()
        x_units_read = x_units_read_Txt.get()
        x_units_read_err = ''
        if x_units_read.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
            x_units_read = float(x_units_read)
        else:
            x_units_read_err = x_units_read + ' is not a float'
        y_units_read = y_units_read_Txt.get()
        y_units_read_err = ''
        if y_units_read.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
            y_units_read = float(y_units_read)
        else:
            y_units_read_err = y_units_read + ' is not a float'
        filetype_write = filetype_write_Txt.get()
        x_units_write = x_units_read_Txt.get()
        x_units_write_err = ''
        if x_units_write.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
            x_units_write = float(x_units_write)
        else:
            x_units_write_err = x_units_write + ' is not a float'
        y_units_write = y_units_write_Txt.get()
        y_units_write_err = ''
        if y_units_write.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
            y_units_write = float(y_units_write)
        else:
            y_units_write_err = y_units_read + ' is not a float'

        data_profile = {'header': profilometry_tool,
                        'filetype_read': '.' + filetype_read, 'x_units_read': x_units_read,
                        'y_units_read': y_units_read,
                        'filetype_write': '.' + filetype_write, 'x_units_write': x_units_write,
                        'y_units_write': y_units_write,
                        }
        header_etch_moitor = header_etch_moitor_Txt.get()
        filetype_read_etcher = filetype_read_etcher_Txt.get()
        filetype_write_etcher = filetype_write_etcher_Txt.get()
        data_etch_monitor = {'header': header_etch_moitor, 'filetype_read': '.' + filetype_read_etcher,
                             'filetype_write': '.' + filetype_write_etcher}

        headr_optical = headr_optical_Txt.get()
        filetype_read_optical = filetype_read_optical_Txt.get()
        filetype_write_optical = filetype_write_optical_Txt.get()
        data_optical = {'header': headr_optical, 'filetype_read': '.' + filetype_read_optical,
                        'filetype_write': '.' + filetype_write_optical}
        headr_misc = headr_misc_Txt.get()
        filetype_read_misc = filetype_read_misc_Txt.get()
        filetype_write_misc = filetype_write_misc_Txt.get()
        data_misc = {'header': headr_misc, 'filetype_read': '.' + filetype_read_misc,
                     'filetype_write': '.' + filetype_write_misc}
        measurement_methods = {'Profilometry': data_profile,
                               'Etch Monitor': data_etch_monitor,
                               'Optical': data_optical,
                               'Misc': data_misc,
                               }
        measurement_methods_err = {'x_units_read_err': x_units_read_err, 'y_units_read_err': y_units_read_err,
                                   'x_units_write_err': x_units_write_err, 'y_units_write_err': y_units_write_err}
        profilo_window.destroy()
    # Read processflow buton lauches the functionio.read_processflow
    kill = tk.Button(profilo_window, text="Set measurements methods", font=('Helvetica', 8), command=lambda: quit())
    kill.place(x=10, y=headr_misc_Label.winfo_y() + headr_misc_Label.winfo_height() + 5)
    profilo_window.update()

wfr = False
def init_Wafer():
    # 4. 'measurements' record the effect of 'processes' on 'features'
    # measurement_methods = io.read_measurement_methods(profilometry=profilometry_tool)
    global wfr
    if features==False:
        try:
            # 5. the 'wafer' structures all of this data as a historical record of 'cause' and 'effect'
            wfr = GraycartWafer(wid=wid,
                                path=base_path,
                                path_results=path_results,
                                designs=designs,
                                features=features,
                                process_flow=process_flow,
                                processes=None,
                                measurement_methods=measurement_methods,
                                )
            Output.delete("1.0", "end")
            Output.insert("1.0", 'Step 3: Wafer initalization' + '\n'
                          + 'Wafer succesfully initialized to:' + '\n'
                          + 'Wafer id: ' + str(wid) + '\n'
                          + 'Base path: ' + base_path + '\n'
                          + 'Results path: ' + path_results + '\n'
                          + 'The process flow given at step 1: ' + '\n'
                          + 'The designs and features given at step 2: ' + '\n'
                          + 'Design indices ' + design_ids_Txt.get() + '\n'
                          + 'Measurments methods: ' + str(measurement_methods)+ '\n'
                          )
        except:
            wfr = False
            Output.delete("1.0", "end")
            Output.insert("1.0", "Step 3: Wafer initalization errors:" + '\n'
                          + "You scrwed up" + '\n'
                          + 'Measurments methods err:- ' + str(measurement_methods) + '\n'
                          )
    else:
        wfr = False
        Output.delete("1.0", "end")
        Output.insert("1.0", "Step 3: Wafer initalization errors:" + '\n'
                      + "You scrwed up" + '\n'
                      + 'Do step 2 first' + '\n'
                      )

"""
4. Read, process, and evaluate the 'measurement_methods' data.

    # the following inputs modify (turn on or off) functions to process profilometry data.
    evaluate_signal_processing: (True or False) Optionally, plot various signal processing methods to smooth profilometry data.
    perform_rolling_on: (True or False) Optionally, perform a rolling operation to smooth data; [[3, 'b1', 25]]

    # the following parameters are passed into smoothing and peak_finding algorithms:
    lambda_peak_rel_height: function to modify the scipy.find_peaks function
    peak_rel_height: if 'lambda_peak_rel_height' is defined, then this variable isn't used.
    downsample: downsample the data to reduce computation.
    width_rel_radius: radial distance, beyond the target radius, to evaluate.
    prominence: the relative height of a peak compared to adjacent peaks.
    fit_func: function that is fit to the profilometry profile to find its peak (center)

    # the following inputs modify plotting functions:
    plot_width_rel_target_radius: plot radius = target_radius * plot_width_rel_target_radius
    save_profilometry_processing_figures = True
    save_merged_profilometry_data = True

Functions:
    evaluate_process_profilometry: iterate through each process and each feature and evaluate profilometry data.
    merge_processes_profilometry: merge all of the profilometry data into a single database.
    merge_exposure_doses_to_process_depths: merge exposure data (dose, focus) with profilometry data (r, z)
"""

# inputs
evaluate_signal_processing = False
lambda_peak_rel_height = lambda x: min([0.95 + x / 100, 0.9875])
perform_rolling_on = False  # [[3, 'b1', 25]]
plot_width_rel_target_radius = 1.2
peak_rel_height = 0.975
downsample = 5
width_rel_radius = 0.01
fit_func = 'parabola'
prominence = 1
save_profilometry_processing_figures = True
save_merged_profilometry_data = True

evaluate_signal_processing = False
def sig_prosOnOff():
    global evaluate_signal_processing
    if evaluate_signal_processing == False:
        evaluate_sig_pros_OnOff.config(image=on)
        evaluate_signal_processing = True
    else:
        evaluate_sig_pros_OnOff.config(image=off)
        evaluate_signal_processing = False

perform_rolling_on = False
def rolling_OnOff():
    global perform_rolling_on
    if perform_rolling_on == False:
        perform_rolling_OnOff.config(image=on)
        perform_rolling_on = True
    else:
        perform_rolling_OnOff.config(image=off)
        perform_rolling_on = False

save_profilometry_processing_figures = False
def save_profi_proc_fig_OnOff():
    global save_profilometry_processing_figures
    if save_profilometry_processing_figures == False:
        save_profilometry_processing_figures_OnOff.config(image=on)
        save_profilometry_processing_figures = True
    else:
        save_profilometry_processing_figures_OnOff.config(image=off)
        save_profilometry_processing_figures = False
# functions

save_merged_profilometry_data = False
def save_merged_prof_data_OnOff():
    global save_merged_profilometry_data
    if save_merged_profilometry_data == False:
        save_merged_profilometry_data_OnOff.config(image=on)
        save_merged_profilometry_data = True
    else:
        save_merged_profilometry_data_OnOff.config(image=off)
        save_merged_profilometry_data = False

def eval_process():
    global lambda_peak_rel_height
    lambda_peak_rel_height_err=''
    global peak_rel_height
    peak_rel_height_err=''
    global downsample
    downsample_err=''
    global width_rel_radius
    width_rel_radius_err=''
    global prominence
    prominence=''
    global fit_func
    fit_func_err=''
    global plot_width_rel_target_radius
    plot_width_rel_target_radius_err=''

    peak_rel_height=peak_rel_height_Txt.get()
    if peak_rel_height.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
        peak_rel_height = float(peak_rel_height)
    else:
        peak_rel_height_err = peak_rel_height + ' is not convertable to float'

    downsample=downsample_Txt.get()
    if downsample.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
        downsample = float(downsample)
    else:
        downsample = downsample + ' is not convertable to float'

    width_rel_radius=width_rel_radius_Txt.get()
    if width_rel_radius.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
        width_rel_radius = float(width_rel_radius)
    else:
        width_rel_radius_err = width_rel_radius + ' is not convertable to float'

    prominence=prominence_Txt.get()
    if prominence.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
        prominence = float(prominence)
    else:
        prominence_err = prominence + ' is not convertable to float'

    plot_width_rel_target_radius=plot_width_rel_target_radius_Txt.get()
    if plot_width_rel_target_radius.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
        plot_width_rel_target_radius = float(plot_width_rel_target_radius)
    else:
        plot_width_rel_target_radius_err = plot_width_rel_target_radius + ' is not convertable to float'

    if wfr==False:
        try:
            wfr.evaluate_process_profilometry(plot_fits=save_profilometry_processing_figures,
                                              perform_rolling_on=perform_rolling_on,
                                              evaluate_signal_processing=evaluate_signal_processing,
                                              plot_width_rel_target=plot_width_rel_target_radius,
                                              peak_rel_height=peak_rel_height,
                                              downsample=downsample,
                                              width_rel_radius=width_rel_radius,
                                              fit_func=fit_func,
                                              prominence=prominence,
                                              )

            wfr.merge_processes_profilometry(export=save_merged_profilometry_data)
            Output.delete("1.0", "end")
            Output.insert("1.0", 'Step 4: Wafer initalization' + '\n'
                          + 'Wafer succesfully evaluated with:' + '\n'
                          + 'peak_rel_height: ' + str(peak_rel_height) + '\n'
                          + 'downsample: ' + str(downsample) + '\n'
                          + 'width_rel_radius: ' + str(width_rel_radius) + '\n'
                          + 'prominence ' + str(prominence) + '\n'
                          + 'fit_func ' + fit_func + '\n'
                          + 'plot_width_rel_target_radius: ' + str(plot_width_rel_target_radius)+ '\n'
                          )
        except:
            Output.delete("1.0", "end")
            Output.insert("1.0", "Step 3: Wafer evaluation errors:" + '\n'
                          + "You scrwed up" + '\n'
                          + 'peak_rel_height err:- ' + peak_rel_height_err + '\n'
                          + 'downsample err:- ' + downsample_err + '\n'
                          + 'width_rel_radius err:- ' + width_rel_radius_err + '\n'
                          + 'prominence err:- ' + prominence_err + '\n'
                          + 'fit_func err:- ' + fit_func_err + '\n'
                          + 'plot_width_rel_target_radius err:- ' + plot_width_rel_target_radius_err + '\n'
                          )
    else:
        Output.delete("1.0", "end")
        Output.insert("1.0", "Step 4: Wafer evaluation errors:" + '\n'
                      + "You scrwed up" + '\n'
                      + 'Do step 3 first' + '\n'
                      )


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
    root.title('Million dollar code')
    root.config(background='light green')
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
    design_lbls_label = tk.Label(root, text="Design labels [(str)]", font=("Helvetica", letter_size))
    design_lbls_label.place(x=step2Label.winfo_x(), y=step2Label.winfo_y() + step2Label.winfo_height())
    root.update()
    design_lbls_Txt = tk.Entry(root, width=15, font=("Helvetica", letter_size))
    design_lbls_Txt.place(x=design_lbls_label.winfo_x() + design_lbls_label.winfo_width() + 10, y=design_lbls_label.winfo_y())
    root.update()

    # target_lbs:not important yet
    target_lbls_Label = tk.Label(root, text="Target labels [(str)]", font=("Helvetica", letter_size))
    target_lbls_Label.place(x=step2Label.winfo_x(),
                            y=design_lbls_Txt.winfo_y() + design_lbls_Txt.winfo_height())  # Apply volt button data button
    root.update()
    target_lbls_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    target_lbls_Txt.place(x=target_lbls_Label.winfo_x() + target_lbls_Label.winfo_width() + 10,
                          y=target_lbls_Label.winfo_y())
    root.update()

    # design_locs: not important yet impletnt x,y coordiantes of the features
    design_locs_Label = tk.Label(root, text="Design locations [(float)]", font=("Helvetica", letter_size))
    design_locs_Label.place(x=step2Label.winfo_x(),
                            y=target_lbls_Txt.winfo_y() + target_lbls_Txt.winfo_height())  # Apply volt button data button
    root.update()
    design_locs_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    design_locs_Txt.place(x=design_locs_Label.winfo_x() + design_locs_Label.winfo_width() + 10,
                         y=design_locs_Label.winfo_y())
    root.update()

    # design_ids: A unique number assigned to each design_lbl (in order to perform numerical operations)
    design_ids_Label = tk.Label(root, text="Design indices [(int)]", font=("Helvetica", letter_size))
    design_ids_Label.place(x=step2Label.winfo_x(), y=design_locs_Txt.winfo_y() + design_locs_Txt.winfo_height())  # Apply volt button data button
    root.update()
    design_ids_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    design_ids_Txt.place(x=design_ids_Label.winfo_x() + design_ids_Label.winfo_width() + 10, y=design_ids_Label.winfo_y())
    root.update()

    # target_depth_profile: (units: microns) defines the axial distance (height) of our target profile.
    target_depth_profile_Label = tk.Label(root, text="Target depth microns (int)", font=('Helvetica', letter_size))
    target_depth_profile_Label.place(x=step2Label.winfo_x(),
                                     y=design_ids_Txt.winfo_y() + design_ids_Txt.winfo_height())
    root.update()
    target_depth_profile_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    target_depth_profile_Txt.place(
        x=target_depth_profile_Label.winfo_x() + target_depth_profile_Label.winfo_width() + 20,
        y=target_depth_profile_Label.winfo_y())
    root.update()
    # design_spacing: not important yet.
    design_spacing_Label = tk.Label(root, text="Design spacing microns (int)", font=("Helvetica", letter_size))
    design_spacing_Label.place(x=step2Label.winfo_x(),
                          y=target_depth_profile_Txt.winfo_y() + target_depth_profile_Txt.winfo_height())  # Apply volt button data button
    root.update()
    design_spacing_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    design_spacing_Txt.place(x=design_spacing_Label.winfo_x() + design_spacing_Label.winfo_width() + 10,
                        y=design_spacing_Label.winfo_y())
    root.update()

    # dose_lbls: A string that identifies which exposure setting (dose) this data belongs to.
    dose_lbls_Label = tk.Label(root, text="Dose labels [(str)]", font=("Helvetica", letter_size))
    dose_lbls_Label.place(x=step2Label.winfo_x(),
                           y=design_spacing_Txt.winfo_y() + design_spacing_Txt.winfo_height())  # Apply volt button data button
    root.update()
    dose_lbls_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    dose_lbls_Txt.place(x=dose_lbls_Label.winfo_x() + dose_lbls_Label.winfo_width() + 10,
                         y=dose_lbls_Label.winfo_y())
    root.update()
    #
    # focus_lbls: A number that identifies which exposure setting (focus) this data belongs to.
    focus_lbls_Label = tk.Label(root, text="Focus labels [(int)]", font=("Helvetica", letter_size))
    focus_lbls_Label.place(x=step2Label.winfo_x(),
                          y=dose_lbls_Txt.winfo_y() + dose_lbls_Txt.winfo_height())  # Apply volt button data button
    root.update()
    focus_lbls_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    focus_lbls_Txt.place(x=focus_lbls_Label.winfo_x() + focus_lbls_Label.winfo_width() + 10,
                        y=focus_lbls_Label.winfo_y())
    root.update()

    # fem_dxdy: not important yet.
    fem_dxdy_Label = tk.Label(root, text="fem_dxdy [(float)]", font=("Helvetica", letter_size))
    fem_dxdy_Label.place(x=step2Label.winfo_x(),
                         y=focus_lbls_Txt.winfo_y() + focus_lbls_Txt.winfo_height())  # Apply volt button data button
    root.update()
    fem_dxdy_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    fem_dxdy_Txt.place(x=fem_dxdy_Label.winfo_x() + fem_dxdy_Label.winfo_width() + 10,
                       y=fem_dxdy_Label.winfo_y())
    root.update()

    # target_radius: (units: microns) defines the radial distance of our target profile.
    target_radius_Label = tk.Label(root, text="Target radius microns (int)", font=('Helvetica', letter_size))
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
    step3Label.place(x=step2Label.winfo_x()+step2Label.winfo_width()+10, y=10)
    root.update()

    # target_radius: (units: microns) defines the radial distance of our target profile.
    profilometry_tool_Label = tk.Button(root, text="Profilometry tool (str)", font=('Helvetica', letter_size),
                                    command=lambda: slect_profilometry(profilometry_tool_Txt.get()))
    profilometry_tool_Label.place(x=step3Label.winfo_x(), y=step3Label.winfo_y() + step3Label.winfo_height())
    root.update()
    profilometry_tool_Txt = tk.Entry(root, width=15, font=("Helvetica", letter_size))
    profilometry_tool_Txt.place(x=profilometry_tool_Label.winfo_x() + profilometry_tool_Label.winfo_width() + 5,
                            y=profilometry_tool_Label.winfo_y())
    root.update()
    # Initailze Features
    initWafer = tk.Button(root, text="Initialize Wafer", font=('Helvetica', letter_size),
                                    command=lambda: init_Wafer())
    initWafer.place(x=step3Label.winfo_x(),
                              y=profilometry_tool_Label.winfo_y() + profilometry_tool_Label.winfo_height() + 5)
    root.update()

    ####### Step 4 #################
    step4Label = tk.Label(root, text="Step 4: Wafer data evaluation", font=('Helvetica', title_size))
    step4Label.place(x=step3Label.winfo_x(), y=initWafer.winfo_y()+initWafer.winfo_height()+5)
    root.update()

    #evaluate_signal_processing: (True or False) Optionally, plot various signal processing methods to smooth profilometry data.
    evaluate_sig_pros_Label = tk.Label(root, text="Smooth profilometry data", font=('Helvetica', letter_size))
    evaluate_sig_pros_Label.place(x=step3Label.winfo_x(), y=step4Label.winfo_y() + step4Label.winfo_height())
    root.update()
    evaluate_sig_pros_OnOff = tk.Button(root, image=off, font=("Helvetica", letter_size),command=lambda: sig_prosOnOff())
    evaluate_sig_pros_OnOff.place(x=evaluate_sig_pros_Label.winfo_x() + evaluate_sig_pros_Label.winfo_width() + 5,
                                y=evaluate_sig_pros_Label.winfo_y())
    root.update()

    # perform_rolling_on: (True or False) Optionally, perform a rolling operation to smooth data; [[3, 'b1', 25]]
    perform_rolling_Label = tk.Label(root, text="Rolling operation", font=('Helvetica', letter_size))
    perform_rolling_Label.place(x=step3Label.winfo_x(), y=evaluate_sig_pros_OnOff.winfo_y() + evaluate_sig_pros_OnOff.winfo_height())
    root.update()
    perform_rolling_OnOff = tk.Button(root, image=off, font=("Helvetica", letter_size),
                                        command=lambda: rolling_OnOff())
    perform_rolling_OnOff.place(x=perform_rolling_Label.winfo_x() + perform_rolling_Label.winfo_width() + 5,
                                  y=perform_rolling_Label.winfo_y())
    root.update()

    #    peak_rel_height: if 'lambda_peak_rel_height' is defined, then this variable isn't used.
    peak_rel_height_Label = tk.Label(root, text="Real peak height microns (float)", font=('Helvetica', letter_size))
    peak_rel_height_Label.place(x=step3Label.winfo_x(), y=perform_rolling_OnOff.winfo_y() + perform_rolling_OnOff.winfo_height())
    root.update()
    peak_rel_height_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    peak_rel_height_Txt.place(x=peak_rel_height_Label.winfo_x() + peak_rel_height_Label.winfo_width() + 5,
                            y=peak_rel_height_Label.winfo_y())
    root.update()

    #    downsample: downsample the data to reduce computation.
    downsample_Label = tk.Label(root, text="Downsample the data (float)", font=('Helvetica', letter_size))
    downsample_Label.place(x=step3Label.winfo_x(),
                                y=peak_rel_height_Label.winfo_y() + peak_rel_height_Label.winfo_height())
    root.update()
    downsample_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    downsample_Txt.place(x=downsample_Label.winfo_x() + downsample_Label.winfo_width() + 5,
                              y=downsample_Label.winfo_y())
    root.update()

    ################# Next side #####################
    # width_rel_radius: radial distance, beyond the target radius, to evaluate.
    width_rel_radius_Label = tk.Label(root, text="Evaluate radius width (float)", font=('Helvetica', letter_size))
    width_rel_radius_Label.place(x=peak_rel_height_Txt.winfo_x()+peak_rel_height_Txt.winfo_width()+10,
                           y=10)
    root.update()
    width_rel_radius_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    width_rel_radius_Txt.place(x=width_rel_radius_Label.winfo_x() + width_rel_radius_Label.winfo_width() + 5,
                         y=10)
    root.update()

    #    prominence: the relative height of a peak compared to adjacent peaks.
    prominence_Label = tk.Label(root, text="Relative peak height (float)", font=('Helvetica', letter_size))
    prominence_Label.place(x=width_rel_radius_Label.winfo_x(),
                                 y=width_rel_radius_Label.winfo_y()+width_rel_radius_Label.winfo_height())
    root.update()
    prominence_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    prominence_Txt.place(x=prominence_Label.winfo_x() + prominence_Label.winfo_width() + 5,
                               y=prominence_Label.winfo_y())
    root.update()
    #fit_func: function that is fit to the profilometry profile to find its peak (center)
    fit_func_Label = tk.Label(root, text="Fit profilometry profile (str)", font=('Helvetica', letter_size))
    fit_func_Label.place(x=prominence_Label.winfo_x(),
                           y=prominence_Label.winfo_y() + prominence_Label.winfo_height())
    root.update()
    fit_func_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    fit_func_Txt.place(x=fit_func_Label.winfo_x() + fit_func_Label.winfo_width() + 5,
                         y=fit_func_Label.winfo_y())
    root.update()

    #    plot_width_rel_target_radius: plot radius = target_radius * plot_width_rel_target_radius
    plot_width_rel_target_radius_Label = tk.Label(root, text="Modify plot func (float)", font=('Helvetica', letter_size))
    plot_width_rel_target_radius_Label.place(x=fit_func_Label.winfo_x(),
                         y=fit_func_Label.winfo_y() + fit_func_Label.winfo_height())
    root.update()
    plot_width_rel_target_radius_Txt = tk.Entry(root, width=10, font=("Helvetica", letter_size))
    plot_width_rel_target_radius_Txt.place(x=plot_width_rel_target_radius_Label.winfo_x() + plot_width_rel_target_radius_Label.winfo_width() + 5,
                       y=plot_width_rel_target_radius_Label.winfo_y())
    root.update()

    #    save_profilometry_processing_figures = True
    save_profilometry_processing_figures_Label = tk.Label(root, text="Save the processing figures (bool)", font=('Helvetica', letter_size))
    save_profilometry_processing_figures_Label.place(x=plot_width_rel_target_radius_Label.winfo_x(),
                                             y=plot_width_rel_target_radius_Label.winfo_y() + plot_width_rel_target_radius_Label.winfo_height())
    root.update()
    save_profilometry_processing_figures_OnOff = tk.Button(root, image=off, font=("Helvetica", letter_size),
                                        command=lambda: save_profi_proc_fig_OnOff())
    save_profilometry_processing_figures_OnOff.place(
        x=save_profilometry_processing_figures_Label.winfo_x() + save_profilometry_processing_figures_Label.winfo_width() + 5,
        y=save_profilometry_processing_figures_Label.winfo_y())
    root.update()

    #     save_merged_profilometry_data = True
    save_merged_profilometry_data_Label = tk.Label(root, text="Save the merged data (bool)",
                                                          font=('Helvetica', letter_size))
    save_merged_profilometry_data_Label.place(x=plot_width_rel_target_radius_Label.winfo_x(),
                                                     y=save_profilometry_processing_figures_OnOff.winfo_y() + save_profilometry_processing_figures_OnOff.winfo_height())
    root.update()
    save_merged_profilometry_data_OnOff = tk.Button(root, image=off, font=("Helvetica", letter_size),
                                                           command=lambda: save_merged_prof_data_OnOff())
    save_merged_profilometry_data_OnOff.place(
        x=save_merged_profilometry_data_Label.winfo_x() + save_merged_profilometry_data_Label.winfo_width() + 5,
        y=save_merged_profilometry_data_Label.winfo_y())
    root.update()

    # Evaluate process button
    initWafer = tk.Button(root, text="Initialize Wafer", font=('Helvetica', letter_size),
                                    command=lambda: eval_process())
    initWafer.place(x=plot_width_rel_target_radius_Label.winfo_x(),
                              y=save_merged_profilometry_data_OnOff.winfo_y() + save_merged_profilometry_data_OnOff.winfo_height() + 5)
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
    Output = tk.Text(root, height=20,width=40,bg="light cyan",font=("Helvetica", letter_size))
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