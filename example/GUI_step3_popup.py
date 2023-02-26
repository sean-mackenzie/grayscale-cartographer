import tkinter as tk
import pandas as pd
from pathlib import Path
class wafer_init:
    def __init__(self,measurement_methods,measurement_methods_err):
        self.measurement_methods=measurement_methods
        self.measurement_methods_err=measurement_methods_err
    def update_meas_metods(self,new_measurement_methods,new_measurement_methods_err):
        self.measurement_methods=new_measurement_methods
        self.measurement_methods_err=new_measurement_methods_err

def slect_profilometry(profilometry_tool,wafer_step3,letter_size,title_size):
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
    filetype_read_Txt = tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    filetype_read_Txt.place(x=filetype_read_Label.winfo_x() + filetype_read_Label.winfo_width() + 5, y=filetype_read_Label.winfo_y())
    profilo_window.update()

    # x units
    x_units_read_Label = tk.Label(profilo_window, text="X units read (float)", font=("Helvetica", letter_size))
    x_units_read_Label.place(x=filetype_read_Txt.winfo_x() + filetype_read_Txt.winfo_width() + 10, y=filetype_read_Label.winfo_y())  # Apply volt button data button
    profilo_window.update()
    x_units_read_Txt =tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    x_units_read_Txt.place(x=x_units_read_Label.winfo_x() + x_units_read_Label.winfo_width() + 5, y=filetype_read_Label.winfo_y())
    profilo_window.update()

    # y units
    y_units_read_Label = tk.Label(profilo_window, text="Y units read (float)", font=('Helvetica', letter_size))
    y_units_read_Label.place(x=x_units_read_Txt.winfo_x() + x_units_read_Txt.winfo_width() + 5, y=filetype_read_Label.winfo_y())
    profilo_window.update()
    y_units_read_Txt =tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    y_units_read_Txt.place(x=y_units_read_Label.winfo_x() + y_units_read_Label.winfo_width() + 5, y=filetype_read_Label.winfo_y())
    profilo_window.update()

    #file name write
    filetype_write_Label = tk.Label(profilo_window, text="Filetype write (str)", font=("Helvetica", letter_size))
    filetype_write_Label.place(x=10, y=filetype_read_Label.winfo_y()+filetype_read_Label.winfo_height()+5)
    profilo_window.update()
    filetype_write_Txt =tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    filetype_write_Txt.place(x=filetype_read_Label.winfo_x() + filetype_read_Label.winfo_width() + 5,
                            y=filetype_write_Label.winfo_y())
    profilo_window.update()

    # x units
    x_units_write_Label = tk.Label(profilo_window, text="X units read (float)", font=("Helvetica", letter_size))
    x_units_write_Label.place(x=filetype_read_Txt.winfo_x() + filetype_read_Txt.winfo_width() + 10,
                             y=filetype_read_Label.winfo_y()+filetype_read_Label.winfo_height()+5)  # Apply volt button data button
    profilo_window.update()
    x_units_write_Txt =tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    x_units_write_Txt.place(x=x_units_read_Label.winfo_x() + x_units_read_Label.winfo_width() + 5,
                           y=filetype_write_Txt.winfo_y()+5)
    profilo_window.update()

    # y units
    y_units_write_Label = tk.Label(profilo_window, text="Y units read (float)", font=('Helvetica', letter_size))
    y_units_write_Label.place(x=x_units_read_Txt.winfo_x() + x_units_read_Txt.winfo_width() + 5,
                             y=filetype_write_Txt.winfo_y()+5)
    profilo_window.update()
    y_units_write_Txt =tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    y_units_write_Txt.place(x=y_units_read_Label.winfo_x() + y_units_read_Label.winfo_width() + 5,
                           y=filetype_write_Txt.winfo_y()+5)
    profilo_window.update()
    ###### Data etch monitor ################
    header_etch_moitor_Label = tk.Label(profilo_window, text="Data Etch monitor: Header (str)", font=("Helvetica", letter_size))
    header_etch_moitor_Label.place(x=10, y=y_units_write_Label.winfo_y() + y_units_write_Label.winfo_height() + 5)
    profilo_window.update()
    header_etch_moitor_Txt =tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    header_etch_moitor_Txt.place(x=header_etch_moitor_Label.winfo_x() + header_etch_moitor_Label.winfo_width() + 5,
                                y=header_etch_moitor_Label.winfo_y())
    profilo_window.update()

    # File name type read ethcer
    filetype_read_etcher_Label = tk.Label(profilo_window, text="File type read (float)", font=("Helvetica", letter_size))
    filetype_read_etcher_Label.place(x=header_etch_moitor_Txt.winfo_x() + header_etch_moitor_Txt.winfo_width() + 10,
                              y=header_etch_moitor_Label.winfo_y() )  # Apply volt button data button
    profilo_window.update()
    filetype_read_etcher_Txt =tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    filetype_read_etcher_Txt.place(x=filetype_read_etcher_Label.winfo_x() + filetype_read_etcher_Label.winfo_width() + 5,
                            y=header_etch_moitor_Label.winfo_y() )
    profilo_window.update()

    # File name type write ethcer
    filetype_write_etcher_Label = tk.Label(profilo_window, text="File type write (float)", font=('Helvetica', letter_size))
    filetype_write_etcher_Label.place(x=filetype_read_etcher_Txt.winfo_x() + filetype_read_etcher_Txt.winfo_width() + 5,
                              y=header_etch_moitor_Label.winfo_y() )
    profilo_window.update()
    filetype_write_etcher_Txt =tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    filetype_write_etcher_Txt.place(x=filetype_write_etcher_Label.winfo_x() + filetype_write_etcher_Label.winfo_width() + 5,
                            y=header_etch_moitor_Label.winfo_y() )
    profilo_window.update()
    ###### Data optical ################
    headr_optical_Label = tk.Label(profilo_window, text="Data Optical: Header (str)", font=("Helvetica", letter_size))
    headr_optical_Label.place(x=10, y=filetype_write_etcher_Txt.winfo_y() + filetype_write_etcher_Txt.winfo_height() + 5)
    profilo_window.update()
    header_optical_Txt =tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    header_optical_Txt.place(x=headr_optical_Label.winfo_x() + headr_optical_Label.winfo_width() + 5,
                                y=headr_optical_Label.winfo_y())
    profilo_window.update()

    # File name type read optical
    filetype_read_optical_Label = tk.Label(profilo_window, text="File type read (float)", font=("Helvetica", letter_size))
    filetype_read_optical_Label.place(x=header_optical_Txt.winfo_x() + header_optical_Txt.winfo_width() + 10,
                              y=headr_optical_Label.winfo_y() )  # Apply volt button data button
    profilo_window.update()
    filetype_read_optical_Txt =tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    filetype_read_optical_Txt.place(x=filetype_read_optical_Label.winfo_x() + filetype_read_optical_Label.winfo_width() + 5,
                            y=headr_optical_Label.winfo_y() )
    profilo_window.update()

    # File name type write optical
    filetype_write_optical_Label = tk.Label(profilo_window, text="File type write (float)", font=('Helvetica', letter_size))
    filetype_write_optical_Label.place(x=filetype_read_optical_Txt.winfo_x() + filetype_read_optical_Txt.winfo_width() + 5,
                              y=headr_optical_Label.winfo_y() )
    profilo_window.update()
    filetype_write_optical_Txt =tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    filetype_write_optical_Txt.place(x=filetype_write_optical_Label.winfo_x() + filetype_write_optical_Label.winfo_width() + 5,
                            y=headr_optical_Label.winfo_y() )
    profilo_window.update()

    ###### Data miscellaneous ################
    headr_misc_Label = tk.Label(profilo_window, text="Data Miscellaneous: Header (str)", font=("Helvetica", letter_size))
    headr_misc_Label.place(x=10,
                              y=filetype_read_optical_Txt.winfo_y() + filetype_read_optical_Txt.winfo_height() + 5)
    profilo_window.update()
    header_misc_Txt =tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    header_misc_Txt.place(x=headr_misc_Label.winfo_x() + headr_misc_Label.winfo_width() + 5,
                            y=headr_misc_Label.winfo_y())
    profilo_window.update()

    # File name type read miscellaneous
    filetype_read_misc_Label = tk.Label(profilo_window, text="File type read (float)", font=("Helvetica", letter_size))
    filetype_read_misc_Label.place(x=header_misc_Txt.winfo_x() + header_misc_Txt.winfo_width() + 10,
                                 y=headr_misc_Label.winfo_y())  # Apply volt button data button
    profilo_window.update()
    filetype_read_misc_Txt = tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    filetype_read_misc_Txt.place(x=filetype_read_misc_Label.winfo_x() + filetype_read_misc_Label.winfo_width() + 5,
                               y=headr_misc_Label.winfo_y())
    profilo_window.update()

    # File name type write miscellaneous
    filetype_write_misc_Label = tk.Label(profilo_window, text="File type write (float)", font=('Helvetica', letter_size))
    filetype_write_misc_Label.place(x=filetype_read_misc_Txt.winfo_x() + filetype_read_misc_Txt.winfo_width() + 5,
                                 y=headr_misc_Label.winfo_y())
    profilo_window.update()
    filetype_write_misc_Txt = tk.Text(profilo_window, width=15,height=1, font=("Helvetica", letter_size))
    filetype_write_misc_Txt.place(x=filetype_write_misc_Label.winfo_x() + filetype_write_misc_Label.winfo_width() + 5,
                               y=headr_misc_Label.winfo_y())
    profilo_window.update()

    ##### initial profilometry tool file load #####
    path = str(Path.cwd())
    parameters_path = path + '/software/profilometry_tools.xlsx'
    try:
        parameters = pd.read_excel(parameters_path)  ## seheet_name='Measurement_methods' net tim
    except:
        pass
    try:
        parameters = pd.read_excel(parameters_path)
    except:
        pass
    try:
        data_profile_list = parameters[profilometry_tool].to_list()
        data_etch_monitor_list = parameters['data_etch_monitor'].to_list()
        data_optical_list = parameters['data_optical'].to_list()
        data_misc = parameters['data_misc'].to_list()
        # Profilometry
        filetype_read_Txt.insert("1.0", data_profile_list[0])
        x_units_read_Txt.insert('1.0', data_profile_list[1])
        y_units_read_Txt.insert('1.0', data_profile_list[2])
        filetype_write_Txt.insert('1.0', data_profile_list[3])
        x_units_write_Txt.insert('1.0', data_profile_list[4])
        y_units_write_Txt.insert('1.0', data_profile_list[5])
        # Data etch monitor
        header_etch_moitor_Txt.insert('1.0', data_etch_monitor_list[0])
        filetype_read_etcher_Txt.insert('1.0', data_etch_monitor_list[1])
        filetype_write_etcher_Txt.insert('1.0', data_etch_monitor_list[2])
        # Data Optical
        header_optical_Txt.insert('1.0', data_optical_list[0])
        filetype_read_optical_Txt.insert('1.0', data_optical_list[1])
        filetype_write_optical_Txt.insert('1.0', data_optical_list[2])
        # Data Misceallenouse
        header_misc_Txt.insert('1.0', data_misc[0])
        filetype_read_misc_Txt.insert('1.0', data_misc[1])
        filetype_write_misc_Txt.insert('1.0', data_misc[2])
    except:
        tk.messagebox.showwarning("showwarning", "profilometry_tools.xlsx is wrong load manually")

    def quit():
        parameters = pd.read_excel(parameters_path)
        ### getting the data and convert intot the talbe that we want to have
        filetype_read = filetype_read_Txt.get("1.0",'end-1c')
        parameters._set_value(0,profilometry_tool,filetype_read)
        x_units_read = x_units_read_Txt.get("1.0",'end-1c')
        x_units_read_err = ''
        if x_units_read.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
            x_units_read = float(x_units_read)
        else:
            x_units_read_err = x_units_read + ' is not a float'
        parameters._set_value(1,profilometry_tool,x_units_read)
        y_units_read = y_units_read_Txt.get("1.0",'end-1c')
        y_units_read_err = ''
        if y_units_read.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
            y_units_read = float(y_units_read)
        else:
            y_units_read_err = y_units_read + ' is not a float'
        parameters._set_value(2,profilometry_tool,y_units_read)

        filetype_write = filetype_write_Txt.get("1.0",'end-1c')
        parameters._set_value(3,profilometry_tool,filetype_write)
        x_units_write = x_units_read_Txt.get("1.0",'end-1c')
        x_units_write_err = ''
        if x_units_write.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
            x_units_write = float(x_units_write)
        else:
            x_units_write_err = x_units_write + ' is not a float'
        parameters._set_value(4,profilometry_tool,x_units_write)

        y_units_write = y_units_write_Txt.get("1.0",'end-1c')
        y_units_write_err = ''
        if y_units_write.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit():
            y_units_write = float(y_units_write)
        else:
            y_units_write_err = y_units_read + ' is not a float'
        parameters._set_value(5,profilometry_tool, y_units_write)
        data_profile = {'header': profilometry_tool,
                        'filetype_read': '.' + filetype_read, 'x_units_read': x_units_read,
                        'y_units_read': y_units_read,
                        'filetype_write': '.' + filetype_write, 'x_units_write': x_units_write,
                        'y_units_write': y_units_write,
                        }
        # Data etch moniotr
        header_etch_moitor = header_etch_moitor_Txt.get("1.0",'end-1c')
        parameters._set_value(0,'data_etch_monitor',header_etch_moitor)
        filetype_read_etcher = filetype_read_etcher_Txt.get("1.0",'end-1c')
        parameters._set_value(1,'data_etch_monitor',filetype_read_etcher)
        filetype_write_etcher = filetype_write_etcher_Txt.get("1.0",'end-1c')
        parameters._set_value(2,'data_etch_monitor', filetype_write_etcher)
        data_etch_monitor = {'header': header_etch_moitor, 'filetype_read': '.' + filetype_read_etcher,
                             'filetype_write': '.' + filetype_write_etcher}
        # Data optical
        headr_optical = header_optical_Txt.get("1.0",'end-1c')
        parameters._set_value(0,'data_optical',headr_optical)
        filetype_read_optical = filetype_read_optical_Txt.get("1.0",'end-1c')
        parameters._set_value(1,'data_optical',filetype_read_optical)
        filetype_write_optical = filetype_write_optical_Txt.get("1.0",'end-1c')
        parameters._set_value(2,'data_optical',filetype_write_optical)
        data_optical = {'header': headr_optical, 'filetype_read': '.' + filetype_read_optical,
                        'filetype_write': '.' + filetype_write_optical}
        # Data miscealleous
        headr_misc = header_misc_Txt.get("1.0",'end-1c')
        parameters._set_value(0,'data_misc',headr_misc)
        filetype_read_misc = filetype_read_misc_Txt.get("1.0",'end-1c')
        parameters._set_value(1,'data_misc',filetype_read_misc)
        filetype_write_misc = filetype_write_misc_Txt.get("1.0",'end-1c')
        parameters._set_value(2,'data_misc',filetype_write_misc)
        data_misc = {'header': headr_misc, 'filetype_read': '.' + filetype_read_misc,
                     'filetype_write': '.' + filetype_write_misc}
        measurement_methods = {'Profilometry': data_profile,
                               'Etch Monitor': data_etch_monitor,
                               'Optical': data_optical,
                               'Misc': data_misc,
                               }
        parameters.to_excel(parameters_path)
        measurement_methods_err = {'x_units_read_err': x_units_read_err, 'y_units_read_err': y_units_read_err,
                                   'x_units_write_err': x_units_write_err, 'y_units_write_err': y_units_write_err}
        wafer_step3.update_meas_metods(new_measurement_methods=measurement_methods,new_measurement_methods_err=measurement_methods_err)
        profilo_window.destroy()

    # Lads the parameter from excel sheet
    load_parameters_Button = tk.Button(profilo_window, text="Load .xlsx or .csv", font=('Helvetica', 8),
                                       command=lambda: load_parameters())
    load_parameters_Button.place(x=10, y=headr_misc_Label.winfo_y() + headr_misc_Label.winfo_height() + 5)
    profilo_window.update()
    load_parameters_Text = tk.Text(profilo_window, width=10,height=1, font=("Helvetica", letter_size))
    load_parameters_Text.place(x=5+load_parameters_Button.winfo_x()+load_parameters_Button.winfo_width(),
                               y=headr_misc_Label.winfo_y() + headr_misc_Label.winfo_height() + 5)
    profilo_window.update()
    # Sets the tiped in parameters and kills the popup window
    kill = tk.Button(profilo_window, text="Set measurements methods", font=('Helvetica', 8), command=lambda: quit())
    kill.place(x=10+load_parameters_Text.winfo_x()+load_parameters_Text.winfo_width(), y=headr_misc_Label.winfo_y() + headr_misc_Label.winfo_height() + 5)

    profilo_window.update()