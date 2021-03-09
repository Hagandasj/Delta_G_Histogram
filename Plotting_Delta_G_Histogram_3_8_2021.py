# -*- coding: utf-8 -*-
"""
@author: hagan

Notes:
    Plot Kernel Density Estimation (KDE) Figures.
    
    This script takes ~30 minutes for a nanopore experiment collected over
    20 minutes at 100 kHz acquisition rate.
"""

#%%

""" ----- Variables to Change ----- """

# Name of files to analyze
files = "JH_L01_11_9_2020_pH7_1MKCl_Azo_1_UV365nm_m200mV_blank_1210"

# Path to the Directory where the data is stored.
path = "D:\\Research\\Data\\Azo_Switch_Data\\November11_2020\\Azo\\1"
# February17_2021\\Azo"

# Data Colelction Frequency
acquisition_rate = 100000

# Times stamp for data to run
start_time = 1
end_time = "Whole_Run" # Inputting a string will plot the whole dataset

# Standardize Data: Sets STD = 1 and Mean = 0
standardize_data = True # False == No; True == Yes

# Set X-Axis Limits
set_x_axis = True # False == No; True == Yes
xlim_values = [-5, 5]

#%%

""" ----- Import Section ----- """

# Modules that are required to run this script
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import time
from sklearn import preprocessing
import seaborn as sns

# Setting styles for seaborn plots
sns.set_style(style = "darkgrid")
sns.set(font_scale = 2, font = "Arial")

# Setting timer for script
run_time = time.time()

# Setting folder name to be generated
folder_to_add = ["Conductance_KDE_Plots"]

#%%

""" ----- Class and Function Section ----- """

class DeltaGHistogram():
    """ This class will be used to extract data from bin files and create
    delta G histogram. """
    
    def __init__(self, data_file, acq_rate, file_path, make_folder = "Plots_Folder", standardize = True, start_t = 1, end_t = "Whole_Run", set_xlim = False, xlim_val = [-4, 4]):
        self.data_file = data_file
        self.acq_rate = acq_rate
        self.star_t = start_t
        self.end_t = end_t
        self.appl_voltage = 0
        self.file_path = file_path
        self.make_folder = make_folder
        self.standardize = standardize
        self.set_xlim = set_xlim
        self.xlim_val = xlim_val
        
        
    def plotHistogram(self):
        """ Function that extracts data and plots everything """
        # Opening the data from bin files
        with open(os.path.join(self.file_path, f"{self.data_file}.bin")) as working_file: # assign working path to location of .bin files

            # Generating a folder where plots will be saved
            try:
                os.mkdir(os.path.join(self.file_path, self.make_folder))
            except OSError as error:
                print(error)
                
            # Extracting data as a double for most accuracy in final values
            raw_data = np.fromfile(working_file, dtype = np.dtype('>d')) 
            
            # Close the opened data
            working_file.close()

        # Calculating applied voltage from bin file data (based on the first 200 datapoints)
        self.appl_voltage = math.ceil(sum(raw_data[5:205:2]) / len(raw_data[5:205:2]))
        
        # Checks if a number or text file is set for the end time to know how much data to analyze.
        if (type(self.end_t) is int) == True or (type(self.end_t) is float) == True:
            end_time = self.end_t 
        elif (type(self.end_t) is str) == True:
            end_time = math.floor(len(raw_data[0::2]) / self.acq_rate)
            
        actual_start = math.floor(self.star_t * self.acq_rate) # formats start time to acquisition rate
        actual_end = math.floor(end_time * self.acq_rate) #  # formats end time to acquisition rate
    
        working_baseline, raw_data = raw_data[0::2][actual_start:actual_end:], []
        
        # Converting current to delta G
        delta_g_values, working_baseline = np.divide(working_baseline, self.appl_voltage), []
        
        # Plotting for standardized data
        if self.standardize == True:
            # Standardizing dataset
            scaled_data, delta_g_values = preprocessing.scale(delta_g_values, with_std = False), []
            print(f"Starting to Plot Data: {time.time() - run_time}")
            fig, ax = plt.subplots(figsize = (9,8))
            
            # SciPy Method
            # x_grid = np.linspace(-4.5, 4.5, 1000)
            # kde = gaussian_kde(scaled_data, bw_method = 0.2 / scaled_data.std(ddof = 1))
            # plt.plot(kde.evaluate(x_grid))
            
            # SciKit Learn Method
            # kde = KernelDensity(kernel = "gaussian", bandwidth = 0.2).fit(scaled_data)
            
            # Seaborn method
            sns.kdeplot(scaled_data, fill = True, palette = "crest", alpha = 0.8, linewidth = 0)
            ax.set(xlabel = "Standardized Conductance")
            
            # Setting limmits for x axis
            if self.set_xlim == True:
                ax.set_xlim(min(self.xlim_val), max(self.xlim_val))
        
        # Plotting for un-standardized data
        elif self.standardize == False:
            print(f"Starting to Plot Data: {time.time() - run_time}")
            
            fig, ax = plt.subplots(figsize = (10,9))
            sns.kdeplot(delta_g_values, fill = True, palette = "crest", alpha = 0.8, linewidth = 0)
            ax.set(xlabel = "Standardized Conductance")
            
            # Setting limmits for x axis
            if self.set_xlim == True:
                ax.set_xlim(min(self.xlim_val), max(self.xlim_val))
        
        # Saving the file section
        if (type(self.end_t) is str) == True:
            elapsed_time = "Whole_Run"
        elif self.end_t - self.star_t < 1:
            elapsed_time = "{:.0f}_m" .format((self.end_t - self.star_t) * 1000)
        else:
            elapsed_time = "{}" .format(self.end_t - self.star_t)
            
        if standardize_data == True:
            standard = "Standardized"
            file_Name = "{}_{}s_{}" .format(self.data_file, elapsed_time, standard)
        else:
            file_Name = "{}_{}s" .format(self.data_file, elapsed_time)
        
        plt.savefig(os.path.join(self.file_path, self.make_folder) + "\\" + file_Name + ".png", dpi = 600)
        
        # Show and close the plots
        plt.show()
        plt.close()


#%%

""" ----- Calling Class and function to plot data ----- """

initialize_data = DeltaGHistogram(data_file = files, acq_rate = acquisition_rate, start_t = start_time, end_t = end_time, file_path = path, make_folder = folder_to_add[0], standardize = standardize_data, set_xlim = set_x_axis, xlim_val = xlim_values)
current_data = DeltaGHistogram.plotHistogram(initialize_data)

#%%

""" ----- Timing Section ----- """

print("This script took {:.5f} seconds to run" .format(time.time() - run_time))