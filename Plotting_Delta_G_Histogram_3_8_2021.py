# -*- coding: utf-8 -*-
"""
@author: hagan

Notes:
    Plotting a standardized histogram for conductance values of current datasets.
"""

#%%

""" ----- Variables to Change ----- """

# Name of files to analyze
files = "JH_L01_2_15_2021_pH7_1MKCl_Azo_2_UV365nm_1_p200mV_10uL_05_uguL_3kbpDNA_GS_1230"

# Path to the Directory where the data is stored.
path = "D:\\Research\\Data\\Azo_Switch_Data\\February17_2021\\Azo"

# Data Colelction Frequency
acquisition_rate = 100000

# Times stamp for data to run
start_time = 1
end_time = "Whole_Run" # Add text to add all data to last data point

# Standardize Data: Sets STD = 1 and Mean = 0
standardize_data = True # False == No; True == Yes

#%%

""" ----- Import Section ----- """

import matplotlib.pyplot as plt
import os
import numpy as np
import math
import time
from sklearn import preprocessing
from scipy.stats import gaussian_kde
import seaborn as sns

sns.set_style("darkgrid")
sns.set(font_scale = 2)
run_time = time.time()
folder_to_add = ["DeltaG_Plots"]

#%%

""" ----- Class and Function Section ----- """

class DeltaGHistogram():
    """ This class will be used to extract data from bin files and create
    delta G histogram. """
    
    def __init__(self, data_file, acq_rate, start_t, end_t, file_path, make_folder, standardize):
        self.data_file = data_file
        self.acq_rate = acq_rate
        self.star_t = start_t
        self.end_t = end_t
        self.appl_voltage = 0
        self.file_path = file_path
        self.make_folder = make_folder
        self.bins = 0
        self.standardize = standardize
        
        
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
            raw_data = np.fromfile(working_file, dtype = np.dtype('>d')) # use numpy module to assign current trace data to [raw_data]
            
            print(f"1: {time.time() - run_time}")
            # Close the opened data
            working_file.close()

        # Calculating applied voltage from bin file data (based on the first 100 datapoints)
        self.appl_voltage = math.ceil(sum(raw_data[5:105:2]) / len(raw_data[5:105:2]))
        
        print(f"2: {time.time() - run_time}")
        if (type(self.end_t) is int) == True or (type(self.end_t) is float) == True:
            end_time = self.end_t # [sec]
        elif (type(self.end_t) is str) == True:
            end_time = math.floor(len(raw_data[0::2]) / self.acq_rate)
            
        actual_start = math.floor(self.star_t * self.acq_rate) # formats start time to acquisition rate
        actual_end = math.floor(end_time * self.acq_rate) #  # formats end time to acquisition rate
    
        working_baseline, raw_data = raw_data[0::2][actual_start:actual_end:], []
        print(f"3: {time.time() - run_time}")
        # Converting current to delta G
        delta_g_values, working_baseline = np.divide(working_baseline, self.appl_voltage), []
        print(f"4: {time.time() - run_time}")
        # bin_number = int(math.log2(len(delta_g_values)) + 1) + 20
        
        if self.standardize == True:
            scaled_data, delta_g_values = preprocessing.scale(delta_g_values), []
            print(f"5: {time.time() - run_time}")
            fig, ax = plt.subplots(figsize = (9,8))
            x_grid = np.linspace(-4.5, 4.5, 1000)
            kde = gaussian_kde(scaled_data, bw_method = 0.2 / scaled_data.std(ddof = 1))
            plt.plot(kde.evaluate(x_grid))
            # kde = KernelDensity(kernel = "gaussian", bandwidth = 0.2).fit(scaled_data)
            # sns.kdeplot(scaled_data, fill = True, palette = "crest", alpha = 0.8, linewidth = 0)
            ax.set(xlabel = "Standardized Conductance")
            print(f"6: {time.time() - run_time}")
        
        elif self.standardize == False:
            fig, ax = plt.subplots(figsize = (9,8))
            sns.kdeplot(delta_g_values, fill = True, palette = "crest", alpha = 0.8, linewidth = 0)
            ax.set(xlabel = "Standardized Conductance")
        
        plt.show()
        plt.close()


#%%

""" ----- Calling Class and function to plot data ----- """

initialize_data = DeltaGHistogram(data_file = files, acq_rate = acquisition_rate, start_t = start_time, end_t = end_time, file_path = path, make_folder = folder_to_add[0], standardize = standardize_data)
current_data = DeltaGHistogram.plotHistogram(initialize_data)

#%%

""" ----- Timing Section ----- """

print("This script took {:.5f} seconds to run" .format(time.time() - run_time))