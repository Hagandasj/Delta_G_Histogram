# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:01:20 2021

@author: hagan
"""

# Plotting the histogram of delta G data


""" ----- Variables to Change ----- """

files = "JH_L01_2_15_2021_pH7_1MKCl_Azo_2_UV365nm_1_White_4_p200mV_blank_1737"


# MAKE A FOLDER TO SAVE THE EVENT ANALYSIS INFO INTO!!! cope path and enter below with "/analysis_parameters" added to the end, or a more specific title if desired
path = 'E:\\Research\\Data\\2021\\February17_2021\\Azo' 
folder = ['Event_Plots']


# Data Colelction Frequency
acquisition_rate = 100000


# Times to run:
set_ylim = 0 # 0 = No; 1 = Yes
start_time = 165
end_time = 370 # "Whole_Run" #"Plot_the_whole_thing" <<< Uncomment this to run the whole baseline


#%%

""" ----- Import Section ----- """

import matplotlib.pyplot as plt
import os
import numpy as np
import math

#%%

""" ----- Class and Function Section ----- """

class DeltaGHistogram():
    """ This class will be used to extract data from bin files and create
    delta G histogram. """
    
    def __init__(self, data_file, acq_rate, start_t, end_t, appl_voltage, col_current, file_path, make_folder):
        self.data_file = data_file
        self.acq_rate = acq_rate
        self.star_t = start_t
        self.end_t = end_t
        self.appl_voltage = 0
        self.col_current = []
        self.file_path = file_path
        self.make_folder = make_folder
        
    def plotHistogram(self):
        # Opening the data from bin files
        with open(os.path.join(self.file_path, f"{self.data_file}.bin")) as working_file: # assign working path to location of .bin files

            # Generating a 
            try:
                os.mkdir(os.path.join(self.file_path, self.make_folder))
            except OSError as error:
                print(error)
                
            raw_data = np.fromfile(working_file, dtype = np.dtype('>d')) # use numpy module to assign current trace data to [raw_data]
            
            working_file.close()
            
        baseline_current = raw_data[0::2] 
        
        
#%%
if set_ylim == 0:
    st = 1
    et = math.floor(len(baseline_current) / frequency)
    
if (type(end_time) is str) == True:
    elapsed_time = "Whole_Run"
    
elif end_Time - start_Time < 1:
    elapsed_time = "{:.0f}_m" .format((end_Time - start_Time) * 1000)
    
else:
    elapsed_time = "{}" .format(end_Time - start_Time)


# File Name
file_Name = "{}_{}s" .format(files[0], elapsed_time)



#%%

plot_data = DeltaGHistogram(data_file = files, acq_rate = , start_t = st, end_t = et, x_lim = , )