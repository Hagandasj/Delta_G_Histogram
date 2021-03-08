# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:01:20 2021

@author: hagan
"""

# Plotting the histogram of delta G data


""" ----- Variables to Change ----- """

files = ["JH_L01_2_15_2021_pH7_1MKCl_Azo_2_UV365nm_1_White_4_p200mV_blank_1737"]


# MAKE A FOLDER TO SAVE THE EVENT ANALYSIS INFO INTO!!! cope path and enter below with "/analysis_parameters" added to the end, or a more specific title if desired
path = 'E:\\Research\\Data\\2021\\February17_2021\\Azo' 
folder = ['Event_Plots']


# Data Colelction Frequency
acquisition_rate = 100000


# Times to run:
set_ylim = 0 # 0 = No; 1 = Yes
start_time = 165
end_time = 370 # "Whole_Run" #"Plot_the_whole_thing" <<< Uncomment this to run the whole baseline


# Set Y-Min
set_ylim = 0 # 0 = No; 1 = Yes
ymin = 3000 # 750
ymax = 5000


#%%

""" ----- Import Section ----- """




class DeltaGHistogram():
    
    import matplotlib.pyplot as plt
    import os
    
    def __init__(self, start_t, end_t, x_lim, y_lim, appl_voltage, col_current, file_path, make_folder):
        self.starT_t = start_t
        self.end_t = end_t
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.appl_voltage = 0
        self.col_current = []
        self.file_path = file_path
        self.make_folder = make_folder
        
    def plotHistogram(self):
        # Opening the data from bin files
        with open(os.path.join(path, analysis_Title + '.bin')) as f: # assign working path to location of .bin files

        # Generating a 
        try:
            os.mkdir(os.path.join(path, folder[0]))
        except OSError as error:
            print(error)
        
        
        
        
        
        
        
#%%

plot_data = DeltaGHistogram()