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
files = "JH_L01_2_15_2021_pH7_1MKCl_Azo_2_UV365nm_1_p200mV_10uL_05_uguL_3kbpDNA_GS_1212"

# Path to the Directory where the data is stored.
path = "D:\\Research\\Data\\Azo_Switch_Data\\February17_2021\\Azo"
# February17_2021\\Azo

# Data Colelction Frequency
acquisition_rate = 100000

# Times stamp for data to run
start_time = 1
end_time = 2 # Inputting a string will plot the whole dataset




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
# sns.set_style(style = "darkgrid")
sns.set(font_scale = 2, font = "Arial")

# Setting timer for script
run_time = time.time()

# Setting folder name to be generated
folder_to_add = ["Conductance_KDE_Plots", "Event_Plots"]

#%%

""" ----- Class and Function Section ----- """

class PlotAnalyzeBaseline():
    """ This class will be used to extract data from bin files and create different plots to 
    display and analyze the data collected. """
    
    def __init__(self, data_file, acq_rate, file_path, start_t = 1, end_t = "Whole_Run"): # , xlim_val = [-4, 4]):
        """ data_file - Name of file to be opened
            acq_rate - Aqcuisition rate for the data collected
            standardize - Tells the program to standardize the dataset
            start_t - Time in the dataset to start at (in seconds)
            end_t - Time in the dataset to end (in seconds). If a string is used it will collect all data to the end
            set_"""
        # Variables provided by user
        self.data_file = data_file
        self.acq_rate = acq_rate
        self.start_t = start_t
        self.end_t = end_t
        self.file_path = file_path
        # self.xlim_val = xlim_val
        
        # Variables extracted from data
        self.appl_voltage = 0
        self.ext_data = []
        self.working_baseline = []
        
        
    def extractBinData(self):
        """ This function is called to extract data from a bin file. For the data from the dwyer
        lab it outputs this as a 2D array with the even numbers being currents and the odd numbers 
        being voltages. """
        # Opening the data from bin files
        with open(os.path.join(self.file_path, f"{self.data_file}.bin")) as working_file: # assign working path to location of .bin files

            
                
            # Extracting data as a double for most accuracy in final values
            self.ext_data = np.fromfile(working_file, dtype = np.dtype('>d')) 
            
        # Close the opened data
        working_file.close()
        
    
    def dataToUse(self):
        """ dataToUse makes another array to use onthe the data that the user wants to extract """
        # Checks if a number or text file is set for the end time to know how much data to analyze.
        if (type(self.end_t) is int) == True or (type(self.end_t) is float) == True:
            end_time = self.end_t 
        elif (type(self.end_t) is str) == True:
            end_time = math.floor(len(self.ext_data[0::2]) / self.acq_rate)
            
        actual_start = math.floor(self.start_t * self.acq_rate) # formats start time to acquisition rate
        actual_end = math.floor(end_time * self.acq_rate) #  # formats end time to acquisition rate
    
        self.working_baseline, self.ext_data = self.ext_data[0::2][actual_start:actual_end:], []
        
    
    def appliedVoltage(self):
        """ appliedVoltage calculates what the applied volatge during the experiment was by looking
        at the extracted voltage data from the bin file. This only works for a datasets that had a
        constant voltage applied throught the experiment. """
        # Calculating applied voltage from bin file data (based on the first 200 datapoints)
        self.appl_voltage = math.ceil(sum(self.ext_data[5:205:2]) / len(self.ext_data[5:205:2]))
        
        
    def standardizeOutput(self, dataset):
        """ StandardizeOutput takes a 1D array of data and standardizes it by converting Mean == 0, but
        it does not effect that standard deviation
                dataset - 1D array that will be standardized. """
        return (preprocessing.scale(dataset, with_std = False))
        
        
        
    def plotHistogram(self, xmin = None, xmax = None):
        """ Function that plots a Kernel Density Estimator plot for the extracted data at a specific
        time scale specified in the when initializing the class. 
                xmin - (Optional) Sets the minimum x value displayed for the x axis.
                xmax - (Optional) Sets the maxinum x value displayed for the x axis."""
        # Extracts all data from bin file
        self.extractBinData()
        
        # Calculates the applied voltage form bin file data (based on first 200 datapoints).
        self.appliedVoltage()
        
        # Uses dataToUse function to truncate total dataset
        self.dataToUse()
        
        # Converting datasets measured current to delta G
        delta_g_values = np.divide(self.working_baseline, self.appl_voltage)
        
        # Plotting for standardized data
        scaled_data, delta_g_values = self.standardizeOutput(delta_g_values), []
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
        ax.set_xlim(xmin, xmax)
        
        # Saving the file section
        if (type(self.end_t) is str) == True:
            elapsed_time = "Whole_Run"
        elif self.end_t - self.start_t < 1:
            elapsed_time = "{:.0f}_m" .format((self.end_t - self.start_t) * 1000)
        else:
            elapsed_time = "{}" .format(self.end_t - self.start_t)
            
        standard = "Standardized"
        file_Name = "{}_{}s_{}" .format(self.data_file, elapsed_time, standard)
            
        # Generating a folder where plots will be saved
        try:
            os.mkdir(os.path.join(self.file_path, "Conductance_KDE_Plots"))
        except OSError as error:
            print(error)
        
        plt.savefig(os.path.join(self.file_path, "Conductance_KDE_Plots") + "\\" + file_Name + ".png", dpi = 600)
        
        # Show and close the plots
        plt.show()
        plt.close()
        
        
    def derivativePlot(self, xmin = None, xmax = None):
        """ derivativePlot takes the derivative of the data and plots it. 
                xmin - (Optional) Sets the minimum x value displayed for the x axis.
                xmax - (Optional) Sets the maxinum x value displayed for the x axis. """
        # Extracts all data from bin file
        self.extractBinData()
        
        # Uses dataToUse function to truncate total dataset
        self.dataToUse()
        
        # Taking the derivative of the data provided
        der_data = np.gradient(self.working_baseline) # , (1 / self.acq_rate))
        # stand_der_data = self.standardizeOutput(der_data)
        fig, ax = plt.subplots(figsize = (9,8))
        
        plt.hist(der_data, bins = 100)
        ax.set_xlim(xmin, xmax)
        plt.show()
        plt.close()
        

    def plotBaseline(self, ymin = None, ymax = None):
        """ plotBaseline makes a line plot of the baseline data extracted
        from the bin file. """
        # Extracts all data from bin file
        self.extractBinData()
        
        # Uses dataToUse function to truncate total dataset
        self.dataToUse()
        
        # Creates a folder for plotting baseline
        try:
            os.mkdir(os.path.join(path, "Event_Plots"))
        except OSError as error:
            print(error)
            
        # Checks if a number or text file is set for the end time to know how much data to analyze.
        if (type(self.end_t) is int) == True or (type(self.end_t) is float) == True:
            end_time = self.end_t 
        elif (type(self.end_t) is str) == True:
            end_time = math.floor(len(self.ext_data[0::2]) / self.acq_rate)
            
        # Calculates the time values for x axis based on acquisition rate and number of datapoints
        current_Trace_Time = np.arange((self.start_t - self.start_t),((math.floor(end_time * self.acq_rate) - math.floor(self.start_t * self.acq_rate)) / self.acq_rate), (1/self.acq_rate))
        
        fig, ax = plt.subplots(figsize = (25,7))  
    
        plt.plot(current_Trace_Time, self.working_baseline, linewidth = 2, color = 'black')
        plt.locator_params(nbins = 7)
        
        ax.set_xlabel("Time (s)", size = 25, fontname = "Arial") #    Time (s)
        ax.set_ylabel("Current(pA)", size = 25, fontname = "Arial")
        
        ax.set_ylim(ymin,ymax)
        
        # Sets parameters for axis labels
        plt.xticks(fontsize = 22, fontname = 'Arial')
        plt.yticks(fontsize = 22, fontname = 'Arial')
        
        # Sets parametesr for plot formatting
        ax.spines['top'].set_visible(False) # Removes top and right lines of plot
        ax.spines['right'].set_visible(False)
        
        ax.spines['left'].set_linewidth(3) # Makes the boarder bold
        ax.xaxis.set_tick_params(width = 3)
        ax.spines['bottom'].set_linewidth(3) # Makes the boarder bold
        ax.yaxis.set_tick_params(width = 3)
        
        fig.tight_layout() # help keep axis titles on the figure
        
        plt.savefig(os.path.join(self.file_path, "Event_Plots") + "\\" + self.data_file[0] + "_" + str(self.start_t) + "_" + str(self.end_t) + ".png", dpi = 1200)
        
        # Displays figure and closes the figure to save memory space
        plt.show()
        plt.close()

#%%

""" ----- Calling Class and function to plot data ----- """

initialize_data = PlotAnalyzeBaseline(data_file = files, acq_rate = acquisition_rate, start_t = start_time, end_t = end_time, file_path = path) # , xlim_val = xlim_values)
#%%
current_data = initialize_data.plotHistogram(xmin = -5, xmax = 5)
#%%
deriv_data = initialize_data.derivativePlot(-100, 100)
#%%

initialize_data.plotBaseline()

#%%

""" ----- Timing Section ----- """

print("This script took {:.5f} seconds to run" .format(time.time() - run_time))