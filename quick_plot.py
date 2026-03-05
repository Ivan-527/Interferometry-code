###################################################################################################
### A little program that reads in the data and plots it
### You can use this as a basis for you analysis software 
###################################################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
import read_data_results3 as rd
import scipy.optimize as opt
from scipy.signal import find_peaks



#Step 1 get the data and the x position

files= str(input()).split(",") #this is the data

def extract(file):    
    results = rd.read_data3('data/' + file)
    
    y1 = np.array(results[0])[1:]
    y2 = np.array(results[1])
    
    x=np.array(results[5])
    
    plt.plot(x,np.log(y2),'o-', markersize=0.1)
    
    return x, y2

def sine(x, a, b, c):
    return a * np.sin(b * x + c) + 8.68e6


extract(files[0]+".txt")

#print(f'rel error: {np.sqrt(p_cov[1, 1])/p_opt[1]}')

    
#plt.figure("Detector 1")
#plt.plot(x,y1,'o-')
#plt.xlabel("Position microsteps")
#plt.ylabel("Signal 1")
#plt.savefig("figures/quick_plot_detector_1.png")

#plt.figure("Detector 2")
#plt.plot(x, y)
#.xlim(0,1e5)

plt.xlabel("Position microsteps")
plt.ylabel("Signal 2")
plt.savefig("figures/quick_plot_detector_2.png")


plt.show()
