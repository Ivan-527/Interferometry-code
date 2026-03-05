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
    

def sine(x, a, b, c):
    return a * np.sin(b * x + c) + 8.68e6



#peaks, _ = find_peaks(y2) #guess fits
#period_guess = np.mean(np.diff(x[peaks]))
#b_guess = 1 / period_guess
#a_guess = 4e4
#b_guess = 4e-4
#c_guess = 5.35#np.pi
#d_guess = (np.max(y2) - np.min(y2))/2 + np.min(y2)

#p_opt, p_cov = opt.curve_fit(sine, x, y2, [a_guess, b_guess, c_guess], maxfev=10000)
#print(f'{p_opt[1]}', f'{np.sqrt(p_cov[1, 1])}')
#y = sine(x, *p_opt)

#print(f'rel error: {np.sqrt(p_cov[1, 1])/p_opt[1]}')

    
#plt.figure("Detector 1")
#plt.plot(x,y1,'o-')
#plt.xlabel("Position microsteps")
#plt.ylabel("Signal 1")
#plt.savefig("figures/quick_plot_detector_1.png")

#plt.figure("Detector 2")
#plt.plot(x, y)
#plt.xlim(-0.1e7,0.1e7)

plt.xlabel("Position microsteps")
plt.ylabel("Signal 2")
extract(files[0]+".txt")
plt.savefig("figures/quick_plot_detector_2.png")


plt.show()
