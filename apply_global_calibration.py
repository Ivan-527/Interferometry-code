import sys
import read_data_results3 as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#import iminuit as im
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi
import Read_spectrum as rdsp

def data(file):
    #Step 1 get the data and the x position
    results = rd.read_data3('data/' + file + '.txt')
    
    # Describe the global calibration used (from either Task 6, or crossing_points.py)
    metres_per_microstep = 3.66e-11 # metres
    # if from Task 6, need to multiple by factor of 2 to account for the mirror movement to path difference conversion
    #metres_per_microstep = 2.0*metres_per_microstep
    
    # We are only going to need data from one detector
    # make sure the index is the right one for your data!
    y1 = np.array(results[1])
    
    # get x-axis data from the results
    # factor of 2 arrives because we assume conversion does not include path difference to motor distance factor
    x = np.array(results[5])*metres_per_microstep
    
    # centre the y-axis on zero by either subtracting the mean
    # or using the Butterworth filter
    y1 = y1 - y1.mean()
    
    # Butterworth filter to correct for offset
    #filter_order = 2
    #freq = 1 #cutoff frequency
    #sampling = 50 # sampling frequency
    #sos = signal.butter(filter_order, freq, 'hp', fs=sampling, output='sos')
    #filtered = signal.sosfilt(sos, y1)
    #y1 = filtered
    
    
    # Hann Window to reduce artifacts
    y2 = y1 * np.hanning(len(y1))
    
    
    # Cubic Spline part - the FFT requires a regular grid on the x-axis
    N = int(1e7) # these are the number of points that you will resample - try changing this and look how well the resampling follows the data.
    xs = np.linspace(x[0], x[-1], N) # x-axis to resample onto
    y = y2[:len(x)] # make sure y axis has same length as x 
    cs = spi.CubicSpline(x, y) # construct the cubic spline function
    
    
    # step 5 FFT to extract spectra
    yf1=spf.fft(cs(xs))
    xf1=spf.fftfreq(len(xs)) # setting the correct x-axis for the fourier transform. Osciallations/step  
    xf1=spf.fftshift(xf1) #shifts to make it easier (google if interested)
    yf1=spf.fftshift(yf1)
    xx1=xf1[int(len(xf1)/2+1):len(xf1)]
    
    # convert x-axis to meaningful units - wavelength
    distance = xs[1:]-xs[:-1]
    # rather than the amplitude
    repx1 = distance.mean()/xx1
    
    return [file, abs(repx1), abs(yf1[int(len(xf1)/2+1):len(xf1)])]


def grating(file):
    results = rdsp.read_data4('data/' + file + '.txt')
    return [file, results[0], results[1]]


files_i = str(input()).split(",") #data from interferogram
files_g = str(input()).split(",") #data from grating
plt.figure('Spectrum using global calibration FFT')
title = 'Data from: '


def plots(files, title, x):
    for file in files:
        if x == 'i':
            item = data(file)
        if x == 'g':
            item = grating(file)
        title += item[0]
        if file != files[:-1]:
            title += ' and '
        print(item[1][np.argmax(item[2])])
        return [item[1], np.array(item[2])/np.array(item[2]).max(), item[0]] #original plotting function
        
def plots_norm(files, title, x):
    for file in files:
        if x == 'i':
            item = data(file)
        if x == 'g':
            item = grating(file)
        title += item[0]
        if file != files[:-1]:
            title += ' and '
        plt.plot(item[1], np.log(item[2]), label=item[0]) #original plotting function
        print(item[1][np.argmax(item[2])])

def atten(files, title):
    item_1 = data(files[0])
    title += item_1[0] + ' and '
    plt.plot(item_1[1], item_1[2], label = item_1[0])
    yellow_y_max = item_1[2].max()
    yellow_x_max = item_1[1][np.argmax(item_1[2])] #yellow peak intensity
    
    item_2 = data(files[1])
    title += item_2[0]
    plt.plot(item_2[1], item_2[2], label = item_2[0])
    x2 = item_2[1]
    y2 = item_2[2]
    
    if x2[0] > x2[-1]:   # decreasing
        x2 = x2[::-1]
        y2 = y2[::-1]
    
    white_equiv = np.interp(yellow_x_max, x2, y2)
    print(white_equiv, yellow_y_max, yellow_x_max)
    atten_ratio = white_equiv/yellow_y_max 
    print(atten_ratio)
    print(np.sum(item_2[2])/np.sum(item_1[2]))

iplot = plots(files_i, title, 'i')
gplot = plots(files_g, title, 'g')
plt.plot(iplot[0], iplot[1], label = iplot[2])
plt.plot(gplot[0], gplot[1], label = gplot[2])
new_g = np.interp(iplot[0], gplot[0], gplot[1]) #interpolate the 'correct data' to get enough data points
plt.plot(iplot[0], iplot[1] - new_g, label = 'difference')
#plots_norm(files_i, title, 'i')
#plots_norm(files_i, title, 'g')
#atten(file_is, title)
plt.title(title)
plt.xlim(3.5e-7, 8e-7)
#plt.xlim(0, 4e-7)
#plt.ylim(0, 0.8e9)
plt.xlabel('Wavelength (m)')
plt.ylabel('Intensity (a.u.)')
plt.legend()
plt.grid()
plt.savefig('figures/temp_data.png')

plt.show()

