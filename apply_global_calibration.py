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

def plots(files, file_type, choice):
    out = []
    for file in files:
        if file_type == 'i':
            item = data(file)
        elif file_type == 'g':
            item = grating(file)
        else:
            raise ValueError("c must be 'i' or 'g'")

        x = item[1]
        y = np.array(item[2])
        
        if choice == 'd':
            y = y/y.max()
        
        name = item[0]

        plt.plot(x, y, label=name)
        out.append(([x, y], name))

    return out

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

plt.figure('Spectrum using global calibration FFT')
plt.title('Normalized Intensity against Wavelength')

def diff():
    while True:
        files = str(input()).split(" ")
        a, name1 = plots([files[0]], 'i', 'd')[0]
        b, name2 = plots([files[1]], 'g', 'd')[0]
        new_b = np.interp(a[0], b[0], b[1]) #interpolate the 'correct data' to get enough data points
        plt.plot(a[0], a[1] - new_b, label = 'Difference between ' + name1 + ' and ' + name2)
        cont = str(input())
        if cont == '':
            continue
        else:
            break

def compare():
    file_type = ''
    while True:
        file_type = input().strip()
        if file_type == 'stop':
            break
    
        files = input().split(" ")
        plots(files, file_type, 'c')
        
#diff()
compare()

plt.figure('Spectrum using global calibration FFT')
plt.title('Normalized Intensity against Wavelength')
plt.xlim(3.5e-7, 8e-7)
plt.xlabel('Wavelength (m)')
plt.ylabel('Normalized Intensity')
plt.legend()
plt.grid()
plt.show()
