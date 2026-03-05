# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:13:16 2023
This version developed by Laura Hollister
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack as spf
import scipy.interpolate as spi


# Import the data

filename = 'mercury_two_filter.txt'  # change this to your filename
# y_ref and y_data may be the other way around depending on which
# beamsplitter has the green reference laser going into it. If you see a
# single peak at the green line wavelength, swap y_ref and y_data here.

y_data, y_ref = np.loadtxt(filename, unpack=True, usecols=(0, 1))

# Set the reference wavelength in units of nm, here the reference wavelength
# is the the grren spectral line
ref_wl = 546/2  # factor 2 as each crossing = half lambda
sampling_frequency = 50  # frequency, in Hz
x = 0.7 * np.arange(0, len(y_ref), 1)/sampling_frequency
'''
find why 0.7 here
'''
# x as distance travelled by motor


# Remove offsets, and the butterworth filter to correct for misaligment
y_ref -= y_ref.mean()  # remove offsets
y_data -= y_data.mean()

filter_order = 2
freq = 1  # cutoff frequency
sampling = 50  # sampling frequency
sos = signal.butter(filter_order, freq, 'hp', fs=sampling, output='sos')
y_ref = signal.sosfilt(sos, y_ref)  # filter the y values
y_data = signal.sosfilt(sos, y_data)

# find the crossing points
crossing_points = []
for i in range(len(y_ref)-1):
    if (y_ref[i] <= 0 and y_ref[i+1] >= 0)\
            or (y_ref[i] >= 0 and y_ref[i+1] <= 0):
        b = (y_ref[i+1] - y_ref[i]/x[i] * x[i+1])/(1-x[i+1]/x[i])
        a = (y_ref[i] - b)/x[i]
        crossing_points.append(-b/a)

# plot crossing points (comment this section out if you don't want the graph)
plt.figure("Find the crossing points")
plt.title("Find the crossing points")
plt.plot(x, y_ref, 'x-', label='Data')
plt.plot(crossing_points, 0*np.array(crossing_points),
         'go', label='Crossing Points')
plt.xlabel('Distance')
plt.ylabel('Intensity, a.u.')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Shift the points so there is equal spacing
index = 0
x_correct_array = []
last_pt = 0
last_pt_correct = 0

for period in range(len(crossing_points)//2-1):
    measured_lam = crossing_points[2*period+2] - crossing_points[2*period]
    shifting_ratio = ref_wl/measured_lam
    while x[index] < crossing_points[2*period+2]:
        x_correct = shifting_ratio*(x[index]-last_pt)+last_pt_correct
        x_correct_array.append(x_correct)
        index += 1
    last_pt = x[index-1]
    last_pt_correct = x_correct_array[-1]

#  Make the Cubic Spline
# these are the number of points that you will resample - try changing this
# and look how well the resampling follows the data.
N = 1000000  # number of data points
x_spline = np.linspace(0, x_correct_array[-1], N)  # x vals of spline points
y = y_data[:len(x_correct_array)]
cs = spi.CubicSpline(x_correct_array, y)  # cubic spline y vals

# plot corrected points and initial points for comparison
plt.figure("Correct the points and resample the points", figsize=(8, 6))
plt.title('Corrected and Resampled Interferogram')
plt.plot(x_correct_array, y, 'go', label='Inital points')
plt.plot(x_spline, cs(x_spline), label="Cubic Spline")
plt.ylabel('Intensity, a.u.')
plt.xlabel('Distance travelled, µsteps')
plt.legend()
plt.grid()
plt.show()

# FFT to extract spectra
distance = x_spline[1:]-x_spline[:-1]  # distance travelled
xf1 = spf.fftshift(spf.fftfreq(len(x_spline)))  # fourier transform and shift
# xvals turns from wavenumber to wavelength, and also use the distance to
# make the x scale nanometres
xvals = abs(2*distance.mean()/xf1[int(len(xf1)/2+1):len(xf1)])

yf1 = spf.fftshift(spf.fft(cs(x_spline)))  # fourier transform and shift
yvals = abs(yf1[int(len(xf1)/2+1):len(xf1)])  # only taking values for +ve x
yvals = yvals/max(yvals)  # normalise intensity

plt.figure("Fully corrected spectrum FFT")
plt.title('Mercury Spectrum')
plt.plot(xvals, yvals, label='Data')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.xlim(572, 585)  # change depending on where you expect to see spectra
plt.grid()
plt.legend()
plt.show()
