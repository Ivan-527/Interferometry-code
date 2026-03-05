#!/usr/bin/python

import sys
import read_data_results3 as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import iminuit as im
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi


# Step 1 get the data and the x position
file='%s'%(sys.argv[1]) #this is the data
results = rd.read_data3(file)

# Step 1.1 - set the reference wavelength. Whatever units you use here will be theunits of your final spectrum
lam_r = 546e-9 # units of m 


# Be carefull!!! Change for the correct detector by swapping one and zero here
# y1 should contain the reference wavelength that needs correcting
y1 = np.array(results[1])
#y2 = np.array(results[1])
#for now remove the mean, or remove the offset with a filter later
#y1 = y1 - y1.mean()
#y2 = y2 - y2.mean()

# create a dummy x-axis (with just integers)
# to figure out where the crossing points are
# note, assumes that each x is regularly spaced within each wavelength
x = np.arange(0, len(y1), 1)

#step 2.1 butterworth filter to correct for misaligment (offset)
filter_order = 2
freq = 1 #cutoff frequency
sampling = 50 # sampling frequency for filter
sos = signal.butter(filter_order, freq, 'hp', fs=sampling, output='sos')
filtered = signal.sosfilt(sos, y1)
y1 = filtered
filtered = signal.sosfilt(sos, y2)
y2 = filtered


#step 2 get the x at which we cross
crossing_pos = []
for i in range(len(y1)-1):
    if (y1[i] <= 0 and y1[i+1] >= 0) or (y1[i] >= 0 and y1[i+1] <= 0) :
    #create the exact crossing point of 0
        xa = x[i]
        ya = y1[i]
        xb = x[i+1]
        yb = y1[i+1]
        b = (yb - ya/xa * xb)/(1-xb/xa)
        a = (ya - b)/xa
        extra = -b/a - xa
        crossing_pos.append(x[i]+extra)

plt.figure("Find the crossing points")
plt.plot(x, y1, 'x-')
plt.plot(crossing_pos, 0*np.array(crossing_pos), 'ko')
#plt.show()

#step 3 shift the points

k = 0
x_corr_array = [0]
last_pt = 0
last_pt_corr = 0

for period in range(len(crossing_pos)//2-1):
    # measured_lam is distance between 3 crossing points (e.g., [0,1,2]) = 1 wavelength
    measured_lam = crossing_pos[2*period+2] - crossing_pos[2*period]
    # define the x-scale using the reference wavelength (lam_r)
    shifting_ratio = lam_r/measured_lam
    while x[k]<crossing_pos[2*period+2]:
        x_corr = shifting_ratio*(x[k]-last_pt)+last_pt_corr
        x_corr_array.append(x_corr)
        k = k+1
    last_pt = x[k-1]
    last_pt_corr = x_corr_array[-1]
x_corr_array = x_corr_array[1:]

# ---------------------------------------------------------------------
# Now apply the corrected x-axis from y1 to the interferogram from y2
# ---------------------------------------------------------------------

#step 4 create a uniform data set 
#Cubic Spline part to map onto corrected onto regular grid
xr = x_corr_array
N = 1000000 # these are the number of points that you will resample - try changing this and look how well the resampling follows the data.
xs = np.linspace(0, x_corr_array[-1], N)
y = y2[:len(x_corr_array)]
cs = spi.CubicSpline(xr, y)

plt.figure("Correct the points and resample  the points")
plt.title('0-crossing - fitted wavelength after CubicSpline \n%s'%file)
plt.plot(xr, y, 'go', label = 'Inital points')
plt.plot(xs,cs(xs), label="Cubic_spline N=%i"%N)
plt.legend()
#plt.show()

distance = xs[1:]-xs[:-1]

# FFT to extract spectra
yf1=spf.fft(cs(xs))
xf1=spf.fftfreq(len(xs)) # setting the correct x-axis for the fourier transform. Osciallations/step  
xf1=spf.fftshift(xf1) #shifts to make it easier (google if interested)
yf1=spf.fftshift(yf1)
xx1=xf1[int(len(xf1)/2+1):len(xf1)]
repx1=distance.mean()/xx1  

plt.figure("Fully corrected spectrum FFT")
plt.title('%s'%file)
#plt.plot(abs(repx0),abs(yf0[int(len(xf0)/2+1):len(xf0)]),label='Original')
#plt.plot(abs(repx),abs(yf[int(len(xf)/2+1):len(xf)]),label='After shifting and uniformising full mercury')
plt.plot(abs(repx1),abs(yf1[int(len(xf1)/2+1):len(xf1)]),label='After shifting and cubicspline N=%i full mercury'%(N))
plt.xlim(300e-9,800e-9)
plt.ylabel('Intensity (a.u.)')
plt.legend()    
plt.show()
plt.savefig("figures/spectrum_from_local_calibration.png")



