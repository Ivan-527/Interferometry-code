# -*- coding: utf-8 -*-
"""
@author: Laura Hollister
email with questions: lh21@ic.ac.uk

Simulation Code for Interferometry Year 2 Lab
Length units are all in metres
"""

#############################################################################
# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as spf

#############################################################################
# Functions go here


def add_line(x, y, wl, amp, width, no_steps):
    """
    This function adds a new line to the interferogram. It assumes each line
    is made of discrete delta functions, assuming a gaussian shape.

    x is the separation between the mirrors
    y is the amplitude of the light
    wl is the wavelength
    amp is the amplitude (arbitrary scale)
    width is the line width (here 1 sigma as we assume gaussian)
    no_steps is the total steps taken
    """

    nsigma = 3
    amplitude = amp*calc_amp(nsigma, no_steps)
    wl_step = nsigma*2.0*width/no_steps  # wavelength moved by one step

    for i in range(len(amplitude)):
        # calculates wavelength at each point in the loops
        wl_i = wl-nsigma*width+i*wl_step
        y = y+amplitude[i]*np.sin(np.pi*2.*x/wl_i)
    return y


def calc_amp(nsigma, no_samps):
    """
    Calculates the amplitude at the various steps according to a gaussian
    distribution.
    """
    y = np.empty(shape=[no_samps])
    step = nsigma*2.0/no_samps
    for i in range(no_samps):
        x = -nsigma+i*step
        y[i] = np.exp(-x*x/4)
    return y


def add_square(x, y, start, amp, width, no_steps):
    step = width/(no_steps-1)
    amplitude = amp/no_steps
    for i in range(no_steps):
        wavelength = start+i*step
        y = y+(amplitude*np.sin(np.pi*2.*x/wavelength)+amplitude)
    return y


#############################################################################
# main code

# interferogram data
no_samps = 660000  # number samples taken of the interferogram
dist_per_step = 30e-9  # distance moved per step of motor - the smaller this
# is, the better your resolution will be
steps_per_sample = 1
dist_in_samp = dist_per_step*steps_per_sample  # distance covered

dist_start = -0.01  # start relevant to null point (1mm back)
dist_end = dist_start+dist_in_samp*no_samps  # dist from null point at end

# setting the x locations of the samples
x = np.linspace(dist_start, dist_end, no_samps)
# make an empty array to store amplitudes in
y = np.zeros(shape=[len(x)])

# here you can generate your signals; uncomment which you want to use

#####
# FOR A SINGLE LINE #
# line1 = 589e-9  # wavelength of spectral line, m
# width1 = 0.01e-9  # width of the spectral line, m
# strength1 = 1  # relative amplitude of first line
# samps1 = 50  # numberof samples in first spectral line
# y = add_line(x, y, line1, strength1, width1, samps1)
#####

#####
# FOR TWO LINES #
# set up multiple lines by generating them separately then summing them
line1 = 589e-9  # wavelength of spectral line, m
width1 = 0.01e-9  # width of the spectral line, m
strength1 = 1  # relative amplitude of first line
samps1 = 50  # numberof samples in first spectral line
line2 = 650.6e-9  # wavelength of a second spectral line, m
width2 = 0.02e-9  # width of a second spectral line, m
strength2 = 1  # relative amplitude of second line
samps2 = 50  # number of samples in second spectral line
y1 = add_line(x, y, line1, strength1, width1, samps1)
y2 = add_line(x, y, line2, strength2, width2, samps2)
y = y1 + y2  # remember to sum all the lines if you add more
#####

#####
# FOR A SQUARE WAVE #
# initial_wl = 590e-9  # start point of square wave
# width = 10e-9  # width of the square wave
# no_samps = 500  # number of samples taken
# amplitude = 1
# y1 = add_square(x, y, 590e-9, 1.0, 10e-9, 500)
# y2 = add_square(x, y, 690e-9, 1.0, 10e-9, 500)
# y = y1+y2
#####

# plot these lines on a graph - zoom in to see more detail on the graph
plt.figure()
plt.title('Interferogram')
plt.plot(x, y, 'b-')
plt.xlabel("Distance from null point (m)")
plt.ylabel("Amplitude (AU)")
plt.grid()
plt.show()


# take a fourier transform - shifts are to align x=0
xf = spf.fftfreq(no_samps)
xf = spf.fftshift(xf)  # xf is now the shifted fourier transform of x
# x is now in oscillations per sample or wavenumber, not wavelength
yf = spf.fft(y)
yf = spf.fftshift(yf)  # yf is now the shifted fourier transform of y

# plot these fourier transforms - the reflection in the y axis shows us we
# can simply take the positive values for the next step
# choose whether to plot the wavenumber graph
plotting = False
if plotting is True:
    fig, ax = plt.subplots(1, 3, figsize=(25, 9))  # eaiser to see the peaks
    fig.suptitle('Amplitude against wavenumber')

    ax[0].plot(xf, np.abs(yf), 'b-')
    ax[0].set_title('Fourier Transform')
    ax[1].plot(xf, np.abs(yf), 'b-')
    ax[1].set_xlim(0.04608, 0.04614)  # change these values to zoom in
    ax[1].set_title('Zoomed in first peak')
    ax[2].plot(xf, np.abs(yf), 'b-')
    ax[2].set_xlim(0.0509, 0.05095)  # change these values to zoom in
    ax[2].set_title('Zoomed in second peak')
    fig.tight_layout()

    for i in ax:
        i.set_xlabel('Wavenumber, 1/m')
        i.set_ylabel('Amplitube, AU ')
        i.grid()

    fig.tight_layout()
    fig.show()

# now reconstruct the original wavelength spectrum
xx = xf[np.where(xf > 0)]  # taking only positive x vals
x_wl = dist_in_samp/xx  # x values in terms of wavelengths
y_wl = abs(yf[np.where(xf > 0)])  # y values in terms of wavelengths
y_wl = y_wl / np.max(y_wl)  # normalise intensity


# plotting
plt.figure()
plt.plot(x_wl, y_wl, 'b')
plt.xlim(550e-9, 700e-9)  # change these values to zoom in on the peak
plt.xlabel("Wavelength (m)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
