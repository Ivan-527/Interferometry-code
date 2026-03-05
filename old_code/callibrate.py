"""
@author: Laura Hollister
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Read in the data (change your filename)
y, x = np.loadtxt('alignmentdata.txt', delimiter=' ', unpack=True,
                  usecols=(1, 5), skiprows=200)

# Use a Butterworth filter to correct offset
filter_order = 2  # type of filter
freq = 1
sampling_freq = 50
# replace original array with new filtered one
y = signal.sosfilt(signal.butter(filter_order, freq, 'hp', fs=sampling_freq,
                                 output='sos'), y)

# Removes artefacts introduced by the butterworth filter
percent_mask = 0.05  # you may be able to make this smaller
y = y[int(len(y)*(percent_mask)):int(len(y)*(1-percent_mask))]
x = x[int(len(x)*(percent_mask)):int(len(x)*(1-percent_mask))]
x -= min(x)  # resets the x values to be centred about 0

# Get the x points at which we cross the x - axis
crossing_points = np.array([])
for i in range(len(y)-1):
    # go through, check for sign change in y
    if (y[i] <= 0 and y[i+1] >= 0) or (y[i] >= 0 and y[i+1] <= 0):
        # create the exact crossing point of 0
        b = (y[i+1] - y[i]/x[i] * x[i+1])/(1-x[i+1]/x[i])
        a = (y[i] - b)/x[i]
        extra = -b/a - x[i]
        crossing_points = np.append(crossing_points, (x[i]+extra))

# Finds the distance between each crossing point
diff = np.array([])
for i in range(len(crossing_points)-1):
    diff = np.append(diff, (np.abs(crossing_points[i+1]-crossing_points[i])))

# Plot Results
fig, ax = plt.subplots(2, 1, figsize=(8, 7))
fig.suptitle('Crossing Point Analysis', fontsize=20)

ax[0].plot(x, y, 'x-', label='Data', color='blue')
ax[0].plot(crossing_points, 0*np.array(crossing_points),
           'ko', label='Crossing Points', color='red')
ax[0].set_xlabel("Position (µsteps)")
ax[0].set_xlim(0, 0.025e7)
ax[0].set_ylim(-500, 500)
ax[0].set_ylabel("Signal")
ax[0].set_title('Position against Signal (zoomed in)')
ax[0].legend(loc='upper right')
ax[1].set_title('Histogram of the Crossing Point Distances')
ax[1].hist(diff, bins=100, color='blue')
ax[1].set_xlabel("Distance between crossings (µsteps)")
ax[1].set_ylabel("Number of entries")

print("The mean difference between crossing points is %.0d and the standard\
      deviation between crossing points is %.0d" % (np.mean(diff),
      np.std(diff)))

fig.tight_layout()
fig.show()


# step 5 - find the wavelength of the laser
mean_diff = np.mean(diff)
wl_musteps = 2*mean_diff  # wavelength in musteps
wl_real = 531.8  # from laser specifications

dist_per_mustep = wl_real / wl_musteps

print('Distance moved per mu step is %.6f nm' % (dist_per_mustep))