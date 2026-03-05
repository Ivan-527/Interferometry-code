import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import read_data_results3 as rd
import sys

# Read in the data
file=str(input()) + ".txt" #this is the data
results = rd.read_data3("data/"+file)
x = np.array(results[5])
y = np.array(results[1])




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
           'ko', label='Crossing Points')
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
plt.savefig("figures/crossing_points.png")
plt.show()

# step 5 - find the global calibration
mean_diff = np.mean(diff) # difference in musteps between crossing points
wl_ref = 532.0e-9  # from laser specifications
# there are 2 crossing points in each wavelength
metres_per_microstep = wl_ref / (2.0*mean_diff)

print('Distance moved per mu step is %.5e metres' % (metres_per_microstep))
