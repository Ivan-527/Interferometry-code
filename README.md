# PhysicsYear2Labs-Interferometry
Repository for all code assiociated with Year 2 Interferometry experiment

The repository contains the following codes. Updates to code to improve clarity only by renaming some variables/functions; updating the comments; spliting 'analysis.py' into two separate codes one for global calibration and the other for local calibration (previously it sort of did both); and retiring duplicated scripts to reduce confusion. Previous versions of code can be found in the old_code folder.

For the simulation:

Simulation.py                 -   to simulate the interferograms from different light sources (based on inter1_b.py).

For the data analysis:

read_data_results3.py         -   reads in and does some tidying of the output file from the detectors.

quick_plot.py                 -   plots the inteferograms from the named output file.

crossing_points.py            -   analyses an interferogram to determine the crossing points - can be used to determine global calibration (based on callibrate.py).

apply_global_calibration.py   -   calculates a spectrum from an interferogram data using a global calibration (metres per microstep) (based on analysis.py).

apply_local_calibration.py    -   runs a local calibration on the interferogram - requires a reference wavelength (based on analysis.py).
