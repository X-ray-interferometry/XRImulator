# XRImulator2

This code is a work in progress end-to-end simulator for a single spacecraft X-ray interferometer. It is an open access, object oriented, python code consisting of four modules and an additional script for executing the desired functions of the simulator. This is a new, corrected, improved and expanded version of the same simulator made by Emily van Hese, which can be found in the following repository: https://github.com/EmilyvHese/XRImulator

Starting with the first ‘end’, the astrophysical source. It is sampled in the Image module (image.py), to generate photons.
Secondly, the whole instrument is created in the Interferometer module (instrument.py). It consists of the main body, within which multiple baselines can be saved, and within each of those is a detector.
Within the Process module (process.py), the sampled source photons go through the initialized interferometer, resulting in deduced photon properties (e.g. impact position, time of arrival and energy). This is pre-processed data and not direct detector output data. The output from the detector will likely be a representation of the detected charge from specific pixels. This is related to and can be calibrated to photon energy and based on the pixels detecting certain amounts of charge, an impact location can be inferred. The readout time, by the computer system controlling the detector, is likely the assigned time of arrival. Converting direct detector output into inferred detected photon properties is still being researched and depends on the detector. This process has therefore been abstracted out and can be accounted for, using noise parameters.
The collection of inferred photon properties can then be analysed, using different techniques for different purposes, in the Analysis module (analysis.py).
A separate file is used to initialize and run the simulator (tests.py). It contains the tests that have so far been performed on the code, along with full experiments. It is a bit of a mess with outdated versions of tests that have not been updated.
leftovers.py contains old versions of methods from the other files that were deemed to potentially have some value later as inspiration.

The other files are not directly related to the X-ray interferometer and were used to test certain pyhton quircks, packages and other things in cleaner environments.
