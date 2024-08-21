# XRImulator
# Emily van Hese

This code is a work in progress end-to-end simulator for a single spacecraft X-ray interferometer.

images.py contains methods that generate photons that are to be observed with the rest of the code.

instrument.py contains methods that generate the interferometers that are to observe the images.
process.py contains the methods that use the interferometers from instrument.py to observe the photons from images.py, and turns them into observational data.

analysis.py contains methods that help with analyzing the data, chief among them the image reconstruction code.

tests.py contains the tests that have so far been performed on the code, along with full experiments. It is a bit of a mess with outdated versions of tests that have not been updated.

leftovers.py contains old versions of methods from the other files that I deemed might have some value later as inspiration.

The other files are not directly related to the x-ray interferometer per se and were either to generate images for my thesis or to test certain pyhton quircks and packages in a cleaner environment.
