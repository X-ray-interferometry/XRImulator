"""
This file contains functions and classes related to the generation images in the form of a number of X-ray photons, 
each with location and time of arrival and energy. 
"""

from PIL import Image
import numpy as np
import scipy.constants as spc
from astropy.io import fits
from astropy.wcs import WCS

# for testing
import matplotlib.pyplot as plt


class image():
    """ 
    Class that defines a data format that all images to be processed by this package should follow. 
    It consists of a number of arrrays of specified size which should contain the energy, arrival time, and ... #TODO define further
    Note that this class only generates an empty image class of specified size.
    Generating the actual photons to fill it up should happen in a seperate function that manipulates an image class object. 
    """

    def __init__(self, size):
        """ Initiation function for the class. Generates arrays of the specified size for each parameter specified in the class docstring.

            Parameters:

            size (int) = number of photons to save in arrays.\n 
        """

        # Abbreviation of 'Times Of Arrival'.
        self.toa = np.zeros(size, dtype=int)
        self.energies = np.zeros(size)

        # photons per area, per energy, per unit time
        self.spectrum = None

        # Array containing coordinates of origin for each photon.
        self.loc = np.zeros((size, 2))

        # Useful to have as shorthand
        self.size = size

        # background photons
        self.bkg_indices = []
        self.bkg_spect = None

def point_source(size, alpha, beta, energy, spectrum = None):
    """
    Function that generates an image of a monochromatic point source according to some specifications.

    Parameters:

    size (int) = number of photons to generate from this source.\n
    alpha (float) = coordinate offset from zero pointing in x-direction (arcsec)\n
    beta (float) = coordinate offset from zero pointing in y-direction (arcsec)\n
    energy (float) = energy of photons to generate (KeV)\n
    spectrum (string) = the file(-path) to be loaded for the source spectrum as counts per energy per time per area\n
    spectrum (2D numpy array) = the pre defined source spectrum as counts per energy per time per area
    """
    im = image(size)

    # save spectrum when given
    if spectrum is not None:

        # load file containing spectrum or copy the spectrum
        if isinstance(spectrum, str):
            im.spectrum = np.loadtxt(spectrum)
        else: 
            im.spectrum = spectrum

        if energy is None:
            energy = np.mean(im.spectrum[:,0])

    elif energy is None:
        # define either spectrum or energy
        raise Exception("ERROR: define either spectrum or energy")
    
    im.energies[:] = energy * spc.eV * 1e3
    im.loc[:] = np.array([alpha, beta]) * 2 * np.pi / (3600 * 360)
    im.toa = np.array([i for i in range(size)])

    return im

def double_point_source(size, alpha, beta, energy, spectrum = None):
    """
    Function that generates an image of two monochromatic point sources according to some specifications.

    Parameters:

    size (int) = number of photons to generate from the sources.\n
    alpha (list-like of floats) = coordinate offsets from zero pointing in x-direction (arcsec)\n
    beta (list-like of floats) = coordinate offsets from zero pointing in y-direction (arcsec)\n
    energy (list-like of floats) = energies of photons to generate (KeV)\n
    spectrum (string) = the file(-path) to be loaded for the source spectrum as counts per energy per time per area\n
    spectrum (2D numpy array) = the pre defined source spectrum as counts per energy per time per area
    """
    im = image(size)

    # save spectrum when given
    if spectrum is not None:

        # load file containing spectrum or copy the spectrum
        if isinstance(spectrum, str):
            im.spectrum = np.loadtxt(spectrum)
        else: 
            im.spectrum = spectrum

        if energy is None:
            energy = np.mean(im.spectrum[:,0])

    elif energy is None:
        # define either spectrum or energy
        raise Exception("ERROR: define either spectrum or energy")
    
    for i in range(0, size):
        source = np.random.randint(0,2)
        im.energies[i] = energy[source] * spc.eV * 1e3
        im.loc[i] = np.array([alpha[source], beta[source]]) * 2 * np.pi / (3600 * 360)
        im.toa[i] = i

    return im

def m_point_sources(size, m, alpha, beta, energy, spectrum = None):
    """
    Function that generates an image of two monochromatic point sources according to some specifications.

    Parameters:

    size (int) = number of photons to generate from the sources.\n
    m (int) = number of point sources to generate.\n
    alpha (list-like of floats) = coordinate offsets from zero pointing in x-direction (arcsec)\n
    beta (list-like of floats) = coordinate offsets from zero pointing in y-direction (arcsec)\n
    energy (list-like of floats) = energies of photons to generate (KeV)\n
    spectrum (string) = the file(-path) to be loaded for the source spectrum as counts per energy per time per area\n
    spectrum (2D numpy array) = the pre defined source spectrum as counts per energy per time per area
    """
    im = image(size)

    # save spectrum when given
    if spectrum is not None:

        # load file containing spectrum or copy the spectrum
        if isinstance(spectrum, str):
            im.spectrum = np.loadtxt(spectrum)
        else: 
            im.spectrum = spectrum

        if energy is None:
            energy = np.mean(im.spectrum[:,0])

    elif energy is None:
        # define either spectrum or energy
        raise Exception("ERROR: define either spectrum or energy")
    
    for i in range(0, size):
        source = np.random.randint(0,m)
        im.energies[i] = energy[source] * spc.eV * 1e3
        im.loc[i] = np.array([alpha[source], beta[source]]) * 2 * np.pi / (3600 * 360)
        im.toa[i] = i

    return im

def point_source_multichromatic_range(size, alpha, beta, energy, spectrum = None):
    """
    Function that generates an image of a multichromatic point source according to some specifications.

    Parameters:

    size (int) = number of photons to generate from this source.\n
    alpha (float) = coordinate offset from zero pointing in x-direction (arcsec)\n
    beta (float) = coordinate offset from zero pointing in y-direction (arcsec)\n
    energy (list-like of floats) = upper and lower bounds for energy of photons to generate (KeV)\n
    spectrum (string) = the file(-path) to be loaded for the source spectrum as counts per energy per time per area\n
    spectrum (2D numpy array) = the pre defined source spectrum as counts per energy per time per area
    """
    im = image(size)

    # save spectrum when given
    if spectrum is not None:

        # load file containing spectrum or copy the spectrum
        if isinstance(spectrum, str):
            im.spectrum = np.loadtxt(spectrum)
        else: 
            im.spectrum = spectrum

        if energy is None:
            energy = np.mean(im.spectrum[:,0])

    elif energy is None:
        # define either spectrum or energy
        raise Exception("ERROR: define either spectrum or energy")
    
    for i in range(0, size):
        im.energies[i] = (np.random.random() * (energy[1] - energy[0]) + energy[0]) * spc.eV * 1e3
        im.loc[i] = np.array([alpha, beta]) * 2 * np.pi / (3600 * 360)
        im.toa[i] = i

    return im

def point_source_multichromatic_gauss(size, alpha, beta, energy, energy_spread):
    """
    Function that generates an image of a multichromatic point source according to some specifications.

    Parameters:

    size (int) = number of photons to generate from this source.\n
    alpha (float) = coordinate offset from zero pointing in x-direction (arcsec)\n
    beta (float) = coordinate offset from zero pointing in y-direction (arcsec)\n
    energy (float) = mean energy of photons to generate (KeV)\n
    energy_spread (float) = spread in energy of photons to generate (KeV)\n
    """
    im = image(size)

    spectrum = None
    
    im.energies = np.random.normal(energy, energy_spread, size) * spc.eV * 1e3
    for i in range(0, size):
        im.loc[i] = np.array([alpha, beta]) * 2 * np.pi / (3600 * 360)
        im.toa[i] = i

    return im

def disc(size, alpha, beta, energy, radius, energy_spread=0., spectrum = None):
    """
    A function that generates photons in the shape of a continuous disk.

    Parameters:

    size (int) = number of photons to generate from this source.\n
    alpha (float) = coordinate offset from zero pointing in x-direction (arcsec)\n
    beta (float) = coordinate offset from zero pointing in y-direction (arcsec)\n
    energy (float) = mean energy of photons to generate (KeV)\n
    radius (float) = radius of the disc (arcsec)\n
    energy_spread (float) = spread in energy of photons to generate (KeV)\n
    spectrum (string) = the file(-path) to be loaded for the source spectrum as counts per energy per time per area\n
    spectrum (2D numpy array) = the pre defined source spectrum as counts per energy per time per area
    """
    im = image(size)

    # save spectrum when given
    if spectrum is not None:

        # load file containing spectrum or copy the spectrum
        if isinstance(spectrum, str):
            im.spectrum = np.loadtxt(spectrum)
        else: 
            im.spectrum = spectrum

        if energy is None:
            energy = np.mean(im.spectrum[:,0])

    elif energy is None:
        # define either spectrum or energy
        raise Exception("ERROR: define either spectrum or energy")
    
    for i in range(0, size):
        im.energies[i] = energy * spc.eV * 1e3
        im.toa[i] = i
        r = np.random.random() * radius
        theta = np.random.random() * 2 * np.pi
        im.loc[i] = np.array([alpha + r * np.cos(theta), beta + r * np.sin(theta)]) * 2 * np.pi / (3600 * 360) 

    if energy_spread > 0.:
        im.energies += np.random.normal(0, energy_spread, size)

    return im

def generate_from_image(image_path, no_photons, img_scale, energy, energy_spread=0., offset=[0,0], spectrum = None, bkg_phot = None, bkg_spect = None, bkg_energy = None):
    """
    Function that generates an image object from any arbitrary input image. 
    useful for testing realistic astrophysical sources without having to include code to simulate them here.
    Just have some other simulator generate an image and use this function to read that out.
    This function uses relative brightness of each part of the input image to generate a pmf defined at each pixel location of the image.
    This pmf is then sampled for however many photons are required.

    Parameters:

    image_path (string) = the image file(-path) name to load in.\n
    no_photons (int) = number of source photons to generate from this source.\n
    img_scale (float) = largest axis size (arcsec)
    energy (float) = mean energy of photons to generate (KeV)\n
    energy_spread (float) = spread in energy of photons to generate (KeV)\n
    offset (list of shape [0,0]) = offset of input image (arcsec)\n
    spectrum (string) = the file(-path) to be loaded for the source spectrum as counts per energy per time per area.\n
    spectrum (2D numpy array) = the pre defined source spectrum as counts per energy per time per area.\n
    bkg_phot (int) = number of background photons to generate on the input image.\n
    *bkg_spect (string) = the file(-path) to be loaded for the background spectrum as counts per energy per time per area.\n
    bkg_spect (2D numpy array) = the pre defined source spectrum as counts per energy per time per area.\n
    bkg_energy (float) = mean energy of photons to generate (KeV)\n

    * not yet implemented (TODO)
    """

    # ensuring sufficient array lengths for source and bkg counts
    if bkg_phot is not None:
        if type(bkg_phot) is float:
            bkg_phot = int(np.around(bkg_phot * no_photons))
            
        # create image instance
        photon_img = image(bkg_phot + no_photons)

    else:
        # create image instance
        photon_img = image(no_photons)

    # save spectrum when given
    if spectrum is not None:

        # load file containing spectrum or copy the spectrum
        if isinstance(spectrum, str):
            photon_img.spectrum = np.loadtxt(spectrum)
        else: 
            photon_img.spectrum = spectrum

        if energy is None:
            energy = np.mean(photon_img.spectrum[:,0])

    elif energy is None:
        # define either spectrum or energy
        raise Exception("ERROR: define either spectrum or energy")

    # Load the image and convert it to grayscale
    img = Image.open(image_path).convert('L')
    
    # Convert the image to a numpy array
    img_array = np.array(img)

    pix_scale = np.array(img_array.shape)

    # Generate a probability mass function from the image
    pmf = img_array / np.sum(img_array)
    
    # Draw N samples from the probability mass function
    source_counts = np.random.choice(
        np.arange(img_array.size),
        size=no_photons,
        p=pmf.flatten()    
    )

    # generate bkg photons
    if bkg_phot is not None:

        # random point of origin within the input image (not based on FoV)
        bkg_counts = np.random.choice(
            np.arange(img_array.size),
            size=bkg_phot
            )
        
        # random bkg vs based on flux
        if bkg_spect == None:

            # rondomize photon TOA, while keeping the respective orders of both source and bkg photons (not based on relative fluxes)
            indices = np.arange(bkg_phot + no_photons)
            pixel_locations = np.concatenate((np.array(bkg_counts), np.array(source_counts)))
            np.random.shuffle(indices)
            
            # keep track of the bkg photon indices
            bkg_i = np.where(np.in1d(indices, np.arange(bkg_phot)))[0]
            photon_img.bkg_indices.extend(bkg_i)

            # Generating photon energies
            photon_img.energies[np.arange(bkg_phot + no_photons)] = energy * spc.eV * 1e3

            if bkg_energy is not None:
                photon_img.energies[bkg_i] = bkg_energy * spc.eV * 1e3

            # Generating times of arrival
            # TODO: make more realistic by making Poisson times when spectra are given
            photon_img.toa[np.arange(bkg_phot + no_photons)] = np.arange(bkg_phot + no_photons)

        else:
            # save spectrum when given
            """TODO: sample the spectrum in process.py"""
            # load file containing spectrum or copy the spectrum
            if isinstance(bkg_spect, str):
                photon_img.bkg_spect = np.loadtxt(bkg_spect)
            else: 
                photon_img.bkg_spect = bkg_spect

            if energy is None:
                energy = np.mean(photon_img.bkg_spect[:,0])

    # no background
    else:
        pixel_locations = source_counts

        # Generating photon energies
        photon_img.energies[np.arange(no_photons)] = energy * spc.eV * 1e3

        # Generating times of arrival
        # TODO: make more realistic by making Poisson times when spectra are given
        photon_img.toa[np.arange(no_photons)] = np.arange(no_photons)
        
    # Convert the flattened indices back into (x,y) coordinates
    pixel_locations = np.column_stack(np.unravel_index(pixel_locations, img_array.shape))

    # Convert the sampled pixel locations to points of origin on the sky
    photon_img.loc = ((pixel_locations - (pix_scale/2)) * img_scale / pix_scale.max() + np.array(offset)) * 2 * np.pi / (3600 * 360)

    # Adds a spread to the energies if given.
    if energy_spread > 0.:
        photon_img.energies += np.random.normal(0, energy_spread * spc.eV * 1e3, no_photons)

    return photon_img, pix_scale

# TODO write a function that adds two (or N) images together