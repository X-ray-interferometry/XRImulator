"""
This file is intended to contain code that can be used to analyse instrument data, in order to be able to draw meaningful conclusions from it.
""" 

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.fft as ft
import scipy.constants as spc
import scipy.stats as stat
import stingray
import time
import threading
import scipy.interpolate as spinterp
from matplotlib import colors

import matplotlib.cm as cm


def hist_data(data, binsno, pixs = False, num = 0):
    """
    Function that makes a histogram of direct output data from an interferometer object.

    Parameter:
    data (interferometer_data class object): Data to be plotted.
    binsno (int): Number of bins in the histogram.
    pixs (Boolean): whether the x-axis is in units of pixels or meters. If true, then in pixels. Default is False.
    """

    if pixs:
        plt.hist(data, binsno, label=f'Baseline {num}')
        plt.xlabel('Detector position (pixels)')
    else:
        plt.hist(data, binsno, label=f'Baseline {num}')
        plt.xlabel('Detector position (micrometers)')
    plt.ylabel('Counts')

def ft_data(y_data, samples, spacing):
    """
    Function that fourier transforms given input data from an interferometer.
    Works by first making a histogram of the positional data to then fourier transform that and obtain spatial frequencies.

    Parameters:
    data (interferometer_data class object): Data to be fourier transformed.
    samples (int): Number of samples for the fourier transform to take.
    """
    ft_x_data = ft.fftfreq(samples, spacing)
    ft_y_data = ft.fft(y_data) / y_data.size

    return ft_x_data, ft_y_data

def plot_ft(ft_x_data, ft_y_data, plot_obj, log=0, num= 0):
    """
    Function to plot fourier transformed interferometer data in a number of ways.

    Parameters:
    ft_x_data (array): fourier transformed data for the x-axis (so spatial frequencies).
    ft_y_data (array): fourier transformed data for the y-axis.
    log (int in [0,2]): indicates how many axes are to be in log scale, with 1 having only the y-axis in log.
    """
    if log == 0:
        plot_obj.plot(ft.fftshift(ft_x_data), (ft.fftshift(ft_y_data)), label=f'Baseline {num}')
    if log == 1:
        plot_obj.semilogy(ft.fftshift(ft_x_data), (ft.fftshift(ft_y_data)), label=f'Baseline {num}')
    if log == 2:
        plot_obj.loglog(ft.fftshift(ft_x_data), (ft.fftshift(ft_y_data)), label=f'Baseline {num}') 

def image_recon_smooth(data, instrument, fov, samples = 512, progress = 0, error = 0.001, recon_type = "IFFT"):
    """
    This function is to be used to reconstruct images from interferometer data.
    Bins input data based on roll angle, which is important to fill out the uv-plane that will be fed into 
    the 2d inverse fourier transform.

    Args:
        data (interferometer_data class object): The interferometer data to recover an image from.
        instrument (interferometer class object): The interferometer used to record the aforementioned data.
        samples (int): N for the NxN matrix that is the uv-plane used for the 2d inverse fourier transform. TODO: outdated
        progress (1 or 0): whether, or not respectively, to show progess of image reconstruction calculations.
        TODO: add fov argument, unknown at this time; largest axis size in as
        
    Returns:
        array, array, array: Three arrays, first of which is the reconstructed image, 
                                second and third of which are the fourier transforms and associated uv coordinates of each roll bin + energy channel combination.
    """    

    
    def inverse_fourier(f_values, uv, fov, error, instrument, energies, recon_type):


        """
        This is a helper function that calculates the inverse fourier transform of the data from all baselines, only to 
        be used at the last step of the parent function. It is sectioned off here for legibility.

        It first defines an image to fill in according to the provided samples size, and calculates the sum
        of all fourier components over the whole image.
        """

        if recon_type == "IFFT":
            wavelengths = spc.h * spc.c / energies
            
            d = (fov * np.pi / (180 * 3600)) / np.max(samples) #1 / (2 * (np.max(np.array([baseline.D for baseline in instrument.baselines])) / np.min(wavelengths)))
            n = np.ceil(1 / (2 * d * (np.min(np.array([baseline.D for baseline in instrument.baselines])) / np.max(wavelengths)) * error)).astype(int) # + 1

            # print(d)
            # print(1 / (2 * (np.max(np.array([baseline.D for baseline in instrument.baselines])) / np.min(wavelengths))))

            # calculate the frequency grid onto which the real sampled frequencies are mapped
            fft_freqs = np.fft.fftfreq(n, d)

            # find closest frequency grid point in numpy.fft.fftfreq of the real sampled frequencies,
            # deviding over frequency bin width: Delta f = 1 / (n * d)
            u_conv = np.around(uv[:,0] * (n * d)).astype(int)
            v_conv = np.around(uv[:,1] * (n * d)).astype(int)

            # create nxn matrix, fourier space image
            image = np.zeros((fft_freqs.size, fft_freqs.size), dtype = np.singlecomplex)#complex)#np.zeros((fft_freqs_u.size, fft_freqs_v.size), dtype = complex)#np.zeros(samples, dtype = complex)

            # add all fourier values to the corresponing frequency coordinate as per numpy.fft.fftfreq
            np.add.at(image, (u_conv, v_conv), f_values)

            fft_image = np.fft.ifft2(image)

            fft_image = np.real(fft_image)
            fft_image = np.roll(fft_image, int(np.around(fft_image.shape[0]/2)), axis=0)
            fft_image = np.roll(fft_image, int(np.around(fft_image.shape[1]/2)), axis=1)

            x_npix_real_image = samples[0] #fov / (0.5 * (3600 * 180 / np.pi) * (d*n))# 1 / np.max(fft_freqs))
            x_lower_bound = np.round((fft_image.shape[0] / 2) - (0.5 * x_npix_real_image)).astype(int)
            x_upper_bound = np.round((fft_image.shape[0] / 2) + (0.5 * x_npix_real_image)).astype(int)
            y_npix_real_image = samples[1] #fov / (0.5 * (3600 * 180 / np.pi) * (d*n))# 1 / np.max(fft_freqs))
            y_lower_bound = np.round((fft_image.shape[0] / 2) - (0.5 * y_npix_real_image)).astype(int)
            y_upper_bound = np.round((fft_image.shape[0] / 2) + (0.5 * y_npix_real_image)).astype(int)

            # prevent indexing problems
            if x_lower_bound < 0:
                x_lower_bound = 0
            if x_upper_bound >= fft_image.shape[0]:
                x_upper_bound = fft_image.shape[0] - 1
            if y_lower_bound < 0:
                y_lower_bound = 0
            if y_upper_bound >= fft_image.shape[0]:
                y_upper_bound = fft_image.shape[0] - 1

            return fft_image[np.ix_(np.arange(x_lower_bound, x_upper_bound, 1), np.arange(y_lower_bound, y_upper_bound, 1))], (samples.max() * (1e6 * 3600 * 360 / (2 * np.pi)) * (d * n))

        else:

            def inverse_fourier_val(x, y, v, u, fourier):

                # This function is the formula for an inverse fourier transform, without the integration.
                # It is included here to make clear that a discrete inverse fourier transform is what is happening, and 
                # to make clear what argument means what and for multi-threading.
                
                global re_im

                # frequency shift + 1 / (d * n)
                re_im += fourier * np.exp(2j * np.pi * ((u) * x + (v) * y))


            global re_im
            re_im = np.zeros(samples, dtype = complex)
            shape = np.array(samples)

            fov_x = fov
            fov_y = fov_x*(shape[0]/shape[1]) # Assumes square pixels

            x_pos = np.linspace(-fov_x/2,fov_x/2,shape[1]+1)*2*np.pi/(360*3600)
            y_pos = np.linspace(-fov_y/2,fov_y/2,shape[0]+1)*2*np.pi/(360*3600)

            # Pixel positions for image calculation correspond to pixel centres:
            x_pix = (x_pos[:-1]+x_pos[1:])/2
            y_pix = (y_pos[:-1]+y_pos[1:])/2
            x_grid, y_grid = np.meshgrid(x_pix, y_pix, indexing='xy')

            usefull = np.nonzero(f_values)[0].astype(int)

            # multi threading
            thread_list = []
            for i, ft_val in enumerate(f_values[usefull]):
                thread = threading.Thread(target = inverse_fourier_val, args = (x_grid, y_grid, uv[usefull][i,0], uv[usefull][i,1], ft_val))
                thread.start()
                thread_list.append(thread)

            # waiting for all the threads to finish
            for thread in thread_list:
                
                thread.join()

            return np.real(re_im), None
    

    # These arrays are all copied locally to reduce the amount of cross-referencing to other objects required.  
    time_data = data.toa
    E_data = data.energies
    base_ind = data.baseline_indices
    pointing = data.pointing
    positional_data = data.pos
    
    # Generating the arrays that will contain the uv coordinates and associated fourier values covered by the interferometer.
    uv = []
    f_values = np.array([])
    
    # Looking only at the baselines that have associated photons
    for k in np.unique(base_ind):

        # Taking only relevant photons from the current baseline
        in_baseline = base_ind == k

        baseline = instrument.baselines[k]

        # Calculating the wavelength of light we are dealing with, and the frequency that this baseline covers in the uv-plane with it.
        lam_baseline = spc.h * spc.c / E_data[in_baseline]
        # ensure only unique energies are used, to avoid calculating all photons seperately when possible
        freq_baseline = baseline.D / lam_baseline

        # Calculating the frequency we will be doing the fourier transform for, which is the frequency we expect the fringes to appear at.
        fourier_freq = 1 / (lam_baseline / baseline.beam_angle)

        # the photon positions in the bin
        data_bin_roll = positional_data[in_baseline]#[in_range]

        # Calculating u and v for middle of current bin by taking a projection of the current frequency
        u = (freq_baseline + 0) * np.sin(pointing[time_data, 2][in_baseline] % (2 * np.pi))
        v = (freq_baseline + 0) * np.cos(pointing[time_data, 2][in_baseline] % (2 * np.pi))
        new_uv_pairs = np.array([u, v]).T

        # Doing the same with the negative frequency
        uv.extend(np.column_stack((new_uv_pairs, -new_uv_pairs)).reshape(-1, new_uv_pairs.shape[1]))

        # Calculating value of the fourier transform for the current frequency and bin
        f_value = np.exp(-2j * np.pi * fourier_freq * data_bin_roll)
        
        # Doing the same with the negative frequency
        f_values = np.append(f_values, np.ravel([f_value, np.conjugate(f_value)], "F"))

    # reshaping the uv coordinate matrix
    uv = np.array(uv).reshape(-1, 2)

    return inverse_fourier(f_values, uv, fov, error, instrument, E_data, recon_type), uv


def periodogram(data, detector, location_data, mono_energetic = True):

    # create positional bins, based on detector precision
    bins = int(np.ceil(abs(detector.pos_range[0] - detector.pos_range[1]) / detector.res_pos))
    edges = np.array([detector.pos_range[0] + i * detector.res_pos for i in range(bins + 1)])
    centres = edges[:-1] + (edges[1:] - edges[:-1])/2

    if mono_energetic:

        # give locations instead of time
        loc_spectrum = stingray.Lightcurve(centres, np.histogram(location_data, edges)[0])
        pow_pectrum = stingray.Powerspectrum(loc_spectrum)#, norm = "rms")

        plt.plot(pow_pectrum.freq * 1e-6, pow_pectrum.power)
        plt.title(r"Periodogram of photon locations")
        plt.xlabel(r"Spatial frequency ($1/\mu m$)")
        plt.ylabel(r"Amplitude")
        # plt.savefig("periodogram_source_bkg.pdf")
        plt.show()

    else:

        for energy in np.unique(data.energies):

            same_energy = data.energies == energy

            # give locations instead of time
            loc_spectrum = stingray.Lightcurve(centres[same_energy], np.histogram(location_data[same_energy], edges[same_energy]))
            pow_pectrum = stingray.Powerspectrum(loc_spectrum)#, norm = "rms")

            plt.plot(pow_pectrum.freq * 1e-6, pow_pectrum.power)
            plt.title("Periodogram of photon locations with an energy of {} J".format(energy))
            plt.xlabel(r"Spatial frequency ($1/\mu m$)")
            plt.ylabel(r"Amplitude")
            plt.show()

    return


def image_compare(source_image, recon_image):

    """Using the Sum of Squared Residuals (SSR), from bachelor thesis Paul Heinemans, for different normailzations"""
    # return np.sum(np.power((recon_image - source_image), 2))
    # return np.sum(np.power((recon_image / np.sum(recon_image)) - (source_image / np.sum(source_image)), 2))
    # return np.sum(np.power((recon_image / np.mean(recon_image)) - (source_image / np.mean(source_image)), 2))
    # return np.sum(np.power(((recon_image - np.min(recon_image)) / np.max(recon_image)) - ((source_image - np.min(source_image)) / np.max(source_image)), 2))
    return np.sum(np.power(((recon_image - np.min(recon_image)) / (np.max(recon_image) - np.min(recon_image))) - ((source_image - np.min(source_image)) / (np.max(source_image) - np.min(source_image))), 2))