"""
This file contains the code that links images.py, instrument.py and analysis.py together.
The functions and classes here pertain to the processing of image data into photon data
at the detector via a simulation of source photons moving through the optical benches.
The main function here is process_image(), with other functions in the file being subsidiary
helper functions. The file also contains the definition for the interferometer_data class,
which acts as the standardised data structure used in the functions in analysis.py.
"""

import warnings
import numpy as np
import scipy.special as sps
import scipy.stats as sampler
import scipy.constants as spc
import scipy.interpolate as interp

# for testing
import matplotlib.pyplot as plt

class interferometer_data():
    """ 
    Class that serves as a container for interferometer output data.
    Constructed as such for ease of use by way of standardization.
    Does not contain manipulation methods, data inside will have to be edited via external methods.
    """

    def __init__(self, instrument, image, eff_area_show = False, pure_diffraction = False, pure_fringes = False):
        """ 
        This function is the main function that takes an image and converts it to instrument data as
        if the instrument had just observed the object te image is a representation of. 
        It models the individual photons coming in each timestep, at what detector they end up, 
        whether they are absorbed along the way, how much noise there is, and also whether the 
        spacecraft the instrument is on wobbles, and possible correction for this.

        Parameters:\n

        instrument (interferometer class object) = Instrument object to be used to simulate observing the image.\n
        image (image class object) = Image object to be observed.\n
        
        eff_area_show (bool) = whether or not to show the effective area of the full instrument, mirrors and detector\n
        pure_diffraction (bool) = sample only the diffraction pattern, without the fringe pattern\n
        pure_fringes (bool) = sample only the fringe pattern, without the diffraction pattern\n
        """
        
        # Useful shorthands
        self.size = image.size

        # image values, for later reference
        self.image_energies = image.energies
        self.image_toa = image.toa

        # simulates the energies of the source photons going through the optical benches until they reach the detector
        self.actual_energies, self.baseline_indices = self.process_photon_e_base(instrument, image, eff_area_show)

        # detector values, which can be subjected to detector effects
        self.energies = self.actual_energies
        self.toa = self.image_toa

        # simulates the positions of the source photons going through the optical benches until they reach the detector
        self.actual_pos = self.process_photon_dpos(instrument, image, pure_diffraction, pure_fringes)
        
        # detector value, which can be subjected to detector effects
        self.pos = self.actual_pos


    def process_photon_e_base(self, instrument, image, eff_area_show = False):
        """
        This function is a helper function for process_image that specifically processes what energy detected photons, which impact
        on the detector, are expected. Not to be used outside the process_image context.

        Parameters:\n

        instrument (interferometer class object) = Instrument object to be used to simulate observing the image.\n
        image (image class object) = Image object to be observed.\n
        
        eff_area_show (bool) = whether or not to show the effective area of the full instrument, mirrors and detector\n
        """

        energies = self.image_energies

        array_lengths = self.size

        # Select random baseline based on effective area per basline
        if image.spectrum is None:
            
            # pure effective area calculation
            eff_areas = np.array([baseline.eff_area(np.unique(energies)) for baseline in instrument.baselines])
            relative_eff_areas = eff_areas / np.sum(eff_areas, axis = 0)

            baseline_indices = np.random.choice(len(instrument.baselines), array_lengths, p = relative_eff_areas.T[0]) #np.random.randint(0, len(instrument.baselines), array_lengths) #
        else:
            # effective area calculation, including source spectra

            cts_perbaseline_persec = []
            all_eCDF = []
            min_val_eCDF = []

            # loop through the baselines
            for baseline in instrument.baselines:

                # pull out detector for its attributes
                detector = baseline.camera

                # define the integration energies, 100 is used to integrate finer then the energy resolution
                # how much better energy sampling, compared to detector energy resolution
                finer_binning = 1e2
                integr_energy = np.linspace(detector.E_range[0], detector.E_range[1], (finer_binning * np.ceil((detector.E_range[1] - detector.E_range[0]) / detector.res_E)).astype(int))

                # get the effective area of the baseline, per energy
                eff_area = baseline.eff_area(integr_energy).T

                # spectra assumed to use keV, turned into SI units of J
                interp_spec = interp.interp1d(image.spectrum[:,0] * 1e3 * spc.eV, image.spectrum[:,1], bounds_error = False, fill_value = 0)(integr_energy)
            
                # integrate the spectrum multiplied with the effective areas over energies, for photons per baseline
                cts_persec = np.sum(interp_spec * eff_area * (integr_energy[-1] - integr_energy[0]) / integr_energy.size)
                cts_perbaseline_persec.append(cts_persec)

                # otherwise difficult to find error
                if (interp_spec == 0).all():
                    raise ValueError("The spectrum hasn't been sampled correctly!!")

                # inversion sampling of photon energies, eCDF calculations
                e_cumsum = np.cumsum(interp_spec * eff_area)
                e_eCDF = e_cumsum / e_cumsum[-1]
                min_val_eCDF.append(np.min(e_eCDF))
                all_eCDF.append(interp.interp1d(e_eCDF, integr_energy))#, bounds_error = False, fill_value = np.nan))

            if eff_area_show:
                plt.plot(integr_energy / (1e3 * spc.eV), eff_area, color = '#ff7f0e', label = "Effective area")
                plt.hlines(np.power(detector.pos_range[1] - detector.pos_range[0], 2), 0, 10, label = "Real area")
                # plt.title("Effective area curve of mirrors and detector")
                plt.ylabel("Effective area ($m^2$)")
                plt.xlabel("Energy ($keV$)")
                plt.xlim(1, 7)
                plt.ylim(1e-12, 1e-2)
                plt.yscale("log")
                # # plt.xscale("log")
                plt.legend()
                plt.savefig("A_eff.pdf")
                plt.show()

            # Select random baseline based on effective area per basline
            baseline_indices = np.random.choice(len(instrument.baselines), array_lengths, p = cts_perbaseline_persec / np.sum(cts_perbaseline_persec, axis = 0)) #np.random.randint(0, len(instrument.baselines), array_lengths) #
        
            # Looking only at the baselines that have associated photons
            for index, baseline in enumerate(instrument.baselines):

                # Taking only relevant photons from the current baseline
                in_baseline = np.array(instrument.baselines)[baseline_indices] == baseline

                # inversion sampling of photon energies, based on the eCDFs
                energies[in_baseline] = all_eCDF[index](np.random.uniform(min_val_eCDF[index], 1, np.sum(in_baseline)))# random(np.sum(in_baseline))) #np.random.uniform(detector.E_range[0], detector.E_range[1], array_lengths)
                
                """Testing plots"""
                # # print(np.sum(energies[in_baseline] == np.max(energies[in_baseline])))
                # plt.hist(energies[in_baseline] / (1000 * spc.eV), (np.ceil((detector.E_range[1] - detector.E_range[0]) / detector.res_E)).astype(int), density = True, label = "Sampled energies")
                # plt.plot(integr_energy / (1000 * spc.eV), interp_spec * eff_area * energies[in_baseline].size / (1000 * np.sum(interp_spec * eff_area)), label = "Interpolated spectrum times the effective area")
                # plt.xlabel("Energy (keV)")
                # plt.ylabel("Normalized counts per energy (keV^-1)")
                # plt.xlim(1, 7)
                # # plt.ylim(0, 1.5)
                # plt.yscale("log")
                # plt.legend()
                # plt.savefig("samp_spec_aeff.pdf")
                # plt.show()
        
        return energies, baseline_indices


    def fre_dif(self, wavelength, baseline, samples, y_pos = None):
        """
        Helper function that calculates the Fresnell difraction pattern for a beam
        such as the case in the interferometer. Also aplicable for, but less efficient at,
        Fraunhoffer diffraction patterns.

        Parameters:\n

        wavelength (array of floats) = the wavelenths for the photons.\n
        baseline (baseline object) = the baseline the photon(s) is/are in.\n
        samples (int) = the number of samples to take.\n
        y_pos (array of floats) = positions to calculate the fresnel diffraction value at.
        """

        # see Willingale (2004) for the definations of the dimensionless coordinate u,
        # used for the Fresnel integrals
        u_0 = baseline.W * np.sqrt(2 / (wavelength * baseline.L))
        u_1 = lambda u, u_0: u - u_0/2
        u_2 = lambda u, u_0: u + u_0/2

        # Only sample the slit size, or set values
        if y_pos is None:
            y_pos = np.linspace(-baseline.W / 2, baseline.W / 2, int(samples))
        u = y_pos * np.sqrt(2 / (wavelength * baseline.L))

        # Fresnel integrals
        S_1, C_1 = sps.fresnel(u_1(u, u_0))
        S_2, C_2 = sps.fresnel(u_2(u, u_0))

        # Fresnel amplitude
        A = ((C_2 - C_1) + 1j*(S_2 - S_1))

        # Fresnel intensity
        A_star = np.conjugate(A)
        I = np.abs(A * A_star)

        return I, u


    def process_photon_dpos(self, instrument, image, pure_diffraction = False, pure_fringes = False):
        """
        This function is a helper function for process_image that specifically processes the locations where photons impact
        on the detector (hence the d(etector)pos(ition) name). Not to be used outside the process_image context.

        Parameters:\n

        instrument (interferometer class object) = Instrument object to be used to simulate observing the image.\n
        image (image class object) = Image object to be observed.\n
        
        pure_diffraction (bool) = sample only the diffraction pattern, without the fringe pattern\n
        pure_fringes (bool) = sample only the fringe pattern, without the diffraction pattern\n
        
        """

        array_lengths = self.size

        actual_pos = np.zeros(array_lengths)

        # Defining the pointing, relative position and off-axis angle for each photon over time.
        # Relative position is useful for the calculation of theta, since the off-axis angle is very dependent on where the axis is.
        self.pointing = instrument.gen_pointing(np.max(self.image_toa))
        pos_rel = self.pointing[self.image_toa, :2] - image.loc
        theta = np.cos(self.pointing[self.image_toa, 2] - np.arctan2(pos_rel[:, 0], pos_rel[:, 1])) * np.sqrt(pos_rel[:, 0]**2 + pos_rel[:, 1]**2)
        
        # only populated baselines are selected
        for baseline_i in np.unique(self.baseline_indices):

            # which indices apply
            photons_in_baseline = self.baseline_indices == baseline_i

            # select baseline
            baseline = instrument.baselines[baseline_i]

            # allow for keeping track of rejected photon properties
            thetas = theta[photons_in_baseline]
            energies = self.actual_energies[photons_in_baseline]
            indices = np.arange(len(thetas))
            times = self.image_toa[photons_in_baseline]

            # initialize the sampled y positions
            y_pos = np.zeros(len(indices))

            # calculate all wavelengths
            wavelengths = spc.h * spc.c / energies

            # accept reject until all photons have been sampled
            while len(indices) > 0:

                # how many photons still need to be sampled in order to reach the specified number,
                # no photons are lost
                samples_left = len(indices)

                # range on the detector where photons can arrive
                width_box = np.array([(-baseline.W * baseline.num_pairs / 2) - baseline.L * thetas, (baseline.W * baseline.num_pairs / 2) - baseline.L * thetas])

                # Tested to be just above the maximum of the Fresnel diffraction,
                # for the height of the box needed by the accept reject method.
                height_box = [0, 2.7 * self.fre_dif(wavelengths, baseline, 1, y_pos = 0)[0]]

                # the fringe spacing, see Willingale (2004) Eq. 1
                fringe_spacing = (wavelengths / baseline.beam_angle)

                # the maximum number of visable fringes
                num_fringes = np.ceil((width_box[1] - width_box[0]) * baseline.num_pairs / fringe_spacing) #np.ceil(baseline.W * baseline.num_pairs / fringe_spacing)

                # location on the detector with pathlength diffrence of zero, assuming the mirrors
                # have been alligned to correct for the distance between the combining mirrors
                off_set_fringes = -(baseline.path_difference_change(times) + baseline.D * np.sin(thetas)) / (2 * np.sin(baseline.beam_angle / 2))

                # find the left most fringe partially or entirely able to reach the detector
                left_fringe = off_set_fringes - (np.ceil((off_set_fringes - baseline.L * thetas + baseline.W * baseline.num_pairs / 2) / fringe_spacing) * fringe_spacing)

                # samples where in the fringe each photon arrives
                rand_cos = 0.5 * fringe_spacing * sampler.cosine.rvs(loc = 2 * np.pi * left_fringe / fringe_spacing, size = samples_left) / np.pi
                
                if pure_diffraction:
                    # selects a random uniform position, resulting in no fringe sampling
                    rand_pos = np.random.uniform(width_box[0], width_box[1], size = samples_left)
                else:
                    # selects wich of the visable fringes each photon arrives in
                    rand_pos = rand_cos + (np.random.randint(low = 0, high = num_fringes + 1, size = samples_left) * fringe_spacing)
                
                # let the user know when the fringes would in practice move outside of the FoV
                if np.logical_or((off_set_fringes < width_box[0]).any(), (off_set_fringes > width_box[1]).any()):
                    warnings.warn("There are fringe centres outside of the FOV!")

                # boolean for wich y posistions don't fall outside of the allowed range,
                # due to sampling an integer number of fringes and not fractional ammounts
                valid_rand_pos = np.logical_and(rand_pos >= width_box[0], rand_pos <= width_box[1]) #np.repeat(True, samples_left) #

                # map sampled positions to the central diffraction pattern
                # the diffraction pattern is assumed to be multiple patterns
                # next to one another without interfering
                if (baseline.num_pairs % 2) == 0:
                    diffrac_pos = (abs(rand_pos + baseline.L * thetas) % baseline.W) - (baseline.W / 2) #rand_pos #
                else:
                    diffrac_pos = (abs(rand_pos + baseline.L * thetas + baseline.W / 2) % baseline.W) - (baseline.W / 2) #rand_pos #

                # accept reject
                diffraction_value, _ = self.fre_dif(wavelengths, baseline, 1, diffrac_pos)
                rand_value = np.random.uniform(height_box[0], height_box[1], size = samples_left)
                if pure_fringes:
                    accept_reject = np.repeat(True, samples_left)
                else:
                    accept_reject = rand_value <= diffraction_value

                # keep the accepted y posistions
                y_pos[indices[accept_reject * valid_rand_pos]] = rand_pos[accept_reject * valid_rand_pos]

                # only keep the rejected theta's and wavelengths for the next round of accept reject
                thetas = thetas[np.invert(accept_reject * valid_rand_pos)]
                wavelengths = wavelengths[np.invert(accept_reject * valid_rand_pos)]
                times = times[np.invert(accept_reject * valid_rand_pos)]
                indices = indices[np.invert(accept_reject * valid_rand_pos)]

            # here the sampled photon positions are saved
            actual_pos[photons_in_baseline] = y_pos

        """Test plots"""
            # plt.hist(actual_pos[photons_in_baseline] * 1e6, bins = 1000)#, weights = 1 / self.fre_dif(spc.h * spc.c / self.image_energies[photons_in_baseline], baseline, 1, (abs(self.actual_pos[photons_in_baseline] + baseline.L * theta[photons_in_baseline] + baseline.W / 2) % baseline.W) - (baseline.W / 2))[0])
            # plt.title("Baseline: {} m".format(baseline.D))
            # plt.xlabel("detector posistion (micron)")
            # plt.ylabel("counts")
            # plt.show()
        # plt.hist(self.actual_pos * 1e6, 1000, density = False)
        # plt.ylabel("counts")
        # plt.xlabel(r"$\mu m$")
        # plt.show()

        return actual_pos