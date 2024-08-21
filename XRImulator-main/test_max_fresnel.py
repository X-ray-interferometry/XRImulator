import numpy as np
import scipy.special as sps
import scipy.constants as spc
import matplotlib.pyplot as plt

from scipy.integrate import simpson


def fre_dif(wavelength, L, W, samples, y_pos = None):
            """
            Helper function that calculates the fresnell difraction pattern for two overlapping
            beams such as is the case in the interferometer. Does so according to a specified number
            of fringes to model out to, and a number of samples to use to interpolate between.
            """
            u_0 = W * np.sqrt(2 / (wavelength * L))
            u_1 = lambda u, u_0: u - u_0/2
            u_2 = lambda u, u_0: u + u_0/2

            # Only sample the slit size, or single value
            if y_pos is None:
                y_pos = np.linspace(-W / 2, W / 2, int(samples))
            u = y_pos * np.sqrt(2 / (wavelength * L))

            S_1, C_1 = sps.fresnel(u_1(u, u_0))
            S_2, C_2 = sps.fresnel(u_2(u, u_0))

            A = (C_2 - C_1 + 1j*(S_2 - S_1))
            A_star = np.conjugate(A)

            I = np.abs(A * A_star)

            # I_pmf = I / sum(I)

            # Repeat the diffraction patern for every slat gap pair, assuming that they don't interfere significantly
            return I, u


energies = np.arange(0.1, 100, 0.5)
wavelengths = spc.h * spc.c / (energies * 1e3 * spc.eV)
L = np.arange(0.1, 20.1, 1) # m
W = np.arange(1, 755, 5) * 1e-6 # m

# Willingale
L = 10
W = 300 * 1e-6

wavelength = 10 * 1e-10

intensity, u_values = fre_dif(wavelength, L, W, int(1e5), y_pos = None)
y = u_values / np.sqrt(2 / (wavelength * L))
plt.plot(y * 1e6, intensity)
plt.ylabel("Intensity")
plt.xlabel(r"Detector position ($\mu m$)")
plt.ylim(0, 3.5)
plt.xlim(-150, 150)
plt.savefig("diff_patten.pdf")
plt.show()

L = np.array([15])
W = np.array([200]) * 1e-6

comb_array = np.array(np.meshgrid(L, W, wavelengths)).T.reshape(-1, 3)
ratio_centre_max = []
ratio_accept = []


tot = len(comb_array)

for counter, triplet in enumerate(comb_array):

    intensity, u_values = fre_dif(triplet[2], triplet[0], triplet[1], int(1e4), y_pos = None)
    y = u_values / np.sqrt(2 / (triplet[2] * triplet[0]))

    max_intnsity = np.max(intensity)
    centre_intensity, _ = fre_dif(triplet[2], triplet[0], triplet[1], 1, y_pos = 0)

    ratio_centre_max.append(np.divide(max_intnsity, centre_intensity))
    ratio_accept.append(simpson(intensity, y) / (2.7 * fre_dif(triplet[2], triplet[0], triplet[1], int(1e4), y_pos = 0)[0] * triplet[1]))

    if (counter + 1) % (tot / 10) == 0:
        print("{}% complete".format((counter + 1) * 100 / tot))

    # print(2.7 * fre_dif(triplet[2], triplet[0], triplet[1], int(1e4), y_pos = 0)[0])

    # plt.axhline(y = 2.7 * fre_dif(triplet[2], triplet[0], triplet[1], int(1e4), y_pos = 0)[0], xmin = -triplet[1] * 1e6 / 2, xmax = triplet[1] * 1e6 / 2, color = "r")
    # plt.plot(y * 1e6, intensity)
    # plt.ylabel("pdf")
    # plt.xlabel(r"$\mu m$")
    # # plt.ylim(0, 10)
    # plt.show()

# maximum ratio between centre point and maximum
print("The biggest possible diffraction centre to maximum ratio is {:.3f}.".format(np.max(np.array(ratio_centre_max))))
print("The worst possible acceptance rate is {:.1f}%.".format(np.min(np.array(ratio_accept)) * 100))
print("The best possible acceptance rate is {:.1f}%.".format(np.max(np.array(ratio_accept)) * 100))

plt.plot(energies, ratio_centre_max)
plt.title("Willingale setup")
plt.xlabel("Energies (keV)")
plt.ylabel("Ratio between difraction center and peak")
plt.show()