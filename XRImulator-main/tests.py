import numpy as np
import scipy.special as sps
import scipy.constants as spc
import scipy.interpolate as spinter
import scipy.optimize as spopt
import scipy.fft as ft
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from PIL import Image

# testing
import random

import images
import instrument
import process
# import process_old_backup as process #
# import analysis_old as analysis
import analysis

from matplotlib.patches import Circle

# np.random.seed(99)

def fourier_transform(y_data, x_data, freq):
    return np.sum(y_data * np.exp(-2j * np.pi * freq * x_data)) / y_data.size

def ps_test():
    image = images.point_source(int(1e5), 0.0001, 0.00, 1.2)

    test_I = instrument.interferometer(0,0,0,0,0)
    test_I.add_baseline(1, 10, 300)#, 17000, 2, 1)

    start = time.time()
    test_data = process.process_image(test_I, image, 0, 10, int(1e4))
    print('Processing this image took ', time.time() - start, ' seconds')

    analysis.hist_data(test_data.discrete_pos, 100)
    ft_x_data, ft_y_data = analysis.ft_data(test_data)
    analysis.plot_ft(ft_x_data, ft_y_data, 2)

def dps_test():
    image = images.double_point_source(10000, [-.001, .001], [0, 0], [1.2, 6])

    test_I = instrument.interferometer(0,0,0,0,0)
    test_I.add_baseline(1, 10, 300)#, 17000, 2, 1)
    # print(image.loc[2,0])

    start = time.time()
    test_data = process.process_image(test_I, image, 0, 10, int(1e4))
    print('Processing this image took ', time.time() - start, ' seconds')

    analysis.hist_data(test_data.discrete_pos, 100)
    ft_x_data, ft_y_data = analysis.ft_data(test_data)
    analysis.plot_ft(ft_x_data, ft_y_data, 0)

def psmc_test():
    image = images.point_source_multichromatic_range(int(1e5), 0.000, 0, [1.2, 1.6])

    # TODO 10 micron is better pixel size
    test_I = instrument.interferometer(.1, 1, 10, np.array([1.2, 6]), np.array([-1500, 1500]))
    test_I.add_baseline(1, 10, 300)#, 17000, 2, 1)

    start = time.time()
    test_data = process.process_image(test_I, image, 0, 10, int(1e4))
    print('Processing this image took ', time.time() - start, ' seconds')
    test_data.discretize_E(test_I)
    test_data.discretize_pos(test_I)

    print(int(np.amax(test_data.discrete_pos) - np.amin(test_data.discrete_pos)))
    analysis.hist_data(test_data.discrete_pos, int(np.amax(test_data.discrete_pos) - np.amin(test_data.discrete_pos)))
    ft_x_data, ft_y_data = analysis.ft_data(test_data.pixel_to_pos(test_I))
    analysis.plot_ft(ft_x_data, ft_y_data, 2)

def Fre_test():
    # u_0 = np.linspace(4, 5, 100)
    u_0 = 4.5
    u_1 = lambda u, u_0: u + u_0/2
    u_2 = lambda u, u_0: u - u_0/2
    u = np.linspace(-5, 5, 1000) 

    u, u_0 = np.meshgrid(u, u_0, indexing='ij') 
    S_1, C_1 = sps.fresnel(u_1(u, u_0))
    S_2, C_2 = sps.fresnel(u_2(u, u_0))

    A = (C_2 - C_1 + 1j*(S_2 - S_1)) * (1 + np.exp(np.pi * 1j * u_0 * u))
    A_star = np.conjugate(A)

    I = A * A_star

    fig = plt.figure(figsize=(8, 6))
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(u, u_0, np.real(I))
    # ax.set_xlabel('u')
    # ax.set_ylabel('u_0')
    # ax.set_zlabel('I')

    plt.plot(u, np.real(I))

    plt.show()

def scale_test():
    func = lambda k, x: 2 + 2 * np.cos(k * x)
    x = np.linspace(-5, 5, 10000)
    k = 2 * np.pi / np.linspace(1, 10, 10000)
    x_grid, k_grid = np.meshgrid(x, k)
    I = func(k_grid, x_grid)

    fig = plt.figure(figsize=(8, 6))
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(x_grid, k_grid, I)
    # ax.set_xlabel('x')
    # ax.set_ylabel('k')
    # ax.set_zlabel('I')

    plt.plot(x, I[0,:])
    plt.plot(x, I[50,:])
    plt.show()

def scale_test2():
    func = lambda k, x: 2 + 2 * np.cos(k * x)
    x = np.linspace(0, 2, 1000)
    plt.plot(x, func(1, x), label="1, x")
    plt.plot(x, func(2, x), label="2, x")
    plt.plot(2*x, func(2, x), label="2, 2x")
    plt.legend()
    plt.show()

def discretize_E_test(E_range, res_E, data):
    """
    Function that discretizes energies of incoming photons into energy channels.

    Parameters:
    data (interferometer-class object): data object containing the energy data to discretize.
    """
    E_edges = np.arange(E_range[0], E_range[1], res_E)
    E_binner = spinter.interp1d(E_edges, E_edges, 'nearest', bounds_error=False)
    return E_binner(data.energies)

def discretize_test():
    data = process.interferometer_data(10)
    data.energies = np.array([1.6, 2.3, 3.1, 4.2, 5.5, 6.0, 7.3, 8.1, 9.9, 10.6])

    print(discretize_E_test([0, 11], 1, data))

#wobble point source
def w_ps_test():
    image = images.point_source_multichromatic_range(int(1e5), 0.0001, 0, [1.2, 1.6])

    # TODO 10 micron is better pixel size
    test_I = instrument.interferometer(.1, .01, 10, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.001, None, instrument.interferometer.smooth_roller, 
                                        .01 * 2 * np.pi, 10, np.pi/4)
    test_I.add_baseline(1, 10, 300)#, 17000, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, int(1e4))
    print('Processing this image took ', time.time() - start, ' seconds')

    analysis.hist_data(test_data.pixel_to_pos(test_I), int(np.amax(test_data.discrete_pos) - np.amin(test_data.discrete_pos)) + 1, False)
    ft_x_data, ft_y_data = analysis.ft_data(test_data.pixel_to_pos(test_I))
    analysis.plot_ft(ft_x_data, ft_y_data, 2)

def willingale_test():
    image = images.point_source_multichromatic_range(int(1e5), 0.00, 0, [1.2, 1.6])

    test_I = instrument.interferometer(.1, .01, 4, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.001, None, instrument.interferometer.smooth_roller, 
                                        .00001 * 2 * np.pi, 10, np.pi/4)
    test_I.add_baseline(.035, 10, 300)#, 1200, 2, 1)
    test_I.add_baseline(.105, 10, 300)#, 3700, 2, 1)
    test_I.add_baseline(.315, 10, 300)#, 11100, 2, 1)
    test_I.add_baseline(.945, 10, 300)#, 33400, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, int(1e4))
    print('Processing this image took ', time.time() - start, ' seconds')

    # for i in range(4):
    #     analysis.hist_data(test_data.pixel_to_pos(test_I)[test_data.baseline_indices == i], 
    #                         int(np.amax(test_data.discrete_pos[test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[test_data.baseline_indices == i])) + 1, False, i)
    # plt.legend()
    # plt.show()

    # for i in range(4):
    #     ft_x_data, ft_y_data, edges = analysis.ft_data(test_data.pixel_to_pos(test_I)[test_data.baseline_indices == i])
    #     analysis.plot_ft(ft_x_data, ft_y_data, 0, i)
    # delta_u = 1 / np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (np.array([1.2, 1.6]) * 1.602177733e-16 * 10))
    # plt.axvline(delta_u[0], 1e-5, 1e4)
    # plt.axvline(delta_u[1], 1e-5, 1e4)
    # plt.legend()
    # plt.xlim(-2 * delta_u[1], 2 * delta_u[1])
    # plt.show()

    # test = np.linspace(-4, 4, 1000)
    # plt.plot(test, test_data.inter_pdf(test))
    # plt.show()

    ft_data, re_im, f_grid = analysis.image_recon_smooth(test_data, test_I, test_data.pointing, .01 * 2 * np.pi)
    # for i in range(4):
    #     ft_base = ft_data[ft_data[:, 3] == i]
    #     plt.plot(ft_base[:, 1], ft_base[:, 2], '.', label=f'baseline {i}')
    # plt.legend()
    # plt.show()

    plt.imshow(abs(f_grid), cmap=cm.Reds)
    plt.show()

    plt.imshow(abs(re_im), cmap=cm.Greens)
    # plt.plot(re_im[:,0], re_im[:,2], label='0-2')
    # plt.legend()
    plt.show()

    # plt.plot(re_im[:,0], re_im[:,1])
    # plt.show()

def image_re_test():
    # TODO Try without diffraction, single baseline, with something with a peak at the centre
    # Try starting simple, add effects until divergence point
    # Maybe don't even throw it through the sampling

    # TODO check fringe generation to be sure

    # TODO check with basic sine wave outside normal code

    # TODO Echt echt echt verder kijken naar de fases

    # TODO kijk naar diagnostic codes
    image = images.double_point_source(int(1e5), [0.0001, -.0001], [0.0001, -.0001], [1.2, 1.2])

    test_I = instrument.interferometer(.1, .01, 4, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        .0005 * 2 * np.pi)
    # test_I.add_baseline(.035, 300, 1200, 2, 1)
    # test_I.add_baseline(.105, 300, 3700, 2, 1)
    # test_I.add_baseline(.315, 300, 11100, 2, 1)
    test_I.add_baseline(.945, 10, 300)#, 33400, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, 512)
    print('Processing this image took ', time.time() - start, ' seconds')

    exp = test_I.baselines[0].F * np.cos(-np.arctan(image.loc[0, 0] / (image.loc[0, 1] + 1e-20))) * np.sqrt(image.loc[0, 0]**2 + image.loc[0, 1]**2) * 1e6
    for i in range(len(test_I.baselines)):
        analysis.hist_data(test_data.actual_pos[test_data.baseline_indices == i], 
                            int(np.ceil(test_I.baselines[i].W / test_I.res_pos)) + 1, False, i)
        # print(test_I.baselines[i].F / test_I.baselines[i].D)
    # print(exp /33400 * 1e-6)
    plt.vlines(-exp, -100, 10000)
    plt.title('Photon impact positions on detector')
    plt.ylim(0, 5000)
    plt.legend()
    plt.show()

    colourlist = ['b', 'orange', 'g', 'r']
    for i in range(len(test_I.baselines)):
        samples = int(np.ceil(test_I.baselines[i].W / test_I.res_pos)) + 1
        ft_x_data, ft_y_data, edges = analysis.ft_data(test_data.pixel_to_pos(test_I)[test_data.baseline_indices == i], samples)
        analysis.plot_ft(ft_x_data, ft_y_data, 0, i)
        delta_u = 1 / np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * 1.602177733e-16 * 10))
        plt.axvline(delta_u, 1e-5, 1e4, color = colourlist[i])
        plt.axvline(-delta_u, 1e-5, 1e4, color = colourlist[i])
    plt.xlim(-4 * delta_u, 4 * delta_u)
    plt.title('Fourier transform of photon positions')
    plt.xlabel('Spatial frequency ($m^{-1}$)')
    plt.ylabel('Fourier magnitude')
    plt.legend()
    plt.show()

    start = time.time()
    test_data_imre = np.zeros((512, 512))
    test_data_imre[256, 256] = 1
    test_data_imre = ft.fft2(test_data_imre)
    re_im, f_grid, test_data_masked = analysis.image_recon_smooth(test_data, test_I, .01 * 2 * np.pi, samples=512, test_data=test_data_imre)
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    test_image = ft.ifft2(test_data_imre)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(abs(ft.fftshift(f_grid)), cmap=cm.Reds)
    ax1.set_title('UV-plane (amplitude)')
    ax2.imshow(abs(1j * ft.fftshift(f_grid)), cmap=cm.Blues)
    ax2.set_title('UV-plane (phase)')
    ax3.imshow(abs(re_im), cmap=cm.Greens)
    ax3.set_title('Reconstructed image')
    ax4.imshow(abs(ft.fftshift(test_data_masked)), cmap=cm.Blues)
    ax4.set_title('UV-plane (amplitude)')
    ax5.imshow(abs(1j * ft.fftshift(test_data_masked)), cmap=cm.Greens)
    ax5.set_title('UV-plane (phase)')
    ax6.imshow(abs(test_image), cmap=cm.Greens)
    ax6.set_title('Reconstructed image')
    plt.show()

def stats_test():
    """This test exists to do some statistical testing"""
    offset = 0e-6
    image = images.point_source(int(1e5), 0.000, offset, 1.2)

    no_sims = 1000
    simulated = np.zeros((no_sims, 4), dtype=np.complex_)
    masked = np.zeros((no_sims, 4), dtype=np.complex_)

    for sim in range(no_sims):
        test_I = instrument.interferometer(.1, 1, .5, np.array([1.2, 6]), np.array([-400, 400]), 
                                            0.00, None, instrument.interferometer.smooth_roller, 
                                            .00000 * 2 * np.pi, roll_init=0/4 * np.pi)
        test_I.add_baseline(.035, 10, 300)#, 1200, 2, 1)
        test_I.add_baseline(.105, 10, 300)#, 3700, 2, 1)
        # test_I.add_baseline(.315, 10, 300)#, 11100, 2, 1)
        # test_I.add_baseline(.945, 10, 300)#, 33400, 2, 1)

        test_data = process.interferometer_data(test_I, image, 10, 512)

        test_data_imre = np.zeros((512, 512))
        test_data_imre[256, 256] = 1
        test_data_imre_fft = ft.fft2(test_data_imre)
        re_im, f_grid, test_data_masked = analysis.image_recon_smooth(test_data, test_I, .02 * 2 * np.pi, samples=test_data_imre.shape, test_data=test_data_imre_fft, exvfast=0)

        simulated[sim] = f_grid[f_grid.nonzero()] / np.sum(np.abs(f_grid[f_grid.nonzero()]))
        masked[sim] = test_data_masked[f_grid.nonzero()] / np.sum(np.abs(test_data_masked[f_grid.nonzero()]))

        if sim % 10 == 0:
            print(f'Done with sim {sim}')

    print(f'Average amplitude of {no_sims} normalised nonzero points in interferometer plane: {np.mean(np.abs(simulated), axis=0)} +/- {np.std(np.abs(simulated), axis=0) / np.sqrt(no_sims)}')
    print(f'Average phase of {no_sims} normalised nonzero points in interferometer plane: {np.mean(np.angle(simulated), axis=0)} +/- {np.std(np.angle(simulated), axis=0) / np.sqrt(no_sims)}')
    print(f'Average amplitude of {no_sims} normalised nonzero points in masked plane: {np.mean(np.abs(masked), axis=0)}  +/- {np.std(np.abs(masked), axis=0) / np.sqrt(no_sims)}')
    print(f'Average phase of {no_sims} normalised nonzero points in masked plane: {np.mean(np.angle(masked), axis=0)}  +/- {np.std(np.angle(masked), axis=0) / np.sqrt(no_sims)}')
    
def image_re_test_uv():
    # m = 10
    # locs = np.linspace(0, 2 * np.pi, m)
    # image = images.m_point_sources(int(1e6), m, [0.000 * np.sin(x) for x in locs], [0.0005 * np.cos(x) for x in locs], [1.2 for x in locs])

    offset = 0e-6
    image = images.point_source(int(1e6), 0.000, offset, 1.2)
    # image = images.double_point_source(int(1e6), [0.000, 0.000], [0.0005, -0.0005], [1.2, 1.2])

    test_I = instrument.interferometer(.1, 1, .5, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        .00000 * 2 * np.pi, roll_init=0/4 * np.pi)
    test_I.add_baseline(.035, 10, 300)#, 1200, 2, 1)
    # test_I.add_baseline(.105, 10, 300)#, 3700, 2, 1)
    # test_I.add_baseline(.315, 10, 300)#, 11100, 2, 1)
    # test_I.add_baseline(.945, 10, 300)#, 33400, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, 100000)
    print('Processing this image took ', time.time() - start, ' seconds')

    colourlist = ['b', 'orange', 'g', 'r']
    for i in range(len(test_I.baselines)):
        exp = test_I.baselines[i].F * np.cos(test_data.pointing[0, 2] - np.arctan2(image.loc[0, 0], (image.loc[0, 1] + 1e-20))) * np.sqrt(image.loc[0, 0]**2 + image.loc[0, 1]**2) * 1e6
        delta_y = np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
        # analysis.hist_data(test_data.actual_pos[(test_data.pointing[test_data.discrete_t, 2] >= test_I.roll_init - .01 * np.pi) * (test_data.pointing[test_data.discrete_t, 2] < test_I.roll_init + .01 * np.pi) * (test_data.baseline_indices == i)], 
        #                     int(np.amax(test_data.discrete_pos[test_data.baseline_indices == i]) - 
        #                     np.amin(test_data.discrete_pos[test_data.baseline_indices == i])) + 1, False, i)
        print(int(np.amax(test_data.discrete_pos[test_data.baseline_indices == i]) - np.amin(test_data.discrete_pos[test_data.baseline_indices == i])) + 1)
        analysis.hist_data(test_data.pixel_to_pos(test_I)[(test_data.pointing[test_data.discrete_t, 2] >= test_I.roll_init - .01 * np.pi) * (test_data.pointing[test_data.discrete_t, 2] < test_I.roll_init + .01 * np.pi) * (test_data.baseline_indices == i)], 
                            int(np.ceil(test_I.baselines[i].W / test_I.res_pos)) + 1, False, i)
        plt.vlines(exp, -100, 10000, color=colourlist[i])
        plt.vlines(exp + (delta_y * np.arange(-5, 5, 1))*1e6, -100, 10000, color=colourlist[i])
        plt.title(f'Photon impact positions on detector at roll of {test_I.roll_init / np.pi} pi rad')
        plt.ylim(0, 8000)
        plt.xlim(-200, 200)
        plt.legend()
        plt.show()

    # test_freq = np.fft.fftfreq(test_data.size)
    # plt.plot(test_freq, np.fft.fftshift(np.fft.fft(test_data.test_data)))
    # plt.show()

    for i in range(len(test_I.baselines)):
        samples = int(np.ceil(test_I.baselines[i].W / test_I.res_pos)) + 1
        binned_data, edges = np.histogram(test_data.actual_pos[:,1][test_data.baseline_indices == i], samples)
        centres = edges[:-1] + (edges[1:] - edges[:-1])/2
        print(centres)
        ft_x_data, ft_y_data = analysis.ft_data(binned_data, samples, edges[1] - edges[0])

        fig, (ax1, ax2) = plt.subplots(2,1)
        delta_u = 1 / np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
        exact = (np.sum(binned_data * np.exp(-2j * np.pi * -delta_u * centres)) / binned_data.size, np.sum(binned_data * np.exp(-2j * np.pi * delta_u * centres)) / binned_data.size)
        fig.suptitle(f'Exact values for amplitude: {np.round(np.abs(exact), 5)}, for phase: {np.round(np.angle(exact), 5)} at offset {np.round(offset, 6)} \"')

        print(exact, np.abs(exact), np.angle(exact))

        analysis.plot_ft(ft_x_data, np.abs(ft_y_data), ax1, 0, i)
        ax1.axvline(delta_u, 1e-5, 1e4, color = 'k', label='Expected frequency')
        ax1.axvline(-delta_u, 1e-5, 1e4, color = 'k')
        ax1.plot([-delta_u, delta_u], np.abs(exact), 'ro', label='Exact value')
        ax1.set_xlim(-4 * delta_u, 4 * delta_u)
        ax1.set_title(f'Fourier transform of photon positions at roll of {test_I.roll_init / (np.pi)} pi rad')
        ax1.set_xlabel('Spatial frequency ($m^{-1}$)')
        ax1.set_ylabel('Fourier magnitude')
        ax1.legend()

        analysis.plot_ft(ft_x_data, np.angle(ft_y_data), ax2, 0, i)
        ax2.axvline(delta_u, 1e-5, 1e4, color = 'k', label='Expected frequency')
        ax2.axvline(-delta_u, 1e-5, 1e4, color = 'k')
        ax2.plot([-delta_u, delta_u], np.angle(exact), 'ro', label='Exact value')
        ax2.set_xlim(-4 * delta_u, 4 * delta_u)
        ax2.set_title(f'Fourier transform of photon positions at roll of {test_I.roll_init / (np.pi)} pi rad')
        ax2.set_xlabel('Spatial frequency ($m^{-1}$)')
        ax2.set_ylabel('Fourier phase')
        ax2.legend()

    # plt.show()

    start = time.time()
    test_data_imre = np.zeros((512, 512))
    test_data_imre[256, 256] = 1
    # test_data_imre[12, 12] = 1
    test_data_imre_fft = ft.fft2(test_data_imre)
    re_im, f_grid, test_data_masked = analysis.image_recon_smooth(test_data, test_I, .02 * 2 * np.pi, samples=test_data_imre.shape, test_data=test_data_imre_fft, exvfast=0)
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    test_image = ft.ifft2(test_data_masked)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    ax1.imshow(np.abs(ft.fftshift(f_grid)), cmap=cm.Reds)
    ax1.set_title('UV-plane (amplitude)')
    ax2.imshow(np.angle(ft.fftshift(f_grid)), cmap=cm.Blues)
    ax2.set_title('UV-plane (phase)')
    ax3.imshow(np.abs(ft.fftshift(re_im)), cmap=cm.Greens)
    ax3.set_title('Reconstructed image')
    # ax3.plot(256, 256, 'r.')
    ax4.imshow(np.abs(ft.fftshift(test_data_masked)), cmap=cm.Blues)
    ax4.set_title('UV-plane (amplitude)')
    ax5.imshow(np.angle(ft.fftshift(test_data_masked)), cmap=cm.Greens)
    ax5.set_title('UV-plane (phase)')
    ax6.imshow(np.abs(test_image), cmap=cm.Greens)
    ax6.set_title('Reconstructed image')

    ax7.imshow(np.abs(ft.fftshift(test_data_imre_fft)), cmap=cm.Blues)
    ax7.set_title('Full UV-plane (amplitude)')
    # ax7.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax7.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax8.imshow(np.angle(ft.fftshift(test_data_imre_fft)), cmap=cm.Greens)
    ax8.set_title('Full UV-plane (phase)')
    # ax8.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax8.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax9.imshow(np.abs(ft.ifft2(test_data_imre_fft)), cmap=cm.Greens)
    ax9.set_title('Image')
    plt.show()

    print(f'Normalised nonzero points in interferometer plane: {f_grid[f_grid.nonzero()] / np.sum(np.abs(f_grid[f_grid.nonzero()]))}')
    print(f'Normalised nonzero points in masked plane: {test_data_masked[f_grid.nonzero()] / np.sum(np.abs(test_data_masked[f_grid.nonzero()]))}')

def image_re_test_parts():
    # m = 10
    # locs = np.linspace(0, 2 * np.pi, m)
    # image = images.m_point_sources(int(1e6), m, [0.000 * np.sin(x) for x in locs], [0.0005 * np.cos(x) for x in locs], [1.2 for x in locs])

    #TODO look at digitize function

    #TODO look at testing with sinusoid instead of point source

    offset = 000e-6
    energy = 1.2
    image = images.point_source(int(1e5), 0.000, offset, energy*5)
    # image = images.point_source_multichromatic_range(int(1e5), 0.000, offset, [energy, energy*2])
    # image_2 = images.point_source(int(1e5), 0.000, offset, energy * 2)
    # image_4 = images.point_source(int(1e5), 0.000, offset, energy * 4)
    # image = images.double_point_source(int(1e5), [0.000, 0.000], [0.001, -0.001], [1.2, 1.2])

    test_I = instrument.interferometer(.1, 1, .2, np.array([1.2, 7]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        .00000 * 2 * np.pi, roll_init=0/4 * np.pi)
    # test_I.add_baseline(.035, 10, 300)#, 1200)
    # test_I.add_baseline(.105, 10, 300)#, 3700)
    # test_I.add_baseline(.315, 10, 300)#, 11100)
    test_I.add_baseline(.945, 10, 300)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 100000, 2)
    # test_data_2 = process.interferometer_data(test_I, image, 100000, 20)
    # test_data_4 = process.interferometer_data(test_I, image_2, 100000)
    print('Processing this image took ', time.time() - start, ' seconds')

    bins = int(np.ceil(abs(test_I.pos_range[0] - test_I.pos_range[1]) / test_I.res_pos))
    edges = np.array([test_I.pos_range[0] + i * test_I.res_pos for i in range(bins + 1)]) * 1e6
    centres = edges[:-1] + (edges[1:] - edges[:-1])/2

    colourlist = ['b', 'orange', 'g', 'r']
    for i in range(len(test_I.baselines)):
        # exp = test_I.baselines[i].F * np.cos(test_data.pointing[0, 2] - np.arctan2(image.loc[0, 0], (image.loc[0, 1]))) * np.sqrt(image.loc[0, 0]**2 + image.loc[0, 1]**2) * 1e6
        # delta_y = test_I.baselines[i].L * spc.h * spc.c / (energy * spc.eV * 1e3 * test_I.baselines[i].W) * 1e6
        # plt.hist(test_data.actual_pos, 
        #                     int(np.ceil(abs(test_I.pos_range[0] - test_I.pos_range[1]) / (test_I.res_pos))), label='No noise')

        baseline = test_I.baselines[i]
        wavelength = spc.c * spc.h / ((energy*5) * spc.eV * 1e3)

        u_0 = baseline.W * np.sqrt(2 / (wavelength * baseline.L))
        u_1 = lambda u, u_0: u + u_0/2
        u_2 = lambda u, u_0: u - u_0/2

        # Times 3 to probe a large area for the later interpolation
        u = np.linspace(-u_0, u_0, 100000)
        y = (u / np.sqrt(2 / (wavelength * baseline.L)) + baseline.F * offset * (2 * np.pi / (3600 * 360)))

        S_1, C_1 = sps.fresnel(u_1(u, u_0))
        S_2, C_2 = sps.fresnel(u_2(u, u_0))

        A = (C_2 - C_1 + 1j*(S_2 - S_1)) * (1 + np.exp(np.pi * 1j * u_0 * u))
        A_star = np.conjugate(A)

        I = np.abs(A * A_star)
        I_pmf = I / sum(I) * 1e5 * 1e5 / (edges[abs(edges) < 300].size)

        hist_1, _ = np.histogram(test_data.pixel_to_pos(test_I)*1e6, edges)
        # hist_2, _ = np.histogram(test_data_4.pixel_to_pos(test_I)*1e6, edges)
        # hist_com = hist_1 + hist_2

        fig = plt.figure(figsize=(8,6))
        # plt.hist(test_data.pixel_to_pos(test_I)*1e6 , edges, label='1.2 keV photons')
        # plt.hist(test_data_4.pixel_to_pos(test_I)*1e6 , edges, label='2.4 keV photons')

        plt.bar(centres, hist_1, label='Simulated photons')
        # plt.bar(centres, hist_2, label='2.4 keV photons', alpha=.5)
        # plt.bar(centres, hist_com, label='Combined photons', alpha=.3)

        plt.plot(0, 1, '.')
        plt.plot(y*1e6, I_pmf, label='pmf', alpha=.8)
        # plt.hist(test_data_2.pixel_to_pos(test_I)*1e6, edges, label='x$_{res}$ = 20 $\\mu$m')
        # plt.hist(test_data_4.pixel_to_pos(test_I)*1e6, edges, label=f'{energy * 4} keV')
        # plt.vlines(exp, -100, 10000, color=colourlist[i])
        # plt.vlines(exp + (delta_y * np.arange(-5, 5, 1)), -100, 10000, color=colourlist[i])
        # plt.title(f'Interferometric fringe pattern with diffraction at offset {offset*1e6} $\\mu$as')
        plt.title(f'Interferometric fringe pattern with diffraction')
        plt.xlabel('Photon impact positions ($\\mu$m)')
        plt.ylabel('Number of photons in bin')
        # plt.ylim(0, 8000)
        # plt.xlim(-400, 400)
        plt.legend(title='E=6 keV\nD = 1 m\n$\\delta$y = 2 $\\mu$m')
    plt.show()

    # test_freq = np.fft.fftfreq(test_data.size)
    # plt.plot(test_freq, np.fft.fftshift(np.fft.fft(test_data.test_data)))
    # plt.show()

    # for i in range(len(test_I.baselines)):
    #     samples = int(np.ceil(abs(test_I.pos_range[0] - test_I.pos_range[1]) / test_I.res_pos)) + 1
    #     binned_data, edges = np.histogram(test_data.actual_pos[test_data.baseline_indices == i], samples)
    #     centres = edges[:-1] + (edges[1:] - edges[:-1])/2
    #     ft_x_data, ft_y_data = analysis.ft_data(binned_data, samples, edges[1] - edges[0])

    #     fig, (ax1, ax2) = plt.subplots(2,1)
    #     delta_u = 1 / (test_I.baselines[i].L * spc.h * spc.c / (energy * spc.eV * 1e3 * test_I.baselines[i].W))
    #     exact = (np.sum(binned_data * np.exp(-2j * np.pi * -delta_u * centres)) / binned_data.size, np.sum(binned_data * np.exp(-2j * np.pi * delta_u * centres)) / binned_data.size)
    #     zero = (np.sum(binned_data) / binned_data.size)
    #     fig.suptitle(f'Exact values for amplitude: {np.round(np.abs(exact), 5)}, for phase: {np.round(np.angle(exact), 5)} at offset {np.round(offset, 6)} \"')

    #     print(np.abs(exact), np.abs(zero), np.abs(exact) / np.abs(zero))

    #     analysis.plot_ft(ft_x_data, np.abs(ft_y_data), ax1, 0, i)
    #     ax1.axvline(delta_u, 1e-5, 1e4, color = 'k', label='Expected frequency')
    #     ax1.axvline(-delta_u, 1e-5, 1e4, color = 'k')
    #     ax1.plot([-delta_u, delta_u], np.abs(exact), 'ro', label='Exact value')
    #     ax1.set_xlim(-4 * delta_u, 4 * delta_u)
    #     ax1.set_title(f'Fourier transform of photon positions at roll of {test_I.roll_init / (np.pi)} pi rad')
    #     ax1.set_xlabel('Spatial frequency ($m^{-1}$)')
    #     ax1.set_ylabel('Fourier magnitude')
    #     ax1.legend()

    #     analysis.plot_ft(ft_x_data, np.angle(ft_y_data), ax2, 0, i)
    #     ax2.axvline(delta_u, 1e-5, 1e4, color = 'k', label='Expected frequency')
    #     ax2.axvline(-delta_u, 1e-5, 1e4, color = 'k')
    #     ax2.plot([-delta_u, delta_u], np.angle(exact), 'ro', label='Exact value')
    #     # ax2.plot([-delta_u * 2, delta_u * 2], [2 * np.pi * exp/delta_y, 2 * np.pi * exp/delta_y], 'g-', label='Expectation phase')
    #     ax2.set_xlim(-4 * delta_u, 4 * delta_u)
    #     ax2.set_title(f'Fourier transform of photon positions at roll of {test_I.roll_init / (np.pi)} pi rad')
    #     ax2.set_xlabel('Spatial frequency ($m^{-1}$)')
    #     ax2.set_ylabel('Fourier phase')
    #     ax2.legend()

    # plt.show()

    # start = time.time()
    # test_data_imre = np.zeros((512, 512))
    # test_data_imre[256, 256] = 1
    # test_data_imre = ft.fft2(test_data_imre)
    # re_im, f_grid, test_data_masked = analysis.image_recon_smooth(test_data, test_I, .02 * 2 * np.pi, samples=[512,512], test_data=test_data_imre, exvfast=0)
    # print('Reconstructing this image took ', time.time() - start, ' seconds')

    # test_image = ft.ifft2(test_data_masked)

    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    # ax1.imshow(np.abs(ft.fftshift(f_grid)), cmap=cm.Reds)
    # ax1.set_title('UV-plane (amplitude)')
    # ax2.imshow(np.angle(ft.fftshift(f_grid)), cmap=cm.Blues)
    # ax2.set_title('UV-plane (phase)')
    # ax3.imshow(np.abs(ft.fftshift(re_im)), cmap=cm.Greens)
    # ax3.set_title('Reconstructed image')
    # # ax3.plot(256, 256, 'r.')
    # ax4.imshow(np.abs(ft.fftshift(test_data_masked)), cmap=cm.Blues)
    # ax4.set_title('UV-plane (amplitude)')
    # ax5.imshow(np.angle(ft.fftshift(test_data_masked)), cmap=cm.Greens)
    # ax5.set_title('UV-plane (phase)')
    # ax6.imshow(np.abs(test_image), cmap=cm.Greens)
    # ax6.set_title('Reconstructed image')
    # plt.show()

def image_re_test_point():
    image = images.point_source(int(1e6), 0.0005, 0.000, 1.2)

    test_I = instrument.interferometer(.1, 1, .5, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        .015 * 2 * np.pi)
    test_I.add_baseline(.035, 10, 300)#, 1200, 2, 1)
    test_I.add_baseline(.105, 10, 300)#, 3700, 2, 1)
    test_I.add_baseline(.315, 10, 300)#, 11100, 2, 1)
    test_I.add_baseline(.945, 10, 300)#, 33400, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, 512)
    print('Processing this image took ', time.time() - start, ' seconds')

    # plt.plot(test_data.test_data)
    # plt.show()

    # colourlist = ['b', 'orange', 'g', 'r']
    # for i in range(len(test_I.baselines)):
    #     exp = test_I.baselines[i].F * np.cos(-np.arctan2(image.loc[0, 0], (image.loc[0, 1] + 1e-20))) * np.sqrt(image.loc[0, 0]**2 + image.loc[0, 1]**2) * 1e6
    #     delta_y = np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
    #     analysis.hist_data(test_data.actual_pos[(test_data.pointing[test_data.discrete_t, 2] > 0) * (test_data.pointing[test_data.discrete_t, 2] < .1 * np.pi) * (test_data.baseline_indices == i)], 
    #                         int(np.amax(test_data.discrete_pos[test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[test_data.baseline_indices == i])) + 1, False, i)
    #     plt.vlines(exp, -100, 10000, color=colourlist[i])
    #     plt.vlines(exp + (delta_y * np.arange(-5, 5, 1))*1e6, -100, 10000, color=colourlist[i])
    #     # print(test_I.baselines[i].F / test_I.baselines[i].D)
    # # print(exp /33400 * 1e-6)
    # plt.title('Photon impact positions on detector')
    # plt.ylim(0, 200)
    # plt.xlim(-200, 200)
    # plt.legend()
    # plt.show()

    # test_freq = np.fft.fftfreq(test_data.size)
    # plt.plot(test_freq, np.fft.fftshift(np.fft.fft(test_data.test_data)))
    # plt.show()

    # for i in range(len(test_I.baselines)):
    #     samples = int(np.amax(test_data.discrete_pos[test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[test_data.baseline_indices == i])) + 1
    #     binned_data, edges = np.histogram(test_data.actual_pos[:,1][test_data.baseline_indices == i], samples)
    #     ft_x_data, ft_y_data = analysis.ft_data(binned_data, samples, edges[1] - edges[0])
    #     analysis.plot_ft(ft_x_data, ft_y_data, 0, i)
    #     delta_u = 1 / np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
    #     # delta_guess = 10**6 / 30.6
    #     plt.axvline(delta_u, 1e-5, 1e4, color = colourlist[i])
    #     plt.axvline(-delta_u, 1e-5, 1e4, color = colourlist[i])
    #     # plt.axvline(delta_guess, 1e-5, 1e4, color = 'r')
    #     # plt.axvline(-delta_guess, 1e-5, 1e4, color = 'r')
    # plt.xlim(-4 * delta_u, 4 * delta_u)
    # plt.title('Fourier transform of photon positions')
    # plt.xlabel('Spatial frequency ($m^{-1}$)')
    # plt.ylabel('Fourier magnitude')
    # plt.legend()
    # plt.show()

    start = time.time()
    test_data_imre = np.zeros((512, 512))
    test_data_imre[264, 256] = 1
    # test_data_imre[261, 261] = 1
    # test_data_imre[12, 12] = 1
    test_data_imre = ft.fft2(test_data_imre)
    re_im, f_grid, test_data_masked = analysis.image_recon_smooth(test_data, test_I, .02 * 2 * np.pi, samples=[512, 512], test_data=test_data_imre)
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    test_image = ft.ifft2(test_data_masked)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(abs(ft.fftshift(f_grid)), cmap=cm.Reds)
    ax1.set_title('UV-plane (amplitude)')
    ax2.imshow(np.imag(ft.fftshift(f_grid)), cmap=cm.Blues)
    ax2.set_title('UV-plane (phase)')
    ax3.imshow(abs(ft.fftshift(re_im)), cmap=cm.Greens)
    ax3.set_title('Reconstructed image')
    # ax3.plot(256, 256, 'r.')
    ax4.imshow(abs(ft.fftshift(test_data_masked)), cmap=cm.Blues)
    ax4.set_title('UV-plane (amplitude)')
    ax5.imshow(np.imag(ft.fftshift(test_data_masked)), cmap=cm.Greens)
    ax5.set_title('UV-plane (phase)')
    ax6.imshow(abs(test_image), cmap=cm.Greens)
    ax6.set_title('Reconstructed image')
    plt.show()

def sinety_test():
    f = 4.387
    phase = np.pi / 4.5
    xend = 3.17
    samples = 1000

    x = np.linspace(0, xend, samples)
    y = np.cos(f * 2 * np.pi * x + phase)
    four_x, fast_y = analysis.ft_data(y, samples, xend/samples)
    four_x = np.fft.fftshift(four_x)
    
    exact_y = np.zeros(four_x.size, dtype=np.complex_)
    for i, freq in enumerate(four_x):
        exact_y[i] = np.sum(y * np.exp(-2j * np.pi * freq * x)) / y.size

    fast_y = np.fft.fftshift(fast_y)
    # y_data, edges = np.histogram(data, samples)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(x, y)
    ax1.set_ylim(-3, 3)
    ax1.set_title(f'Cosine with frequency pi and phase {phase / np.pi} pi')

    ax2.plot(four_x, np.abs(fast_y), '--', label='Fast')
    ax2.plot(four_x, np.abs(exact_y), '-.', label='Exact')
    ax2.plot([-f, f], [np.abs(np.sum(y * np.exp(-2j * np.pi * -f * x)) / y.size), np.abs(np.sum(y * np.exp(-2j * np.pi * f * x)) / y.size)], 'ro', label='Extra exact')

    # ax2.vlines([-5, 5], [-2, -2], [2, 2], colors='r')
    ax2.set_ylim(-1, 1)
    ax2.set_xlim(-6, 6)
    ax2.set_title('Amplitude of fourier transform')
    ax2.legend()

    ax3.plot(four_x, np.angle(fast_y), '--', label='Fast')
    ax3.plot(four_x, np.angle(exact_y), '-.', label='Exact')
    ax3.plot(four_x, np.angle(exact_y), '.', label='Exact')
    ax3.plot(four_x, [phase for i in four_x], 'r:', label='Phase')
    ax3.plot(four_x, [-phase for i in four_x], 'r:')
    ax3.plot([-f, f], [np.angle(np.sum(y * np.exp(-2j * np.pi * -f * x)) / y.size), np.angle(np.sum(y * np.exp(-2j * np.pi * f * x)) / y.size)], 'ro', label='Extra exact')
    ax3.set_ylim(-np.pi, np.pi)
    ax3.set_xlim(-6, 6)
    ax3.set_title('Phase of fourier transform')
    ax3.legend()
    plt.show()


def sinetier_test():
    f = 4.387
    phase = np.pi / 4.5
    xend = 3.17
    samples = 1000

    x = np.linspace(0, xend, samples)
    y = np.cos(f * 2 * np.pi * x + phase)
    four_x, fast_y = analysis.ft_data(y, samples, xend/samples)
    four_x = np.fft.fftshift(four_x)
    
    exact_y = np.zeros(four_x.size, dtype=np.complex_)
    for i, freq in enumerate(four_x):
        exact_y[i] = np.sum(y * np.exp(-2j * np.pi * freq * x)) / y.size

    fast_y = np.fft.fftshift(fast_y)
    # y_data, edges = np.histogram(data, samples)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(x, y)
    ax1.set_ylim(-3, 3)
    ax1.set_title(f'Cosine with frequency pi and phase {phase / np.pi} pi')

    ax2.plot(four_x, np.abs(fast_y), '--', label='Fast')
    ax2.plot(four_x, np.abs(exact_y), '-.', label='Exact')
    ax2.plot([-f, f], [np.abs(np.sum(y * np.exp(-2j * np.pi * -f * x)) / y.size), np.abs(np.sum(y * np.exp(-2j * np.pi * f * x)) / y.size)], 'ro', label='Extra exact')

    # ax2.vlines([-5, 5], [-2, -2], [2, 2], colors='r')
    ax2.set_ylim(-1, 1)
    ax2.set_xlim(-6, 6)
    ax2.set_title('Amplitude of fourier transform')
    ax2.legend()

    ax3.plot(x, np.fft.ifft(np.fft.fftshift(fast_y)), '--', label='Fast')
    ax3.plot(x, np.fft.ifft(np.fft.fftshift(exact_y)), '-.', label='Exact')
    ax3.set_title('Reconstructed cosine')
    ax3.legend()

    ax4.plot(four_x, np.angle(fast_y), '--', label='Fast')
    ax4.plot(four_x, np.angle(exact_y), '-.', label='Exact')
    ax4.plot(four_x, np.angle(exact_y), '.', label='Exact')
    ax4.plot(four_x, [phase for i in four_x], 'r:', label='Phase')
    ax4.plot(four_x, [-phase for i in four_x], 'r:')
    ax4.plot([-f, f], [np.angle(np.sum(y * np.exp(-2j * np.pi * -f * x)) / y.size), np.angle(np.sum(y * np.exp(-2j * np.pi * f * x)) / y.size)], 'ro', label='Extra exact')
    ax4.set_ylim(-np.pi, np.pi)
    ax4.set_xlim(-6, 6)
    ax4.set_title('Phase of fourier transform')
    ax4.legend()
    plt.show()

def image_re_test_multiple():
    # image = images.double_point_source(int(1e6), [0.0002, -0.0002], [0.0002, -0.0002], [1.2, 1.2])
    image1 = images.point_source(int(1e5), 0.0002, 0.0002, 1.2)
    image2 = images.point_source(int(1e5), -0.0002, -0.0002, 1.2)

    test_I = instrument.interferometer(.1, .01, .5, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        .0005 * 2 * np.pi)
    # test_I.add_baseline(.035, 10, 300)#, 1200, 2, 1)
    # test_I.add_baseline(.105, 10, 300)#, 3700, 2, 1)
    # test_I.add_baseline(.315, 10, 300)#, 11100, 2, 1)
    test_I.add_baseline(.945, 10, 300)#, 33400, 2, 1)

    start = time.time()
    test_data1 = process.interferometer_data(test_I, image1, 10, 512)
    print('Processing this first image took ', time.time() - start, ' seconds')

    start = time.time()
    test_data2 = process.interferometer_data(test_I, image2, 10, 512)
    print('Processing this second image took ', time.time() - start, ' seconds')

    # plt.plot(test_data.test_data)
    # plt.show()

    # exp = test_I.baselines[0].F * np.cos(-np.arctan(image.loc[0, 0] / (image.loc[0, 1] + 1e-20))) * np.sqrt(image.loc[0, 0]**2 + image.loc[0, 1]**2) * 1e6
    # for i in range(len(test_I.baselines)):
    #     analysis.hist_data(test_data.actual_pos[test_data.baseline_indices == i], 
    #                         int(np.amax(test_data.discrete_pos[test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[test_data.baseline_indices == i])) + 1, False, i)
    #     # print(test_I.baselines[i].F / test_I.baselines[i].D)
    # # print(exp /33400 * 1e-6)
    # # plt.vlines(-exp, -100, 10000)
    # plt.title('Photon impact positions on detector')
    # # plt.ylim(0, 500)
    # plt.legend()
    # plt.show()

    # test_freq = np.fft.fftfreq(test_data.size)
    # plt.plot(test_freq, np.fft.fftshift(np.fft.fft(test_data.test_data)))
    # plt.show()

    # colourlist = ['b', 'orange', 'g', 'r']
    # for i in range(len(test_I.baselines)):
    #     samples = int(np.amax(test_data.discrete_pos[test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[test_data.baseline_indices == i])) + 1
    #     binned_data, edges = np.histogram(test_data.actual_pos[:,1][test_data.baseline_indices == i], samples)
    #     ft_x_data, ft_y_data = analysis.ft_data(binned_data, samples, edges[1] - edges[0])
    #     analysis.plot_ft(ft_x_data, ft_y_data, 0, i)
    #     delta_u = 1 / np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
    #     # delta_guess = 10**6 / 30.6
    #     plt.axvline(delta_u, 1e-5, 1e4, color = colourlist[i])
    #     plt.axvline(-delta_u, 1e-5, 1e4, color = colourlist[i])
    #     # plt.axvline(delta_guess, 1e-5, 1e4, color = 'r')
    #     # plt.axvline(-delta_guess, 1e-5, 1e4, color = 'r')
    # plt.xlim(-4 * delta_u, 4 * delta_u)
    # plt.title('Fourier transform of photon positions')
    # plt.xlabel('Spatial frequency ($m^{-1}$)')
    # plt.ylabel('Fourier magnitude')
    # plt.legend()
    # plt.show()

    re_im1, f_grid1, test_data_masked = analysis.image_recon_smooth(test_data1, test_I, .01 * 2 * np.pi, samples=512)
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    re_im2, f_grid2, test_data_masked = analysis.image_recon_smooth(test_data1, test_I, .01 * 2 * np.pi, samples=512)
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(abs(ft.fftshift(f_grid1)), cmap=cm.Reds)
    ax1.set_title('UV-plane (amplitude)')
    ax2.imshow(np.imag(ft.fftshift(f_grid1)), cmap=cm.Blues)
    ax2.set_title('UV-plane (phase)')
    ax3.imshow(abs(ft.fftshift(re_im1)), cmap=cm.Greens)
    ax3.set_title('Reconstructed image')
    ax4.imshow(abs(ft.fftshift(f_grid2)), cmap=cm.Reds)
    ax4.set_title('UV-plane (amplitude)')
    ax5.imshow(np.imag(ft.fftshift(f_grid2)), cmap=cm.Blues)
    ax5.set_title('UV-plane (phase)')
    ax6.imshow(abs(ft.fftshift(re_im2)), cmap=cm.Greens)
    ax6.set_title('Reconstructed image')

    plt.show()

def full_image_test(test_code):
    image_path = r"Models\galaxy_lobes.png"
    # image_path = r"Pictures\Funky  mode.png"
    img_scale = 2.2 * .75 * 6.957 * 1e8 / (9.714 * spc.parsec)
    # img_scale = .00015
    image, pix_scale = images.generate_from_image(image_path, int(1e6), img_scale, 1.2)
    # image, pix_scale = images.generate_from_image(image_path, int(1e6), img_scale)

    #TODO find total FOV of instrument

    histedimage, _, __ = np.histogram2d(image.loc[:,0], image.loc[:,1], np.array([np.linspace(-img_scale/2, img_scale/2, pix_scale[0]), 
                                                                                  np.linspace(-img_scale/2, img_scale/2, pix_scale[1])]) * 2 * np.pi / (3600 * 360)) 
    plt.imshow(histedimage, cmap=cm.Greens)
    plt.xlabel('x-axis angular offset from optical axis (arcsec)')
    plt.ylabel('y-axis angular offset from optical axis (arcsec)')
    plt.show()

    # image = images.point_source(int(1e5), 0.000, 0.0005, 1.2)

    test_I = instrument.interferometer(.1, 1, .5, np.array([1.2, 6]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        0.003 * 2 * np.pi, roll_init=0.)
    test_I.add_baseline(1, .035, 10, 300)#, 1200, 2, 1)
    test_I.add_baseline(1, .05, 10, 300)#, 1800, 2, 1)
    test_I.add_baseline(1, .001, 10, 300)#, 20, 2, 1)
    test_I.add_baseline(1, .010, 10, 300)#, 400, 2, 1)
    test_I.add_baseline(1, .020, 10, 300)#, 800, 2, 1)
    test_I.add_baseline(1, .005, 10, 300)#, 100, 2, 1)
    test_I.add_baseline(1, .5, 10, 300)#, 18000, 2, 1)
    test_I.add_baseline(1, .75, 10, 300)#, 26000, 2, 1)
    test_I.add_baseline(1, .105, 10, 300)#, 3700, 2, 1)
    test_I.add_baseline(1, .315, 10, 300)#, 11100, 2, 1)
    test_I.add_baseline(1, .945, 10, 300)#, 33400, 2, 1)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 10, 100000)
    print('Processing this image took ', time.time() - start, ' seconds')

    # exp = test_I.baselines[0].F * np.cos(-np.arctan(image.loc[0, 0] / (image.loc[0, 1] + 1e-20))) * np.sqrt(image.loc[0, 0]**2 + image.loc[0, 1]**2) * 1e6
    # for i in range(len(test_I.baselines)):
    #     analysis.hist_data(test_data.actual_pos[test_data.baseline_indices == i], 
    #                         int(np.amax(test_data.discrete_pos[test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[test_data.baseline_indices == i])) + 1, False, i)
    #     # print(test_I.baselines[i].F / test_I.baselines[i].D)
    # # print(exp /33400 * 1e-6)
    # # plt.vlines(-exp, -100, 10000)
    # plt.title('Photon impact positions on detector')
    # # plt.ylim(0, 500)
    # plt.legend()
    # plt.show()

    # test_freq = np.fft.fftfreq(test_data.size)
    # plt.plot(test_freq, np.fft.fftshift(np.fft.fft(test_data.test_data)))
    # plt.show()

    # colourlist = ['b', 'orange', 'g', 'r']
    # for i in range(len(test_I.baselines)):
    #     samples = int(np.amax(test_data.discrete_pos[test_data.baseline_indices == i]) - 
    #                         np.amin(test_data.discrete_pos[test_data.baseline_indices == i])) + 1
    #     binned_data, edges = np.histogram(test_data.actual_pos[:,1][test_data.baseline_indices == i], samples)
    #     ft_x_data, ft_y_data = analysis.ft_data(binned_data, samples, edges[1] - edges[0])
    #     analysis.plot_ft(ft_x_data, ft_y_data, 0, i)
    #     delta_u = 1 / np.sqrt(test_I.baselines[i].L * spc.h * spc.c / (1.2 * spc.eV * 1e3 * 10))
    #     # delta_guess = 10**6 / 30.6
    #     plt.axvline(delta_u, 1e-5, 1e4, color = colourlist[i])
    #     plt.axvline(-delta_u, 1e-5, 1e4, color = colourlist[i])
    #     # plt.axvline(delta_guess, 1e-5, 1e4, color = 'r')
    #     # plt.axvline(-delta_guess, 1e-5, 1e4, color = 'r')
    # plt.xlim(-4 * delta_u, 4 * delta_u)
    # plt.title('Fourier transform of photon positions')
    # plt.xlabel('Spatial frequency ($m^{-1}$)')
    # plt.ylabel('Fourier magnitude')
    # plt.legend()
    # plt.show()

    start = time.time()
    test_data_imre = np.array(Image.open(image_path).convert('L'))
    # plt.imshow(test_data_imre)
    # plt.show()
    shap = np.array(test_data_imre.shape)
    # test_data_imre[np.array(test_data_imre.shape)//2] = 1
    # test_data_imre[261, 261] = 1
    # test_data_imre[12, 12] = 1
    test_data_imre_fft = ft.fft2(test_data_imre)
    re_im, f_grid, test_data_masked = analysis.image_recon_smooth(test_data, test_I, .01 * 2 * np.pi, img_scale,
                                                                  samples=np.array(test_data_imre.shape), test_data=test_data_imre_fft)
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    test_image = ft.ifft2(test_data_masked)

    """
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    ax1.imshow(np.abs(ft.fftshift(f_grid)), cmap=cm.Reds)
    ax1.set_title('UV-plane (amplitude)')
    # ax1.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax1.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax2.imshow(np.angle(ft.fftshift(f_grid)), cmap=cm.Blues)
    ax2.set_title('UV-plane (phase)')
    # ax2.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax2.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax3.imshow(np.abs(ft.ifftshift(re_im)), cmap=cm.Greens)
    ax3.set_title('Reconstructed image')

    ax4.imshow(np.abs(ft.fftshift(test_data_masked)), cmap=cm.Blues)
    ax4.set_title('UV-plane (amplitude)')
    # ax4.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax4.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax5.imshow(np.angle(ft.fftshift(test_data_masked)), cmap=cm.Greens)
    ax5.set_title('UV-plane (phase)')
    # ax5.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax5.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax6.imshow(np.abs(test_image), cmap=cm.Greens)
    ax6.set_title('Ifft of masked image')

    ax7.imshow(np.abs(ft.fftshift(test_data_imre_fft)), cmap=cm.Blues)
    ax7.set_title('Full UV-plane (amplitude)')
    # ax7.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax7.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax8.imshow(np.angle(ft.fftshift(test_data_imre_fft)), cmap=cm.Greens)
    ax8.set_title('Full UV-plane (phase)')
    # ax8.set_xlim(shap[0] // 2 - 100, shap[0] // 2 + 100)
    # ax8.set_ylim(shap[1] // 2 - 100, shap[1] // 2 + 100)
    ax9.imshow(np.abs(ft.ifft2(test_data_imre_fft)), cmap=cm.Greens)
    ax9.set_title('Image')
    plt.show()
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Look at different normalisations`
    rel_grid = test_data_masked / np.sum(test_data_masked) - f_grid / np.sum(f_grid)

    ax1.imshow(np.abs(ft.fftshift(re_im)), cmap=cm.Reds)
    ax1.set_title('Recreated image')
    # ax1.set_xlim(shap[0] // 2 - 25, shap[0] // 2 + 25)
    # ax1.set_ylim(shap[1] // 2 - 25, shap[1] // 2 + 25)

    ax2.imshow(np.abs(test_image), cmap=cm.Greens)
    ax2.set_title('theoretical image')
    # ax2.set_xlim(shap[0] // 2 - 25, shap[0] // 2 + 25)
    # ax2.set_ylim(shap[1] // 2 - 25, shap[1] // 2 + 25)
    # ax3.imshow(np.abs(ft.ifft2(rel_grid)))
    # ax3.set_title('Image difference')

    plt.show()

    f_grid[f_grid.nonzero()] = 1

    plt.imshow(np.abs(ft.ifftshift(f_grid)), cmap=cm.Greens)
    plt.show()

    plt.imshow(np.abs(ft.ifftshift(re_im)), cmap=cm.Greens)
    plt.show()
    # print('Relative non-zero data points in test data: \n', np.abs(test_data_masked[test_data_masked.nonzero()] / np.amax(test_data_masked[test_data_masked.nonzero()])).astype(float))
    # print('Relative non-zero data points in interferometer data: \n', np.abs(f_grid[f_grid.nonzero()] / np.amax(f_grid[f_grid.nonzero()])).astype(float))

def image_re_test_exact():
    offset = 0e-6
    # image = images.point_source(int(1e5), 0.000, offset, 1.2)
    # image = images.m_point_sources(int(1e6), 4, [0.000, -0.000, -.0004, .00085], [0.000236, -0.00065, 0., 0.], [1.2, 1.2, 1.2, 1.2])

    # Code for a plot of cyg X-1
    image_path = r"Models\hmxb.jpg"
    img_scale = .00055

    # Code for AU mic 
    # Image is big though, so expect a long wait
    # image_path = r"models\au_mic.png"
    # img_scale = 0.0013

    # Code for sgr A*
    # Remember to add // 5 to pix_scale to make sure there aren't too many useless pixels taken into account
    # image_path = r"models\bhdisk.png"
    # img_scale = 0.00037

    # image_path = r"models\compact.png"
    # img_scale = 0.000512 * 2

    image, pix_scale = images.generate_from_image(image_path, int(1e7), img_scale, 6.4)

    # plt.plot(image.loc[:,1] * (3600*360 / (2 * np.pi)), -image.loc[:,0] * (3600*360 / (2 * np.pi)), '.', alpha=.2)
    # histed_photons, _, __ = np.histogram2d(image.loc[:,0], image.loc[:,1], pix_scale)
    # plt.imshow(histed_photons, cmap=cm.Greens)
    # plt.show()

    test_I = instrument.interferometer(.1, 1, 2, np.array([.1, 10]), np.array([-400, 400]),  
                                        roller = instrument.interferometer.smooth_roller, roll_speed=.00001 * 2 * np.pi)
    for D in np.linspace(.05, 1, 10):
        test_I.add_willingale_baseline(D)

    start = time.time()
    test_data = process.interferometer_data(test_I, image, 100000, 2, .15 / (2 * np.sqrt(2*np.log(2))))
    # test_data = process.interferometer_data(test_I, image, 100000)
    print('Processing this image took ', time.time() - start, ' seconds')

    start = time.time()
    re_im, f_values, uv = analysis.image_recon_smooth(test_data, test_I, .02 * 2 * np.pi, img_scale, samples=pix_scale)
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    fig = plt.figure(figsize=(6,6))
    plt.imshow(re_im, cmap=cm.cubehelix)
    plt.xlabel('x ($\mu$as)')
    plt.ylabel('y ($\mu$as)')
    plt.show()

    # fig = plt.figure(figsize=(6,6))
    # plt.plot(uv[:, 0], uv[:, 1], 'g.')
    # plt.xlim(-np.max(uv) * 1.2, np.max(uv) * 1.2)
    # plt.ylim(-np.max(uv) * 1.2, np.max(uv) * 1.2)
    # plt.show()

def locate_test(offset, no_Ns, total_photons, energy, D):
    """This test exists to do some statistical testing"""
    test_I = instrument.interferometer(.1, 1, 2, np.array([.1, 10]), np.array([-399, 399]),  
                                        roller = instrument.interferometer.smooth_roller, roll_speed=.00000 * 2 * np.pi)
    test_I.add_willingale_baseline(D)

    Ns = np.logspace(2, 5, no_Ns)
    sigmas = np.zeros((no_Ns, 2))
    freq = 1 / (test_I.baselines[0].L * spc.h * spc.c / (energy * spc.eV * 1e3 * test_I.baselines[0].W))

    bins = int(np.ceil(abs(test_I.pos_range[0] - test_I.pos_range[1]) / test_I.res_pos))
    edges = np.array([test_I.pos_range[0] + i * test_I.res_pos for i in range(bins + 1)])
    centres = edges[:-1] + (edges[1:] - edges[:-1])/2

    # image = images.point_source(int(total_photons), 0., offset, energy)
    # for i, N in enumerate(Ns):
    #     print(f'Now doing photon count {N}, which is test {i + 1}')
    #     number_of_sims = int(total_photons // (N))

    #     test_data = process.interferometer_data(test_I, image, int(1e5))
    #     pos_data = test_data.pixel_to_pos(test_I)

    #     phases = np.zeros((number_of_sims))
    #     for sim in range(phases.size):
    #         y_data, _ = np.histogram(pos_data[int(N*sim):int(N*(sim+1))], edges)

    #         phases[sim] = np.angle(np.sum(y_data * np.exp(-2j * np.pi * freq * centres)) / N)

    #     sigmas[i] = np.array([np.mean(phases), np.std(phases)])


    for j, N in enumerate(Ns):
        image = images.point_source(int(total_photons), 0., offset, energy)
        print(f'Now doing photon count {N}, which is test {j + 1}')

        test_data = process.interferometer_data(test_I, image, int(1e5), 0, 0)
        # test_data = process.interferometer_data(test_I, image, int(1e5))
        pos_data = test_data.pixel_to_pos(test_I)

        wavelength = spc.h * spc.c / (np.mean(test_data.channel_to_E(test_I)))
        freq = 1 / (test_I.baselines[0].L * wavelength / (test_I.baselines[0].W))

        number_of_sims = int(total_photons // (N))
        phases = np.zeros((number_of_sims))
        for sim in range(phases.size):
            y_data, _ = np.histogram(pos_data[int(N*sim):int(N*(sim+1))], edges)

            phases[sim] = np.angle(fourier_transform(y_data, centres, freq))
                
        sigmas[j] = np.array([np.mean(phases), np.std(phases)])

    # res_I = (spc.h * spc.c / (energy * spc.eV * 1e3)) / (2 * D) * (3600 * 360 / (2 * np.pi))
    # res_diff = 1.22 * (spc.h * spc.c / (energy * spc.eV * 1e3)) / D * (3600 * 360 / (2 * np.pi))
    # print(res_I)
    sigmas *= (spc.h * spc.c / (energy * spc.eV * 1e3)) / test_I.baselines[0].D * (3600 * 360 / (2 * np.pi)**2)
    fit_func = lambda N, a: a / np.sqrt(N)
    fit_p, fit_cov = spopt.curve_fit(fit_func, Ns, sigmas[:,1], p0=(1))
    print(fit_p)

    significants = abs(sigmas[:,0]) > sigmas[:, 1]

    plt.semilogx(Ns, sigmas[:, 0], '.', label='Mean of calculated offsets')
    plt.errorbar(Ns, sigmas[:,0], yerr=sigmas[:,1], ls='', marker='.')
    plt.plot([0, Ns[-1]], [-offset, -offset], 'g--', label='Actual offset')
    plt.legend()
    plt.show()

    plt.semilogx(Ns, fit_func(Ns, *fit_p), label=r'Fit of $\frac{a}{\sqrt{N}}$')
    # plt.semilogx(Ns, fit_func(Ns, res_I), label=r'Fit function with a = $\theta_I$')
    plt.semilogx(Ns[significants == False], sigmas[significants == False, 1], 'r.', label='Points indistinguishable from 0')
    plt.semilogx(Ns[significants], sigmas[significants, 1], 'g.', label='Points distinguishable from 0')
    plt.xlabel('Number of Photons')
    plt.ylabel('Positional uncertainty (as)')
    plt.legend()
    plt.show()

def locate_test_multiple_D(offset, no_Ns, total_photons, energy, Ds):
    """This test exists to do some statistical testing"""
    test_I = instrument.interferometer(.01, 1, 2, np.array([1.2, 7]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        .00000 * 2 * np.pi, roll_init=0/4 * np.pi)
    
    Ns = np.logspace(2, 5, no_Ns)
    artificial_Ns = np.logspace(2, 5, 10000)
    sigmas = np.zeros((len(Ds), no_Ns, 2))

    bins = int(np.ceil(abs(test_I.pos_range[0] - test_I.pos_range[1]) / test_I.res_pos))
    edges = np.array([test_I.pos_range[0] + i * test_I.res_pos for i in range(bins + 1)])
    centres = edges[:-1] + (edges[1:] - edges[:-1])/2

    image = images.point_source(int(total_photons), 0., offset, energy)
    wavelength = spc.h * spc.c / (energy * spc.eV * 1e3)
    for i, D in enumerate(Ds):
        test_I.clear_baselines()
        test_I.add_willingale_baseline(D)
        freq = 1 / (test_I.baselines[0].L * wavelength / (test_I.baselines[0].W))

        for j, N in enumerate(Ns):
            print(f'Now doing photon count {N}, which is test {j + 1}')
            number_of_sims = int(total_photons // (N))

            test_data = process.interferometer_data(test_I, image, int(1e5))
            pos_data = test_data.pixel_to_pos(test_I)

            phases = np.zeros((number_of_sims))
            for sim in range(phases.size):
                y_data, _ = np.histogram(pos_data[int(N*sim):int(N*(sim+1))], edges)

                phases[sim] = np.angle(np.sum(y_data * np.exp(-2j * np.pi * freq * centres)) / N)

            sigmas[i, j] = np.array([np.mean(phases), np.std(phases)])

        res_I = (wavelength) / (2 * D) * (3600 * 360 / (2 * np.pi))
        fit_func = lambda N, a: a * (wavelength/ D) / np.sqrt(N)
        sigmas[i] *= (test_I.baselines[0].L * wavelength / (test_I.baselines[0].W)) / test_I.baselines[0].F * (3600 * 360 / (2 * np.pi)**2)
        fit_p, fit_cov = spopt.curve_fit(fit_func, Ns, sigmas[i, :, 1], p0=(res_I))

        plt.semilogx(artificial_Ns, fit_func(artificial_Ns, *fit_p)*1e6, label='Fit of' + f'{fit_p[0]}' + r'$\cdot \frac{\lambda}{D \cdot \sqrt{N}}$ for D = ' + f'{D:.3f}')
        plt.semilogx(Ns, sigmas[i, :, 1]*1e6, '.', label=f'Data for D = {D:.3f}')

    plt.title(f'Positional uncertainty determined with {total_photons} monochromatic photons of energy {energy} keV at offset {offset*1e6} $\\mu$as')
    plt.xlabel('Number of Photons')
    plt.ylabel(r'Positional uncertainty ($\mu$as)')
    plt.legend()
    plt.show()

def locate_test_multiple_E(offset, no_Ns, total_photons, energies, D, pos_noise = 0, E_noise = 0):
    """This test exists to do some statistical testing"""
    time_step = 1
    
    # making the interferometer itself
    test_I = instrument.interferometer(time_step, roller = instrument.interferometer.smooth_roller,
                                       roll_init = 0, roll_speed = 0)
    # ,
    #                                    res_E = 0.01, res_t = 0.01, res_pos = 0.02,
    #                                    E_range = np.array([1, 10]), pos_range = np.array([-400, 400]))
    # test_I.add_willingale_baseline(D)
    test_I.add_baseline(num_pairs = 1, D = D, L = 10, W = 300)
    test_I.baselines[0].add_custom_detector(res_E = 0.01, res_t = 0.01, res_pos = 0.02, pos_noise = pos_noise, E_noise = E_noise, E_range = np.array([1, 10]), pos_range = np.array([-400, 400]))

    
    Ns = np.logspace(2, 5, no_Ns)
    artificial_Ns = np.logspace(2, 5, 10000)
    sigmas = np.zeros((len(energies), no_Ns, 2))

    bins = int(np.ceil(abs(test_I.baselines[0].camera.pos_range[0] - test_I.baselines[0].camera.pos_range[1]) / test_I.baselines[0].camera.res_pos))
    edges = np.array([test_I.baselines[0].camera.pos_range[0] + i * (test_I.baselines[0].camera.res_pos) for i in range(bins + 1)])
    centres = edges[:-1] + (edges[1:] - edges[:-1])/2

    colour_list = ['b', 'orange', 'g', 'r', 'purple']
    p0 = np.zeros((len(energies)))
    fig = plt.figure(figsize=(8,6))

    for i, energy in enumerate(energies):
        # image = images.point_source_multichromatic_gauss(int(total_photons), 0., offset, energy, .08 / 2.355)
        image = images.point_source(int(total_photons), 0., offset, energy)

        for j, N in enumerate(Ns):
            print(f'Now doing photon count {N}, which is test {j + 1}')

            test_data = process.interferometer_data(test_I, image)#, pure_fringes=True)
            # test_data = process.interferometer_data(test_I, image, int(1e5))

            test_data = test_I.baselines[0].camera.add_detector_effects(test_data, np.repeat(True, test_data.pos.size))

            pos_data = test_data.pos

            wavelength = spc.h * spc.c / (np.mean(test_data.energies))
            freq = 1 / (test_I.baselines[0].L * wavelength / (test_I.baselines[0].W))

            number_of_sims = int(total_photons // (N))
            phases = np.zeros((number_of_sims))
            for sim in range(phases.size):
                y_data, _ = np.histogram(pos_data[int(N*sim):int(N*(sim+1))], edges)

                phases[sim] = np.angle(fourier_transform(y_data, centres, freq))
                    
            sigmas[i, j] = np.array([np.mean(phases), np.std(phases)])

        fit_func = lambda N, a: a * (wavelength / D) / np.sqrt(N) * 3600 * 360 / (2 * np.pi)
        sigmas[i] *= wavelength / test_I.baselines[0].D * (3600 * 360 / (2 * np.pi)**2)
        fit_p, fit_cov = spopt.curve_fit(fit_func, Ns, sigmas[i, :, 1], p0=(1))
        p0[i] = fit_p[0]
        
        plt.semilogx(artificial_Ns, fit_func(artificial_Ns, p0[i])*1e6, colour_list[i])
        plt.semilogx(Ns, sigmas[i, :, 1]*1e6, '.', color=colour_list[i], label=f'{energy:.1f} keV, a = {p0[i]:.4f}')

    plt.title(f'Determined with $10^{int(np.log10(total_photons))}$ monochromatic photons per datapoint')
    plt.xlabel('Number of photons')
    plt.ylabel(r'Positional uncertainty ($\mu$as)')
    plt.ylim(bottom = 0)
    plt.legend()#title=f'D = {1} m \noffset = {offset} $\\mu$as\nFWHM = {E_noise * 2.355:.3f} keV\npixel = {pos_noise:.1f} $\mu$m')
    plt.savefig("astro_presice_withdetect.pdf")
    plt.show()

    # Calculation of fringe spacing for each energy, useful to say something on how well sampled each energy is with specific pixel size.
    # print(spc.h * spc.c * test_I.baselines[0].L / (np.array(energies) * spc.eV * 1e3 * test_I.baselines[0].W) * 1e6)

def visibility_test_E(no_Ds, no_sims, energy):
    test_I = instrument.interferometer(1, roller = instrument.interferometer.smooth_roller,
                                       roll_init = 0, roll_speed = 0)
    # test_I = instrument.interferometer(.01, 1, 2, np.array([1.2, 7]), np.array([-400, 400]), 
    #                                     0.00, None, instrument.interferometer.smooth_roller, 
    #                                     .00000 * 2 * np.pi, roll_init=0/4 * np.pi)
    img_scale = .0004
    image = images.disc(int(1e5), 0, 0, energy, img_scale / 2)
    image_2 = images.disc(int(1e5), 0, 0, 2 * energy, img_scale / 2)
    image_3 = images.disc(int(1e5), 0, 0, 3 * energy, img_scale / 2)
    image_4 = images.disc(int(1e5), 0, 0, 4 * energy, img_scale / 2)

    Ds = np.linspace(.005, 1, no_Ds)

    def calc_vis(image, Ds):
        vis = np.zeros((no_Ds, 2))
        for i, D in enumerate(Ds):
            print(f'Now doing baseline length {D}, which is test {i + 1}')
            test_I.clear_baselines()
            test_I.add_baseline(num_pairs = 1, D = D, L = 10, W = 300)
            test_I.baselines[0].add_custom_detector(res_E = 0.01, res_t = 0.01, res_pos = 0.02, pos_noise = 2, E_noise = .15 / (2.355), E_range = np.array([1.2, 7]), pos_range = np.array([-400, 400]))

            bins = int(np.ceil(abs(test_I.baselines[0].camera.pos_range[0] - test_I.baselines[0].camera.pos_range[1]) / test_I.baselines[0].camera.res_pos))
            edges = np.array([test_I.baselines[0].camera.pos_range[0] + i * (test_I.baselines[0].camera.res_pos) for i in range(bins + 1)])
            centres = edges[:-1] + (edges[1:] - edges[:-1])/2

            freq = 1 / (test_I.baselines[0].L * spc.h * spc.c / (image.energies[0] * test_I.baselines[0].W))
            amps = np.zeros((no_sims, 2))

            for sim in range(no_sims):
                test_data = process.interferometer_data(test_I, image)
                # test_data = test_I.baselines[0].camera.add_detector_effects(test_data, np.repeat(True, test_data.pos.size))

                y_data, _ = np.histogram(test_data.pos, edges)

                amps[sim, 0] = np.abs(np.sum(y_data * np.exp(-2j * np.pi * freq * centres)) / y_data.size)
                amps[sim, 1] = np.abs(np.sum(y_data) / y_data.size)

            vis[i, 0] = np.mean(amps[:, 0]) / np.mean(amps[:, 1])
            vis[i, 1] = np.std(amps[:, 0]) / np.mean(amps[:, 1])

        return vis * 2
    
    vis = calc_vis(image, Ds)
    vis_2 = calc_vis(image_2, Ds)
    vis_3 = calc_vis(image_3, Ds)
    vis_4 = calc_vis(image_4, Ds)

    D_theory = (spc.h * spc.c / (energy * spc.eV * 1e3)) / (img_scale * 2 * np.pi / (3600 * 360))
    D_theory_2 = (spc.h * spc.c / (2 * energy * spc.eV * 1e3)) / (img_scale * 2 * np.pi / (3600 * 360))
    D_theory_3 = (spc.h * spc.c / (3 * energy * spc.eV * 1e3)) / (img_scale * 2 * np.pi / (3600 * 360))
    D_theory_4 = (spc.h * spc.c / (4 * energy * spc.eV * 1e3)) / (img_scale * 2 * np.pi / (3600 * 360))

    plt.errorbar(Ds, vis[:, 0], yerr=vis[:, 1], marker='.', ls='-', label=f'{energy:.1f} keV')
    plt.vlines(D_theory, vis[:, 0].min(), vis[:, 0].max(), 'b', alpha=.3)
    plt.errorbar(Ds, vis_2[:, 0], yerr=vis_2[:, 1], marker= '.', ls='-.', label=f'{2 * energy:.1f} keV')
    plt.vlines(D_theory_2, vis_2[:, 0].min(), vis_2[:, 0].max(), 'orange', alpha=.3)
    plt.errorbar(Ds, vis_3[:, 0], yerr=vis_3[:, 1], marker= '.', ls=':', label=f'{3 * energy:.1f} keV')
    plt.vlines(D_theory_3, vis_3[:, 0].min(), vis_3[:, 0].max(), 'g', alpha=.3)
    plt.errorbar(Ds, vis_4[:, 0], yerr=vis_4[:, 1], marker= '.', ls='--', label=f'{4 * energy:.1f} keV')
    plt.vlines(D_theory_4, vis_4[:, 0].min(), vis_4[:, 0].max(), 'r', alpha=.3)
    plt.title(f'Average of {no_sims} observations of uniform disc with {img_scale/2:.4f} radius at variable keV')
    plt.xlabel('Baseline length (m)')
    plt.ylabel('Visibility')
    plt.legend()
    plt.show()

def visibility_test_scale(no_Ds, no_sims, energy):
    test_I = instrument.interferometer(.1, 1, 2, np.array([.1, 7]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        .00000 * 2 * np.pi, roll_init=0/4 * np.pi)
    img_scale = .0004
    image = images.disc(int(1e5), 0, 0, energy, img_scale / 2)
    image_2 = images.disc(int(1e5), 0, 0, energy, 2 * img_scale / 2)
    image_3 = images.disc(int(1e5), 0, 0, energy, 3 * img_scale / 2)
    image_4 = images.disc(int(1e5), 0, 0, energy, 4 * img_scale / 2)

    # image = images.double_point_source(int(1e6), [0,0], [-img_scale/2, img_scale/2], [energy, energy])
    # image_2 = images.double_point_source(int(1e6), [0,0], [-img_scale/2 * 2, img_scale/2 * 2], [energy, energy])
    # image_3 = images.double_point_source(int(1e6), [0,0], [-img_scale/2 * 3, img_scale/2 * 3], [energy, energy])
    # image_4 = images.double_point_source(int(1e6), [0,0], [-img_scale/2 * 4, img_scale/2 * 4], [energy, energy])

    Ds = np.linspace(.005, 1, no_Ds)

    bins = int(np.ceil(abs(test_I.pos_range[0] - test_I.pos_range[1]) / test_I.res_pos))
    edges = np.array([test_I.pos_range[0] + i * test_I.res_pos for i in range(bins + 1)])
    centres = edges[:-1] + (edges[1:] - edges[:-1])/2

    fig = plt.figure(figsize=(8,6))
    def calc_vis(image, Ds):
        vis = np.zeros((no_Ds, 2))
        for i, D in enumerate(Ds):
            print(f'Now doing baseline length {D}, which is test {i + 1}')
            test_I.clear_baselines()
            test_I.add_willingale_baseline(D)

            freq = 1 / (test_I.baselines[0].L * spc.h * spc.c / (image.energies[0] * test_I.baselines[0].W))
            amps = np.zeros((no_sims, 2))

            for sim in range(no_sims):
                test_data = process.interferometer_data(test_I, image, 100000, 2, .15 / (2*np.sqrt(2*np.log(2))))
                # test_data = process.interferometer_data(test_I, image, 100000)

                y_data, _ = np.histogram(test_data.pixel_to_pos(test_I), edges)
                # y_data, _ = np.histogram(test_data.pos, edges)

                amps[sim, 0] = np.abs(np.sum(y_data * np.exp(-2j * np.pi * freq * centres)) / y_data.size)
                amps[sim, 1] = np.abs(np.sum(y_data) / y_data.size)

            vis[i, 0] = np.mean(amps[:, 0]) / np.mean(amps[:, 1])
            vis[i, 1] = np.std(amps[:, 0]) / np.mean(amps[:, 1])

        return vis * 2
    
    vis = calc_vis(image, Ds)
    vis_2 = calc_vis(image_2, Ds)
    vis_3 = calc_vis(image_3, Ds)
    vis_4 = calc_vis(image_4, Ds)

    # D_theory = (spc.h * spc.c / (energy * spc.eV * 1e3)) / (img_scale * 2 * np.pi / (3600 * 360)) * 1.22 * 2
    # D_theory_2 = (spc.h * spc.c / (energy * spc.eV * 1e3)) / (2 * img_scale * 2 * np.pi / (3600 * 360)) * 1.22 * 2
    # D_theory_3 = (spc.h * spc.c / (energy * spc.eV * 1e3)) / (3 * img_scale * 2 * np.pi / (3600 * 360)) * 1.22 * 2
    # D_theory_4 = (spc.h * spc.c / (energy * spc.eV * 1e3)) / (4 * img_scale * 2 * np.pi / (3600 * 360)) * 1.22 * 2

    # D_theory = (spc.h * spc.c / (energy * spc.eV * 1e3)) / (img_scale * 2 * np.pi / (3600 * 360)) / 2
    # D_theory_2 = (spc.h * spc.c / (energy * spc.eV * 1e3)) / (2 * img_scale * 2 * np.pi / (3600 * 360)) / 2
    # D_theory_3 = (spc.h * spc.c / (energy * spc.eV * 1e3)) / (3 * img_scale * 2 * np.pi / (3600 * 360)) / 2
    # D_theory_4 = (spc.h * spc.c / (energy * spc.eV * 1e3)) / (4 * img_scale * 2 * np.pi / (3600 * 360)) / 2

    plt.errorbar(Ds, vis[:, 0], yerr=vis[:, 1], marker='.', ls='-', label=f'{img_scale*1e6:.1f} $\\mu$as')
    # plt.vlines(D_theory, vis_4[:, 0].min(), vis_4[:, 0].max(), 'b', alpha=.3)
    plt.errorbar(Ds, vis_2[:, 0], yerr=vis_2[:, 1], marker= '.', ls='-.', label=f'{2 * img_scale*1e6:.1f} $\\mu$as')
    # plt.vlines(D_theory_2, vis_4[:, 0].min(), vis_4[:, 0].max(), 'orange', alpha=.3)
    plt.errorbar(Ds, vis_3[:, 0], yerr=vis_3[:, 1], marker= '.', ls=':', label=f'{3 * img_scale*1e6:.1f} $\\mu$as')
    # plt.vlines(D_theory_3, vis_4[:, 0].min(), vis_4[:, 0].max(), 'g', alpha=.3)
    plt.errorbar(Ds, vis_4[:, 0], yerr=vis_4[:, 1], marker= '.', ls='--', label=f'{4 * img_scale*1e6:.1f} $\\mu$as')
    # plt.vlines(D_theory_4, vis_4[:, 0].min(), vis_4[:, 0].max(), 'r', alpha=.3)
    plt.title(f'Mean of {no_sims} observations of uniform discs of variable radii')
    plt.xlabel('Baseline length (m)')
    plt.ylabel('Visibility')
    plt.legend(title=f'E = {energy} keV\nFWHM = .15 keV\n$\\delta y$ = 2 $\mu$m')
    plt.ylim(0, 1)
    plt.xlim(0, 1.01)
    plt.show()
    
def visibility_test_2(no_Ds, no_sims, energy):
    test_I = instrument.interferometer(.01, 1, 2, np.array([1.2, 7]), np.array([-400, 400]), 
                                        0.00, None, instrument.interferometer.smooth_roller, 
                                        .00000 * 2 * np.pi, roll_init=0/4 * np.pi)
    img_scale = .0004
    image = images.disc(int(1e5), 0, 0, energy, img_scale / 2)

    Ds = np.linspace(0.05, 5, no_Ds)
    vis = np.zeros((no_Ds, 2))

    bins = int(np.ceil(abs(test_I.pos_range[0] - test_I.pos_range[1]) / test_I.res_pos))
    edges = np.array([test_I.pos_range[0] + i * test_I.res_pos for i in range(bins + 1)])
    centres = edges[:-1] + (edges[1:] - edges[:-1])/2

    freq = 1 / (test_I.baselines[0].L * spc.h * spc.c / (energy * spc.eV * 1e3 * test_I.baselines[0].W))

    for i, D in enumerate(Ds):
        print(f'Now doing baseline length {D}, which is test {i + 1}')
        test_I.clear_baselines()
        test_I.add_willingale_baseline(D)

        amps = np.zeros((no_sims, 2))

        for sim in range(no_sims):
            test_data = process.interferometer_data(test_I, image, 100000)
            y_data, _ = np.histogram(test_data.pixel_to_pos(test_I), edges)

            amps[sim, 0] = np.abs(np.sum(y_data * np.exp(-2j * np.pi * freq * centres)) / y_data.size)
            amps[sim, 1] = np.abs(np.sum(y_data) / y_data.size)

        vis[i, 0] = np.mean(amps[:, 0]) / np.mean(amps[:, 1])
        vis[i, 1] = np.std(amps[:, 0]) / np.mean(amps[:, 1])

    D_theory = 1.22 *  (spc.h * spc.c / (energy * spc.eV * 1e3)) / (img_scale * 2 * np.pi / (3600 * 360))

    plt.errorbar(Ds, vis[:, 0], yerr=vis[:, 1], marker='.', ls='', label=f'{energy} eV')
    plt.vlines(D_theory, vis[:, 0].min(), vis[:, 0].max(), 'b')
    plt.xlabel('Baseline length (m)')
    plt.ylabel('Visibility')
    plt.legend()
    plt.show()

def fringes_plots():

    # offsets = [x * 1e-6 for x in np.linspace(-50, 50, 25)]
    offsets = [x * 1e-6 for x in np.linspace(-1000, 1000, 25)]
    energy = 1.2398425

    test_I = instrument.interferometer(.1, 1, 2, np.array([.1, 10]), np.array([-400, 400]),  
                                        roller = instrument.interferometer.smooth_roller, roll_speed=.00000 * 2 * np.pi)
    test_I.add_baseline(1, 10, 300)#, 33400)

    bins = int(np.ceil(abs(test_I.pos_range[0] - test_I.pos_range[1]) / test_I.res_pos))
    edges = np.array([test_I.pos_range[0] + i * test_I.res_pos for i in range(bins + 1)]) * 1e6

    colourlist = ['b', 'orange', 'g', 'r']
    for i, offset in enumerate(offsets):

        image = images.point_source(int(1e5), 0.000, offset, energy)
        test_data = process.interferometer_data(test_I, image, 100000)
        plt.hist(test_data.pixel_to_pos(test_I)*1e6, edges, label=f'{energy} keV')
        plt.title(f'Interferometric fringe pattern with diffraction')
        plt.xlabel('Photon impact positions ($\\mu$m)')
        plt.ylabel('Number of photons in bin')
        plt.ylim(0, 2200)
        plt.legend(title=f'D = 1 m\noffset = {offset*1e6:.1f} $\\mu$as')
        plt.show()
        # plt.savefig(f'frame-{i}.png')
        # plt.close()

def Willingale_plot7():

    energy = 1.2398425

    test_I = instrument.interferometer(.1, 1, 2, np.array([.1, 10]), np.array([-1000, 1000]),  
                                        roller = instrument.interferometer.smooth_roller, roll_speed=.00000 * 2 * np.pi)
    test_I.add_baseline(1, 10, 300)#, 33400)

    bins = int(np.ceil(abs(test_I.pos_range[0] - test_I.pos_range[1]) / test_I.res_pos))
    edges = np.array([test_I.pos_range[0] + i * test_I.res_pos for i in range(bins + 1)]) * 1e6

    photons_counts = int(3e5)

    image = images.double_point_source(photons_counts, [0.000, 0.000], [-5 * 1e-3, 5 * 1e-3], [energy, energy])
    test_data = process.interferometer_data(test_I, image, photons_counts)
    plt.hist(test_data.pixel_to_pos(test_I)*1e6, edges, label=f'{energy} keV')
    plt.title(f'Figure 7 of Willingale')
    plt.xlabel('Photon impact positions ($\\mu$m)')
    plt.ylabel('Number of photons in bin')
    plt.ylim(0, 2200)
    plt.show()
    #plt.savefig(f'frame-{i}.png')
    # plt.close()


def Correct_diff_fringes():

    # energy in keV: 1.2398425keV or 10A from Willingale (2004),
    # 3.6 keV found as optimum by Emily in her Master thesis,
    # and 6.4 keV the iron K-alpha line
    energy = 1.2398425 #3.6 #6.4 #

    # seemingly arbetrary time step?
    time_step = 1

    # the chosen number of source counts
    photons_counts = int(1e5)

    # making the interferometer itself
    test_I = instrument.interferometer(time_step, roller = instrument.interferometer.smooth_roller, roll_speed = 0)
    
    # adding a baseline, values based on discussions
    test_I.add_baseline(D = 1, L = 10, W = 300, num_pairs = 2)

    # # taking the standard detector, it can be changed
    # detector = test_I.baselines[0].camera

    # # binning the entire detector, based on the detector charateristics
    # bins = int(np.ceil(abs(detector.pos_range[0] - detector.pos_range[1]) / detector.res_pos))
    # edges = np.array([detector.pos_range[0] + i * detector.res_pos for i in range(bins + 1)])

    # creating the source
    off_axis = 0.0009 # arcsec
    image = images.point_source(photons_counts, 0.000, off_axis, energy)

    # observing the source using the optical bench
    test_data = process.interferometer_data(test_I, image, photons_counts)#, pure_diffraction=True)

    # plotting the binned photon positions
    # plt.rc('font', size=50)
    # fig = plt.figure(figsize=(22,15))
    plt.hist(test_data.pos * 1e6, 1000, label=f'{energy} keV')#edges)
    plt.vlines((test_I.baselines[0].L * (off_axis * np.pi / (180 * 3600))) * 1e6, 0, 2000, color="red", label="Diffraction pattern center")
    plt.vlines(((test_I.baselines[0].D * np.sin((off_axis * np.pi / (180 * 3600)))) / (2 * np.sin(test_I.baselines[0].beam_angle / 2))) * 1e6, 0, 500, color="orange", label="Central fringe")
    plt.xlabel('Photon impact positions ($\\mu$m)')
    plt.ylabel('Number of photons in bin')
    plt.ylim(0, 300) #3500) #
    # plt.xlim(-250, 250)
    plt.legend()
    plt.savefig("full_pattern_00009.pdf")
    plt.show()


def image_reconstruction():

    def fourier2D(matrix_in):
        """
        Calculate the 2D Fourier transform of a matrix

        Input:
            matrix_in: 2D numpy array
        
        Output:
            matrix_out: 2D Fourier transform of matrix_in
        """
    
        matrix_out = np.absolute(np.fft.fft2(matrix_in))
        matrix_out = np.roll(matrix_out, int(matrix_out.shape[0]/2), axis=0)
        matrix_out = np.roll(matrix_out, int(matrix_out.shape[1]/2), axis=1)
        return matrix_out


    """Point source"""
    # offset = 0.0009
    # image = images.point_source(int(1e4), 0.000, offset, 1.2398425) # image = images.m_point_sources(int(1e6), 4, [0.000, -0.000, -.0004, .00085], [0.000236, -0.00065, 0., 0.], [1.2, 1.2, 1.2, 1.2])
    # # img_scale is unclear to me what it's based on. I presume it to be based on pixel size compared to source size
    # img_scale = 1
    # pix_scale =  np.array([330, 550])

    """All models are those from the dropbox 'models'."""

    """Code for a plot of cyg X-1""" #works
    image_path = r"Models\hmxb.jpg"
    img_scale = .00055 # largest axis size in as

    tranmis_ISM = np.load("tbabs_transm_NH0p1e22.npy")
    # plt.plot(tranmis_ISM[0], tranmis_ISM[1])
    # plt.ylabel("Transmission coefficient (0-1)")
    # plt.xlabel("Energy (keV)")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.ylim(1e-4, 1)
    # plt.show()

    nH = 6e21 # between 5.4e21 and 7e21 cm^2
    gamma = 2

    sample_energies = np.linspace(1, 7, 1000) # keV

    transm_fact = spinter.interp1d(tranmis_ISM[0], tranmis_ISM[1])(sample_energies)

    pow_spect = np.power(sample_energies, - gamma)

    transm_spec = pow_spect * np.power(transm_fact, nH / 1e21) #transm_fact
    # rescale_transmis = np.power(transm_spec, nH / 1e21)

    spectrum = np.array([sample_energies, transm_spec]).T # rescale_transmis

    # plt.plot(spectrum[:,0] / (1e3 * spc.eV), transm_spec)#rescale_transmis)#np.power( * 1e21, 6))#np.power(spectrum[:,0] / (1e3 * spc.eV), 2) * pow_spect * rescale_transmis)
    # plt.ylabel("Non-normalized counts (s^-1 keV^-1 m^-2)")
    # plt.xlabel("Energy (keV)")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()

    # spectrum = None

    """Code for AU mic""" #DOESN'T work
    # Image is big though, so expect a long wait
    # image_path = r"Models\au_mic.png"
    # img_scale = 0.0013

    """Code for exop model""" #works
    # image_path = r"Models\exop.jpg"
    # img_scale = 0.5 * 1e-3

    """Code for spiral model""" #DOESN'T work
    # image_path = r"Models\spiral.jpg"
    # img_scale = 5 * 1e-3

    """Code for sgr A*""" #works
    # Remember to add // 5 to pix_scale to make sure there aren't too many useless pixels taken into account
    # image_path = r"Models\bhdisk.png"
    # img_scale = 0.00037

    """Code for 'compact' model""" #Uncertain
    # image_path = r"Models\compact.png"
    # img_scale = 0.000512 * 2

    """This samples the source, when an image is given"""
    spectrum = r"Models\Spectra\Test_Erik.txt"
    # spectrum = np.array([sample_energies, np.repeat(1, sample_energies.size)]).T
    spectrum = None
    source_counts = 1e6
    image, pix_scale = images.generate_from_image(image_path, int(source_counts), img_scale, spectrum = spectrum, energy = 6.4)#3.6)#1.2398425)#, energy_spread = 0.1)#3.6)#6.4)#

    # plots the sampled points of origin of the source photons
    # plt.plot(image.loc[:,1] * (3600*360 / (2 * np.pi)), -image.loc[:,0] * (3600*360 / (2 * np.pi)), '.', alpha=.2)
    # histed_photons, _, _ = np.histogram2d(image.loc[:,0], image.loc[:,1], pix_scale)
    # plt.imshow(histed_photons, cmap=cm.Greens)
    # plt.show()

    # seemingly arbetrary time step?
    time_step = 1
    
    # making the interferometer itself
    test_I = instrument.interferometer(time_step, roller = instrument.interferometer.smooth_roller,
                                       roll_init = 0, roll_speed = np.pi / 1e3)#(np.max(image.toa) * time_step)) #.00001 * 2 * np.pi) #

    reflec_list = [r"Models\Mirror reflectivity Si\Angle_0.01deg.txt", r"Models\Mirror reflectivity Si\Angle_0.5deg.txt", r"Models\Mirror reflectivity Si\Angle_0.8deg.txt", r"Models\Mirror reflectivity Si\Angle_1.0deg.txt", r"Models\Mirror reflectivity Si\Angle_1.3deg.txt", r"Models\Mirror reflectivity Si\Angle_1.5deg.txt", r"Models\Mirror reflectivity Si\Angle_2.0deg.txt", r"Models\Mirror reflectivity Si\Angle_2.5deg.txt", r"Models\Mirror reflectivity Si\Angle_5.0deg.txt"]
    
    # adding the baselines, values based trying different things
    # baseline_Ds = [0.001, 0.01, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # test_I.add_baseline(num_pairs = 5, D = 0.5, L = 10, W = 400)
    for D in np.linspace(0.05, 1, 10): #baseline_Ds: #np.geomspace(0.10, 4.0, 10):
        # test_I.add_baseline(num_pairs = 1, D = D, L = 10, W = 300)
        # test_I.add_baseline(num_pairs = 5, D = D, L = 10, W = 300)
        # test_I.add_baseline(num_pairs = 5, D = D, L = 10, W = 400)
        # test_I.add_baseline(num_pairs = 5, D = D, L = 10, W = 400, grazing_angle = 2 * D, mirr_mater = reflec_list)
        # test_I.add_baseline(num_pairs = 5, D = D, L = 10, W = 400, grazing_angle = (0.7598) * np.pi / 180, mirr_mater = reflec_list) # most realistic atm based on discussions
        test_I.add_willingale_baseline(D)

    # observing the source using the optical benches and timing it
    start = time.time()
    # potentially adding positional and energy noise
    test_data = process.interferometer_data(test_I, image)#, 2, .15 / (2 * np.sqrt(2*np.log(2))))

    # adding detector effects if desired
    # test_data.detector_effects(test_I, [0, 3, 9])

    print('Processing this image took ', time.time() - start, ' seconds')

    # reconstruction of the source and timing it
    start = time.time()
    # print(pix_scale)
    recon_image_info, uv = analysis.image_recon_smooth(test_data, test_I, img_scale, samples = pix_scale, error = 0.1)#, recon_type = "eIFT") 
    re_im, max_size = recon_image_info
    print('Reconstructing this image took ', time.time() - start, ' seconds')

    # showing the sampled points in the uv-plane
    # plt.rc('font', size=50)
    fig = plt.figure(figsize=(6,6))#15,15
    plt.plot(uv[:, 0], uv[:, 1], 'g.')#, markersize = 15)
    plt.xlim(-np.max(uv) * 1.2, np.max(uv) * 1.2)
    plt.ylim(-np.max(uv) * 1.2, np.max(uv) * 1.2)
    plt.title("uv-plane sampling")
    # plt.savefig("uv_sparce.png")
    plt.show()

    # showing the log of the Fourier transform of the reconstructed image
    fig = plt.figure(figsize=(6,6))
    plt.imshow(np.log10(fourier2D(re_im)), cmap=cm.cubehelix)
    plt.colorbar()
    plt.show()

    # showing the reconstructed image
    fig = plt.figure(figsize=(20,20))
    # plt.rc('font', size=50)
    plt.imshow(re_im, extent = [0, img_scale * 1e6, img_scale * 1e6 * pix_scale[0] / pix_scale[1], 0], cmap=cm.cubehelix)
    plt.xlabel('x ($\mu$as)')
    plt.ylabel('y ($\mu$as)')
    # plt.colorbar()
    # plt.savefig("eIFT_recon.png")#.format(source_counts))
    plt.show()

    # showing the log of the reconstructed image
    # fig = plt.figure(figsize=(6,6))
    # plt.imshow(np.log10(re_im), cmap=cm.cubehelix)
    # plt.colorbar()
    # plt.xlabel('x ($\mu$as)')
    # plt.ylabel('y ($\mu$as)')
    # plt.show()

    # checking image reconstruction
    ssr = analysis.image_compare(np.array(Image.open(image_path).convert('L')), re_im)

    # show residual
    input_im = np.array(Image.open(image_path).convert('L'))
    print("The Sum of Squared Residuals (SSR) for this reconstructed image is {}.".format(ssr))
    fig = plt.figure(figsize=(20,20))#6,6))#
    plt.rc('font', size=15)
    plt.imshow(((re_im - np.min(re_im)) / (np.max(re_im) - np.min(re_im))) - ((input_im - np.min(input_im)) / (np.max(input_im) - np.min(input_im))), extent = [0, img_scale * 1e6, img_scale * 1e6 * pix_scale[0] / pix_scale[1], 0], cmap=cm.cubehelix)
    plt.xlabel('x ($\mu$as)')
    plt.ylabel('y ($\mu$as)')
    plt.colorbar()
    plt.savefig("residual_Phil.pdf")#.format(source_counts))
    plt.show()


def test_periodgram():

    # energy in keV: 1.2398425keV or 10A from Willingale (2004),
    # 3.6 keV found as optimum by Emily in her Master thesis,
    # and 6.4 keV the iron K-alpha line
    energy = 6.4 #3.6 #1.2398425 #

    # making the interferometer itself
    test_I = instrument.interferometer(1, roller = instrument.interferometer.smooth_roller, roll_speed = 2 * np.pi / 1e3)
    
    # adding a baseline, values based on discussions
    test_I.add_baseline(D = 0.05, L = 10, W = 300, num_pairs = 2)

    # taking the standard detector, it can be changed
    detector = test_I.baselines[0].camera

    # creating the source
    # image = images.point_source(int(1e3), 0.000, 0.0009, energy)
    """Code for a plot of cyg X-1""" #works
    image_path = r"Models\hmxb.jpg"
    img_scale = .00055 # largest axis size in mas
    spectrum = None
    source_counts =  int(1e2) #0 #
    bkg_counts = 0 #int(1e2) #
    image, pix_scale = images.generate_from_image(image_path, int(source_counts), img_scale, spectrum = spectrum, energy = 6.4, bkg_phot=bkg_counts)#3.6)#1.2398425)#, energy_spread = 0.1)#3.6)#6.4)#

    
    # observing the source using the optical bench
    test_data = process.interferometer_data(test_I, image)#, pure_fringes=True)
    
    # binning the entire detector, based on the detector charateristics
    bins = int(np.ceil(abs(detector.pos_range[0] - detector.pos_range[1]) / detector.res_pos))
    edges = np.array([detector.pos_range[0] + i * detector.res_pos for i in range(bins + 1)]) * 1e6

    # plotting the binned photon positions
    # plt.hist(test_data.pos * 1e6, edges, label=f'{energy} keV')
    # plt.xlabel('Photon impact positions ($\\mu$m)')
    # plt.ylabel('Number of photons in bin')
    # # plt.ylim(0, 2200)
    # plt.show()

    # making a periodogram of the above photon positions
    analysis.periodogram(test_data, detector, test_data.pos)



def background_test():

    def fourier2D(matrix_in):
        """
        Calculate the 2D Fourier transform of a matrix

        Input:
            matrix_in: 2D numpy array
        
        Output:
            matrix_out: 2D Fourier transform of matrix_in
        """
    
        matrix_out = np.absolute(np.fft.fft2(matrix_in))
        matrix_out = np.roll(matrix_out, int(matrix_out.shape[0]/2), axis=0)
        matrix_out = np.roll(matrix_out, int(matrix_out.shape[1]/2), axis=1)
        return matrix_out

    """Point source"""
    # offset = 0.0009
    # image = images.point_source(int(1e7), 0.000, offset, 1.2398425) # image = images.m_point_sources(int(1e6), 4, [0.000, -0.000, -.0004, .00085], [0.000236, -0.00065, 0., 0.], [1.2, 1.2, 1.2, 1.2])
    # # img_scale is unclear to me what it's based on. I presume it to be based on pixel size compared to source size
    # img_scale = 1
    # pix_scale =  np.array([330, 550])

    """All models are those from the dropbox 'models'."""

    """Code for a plot of cyg X-1""" #works
    image_path = r"Models\hmxb.jpg"
    img_scale = .00055

    tranmis_ISM = np.load(r"tbabs_transm_NH0p1e22.npy")
    # plt.plot(tranmis_ISM[0], tranmis_ISM[1])
    # plt.ylabel("Transmission coefficient (0-1)")
    # plt.xlabel("Energy (keV)")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.ylim(1e-4, 1)
    # plt.show()

    nH = 6e21 # between 5.4e21 and 7e21 cm^2
    gamma = 2

    sample_energies = np.linspace(1, 7, 1000) # keV

    transm_fact = spinter.interp1d(tranmis_ISM[0], tranmis_ISM[1])(sample_energies)

    pow_spect = np.power(sample_energies, - gamma)

    transm_spec = pow_spect * np.power(transm_fact, nH / 1e21) #transm_fact
    # rescale_transmis = np.power(transm_spec, nH / 1e21)

    spectrum = np.array([sample_energies, transm_spec]).T # rescale_transmis

    # plt.plot(spectrum[:,0] / (1e3 * spc.eV), transm_spec)#rescale_transmis)#np.power( * 1e21, 6))#np.power(spectrum[:,0] / (1e3 * spc.eV), 2) * pow_spect * rescale_transmis)
    # plt.ylabel("Non-normalized counts (s^-1 keV^-1 m^-2)")
    # plt.xlabel("Energy (keV)")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()

    # spectrum = None

    """Code for AU mic""" #DOESN'T work
    # Image is big though, so expect a long wait
    # image_path = r"Models\au_mic.png"
    # img_scale = 0.0013

    """Code for exop model""" #works
    # image_path = r"Models\exop.jpg"
    # img_scale = 0.5 * 1e-3

    """Code for spiral model""" #DOESN'T work
    # image_path = r"Models\spiral.jpg"
    # img_scale = 5 * 1e-3

    """Code for sgr A*""" #works
    # Remember to add // 5 to pix_scale to make sure there aren't too many useless pixels taken into account
    # image_path = r"Models\bhdisk.png"
    # img_scale = 0.00037

    """Code for 'compact' model""" #Uncertain
    # image_path = r"Models\compact.png"
    # img_scale = 0.000512 * 2

    """This samples the source, when an image is given"""
    # spectrum = r"Models\Spectra\Test_Erik.txt"
    spectrum = None
    image, pix_scale = images.generate_from_image(image_path, int(1e6), img_scale, spectrum = spectrum, energy = 6.4, bkg_phot = int(1e6), bkg_energy = 6.4)
    

    # plots the sampled points of origin of the source photons
    # plt.plot(image.loc[:,1][image.bkg_indices] * (3600*360 / (2 * np.pi)), -image.loc[:,0][image.bkg_indices] * (3600*360 / (2 * np.pi)), '.', alpha=.2)
    histed_photons, _, _ = np.histogram2d(image.loc[:,0][image.bkg_indices], image.loc[:,1][image.bkg_indices], pix_scale)
    plt.imshow(histed_photons, cmap=cm.Reds, alpha = 0.5, extent = [0, img_scale * 1e6, img_scale * 1e6 * pix_scale[0] / pix_scale[1], 0])
    mask = np.ones(image.loc[:,0].size, dtype=bool)
    mask[image.bkg_indices] = False
    # plt.plot(image.loc[:,1][mask] * (3600*360 / (2 * np.pi)), -image.loc[:,0][mask] * (3600*360 / (2 * np.pi)), '.', alpha=.2)
    histed_photons, _, _ = np.histogram2d(image.loc[:,0][mask], image.loc[:,1][mask], pix_scale)
    plt.imshow(histed_photons, cmap=cm.Greens, alpha = 0.5, extent = [0, img_scale * 1e6, img_scale * 1e6 * pix_scale[0] / pix_scale[1], 0])
    plt.xlabel('x ($\mu$as)')
    plt.ylabel('y ($\mu$as)')
    plt.savefig("samp_bkg_on.pdf")
    plt.show()

    # seemingly arbetrary time step?
    time_step = 1
    
    # making the interferometer itself
    test_I = instrument.interferometer(time_step, roller = instrument.interferometer.smooth_roller,
                                       roll_init = 0, roll_speed = np.pi / (1e3 * time_step)) #np.max(image.toa)#.00001 * 2 * np.pi) #

    # adding the baselines, values based trying different things
    # baseline_Ds = [0.001, 0.01, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    reflec_list = [r"Models\Mirror reflectivity Si\Angle_0.01deg.txt", r"Models\Mirror reflectivity Si\Angle_0.5deg.txt", r"Models\Mirror reflectivity Si\Angle_0.8deg.txt", r"Models\Mirror reflectivity Si\Angle_1.0deg.txt", r"Models\Mirror reflectivity Si\Angle_1.3deg.txt", r"Models\Mirror reflectivity Si\Angle_1.5deg.txt", r"Models\Mirror reflectivity Si\Angle_2.0deg.txt", r"Models\Mirror reflectivity Si\Angle_2.5deg.txt", r"Models\Mirror reflectivity Si\Angle_5.0deg.txt"]
    for D in np.linspace(0.05, 1, 10): #baseline_Ds: #np.geomspace(0.10, 4.0, 10):
        # test_I.add_baseline(num_pairs = 1, D = D, L = 10, W = 300)
        # test_I.add_baseline(num_pairs = 5, D = D, L = 10, W = 300)
        # test_I.add_baseline(num_pairs = 5, D = D, L = 10, W = 400)
        # test_I.add_baseline(num_pairs = 5, D = D, L = 10, W = 400, grazing_angle = 2 * D, mirr_mater = reflec_list)
        # test_I.add_baseline(num_pairs = 5, D = D, L = 10, W = 400, grazing_angle = (0.7598) * np.pi / 180, mirr_mater = reflec_list) # most realistic atm based on discussions
        test_I.add_willingale_baseline(D)

    # observing the source using the optical benches and timing it
    start = time.time()
    # potentially adding positional and energy noise
    test_data = process.interferometer_data(test_I, image)#, 2, .15 / (2 * np.sqrt(2*np.log(2))))

    # adding detector effects if desired
    # test_data.detector_effects(test_I, [0, 3, 9])

    print('Processing this image took ', time.time() - start, ' seconds')

    # reconstruction of the source and timing it
    # start = time.time()
    # re_im, uv = analysis.image_recon_smooth(test_data, test_I, img_scale, samples = pix_scale) 
    # print('Reconstructing this image took ', time.time() - start, ' seconds')
    recon_image_info, uv = analysis.image_recon_smooth(test_data, test_I, img_scale, samples = pix_scale, error = 0.1) 
    re_im, max_size = recon_image_info
    print('Reconstructing this image took ', time.time() - start, ' seconds')


    # showing the sampled points in the uv-plane
    fig = plt.figure(figsize=(6,6))
    plt.plot(uv[:, 0], uv[:, 1], 'g.')
    plt.xlim(-np.max(uv) * 1.2, np.max(uv) * 1.2)
    plt.ylim(-np.max(uv) * 1.2, np.max(uv) * 1.2)
    plt.title("uv-plane sampling")
    # plt.savefig("uv_plane_img2")
    plt.show()

    # showing the log of the Fourier transform of the reconstructed image
    fig = plt.figure(figsize=(6,6))
    plt.imshow(np.log10(fourier2D(re_im)), cmap=cm.cubehelix)
    plt.colorbar()
    # plt.savefig("fourier_img2")
    plt.show()

    # showing the reconstructed image
    fig = plt.figure(figsize=(6,6))#(20,20))
    # plt.rc('font', size=50)
    plt.imshow(re_im, extent = [0, img_scale * 1e6, img_scale * 1e6 * pix_scale[0] / pix_scale[1], 0], cmap=cm.cubehelix)
    plt.xlabel('x ($\mu$as)')
    plt.ylabel('y ($\mu$as)')
    # plt.colorbar()
    # plt.savefig("BKG_on.png")#.format(source_counts))
    plt.show()

    # showing the log of the reconstructed image
    # fig = plt.figure(figsize=(6,6))
    # plt.imshow(np.log10(re_im), cmap=cm.cubehelix)
    # plt.colorbar()
    # plt.xlabel('x ($\mu$as)')
    # plt.ylabel('y ($\mu$as)')
    # plt.show()

    # checking image reconstruction
    ssr = analysis.image_compare(np.array(Image.open(image_path).convert('L')), re_im)
    print("The Sum of Squared Residuals (SSR) for this reconstructed image is {:.3f}.".format(ssr))

    # taking the standard detector, it can be changed
    # detector = test_I.baselines[0].camera

    # making a periodogram of the above photon positions
    # analysis.periodogram(test_data, detector, test_data.pos)



def speed_test():

    # seemingly arbetrary time step?
    time_step = 1
    # making the interferometer itself
    test_I = instrument.interferometer(time_step, roller = instrument.interferometer.smooth_roller,
                                       roll_init = 0, roll_speed = np.pi / (1e3)) #.00001 * 2 * np.pi) #

    # adding the baselines, values based trying different things
    for D in np.linspace(0.05, 1, 10):
        test_I.add_willingale_baseline(D)

    all_timing = []
    number = 1

    for recon_type in ["eIFT"]: #"IFFT", 

        recon_timing = []

        for spectrum in [None]:

            if spectrum is not None:
                tranmis_ISM = np.load(r"tbabs_transm_NH0p1e22.npy")
                # plt.plot(tranmis_ISM[0], tranmis_ISM[1])
                # plt.ylabel("Transmission coefficient (0-1)")
                # plt.xlabel("Energy (keV)")
                # plt.xscale("log")
                # plt.yscale("log")
                # plt.ylim(1e-4, 1)
                # plt.show()

                nH = 6e21 # between 5.4e21 and 7e21 cm^2
                gamma = 2

                sample_energies = np.linspace(1, 7, 1000) # keV

                transm_fact = spinter.interp1d(tranmis_ISM[0], tranmis_ISM[1])(sample_energies)

                pow_spect = np.power(sample_energies, - gamma)

                transm_spec = pow_spect * np.power(transm_fact, nH / 1e21) #transm_fact
                # rescale_transmis = np.power(transm_spec, nH / 1e21)

                spectrum = np.array([sample_energies, transm_spec]).T # rescale_transmis

            spectral_timing = []

            """All models are those from the dropbox 'models'."""
            for model in ["sgr A*"]: #"cyg X-1", "exoplanet", 
                model_timing = []
                if model == "cyg X-1":
                    """Code for a plot of cyg X-1""" #works
                    image_path = r"Models\hmxb.jpg"
                    img_scale = .00055 # x-axis size in mas
                elif model == "exoplanet":
                    """Code for exoplanet model""" #works
                    image_path = r"Models\exop.jpg"
                    img_scale = 0.5 * 1e-3
                else:
                    """Code for sgr A*""" #works
                    image_path = r"Models\bhdisk.png"
                    img_scale = 0.00037

                for source_count in [1e3, 1e4, 1e5]:#, 1e6, 1e7]:

                    count_timing = []

                    # repeat
                    for i in range(3):

                        repeat_timing = []

                        """This samples the source, when an image is given"""
                        image, pix_scale = images.generate_from_image(image_path, int(source_count), img_scale, spectrum = spectrum, energy = 6.4)#3.6)#1.2398425)#, energy_spread = 0.1)#3.6)#6.4)#

                        # potentially adding positional and energy noise
                        test_data = process.interferometer_data(test_I, image)#, 2, .15 / (2 * np.sqrt(2*np.log(2))))

                        # reconstruction of the source and timing it
                        start = time.time()
                        # print(pix_scale)
                        recon_image_info, uv = analysis.image_recon_smooth(test_data, test_I, img_scale, samples = pix_scale, error = 0.1, recon_type = recon_type) 
                        re_im, max_size = recon_image_info
                        repeat_timing.append(time.time() - start)
                        print('Reconstructing this image took ', time.time() - start, ' seconds')
                        print("Reconstruction {} / 90".format(number))
                        number += 1

                    count_timing.append(repeat_timing)
                model_timing.append(count_timing)
            spectral_timing.append(model_timing)
        recon_timing.append(spectral_timing)
    all_timing.append(spectral_timing)

    file = open('timing.txt','w')
    for timing in all_timing:
        file.write(timing+"\n")
    file.close()





def recon_qually_test():

    # seemingly arbetrary time step?
    time_step = 1
    # making the interferometer itself
    test_I = instrument.interferometer(time_step, roller = instrument.interferometer.smooth_roller,
                                       roll_init = 0, roll_speed = np.pi / (1e3)) #.00001 * 2 * np.pi) #

    # adding the baselines, values based trying different things
    for D in np.linspace(0.05, 1, 10):
        test_I.add_willingale_baseline(D)

    all_ssr = []
    number = 1

    spectrum = None

    for energy in [6.4]:#1.2398425, 3.6, 

        model_ssr = []

        """All models are those from the dropbox 'models'."""
        for model in ["cyg X-1"]:#, "exoplanet", "sgr A*"]:#
            if model == "cyg X-1":
                """Code for a plot of cyg X-1""" #works
                image_path = r"Models\hmxb.jpg"
                img_scale = .00055 # x-axis size in mas
            elif model == "exoplanet":
                """Code for exoplanet model""" #works
                image_path = r"Models\exop.jpg"
                img_scale = 0.5 * 1e-3
            else:
                """Code for sgr A*""" #works
                image_path = r"Models\bhdisk.png"
                img_scale = 0.00037

            count_ssr = []

            for source_count in [1e3, 1e4, 1e5, 1e6, 1e7]:

                repeat_ssr = []

                # repeat
                for i in range(3):

                    """This samples the source, when an image is given"""
                    image, pix_scale = images.generate_from_image(image_path, int(source_count), img_scale, spectrum = spectrum, energy = energy)#3.6)#1.2398425)#, energy_spread = 0.1)#3.6)#6.4)#

                    # potentially adding positional and energy noise
                    test_data = process.interferometer_data(test_I, image)#, 2, .15 / (2 * np.sqrt(2*np.log(2))))

                    # reconstruction of the source
                    recon_image_info, uv = analysis.image_recon_smooth(test_data, test_I, img_scale, samples = pix_scale, error = 0.1) 
                    re_im, max_size = recon_image_info

                    # checking image reconstruction
                    ssr = analysis.image_compare(np.array(Image.open(image_path).convert('L')), re_im)
                    repeat_ssr.append(ssr)


                    # showing the reconstructed image
                    fig = plt.figure(figsize=(6,6))#20,20))
                    # plt.rc('font', size=50)
                    plt.imshow(re_im, extent = [0, img_scale * 1e6, img_scale * 1e6 * pix_scale[0] / pix_scale[1], 0], cmap=cm.cubehelix)
                    plt.xlabel('x ($\mu$as)')
                    plt.ylabel('y ($\mu$as)')
                    plt.savefig(r"ssr_nphot\recon_{}_{}.pdf".format(source_count, i + 1))
                    plt.close()

                    # show residual
                    input_im = np.array(Image.open(image_path).convert('L'))
                    # showing the reconstructed image
                    fig = plt.figure(figsize=(6,6))#20,20))
                    # plt.rc('font', size=50)
                    plt.imshow(((re_im - np.min(re_im)) / (np.max(re_im) - np.min(re_im))) - ((input_im - np.min(input_im)) / (np.max(input_im) - np.min(input_im))), extent = [0, img_scale * 1e6, img_scale * 1e6 * pix_scale[0] / pix_scale[1], 0], cmap=cm.cubehelix)
                    plt.xlabel('x ($\mu$as)')
                    plt.ylabel('y ($\mu$as)')
                    plt.colorbar()
                    plt.savefig(r"ssr_nphot\residual_{}_{}.pdf".format(source_count, i + 1))
                    plt.close()

                    print("The Sum of Squared Residuals (SSR) for this reconstructed image is {}.".format(ssr))
                    print("Reconstruction {} / 15".format(number))
                    number += 1

                count_ssr.append(repeat_ssr)
            model_ssr.append(count_ssr)
        all_ssr.append(model_ssr)

    print(all_ssr)

    np.savetxt("SSR.txt", np.array(all_ssr))




    
def eff_area_test():

    sample_energies = np.linspace(1, 7, 1000) # keV

    spectrum = np.array([sample_energies, np.repeat(1, sample_energies.size)]).T 

    """Point source"""
    offset = 0.0009
    image = images.point_source(int(1e4), 0.000, offset, 1.2398425, spectrum = spectrum) # image = images.m_point_sources(int(1e6), 4, [0.000, -0.000, -.0004, .00085], [0.000236, -0.00065, 0., 0.], [1.2, 1.2, 1.2, 1.2])
    # img_scale is unclear to me what it's based on. I presume it to be based on pixel size compared to source size
    img_scale = 1
    pix_scale =  np.array([330, 550])

    # plt.plot(spectrum[:,0] / (1e3 * spc.eV), spectrum[:,1])#rescale_transmis)#np.power( * 1e21, 6))#np.power(spectrum[:,0] / (1e3 * spc.eV), 2) * pow_spect * rescale_transmis)
    # plt.ylabel("Non-normalized counts (s^-1 keV^-1 m^-2)")
    # plt.xlabel("Energy (keV)")
    # plt.show()

    # seemingly arbetrary time step?
    time_step = 1
    
    # making the interferometer itself
    test_I = instrument.interferometer(time_step, roller = instrument.interferometer.smooth_roller,
                                       roll_init = 0, roll_speed = np.pi / (np.max(image.toa) * time_step)) #.00001 * 2 * np.pi) #

    reflec_list = [r"Models\Mirror reflectivity Si\Angle_0.01deg.txt", r"Models\Mirror reflectivity Si\Angle_0.5deg.txt", r"Models\Mirror reflectivity Si\Angle_0.8deg.txt", r"Models\Mirror reflectivity Si\Angle_1.0deg.txt", r"Models\Mirror reflectivity Si\Angle_1.3deg.txt", r"Models\Mirror reflectivity Si\Angle_1.5deg.txt", r"Models\Mirror reflectivity Si\Angle_2.0deg.txt", r"Models\Mirror reflectivity Si\Angle_2.5deg.txt", r"Models\Mirror reflectivity Si\Angle_5.0deg.txt"]
    
    # adding the baselines, values based trying different things
    for D in np.linspace(0.05, 1, 10):
        test_I.add_baseline(num_pairs = 4, D = D, L = 10, W = 400, grazing_angle = (0.9) * np.pi / 180, mirr_mater = reflec_list) # most realistic atm based on discussions
        

    # observing the source using the optical benches and timing it
    test_data = process.interferometer_data(test_I, image, eff_area_show = True)
    

def visibility_test_repeat(no_Ds, no_sims, energy):
    test_I = instrument.interferometer(1, roller = instrument.interferometer.smooth_roller,
                                       roll_init = 0, roll_speed = 0)
    # test_I = instrument.interferometer(.01, 1, 2, np.array([1.2, 7]), np.array([-400, 400]), 
    #                                     0.00, None, instrument.interferometer.smooth_roller, 
    #                                     .00000 * 2 * np.pi, roll_init=0/4 * np.pi)
    img_scale = .0004
    image = images.double_point_source(int(1e5), [0., 0.], [120 * 1e-6 / 2, 120 * 1e-6 / 2], [energy, energy]) #images.disc(int(1e5), 0, 0, energy, img_scale / 2)
    image_2 = images.double_point_source(int(1e5), [0., 0.], [240 * 1e-6 / 2, 240 * 1e-6 / 2], [energy, energy]) #images.disc(int(1e5), 0, 0, 2 * energy, img_scale / 2)
    image_3 = images.double_point_source(int(1e5), [0., 0.], [360 * 1e-6 / 2, 360 * 1e-6 / 2], [energy, energy]) #images.disc(int(1e5), 0, 0, 3 * energy, img_scale / 2)
    image_4 = images.double_point_source(int(1e5), [0., 0.], [480 * 1e-6 / 2, 480 * 1e-6 / 2], [energy, energy]) #images.disc(int(1e5), 0, 0, 4 * energy, img_scale / 2)

    Ds = np.linspace(.005, 1, no_Ds)

    def calc_vis(image, Ds):
        vis = np.zeros((no_Ds, 2))
        for i, D in enumerate(Ds):
            print(f'Now doing baseline length {D}, which is test {i + 1}')
            test_I.clear_baselines()
            test_I.add_baseline(num_pairs = 1, D = D, L = 10, W = 300)
            test_I.baselines[0].add_custom_detector(res_E = 0.01, res_t = 0.01, res_pos = 0.02, pos_noise = 2, E_noise = .15 / (2.355), E_range = np.array([1.2, 7]), pos_range = np.array([-400, 400]))

            bins = int(np.ceil(abs(test_I.baselines[0].camera.pos_range[0] - test_I.baselines[0].camera.pos_range[1]) / test_I.baselines[0].camera.res_pos))
            edges = np.array([test_I.baselines[0].camera.pos_range[0] + i * (test_I.baselines[0].camera.res_pos) for i in range(bins + 1)])
            centres = edges[:-1] + (edges[1:] - edges[:-1])/2

            freq = 1 / (test_I.baselines[0].L * spc.h * spc.c / (image.energies[0] * test_I.baselines[0].W))
            amps = np.zeros((no_sims, 2))

            for sim in range(no_sims):
                test_data = process.interferometer_data(test_I, image)#, pure_fringes = True)
                # test_data = test_I.baselines[0].camera.add_detector_effects(test_data, np.repeat(True, test_data.pos.size))

                y_data, _ = np.histogram(test_data.pos, edges)

                amps[sim, 0] = np.abs(np.sum(y_data * np.exp(-2j * np.pi * freq * centres)) / y_data.size)
                amps[sim, 1] = np.abs(np.sum(y_data) / y_data.size)

            vis[i, 0] = np.mean(amps[:, 0]) / np.mean(amps[:, 1])
            vis[i, 1] = np.std(amps[:, 0]) / np.mean(amps[:, 1])

        return vis * 2
    
    vis = calc_vis(image, Ds)
    vis_2 = calc_vis(image_2, Ds)
    vis_3 = calc_vis(image_3, Ds)
    vis_4 = calc_vis(image_4, Ds)

    D_theory = (spc.h * spc.c / (energy * spc.eV * 1e3)) / (img_scale * 2 * np.pi / (3600 * 360))
    D_theory_2 = (spc.h * spc.c / (2 * energy * spc.eV * 1e3)) / (img_scale * 2 * np.pi / (3600 * 360))
    D_theory_3 = (spc.h * spc.c / (3 * energy * spc.eV * 1e3)) / (img_scale * 2 * np.pi / (3600 * 360))
    D_theory_4 = (spc.h * spc.c / (4 * energy * spc.eV * 1e3)) / (img_scale * 2 * np.pi / (3600 * 360))

    plt.errorbar(Ds, vis[:, 0], yerr=vis[:, 1], marker='.', ls='-', label=f'120 mu mas')
    plt.vlines(D_theory, vis[:, 0].min(), vis[:, 0].max(), 'b', alpha=.3)
    plt.errorbar(Ds, vis_2[:, 0], yerr=vis_2[:, 1], marker= '.', ls='-.', label=f'240 mu mas')
    plt.vlines(D_theory_2, vis_2[:, 0].min(), vis_2[:, 0].max(), 'orange', alpha=.3)
    plt.errorbar(Ds, vis_3[:, 0], yerr=vis_3[:, 1], marker= '.', ls=':', label=f'360 mu mas')
    plt.vlines(D_theory_3, vis_3[:, 0].min(), vis_3[:, 0].max(), 'g', alpha=.3)
    plt.errorbar(Ds, vis_4[:, 0], yerr=vis_4[:, 1], marker= '.', ls='--', label=f'480 mu mas')
    plt.vlines(D_theory_4, vis_4[:, 0].min(), vis_4[:, 0].max(), 'r', alpha=.3)
    plt.title(f'Average of {no_sims} observations of dual point sources at variable seperations')
    plt.xlabel('Baseline length (m)')
    plt.ylabel('Visibility')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    # willingale_test() # outdated
    # image_re_test() # outdated
    # image_re_test_multiple() # outdated
    # w_ps_test() # outdated
    # Fre_test()
    # scale_test2()
    # discretize_test() # outdated
    # sinety_test()
    # sinetier_test()
    # full_image_test(0) # bugged
    # image_re_test_point() # outdated
    # image_re_test_parts()
    # image_re_test_exact()
    # image_re_test_uv() # outdated
    # full_image_test(0) # bugged
    # stats_test() # outdated
    # locate_test(-400e-6, 10, 1e6, 1.2, 1)
    # locate_test_multiple_E(1e-6, 10, 1e6, [1.2, 2.4, 3.6, 4.8, 6], 1, 2, .15 / (2 * np.sqrt(2 * np.log(2)))) #[1.2, 2.4, 3.6, 4.8, 6]
    # visibility_test_E(20, 5, 1.2)
    # visibility_test_scale(40, 5, 1.2)
    # visibility_test_2(50, 10, 1.2) # bugged
    # fringes_plots()
    # Willingale_plot7()
    # Correct_diff_fringes() # Send to Phil in notebook
    # image_reconstruction() # Send to Phil in notebook
    # test_periodgram() # Send to Phil in notebook
    # background_test() # Send to Phil in notebook
    # speed_test()
    # recon_qually_test()
    # eff_area_test()
    # visibility_test_repeat(50, 50, 1.2)

    number_photons = np.array([1e3, 1e4, 1e5, 1e6, 1e7])

    # IFFT_Cyg_3 = np.array([1.9631474018096924, 1.964709758758545, 1.9854674339294434])
    # IFFT_Cyg_4 = np.array([2.2012534141540527, 1.966874361038208, 2.0153300762176514])
    # IFFT_Cyg_5 = np.array([2.2482523918151855, 2.269254446029663, 2.236830949783325])
    # IFFT_Cyg_6 = np.array([5.281658172607422, 5.194562196731567, 5.110870122909546])
    # IFFT_Cyg_7 = np.array([76.57766318321228, 60.02653241157532, 48.7470543384552])

    # IFFT_Exo_3 = np.array([60.394174098968506, 56.220364570617676, 47.45090055465698])
    # IFFT_Exo_4 = np.array([47.061010122299194, 46.749977111816406, 48.105910778045654])
    # IFFT_Exo_5 = np.array([45.7546546459198, 50.226318359375, 79.85661053657532])
    # IFFT_Exo_6 = np.array([55.69122290611267, 59.75592279434204, 51.28024196624756])
    # IFFT_Exo_7 = np.array([89.45436525344849, 90.06537652015686, 100.69448661804199])

    # IFFT_SgA_3 = np.array([1144.2079203128815, 996.4088551998138, 891.9247572422028])
    # IFFT_SgA_4 = np.array([1007.846200466156, 917.9999792575836, 940.4920704364777])
    # IFFT_SgA_5 = np.array([874.2314040660858, 942.5474908351898, 889.7235646247864])
    # IFFT_SgA_6 = np.array([907.2256791591644, 878.0419678688049, 880.2273998260498])
    # IFFT_SgA_7 = np.array([701.7326200008392, 668.9526920318604, 662.4195463657379])


    # eIFT_Cyg_3 = np.array([23.580647706985474, 25.362747192382812, 22.795028924942017])
    # eIFT_Cyg_4 = np.array([210.63913583755493, 210.20079231262207, 210.90441608428955])
    # eIFT_Cyg_5 = np.array([3434.2364015579224, 3433.2942633628845, 3471.781613588333])
    # eIFT_Cyg_6 = np.array([])
    # eIFT_Cyg_7 = np.array([])

    # eIFT_Exo_3 = np.array([183.58709502220154, 196.87431049346924, 173.3073971271515])
    # eIFT_Exo_4 = np.array([1749.9228456020355, 1753.175392627716, 1728.207153081894])
    # eIFT_Exo_5 = np.array([18495.82885479927, 15852.687741041183, 15807.301728248596])
    # eIFT_Exo_6 = np.array([])
    # eIFT_Exo_7 = np.array([])

    # eIFT_SgA_3 = np.array([138.07716751098633, 136.75148916244507, 137.3305847644806])
    # eIFT_SgA_4 = np.array([1435.3916058540344, 1389.2145638465881, 1408.7995910644531])
    # eIFT_SgA_5 = np.array([14898.095885276794, 14824.884345769882, 14773.56988620758])
    # eIFT_SgA_6 = np.array([])
    # eIFT_SgA_7 = np.array([])



    # IFFT_Cyg_timing = np.array([np.mean(IFFT_Cyg_3), np.mean(IFFT_Cyg_4), np.mean(IFFT_Cyg_5), np.mean(IFFT_Cyg_6), np.mean(IFFT_Cyg_7)])
    # IFFT_Cyg_error = np.array([np.std(IFFT_Cyg_3), np.std(IFFT_Cyg_4), np.std(IFFT_Cyg_5), np.std(IFFT_Cyg_6), np.std(IFFT_Cyg_7)])
    
    # IFFT_Exo_timing = np.array([np.mean(IFFT_Exo_3), np.mean(IFFT_Exo_4), np.mean(IFFT_Exo_5), np.mean(IFFT_Exo_6), np.mean(IFFT_Exo_7)])
    # IFFT_Exo_error = np.array([np.std(IFFT_Exo_3), np.std(IFFT_Exo_4), np.std(IFFT_Exo_5), np.std(IFFT_Exo_6), np.std(IFFT_Exo_7)])
    
    # IFFT_SgA_timing = np.array([np.mean(IFFT_SgA_3), np.mean(IFFT_SgA_4), np.mean(IFFT_SgA_5), np.mean(IFFT_SgA_6), np.mean(IFFT_SgA_7)])
    # IFFT_SgA_error = np.array([np.std(IFFT_SgA_3), np.std(IFFT_SgA_4), np.std(IFFT_SgA_5), np.std(IFFT_SgA_6), np.std(IFFT_SgA_7)])
    
    # eIFT_Cyg_timing = np.array([np.mean(eIFT_Cyg_3), np.mean(eIFT_Cyg_4), np.mean(eIFT_Cyg_5), np.mean(eIFT_Cyg_6), np.mean(eIFT_Cyg_7)])
    # eIFT_Cyg_error = np.array([np.std(eIFT_Cyg_3), np.std(eIFT_Cyg_4), np.std(eIFT_Cyg_5), np.std(eIFT_Cyg_6), np.std(eIFT_Cyg_7)])
    
    # eIFT_Exo_timing = np.array([np.mean(eIFT_Exo_3), np.mean(eIFT_Exo_4), np.mean(eIFT_Exo_5), np.mean(eIFT_Exo_6), np.mean(eIFT_Exo_7)])
    # eIFT_Exo_error = np.array([np.std(eIFT_Exo_3), np.std(eIFT_Exo_4), np.std(eIFT_Exo_5), np.std(eIFT_Exo_6), np.std(eIFT_Exo_7)])
    
    # eIFT_SgA_timing = np.array([np.mean(eIFT_SgA_3), np.mean(eIFT_SgA_4), np.mean(eIFT_SgA_5), np.mean(eIFT_SgA_6), np.mean(eIFT_SgA_7)])
    # eIFT_SgA_error = np.array([np.std(eIFT_SgA_3), np.std(eIFT_SgA_4), np.std(eIFT_SgA_5), np.std(eIFT_SgA_6), np.std(eIFT_SgA_7)])


    # # plt.figure().patch.set_facecolor('white')
    # # plt.axes().set_facecolor('white')


    # plt.errorbar(number_photons, IFFT_Cyg_timing, IFFT_Cyg_error, linestyle = "-", color = '#1f77b4', marker = "s", label="IFFT, Cygnus X-1 model")
    # plt.errorbar(number_photons, IFFT_Exo_timing, IFFT_Exo_error, linestyle = "-", color = '#ff7f0e', marker = "d", label="IFFT, AU Microscopii model")
    # plt.errorbar(number_photons, IFFT_SgA_timing, IFFT_SgA_error, linestyle = "-", color = '#2ca02c', marker = "o", label="IFFT, Sagittarius A* model")

    # plt.errorbar(number_photons, eIFT_Cyg_timing, eIFT_Cyg_error, linestyle = "--", color = '#1f77b4', marker = "s", label="eIFT, Cygnus X-1 model")
    # plt.errorbar(number_photons, eIFT_Exo_timing, eIFT_Exo_error, linestyle = "--", color = '#ff7f0e', marker = "d", label="eIFT, AU Microscopii model")
    # plt.errorbar(number_photons, eIFT_SgA_timing, eIFT_SgA_error, linestyle = "--", color = '#2ca02c', marker = "o", label="eIFT, Sagittarius A* model")

    # plt.rc('font', size=15)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # # plt.title('IFFT vs eIFT timing')
    # plt.xlabel('Number of source photons', fontsize = 20)
    # plt.ylabel('Reconstruction time (s)', fontsize = 20)
    # plt.yscale("log")
    # plt.xscale("log")
    # # plt.xlim(0, 800)
    # # plt.ylim(0, 6)
    # plt.legend()

    # # plt.savefig('timing.pdf')
    # plt.show()







    Cyg_124_3 = np.array([1236114597.848747, 1236114597.6002932, 1236114597.020531])
    Cyg_124_4 = np.array([1236114492.9688253, 1236114488.7199087, 1236114492.5819526])
    Cyg_124_5 = np.array([1236113430.8753245, 1236113427.6110616, 1236113422.1707175])
    Cyg_124_6 = np.array([1236102847.1334524, 1236102838.8741727, 1236102813.2981594])
    Cyg_124_7 = np.array([1235996821.2452385, 1235996768.553115, 1235996612.8158727])

    Exo_124_3 = np.array([])
    Exo_124_4 = np.array([])
    Exo_124_5 = np.array([])
    Exo_124_6 = np.array([])
    Exo_124_7 = np.array([])

    SgA_124_3 = np.array([])
    SgA_124_4 = np.array([])
    SgA_124_5 = np.array([])
    SgA_124_6 = np.array([])
    SgA_124_7 = np.array([])


    Cyg_36_3 = np.array([1236114571.7961109, 1236114582.2834003, 1236114569.3785374])
    Cyg_36_4 = np.array([1236114228.8148422, 1236114245.5130758, 1236114247.5420103])
    Cyg_36_5 = np.array([1236110826.4788322, 1236110882.232902, 1236110849.6094344])
    Cyg_36_6 = np.array([1236076554.4684148, 1236076670.1217275, 1236076816.7994378])
    Cyg_36_7 = np.array([1235734826.7889733, 1235735260.4469361, 1235734612.0447907])

    Exo_36_3 = np.array([17391007664.81562, 17391007658.25832, 17391007644.369984])
    Exo_36_4 = np.array([17391007151.393616, 17391007161.457447, 17391007155.635056])
    Exo_36_5 = np.array([17391002243.27637, 17391002247.73468, 17391002257.896156])
    Exo_36_6 = np.array([17390952804.444717, 17390953069.74539, 17390952901.365723])
    Exo_36_7 = np.array([])

    SgA_36_3 = np.array([])
    SgA_36_4 = np.array([])
    SgA_36_5 = np.array([])
    SgA_36_6 = np.array([])
    SgA_36_7 = np.array([])


    Cyg_64_3 = np.array([1236114542.8240502, 1236114547.4153028, 1236114564.8724923])
    Cyg_64_4 = np.array([1236113935.5952806, 1236113947.9706137, 1236113896.0695245])
    Cyg_64_5 = np.array([1236107895.228716, 1236107799.034359, 1236107870.7725048])
    Cyg_64_6 = np.array([1236045421.558281, 1236045875.9858615, 1236045629.3317094])
    Cyg_64_7 = np.array([1235427341.2347097, 1235428444.4732823, 1235426343.8226888])

    Exo_64_3 = np.array([17391007624.97117, 17391007612.844875, 17391007634.974373])
    Exo_64_4 = np.array([17391006696.68545, 17391006651.355335, 17391006739.965458])
    Exo_64_5 = np.array([17390997835.589954, 17390997881.423244, 17390997941.98285])
    Exo_64_6 = np.array([17390911139.608047, 17390910906.227394, 17390910060.975716])
    Exo_64_7 = np.array([17390031223.367237, 17390030592.809185, 17390028582.729874])

    SgA_64_3 = np.array([1489423982.971174, 1489423983.7859428, 1489423979.137187])
    SgA_64_4 = np.array([1489423845.8074262, 1489423858.207316, 1489423848.2682767])
    SgA_64_5 = np.array([1489422533.8384407, 1489422529.461073, 1489422552.7962255])
    SgA_64_6 = np.array([1489409347.454425, 1489409399.8325233, 1489409419.0246832])
    SgA_64_7 = np.array([1489277725.0397675, 1489277921.3032167, 1489277579.0429018])

    # normalize sum to 1
    Cyg_64_3_new = np.array([3.208651194433485 * 1e-05, 3.07335975999829 * 1e-05, 3.367467494571502 * 1e-05])
    Cyg_64_4_new = np.array([2.7745310188525703 * 1e-05, 2.864349027276037 * 1e-05, 2.7148755401509657 * 1e-05])
    Cyg_64_5_new = np.array([2.8139639900840082 * 1e-05, 2.8278655755612655 * 1e-05, 2.7904435195362887 * 1e-05])
    Cyg_64_6_new = np.array([2.7889884636435872 * 1e-05, 2.803921889893723 * 1e-05, 2.7896191977188387 * 1e-05])
    Cyg_64_7_new = np.array([2.795604635551624 * 1e-05, 2.7906015187859024 * 1e-05, 2.793854082794949 * 1e-05])

    # normalize mean to 1
    Cyg_64_3_new = np.array([1035314.8564289133, 1090575.2998458548, 1060631.3351250472])
    Cyg_64_4_new = np.array([959132.4402738251, 933511.876965703, 942005.8091980561])
    Cyg_64_5_new = np.array([925471.0842995438, 921584.2152094619, 926589.9650561545])
    Cyg_64_6_new = np.array([922116.7173445274, 921735.9708625609, 925603.227290945])
    Cyg_64_7_new = np.array([921195.1375795823, 922497.7342393314, 922520.8545621582])

    # normalize by subtracting minimum and dividing by maximum
    Cyg_64_3_new = np.array([68243.24108901877, 133480.41471306005, 56030.558339935305])
    Cyg_64_4_new = np.array([33849.459707714625, 39484.847652377284, 24112.870323100407])
    Cyg_64_5_new = np.array([25592.900939951185, 20580.79092877594, 24032.348023837334])
    Cyg_64_6_new = np.array([22517.503722998914, 23010.743296383607, 23925.135100078103])
    Cyg_64_7_new = np.array([22624.169950505377, 23335.40806607, 23496.365693644617])

    # normalize by Phil equation
    Cyg_64_3_new = np.array([42828.458816524726, 33503.36914274371, 46978.78301213146])
    Cyg_64_4_new = np.array([28969.68194029493, 23250.476151652685, 17179.89022678827])
    Cyg_64_5_new = np.array([21933.580314071376, 22592.45299844117, 22179.022891041142])
    Cyg_64_6_new = np.array([21042.891867355058, 21256.150203939804, 19600.885579116533])
    Cyg_64_7_new = np.array([21272.28941244448, 21365.088373019546, 21722.79599986528])



    Cyg_124_SSR = np.array([np.mean(Cyg_124_3), np.mean(Cyg_124_4), np.mean(Cyg_124_5), np.mean(Cyg_124_6), np.mean(Cyg_124_7)])
    Cyg_124_SSR_error = np.array([np.std(Cyg_124_3), np.std(Cyg_124_4), np.std(Cyg_124_5), np.std(Cyg_124_6), np.std(Cyg_124_7)])
    
    Exo_124_SSR = np.array([np.mean(Exo_124_3), np.mean(Exo_124_4), np.mean(Exo_124_5), np.mean(Exo_124_6), np.mean(Exo_124_7)])
    Exo_124_SSR_error = np.array([np.std(Exo_124_3), np.std(Exo_124_4), np.std(Exo_124_5), np.std(Exo_124_6), np.std(Exo_124_7)])
    
    SgA_124_SSR = np.array([np.mean(SgA_124_3), np.mean(SgA_124_4), np.mean(SgA_124_5), np.mean(SgA_124_6), np.mean(SgA_124_7)])
    SgA_124_SSR_error = np.array([np.std(SgA_124_3), np.std(SgA_124_4), np.std(SgA_124_5), np.std(SgA_124_6), np.std(SgA_124_7)])


    Cyg_36_SSR = np.array([np.mean(Cyg_36_3), np.mean(Cyg_36_4), np.mean(Cyg_36_5), np.mean(Cyg_36_6), np.mean(Cyg_36_7)])
    Cyg_36_SSR_error = np.array([np.std(Cyg_36_3), np.std(Cyg_36_4), np.std(Cyg_36_5), np.std(Cyg_36_6), np.std(Cyg_36_7)])
    
    Exo_36_SSR = np.array([np.mean(Exo_36_3), np.mean(Exo_36_4), np.mean(Exo_36_5), np.mean(Exo_36_6), np.mean(Exo_36_7)])
    Exo_36_SSR_error = np.array([np.std(Exo_36_3), np.std(Exo_36_4), np.std(Exo_36_5), np.std(Exo_36_6), np.std(Exo_36_7)])
    
    SgA_36_SSR = np.array([np.mean(SgA_36_3), np.mean(SgA_36_4), np.mean(SgA_36_5), np.mean(SgA_36_6), np.mean(SgA_36_7)])
    SgA_36_SSR_error = np.array([np.std(SgA_36_3), np.std(SgA_36_4), np.std(SgA_36_5), np.std(SgA_36_6), np.std(SgA_36_7)])


    Cyg_64_SSR = np.array([np.mean(Cyg_64_3), np.mean(Cyg_64_4), np.mean(Cyg_64_5), np.mean(Cyg_64_6), np.mean(Cyg_64_7)])
    Cyg_64_SSR_error = np.array([np.std(Cyg_64_3), np.std(Cyg_64_4), np.std(Cyg_64_5), np.std(Cyg_64_6), np.std(Cyg_64_7)])
    
    Exo_64_SSR = np.array([np.mean(Exo_64_3), np.mean(Exo_64_4), np.mean(Exo_64_5), np.mean(Exo_64_6), np.mean(Exo_64_7)])
    Exo_64_SSR_error = np.array([np.std(Exo_64_3), np.std(Exo_64_4), np.std(Exo_64_5), np.std(Exo_64_6), np.std(Exo_64_7)])
    
    SgA_64_SSR = np.array([np.mean(SgA_64_3), np.mean(SgA_64_4), np.mean(SgA_64_5), np.mean(SgA_64_6), np.mean(SgA_64_7)])
    SgA_64_SSR_error = np.array([np.std(SgA_64_3), np.std(SgA_64_4), np.std(SgA_64_5), np.std(SgA_64_6), np.std(SgA_64_7)])



    Cyg_64_SSR_new = np.array([np.mean(Cyg_64_3_new), np.mean(Cyg_64_4_new), np.mean(Cyg_64_5_new), np.mean(Cyg_64_6_new), np.mean(Cyg_64_7_new)])
    Cyg_64_SSR_error_new = np.array([np.std(Cyg_64_3_new), np.std(Cyg_64_4_new), np.std(Cyg_64_5_new), np.std(Cyg_64_6_new), np.std(Cyg_64_7_new)])
    
    Cyg_64_SSR_new = np.array([np.mean(Cyg_64_3_new), np.mean(Cyg_64_4_new), np.mean(Cyg_64_5_new), np.mean(Cyg_64_6_new), np.mean(Cyg_64_7_new)])
    Cyg_64_SSR_error_new = np.array([np.std(Cyg_64_3_new), np.std(Cyg_64_4_new), np.std(Cyg_64_5_new), np.std(Cyg_64_6_new), np.std(Cyg_64_7_new)])
    

    fig = plt.figure(figsize=(10,10))
    plt.rc('font', size=15)
    # plt.figure().patch.set_facecolor('white')
    # plt.axes().set_facecolor('white')


    # plt.errorbar(number_photons, Cyg_124_SSR, Cyg_124_SSR_error, linestyle = "-", color = "b", marker = "s", label="1.24 keV, Cygnus-X1 model")
    # plt.errorbar(number_photons, Exo_124_SSR, Exo_124_SSR_error, linestyle = "-", color = "orange", marker = "s", label="1.24 keV, Star with exoplanet model")
    # plt.errorbar(number_photons, SgA_124_SSR, SgA_124_SSR_error, linestyle = "-", color = "r", marker = "s", label="1.24 keV, Sagetarius A* model")

    # plt.errorbar(number_photons, Cyg_36_SSR, Cyg_36_SSR_error, linestyle = "-", color = "b", marker = "s", label="3.6 keV, Cygnus-X1 model")
    # plt.errorbar(number_photons, Exo_36_SSR, Exo_36_SSR_error, linestyle = "-", color = "orange", marker = "s", label="3.6 keV, Star with exoplanet model")
    # plt.errorbar(number_photons, SgA_36_SSR, SgA_36_SSR_error, linestyle = "-", color = "r", marker = "s", label="3.6 keV, Sagetarius A* model")

    # plt.errorbar(number_photons, Cyg_64_SSR, Cyg_64_SSR_error, linestyle = "-", color = "b", marker = "s", label="6.4 keV, Cygnus-X1 model")
    # plt.errorbar(number_photons, Exo_64_SSR, Exo_64_SSR_error, linestyle = "-", color = "orange", marker = "s", label="6.4 keV, Star with exoplanet model")
    # plt.errorbar(number_photons, SgA_64_SSR, SgA_64_SSR_error, linestyle = "-", color = "r", marker = "s", label="6.4 keV, Sagetarius A* model")
    
    plt.errorbar(number_photons, Cyg_64_SSR_new, Cyg_64_SSR_error_new, linestyle = "-", marker = "s", label="6.4 keV, Cygnus X-1 model")
    
    # plt.title('IFFT vs eIFT timing')
    plt.xlabel('Number of source photons')
    plt.ylabel('SSR')
    # plt.yscale("log")
    plt.xscale("log")
    # plt.xlim(0, 800)
    plt.ylim(0, 60000)
    # plt.legend()

    plt.savefig('scaling_relation.pdf')
    plt.show()



    pass