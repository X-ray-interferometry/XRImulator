import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2, 1000)
y1 = np.sin(x * 10 * 2 * np.pi)
y2 = np.sin(x * 2  * 2 * np.pi)
y3 = np.random.random(x.size) * 20 - 10
y4 = y1 + y2 + y3

fig = plt.figure(figsize=(12,6))
sfigs = fig.subfigures(1, 2)
gs = sfigs[0].add_gridspec(4, hspace=0)
axs = gs.subplots(sharex=True)
sfigs[0].suptitle('Signal construction')

axs[0].plot(x, y1, '.', label='Freq=10')
axs[1].plot(x, y2, 'r.', label='Freq=2')
axs[2].plot(x, y3, 'g.', label='Noise')
axs[3].plot(x, y4, 'k.', label='Combination')
axs[3].set_xlabel('x')


# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()
    ax.set_ylim(-12, 12)
    ax.legend()
    ax.set_ylabel('y')

gs2 = sfigs[1].add_gridspec(1)
ax2 = gs2.subplots()
sfigs[1].suptitle('Fourier transformation')

fourier = np.fft.fft(y4) / y4.size
freqs = np.fft.fftfreq(y4.size, x[1])

ax2.plot(freqs[2:freqs.size//2], abs(fourier[2:freqs.size//2]), label='Fourier')
ax2.set_xlim(0, 15)
ax2.set_ylim(0, 1)
ax2.axvline(2, ymin=-5, ymax=5, color='r', linestyle='--', label='Freq=2')
ax2.axvline(10, ymin=-5, ymax=5, linestyle='--', label='Freq=10')
ax2.set_xlabel('Frequency (1/x)')
ax2.set_ylabel('Amplitude')
ax2.legend()

plt.show()