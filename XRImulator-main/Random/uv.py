import matplotlib.pyplot as plt
import numpy as np

zero = np.array([0, 0])
freq = np.array([1, 1/2])
zerotofreq = np.array([[0,1], [0, 1/2]])

meet_point = [3/4, 3/4]
arc_rad = np.linspace(0, np.pi/4, 100)
arc_coord = np.stack((np.array([.5*np.cos(i) for i in arc_rad]), np.array([.5*np.sin(i) for i in arc_rad])), axis=-1)

fig = plt.figure(figsize=(6,6))
plt.plot(zerotofreq[0,:], zerotofreq[1,:], 'r')
plt.plot(freq[0], freq[1], 'ro')
plt.plot([0, freq[0]], [0, 0], 'g')
plt.plot([0, 0], [0, freq[1]], 'g')
plt.plot([freq[0], freq[0]], [0, freq[1]], 'k--')
plt.plot([0, freq[0]], [freq[1], freq[1]], 'k--')
plt.plot(freq[0], 0, 'go')
plt.plot(0, freq[1], 'go')
plt.xlabel('$u$ $(m^-1)$ ')
plt.ylabel('$v$ $(m^-1)$ ')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.show()


angles = np.linspace(0, 2*np.pi, 64)
coords = np.array([[r * np.cos(angles), r * np.sin(angles)] for r in [5, 15, 30, 50]])
fig = plt.figure(figsize=(6,6))
for i, coord in enumerate(coords):
    plt.plot(coord[0,:], coord[1,:], '.', label=f'Baseline {i + 1}')

plt.legend()
plt.xlabel('$u$ $(m^-1)$ ')
plt.ylabel('$v$ $(m^-1)$ ')
plt.xlim(-75, 75)
plt.ylim(-75, 75)
plt.grid(True)
plt.show()