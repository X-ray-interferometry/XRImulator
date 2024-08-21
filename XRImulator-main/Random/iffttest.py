from scipy.fft import ifft2, fft2, fftfreq, fftshift
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

N = 1000

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')

x = np.zeros((N,N))
x[N//2 - 1:N//2 + 1, N//2 - 1 :N//2 + 1] = 1
# x[N//2 - 1 + 5:N//2 + 1 + 5, N//2 - 1 :N//2 + 1] = 1
# x[N//2 - 1 - 5:N//2 + 1 - 5, N//2 - 1 :N//2 + 1] = 1
ax1.imshow(x, cmap=cm.Greens)

def masker(radius):
    mask = np.zeros((N,N))
    for theta in np.arange(0, 2 * np.pi, .01 * np.pi):
        x, y = np.array(np.cos(theta) * radius + N/2, dtype=np.int_), np.array(np.sin(theta) * radius + N/2, dtype=np.int_)
        mask[x,y] = 1
    return mask

def weird_masker(radius):
    mask = np.zeros((N,N))
    for theta in np.arange(0, 2 * np.pi, .01 * np.pi):
        x, y = np.array(np.cos(theta) * radius, dtype=np.int_), np.array(np.sin(theta) * radius, dtype=np.int_)
        mask[x,y] = 1
    return mask


xf = fft2(x)
x_freq = fftfreq(x[:, 0].size, 1)
y_freq = fftfreq(x[:, 1].size, 1)
# freq = np.zeros((N,N))
# for i, x_val in enumerate(x_freq):
#     for j, y_val in enumerate(y_freq):
#         freq[i, j] = np.random.random()
mask = weird_masker(np.array([1, 2, 5, 10, 15, 20, 25, 30, 40, 49]))

# ax4.imshow(freq, cmap=cm.Greens)
ax2.imshow(abs(xf * mask), cmap=cm.Oranges)
ax5.imshow(abs(xf), cmap=cm.Oranges)

Z = ifft2(mask)
Z_mask = ifft2(xf * mask)
ax3.imshow(abs(Z_mask), cmap=cm.Reds)
ax6.imshow(abs(fftshift(Z)), cmap=cm.Reds)

# x2 = np.zeros((N,N))
# x2[1:5, :] = 1
# # x2[12, (12, 18)] = 1
# # x2[15:17, :] = 1
# # ax4.imshow(x2, cmap=cm.Greens)

# xf2 = fftn(x2)
# # print(xf2)

# Z2 = ifftn(xf2)
# ax6.imshow(abs(Z2), cmap=cm.Reds)

# xf = np.zeros((N,N))

# for theta in np.arange(0, 2 * np.pi, .01 * np.pi):
#     x, y = np.cos(theta) * 5 + 15, np.sin(theta) * 5 + 15
#     xf[int(x),int(y)] = 1

# Z = ifftn(xf)
# ax1.imshow(xf, cmap=cm.Reds)
# ax4.imshow(np.real(Z), cmap=cm.gray)

# xf = np.zeros((N, N))
# xf[5, 0] = 1
# xf[N-5, 0] = 1
# Z = ifftn(xf)
# ax2.imshow(xf, cmap=cm.Reds)
# ax5.imshow(np.real(Z), cmap=cm.gray)

# xf = np.zeros((N, N))
# xf[5, 10] = 1
# xf[N-5, N-10] = 1
# Z = ifftn(xf)
# ax3.imshow(xf, cmap=cm.Reds)
# ax6.imshow(np.real(Z), cmap=cm.gray)

plt.show()