import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

img = np.array(Image.open(r'C:\Users\nielz\Documents\Uni\Master\Thesis\Simulator\vri\models\crux.png').convert('L'))

# plt.imshow(img, cmap=cm.Greys)
# plt.show()

img_fft = np.fft.fftshift(np.fft.fft2(img))

# plt.imshow(np.abs(img_fft), cmap=cm.Greens)
# plt.show()

img_shape = (np.array(img_fft.shape))
mask = np.zeros(img_shape)
masked_coords_x = np.array([[y] for x in np.linspace(0, .000000000001 * np.pi, 800) for y in np.linspace(-80, 80, 800)]) + img_shape[0]/2
masked_coords_y = np.array([[y]  for x in np.linspace(0, .000000000001 * np.pi, 800) for y in np.linspace(-80, 80, 800)]) + img_shape[1]/2
mask[masked_coords_x.astype(int), masked_coords_y.astype(int)] = 1
img_fft_masked = img_fft * mask

plt.imshow(mask, cmap=cm.Greys)
plt.show()

plt.imshow(np.abs(img_fft_masked), cmap=cm.Greens)
plt.show()

masked_image = np.fft.ifft2(np.fft.ifftshift(img_fft_masked))
plt.imshow(np.abs(masked_image), cmap=cm.Greys)
plt.show()

# plt.imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft))), cmap=cm.Greys)
# plt.show()