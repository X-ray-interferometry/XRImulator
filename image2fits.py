

import sys
import numpy as np
from PIL import Image
from scipy import constants as sc
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

def convert_image_to_fits(
    image_path,
    output_fits,
    ra_center=0.0,          # degrees
    dec_center=0.0,         # degrees
    fov_mas=(1., 1.)        # total field of view (mas)
):
    # Load image and convert to grayscale array
    img = Image.open(image_path).convert("L")
    data = np.array(img)
    mask = data > 0         # Mask to ignore zero pixels if needed
    data[mask] -= 1
    data = np.flipud(data)  # Flip the image vertically, optional

    ny, nx = data.shape

    # Compute pixel scale (mas/pixel)
    pixscale_x = fov_mas[0] / nx
    pixscale_y = fov_mas[1] / ny

    # Create a WCS header
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2, ny / 2]                                          # reference pixel at image center
    w.wcs.cdelt = np.array([-pixscale_x, pixscale_y]) * sc.arcsec / 1e3     # negative CDELT1: East left
    w.wcs.crval = [ra_center, dec_center]                                   # RA, Dec of center
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Create FITS header
    header = w.to_header()
    header["BUNIT"] = "adu"                                                 # or leave as is
    header["COMMENT"] = "Generated from image {}".format(image_path)

    # Create FITS HDU
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(output_fits, overwrite=True)

    print(f"✅ FITS file written to: {output_fits}")
    print(f"Image shape: {data.shape}, pixel scale: {(pixscale_x):.6f} mas/pix")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python image2fits.py input.jpg output.fits [RA] [Dec] [FOV_mas]")
        sys.exit(1)

    img_path = sys.argv[1]
    fits_path = sys.argv[2]
    ra = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0
    dec = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    if len(sys.argv) > 5:
        fov = tuple(map(float, sys.argv[5].split(',')))
        if len(fov) != 2:
            print("FOV must be two comma-separated floats, e.g. 1.0,1.0")
            sys.exit(1)
    else:
        fov = (1.0, 1.0)

    convert_image_to_fits(img_path, fits_path, ra_center=ra, dec_center=dec, fov_mas=fov)
