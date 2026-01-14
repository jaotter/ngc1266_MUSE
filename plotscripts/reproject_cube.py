from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, utils
from reproject import reproject_interp, reproject_exact

import astropy.units as u
import numpy as np


cube_fl = fits.open('/Users/jotter/highres_psbs/ngc1266_data/MUSE/ADP.2019-02-25T15 20 26.375.fits')
cube_header = cube_fl[1].header
cube_orig = cube_fl[1].data
cube_wcs = WCS(cube_header)

hdu0 = cube_fl[0]

cube_fl.close()

#cube fitting range
y_i, y_f = 105,205
x_i, x_f = 105,205

cube_fit = cube_orig[:, y_i:y_f, x_i:x_f]
cube_fit_wcs = cube_wcs[:, y_i:y_f, x_i:x_f]

pix_scales = utils.proj_plane_pixel_scales(cube_fit_wcs.celestial)

cube_fit_sb = cube_fit / (pix_scales[0]*u.degree * pix_scales[1]*u.degree)

reproj_factor = 4

#print(cube_wcs)
#print(cube_header)

#reproj_header = cube_header.copy(strip=True)
reproj_header = cube_fit_wcs[:, ::reproj_factor, ::reproj_factor].to_header()

#print(WCS(reproj_header))

#reproj_header['CD1_1'] = cube_header['CD1_1'] * reproj_factor
#reproj_header['CD2_2'] = cube_header['CD2_2'] * reproj_factor
reproj_header['NAXIS'] = 3
reproj_header['NAXIS1'] = int((y_f - y_i) / round(reproj_factor))
reproj_header['NAXIS2'] = int((x_f - x_i) / round(reproj_factor))
reproj_header['NAXIS3'] = cube_header['NAXIS3']

#reproj_wcs = WCS(reproj_header)

#print(reproj_header)
#reproj_header[]

reproj_cube_sb, ftprnt = reproject_interp((cube_fit_sb, cube_fit_wcs), reproj_header)
#reproj_cube = cube_fit

new_pix_scales = utils.proj_plane_pixel_scales(WCS(reproj_header).celestial)
reproj_cube = reproj_cube_sb * (new_pix_scales[0] * u.degree * new_pix_scales[1] * u.degree)

new_header = cube_header.copy()

for card in reproj_header:
	new_header[card] = reproj_header[card]

new_hdu = fits.PrimaryHDU(data=reproj_cube.value, header=new_header)

new_hdu.writeto(f'/Users/jotter/highres_psbs/ngc1266_data/MUSE/MUSE_cube_center_reproj{reproj_factor}.fits', overwrite=True)



