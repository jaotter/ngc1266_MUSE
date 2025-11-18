import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
from matplotlib.gridspec import GridSpec

from astropy.io import fits
from astropy.table import Table

from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.patches import Rectangle, Ellipse
from spectral_cube import SpectralCube
from radio_beam import Beam

z = 0.007214
ngc1266_vel = (const.c.to(u.km/u.s)).value*z
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

def load_maps(path, ncomp=3, nflux=2):
	maps_fl = fits.open('/Users/jotter/highres_PSBs/ngc1266_MUSE/output/fitsimages/NGC1266_maps_run4_sortmid2.fits')

	maps_header = maps_fl[0].header
	maps_wcs = WCS(maps_fl[0].header).celestial
	maps_data = maps_fl[0].data
	maps_fl.close()

	num_map = maps_data[0,:,:]

	if ncomp == 3:
		vel_c1 = maps_data[2,:,:]
		vel_c2 = maps_data[3,:,:]
		vel_c3 = maps_data[4,:,:]
		vels = [vel_c1, vel_c2, vel_c3]

		sig_c1 = maps_data[5,:,:]
		sig_c2 = maps_data[6,:,:]
		sig_c3 = maps_data[7,:,:]
		sigs = [sig_c1, sig_c2, sig_c3]

		if nflux == 1:
			flux_c1 = maps_data[8,:,:]
			flux_c2 = maps_data[9,:,:]
			flux_c3 = maps_data[10,:,:]

			flux_tot = np.nansum([flux_c1, flux_c2, flux_c3], axis=0)

			fluxes = [flux_tot]

		if nflux == 2:
			fluxA_c1 = maps_data[8,:,:]
			fluxA_c2 = maps_data[10,:,:]
			fluxA_c3 = maps_data[12,:,:]

			fluxB_c1 = maps_data[9,:,:]
			fluxB_c2 = maps_data[11,:,:]
			fluxB_c3 = maps_data[13,:,:]

			fluxA_tot = np.nansum([fluxA_c1, fluxA_c2, fluxA_c3], axis=0)
			fluxB_tot = np.nansum([fluxB_c1, fluxB_c2, fluxB_c3], axis=0)

			fluxes = [fluxA_tot, fluxB_tot]

	return vels, sigs, fluxes, wcs


def plot_vel_3comp(vels, wcs, plotname):

	vel_c1, vel_c2, vel_c3 = vels[0], vels[1], vels[2]

	fig = plt.figure(figsize=(26,8))
	gs = GridSpec(1,3, wspace=0.15)

	ax0 = fig.add_subplot(gs[0,0], projection=wcs)

	col = ax0.imshow(vel_c1, cmap='RdBu_r', vmin=-600, vmax=600, origin='lower')
	cb0 = fig.colorbar(col, label='km/s', ax=ax0)

	cb0.ax.tick_params(axis='y', labelsize=16)
	cb0.ax.set_ylabel('km/s', fontsize=16)

	ax0.set_title('Narrow low velocity component', fontsize=20)
	ax0.set_ylabel('Dec.', fontsize=20)
	ax0.set_xlabel('RA', fontsize=20)

	ax0.tick_params(axis='both', labelsize=16)


	ax1 = fig.add_subplot(132, projection=cube_wcs)

	col1 = ax1.imshow(vel_c3, cmap='RdBu_r', vmin=-600, vmax=600, origin='lower')#, transform=ax1.get_transform(cube_wcs))


	ax2 = fig.add_subplot(133, projection=cube_wcs)
	col2 = ax2.imshow(vel_c2, cmap='RdBu_r', vmin=-600, vmax=600, origin='lower')

	cb1 = fig.colorbar(col1, label='km/s', ax=ax1)
	cb2 = fig.colorbar(col2, label='km/s', ax=ax2)

	cb1.ax.tick_params(axis='y', labelsize=16)
	cb1.ax.set_ylabel('km/s', fontsize=16)

	cb2.ax.tick_params(axis='y', labelsize=16)
	cb2.ax.set_ylabel('km/s', fontsize=16)

	ax1.set_title('Narrow high velocity component', fontsize=20)
	ax1.set_ylabel(' ', fontsize=14)
	ax1.set_xlabel('RA', fontsize=20)

	ax1.tick_params(axis='y', labelleft=False)
	ax1.tick_params(axis='x', labelsize=16)

	ax2.set_title('Broad component', fontsize=20)
	ax2.set_ylabel(' ', fontsize=14)
	ax2.set_xlabel('RA', fontsize=20)

	ax2.tick_params(axis='y', labelleft=False)
	ax2.tick_params(axis='x', labelsize=16)

	plt.savefig(f'/Users/jotter/highres_PSBs/plots/MUSE_velmaps_{plotname}.png', dpi=500, bbox_inches='tight')


def plot_sigma_3comp(sigs, wcs, plotname):

	sig_c1, sig_c2, sig_c3 = sigs[0], sigs[1], sigs[2]
	fig = plt.figure(figsize=(26,8))
	gs = GridSpec(1,3, wspace=0.15)

	ax0 = fig.add_subplot(gs[0,0], projection=wcs)

	col = ax0.imshow(sig_c1, cmap='inferno', vmin=0, vmax=500, origin='lower')
	cb0 = fig.colorbar(col, label='km/s', ax=ax0)

	cb0.ax.tick_params(axis='y', labelsize=16)
	cb0.ax.set_ylabel('km/s', fontsize=16)

	ax0.set_title('Narrow low velocity component', fontsize=20)
	ax0.set_ylabel('Dec.', fontsize=20)
	ax0.set_xlabel('RA', fontsize=20)
	ax0.tick_params(axis='both', labelsize=16)

	ax1 = fig.add_subplot(132, projection=cube_wcs)

	col1 = ax1.imshow(sig_c2, cmap='inferno', vmin=0, vmax=500, origin='lower')#, transform=ax1.get_transform(cube_wcs))

	ax1.set_title('Narrow high velocity component', fontsize=20)
	ax1.set_ylabel(' ', fontsize=14)
	ax1.set_xlabel('RA', fontsize=20)

	ax1.tick_params(axis='y', labelleft=False)
	ax1.tick_params(axis='x', labelsize=16)

	ax2 = fig.add_subplot(133, projection=cube_wcs)
	col2 = ax2.imshow(sig_c3, cmap='inferno', vmin=0, vmax=500, origin='lower')

	cb1 = fig.colorbar(col1, label='km/s', ax=ax1)
	cb2 = fig.colorbar(col2, label='km/s', ax=ax2)

	cb1.ax.tick_params(axis='y', labelsize=16)
	cb1.ax.set_ylabel('km/s', fontsize=16)

	cb2.ax.tick_params(axis='y', labelsize=16)
	cb2.ax.set_ylabel('km/s', fontsize=16)

	ax2.set_title('Broad Component', fontsize=20)
	ax2.set_ylabel(' ', fontsize=14)
	ax2.set_xlabel('RA', fontsize=20)
	ax2.tick_params(axis='y', labelleft=False)
	ax2.tick_params(axis='x', labelsize=16)

	plt.savefig(f'/Users/jotter/highres_PSBs/plots/MUSE_sigmamaps_{plotname}.png', dpi=500, bbox_inches='tight')


def flux_maps



