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
from astropy.cosmology import FlatLambdaCDM

z = 0.007214
ngc1266_vel = (const.c.to(u.km/u.s)).value*z
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

def load_maps(path, ncomp=3, nflux=2):
	maps_fl = fits.open(path)

	maps_header = maps_fl[0].header
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

	return vels, sigs, fluxes


def calc_line_ratios():

	Ha_vels, Ha_sigs, HaNii_fluxes = load_maps('/Users/jotter/highres_PSBs/ngc1266_MUSE/output/fitsimages/NGC1266_maps_run4_sortmid2.fits', nflux=2)
	Hb_vels, Hb_sigs, Hb_fluxes = load_maps('/Users/jotter/highres_PSBs/ngc1266_MUSE/output/fitsimages/NGC1266_maps_Hb_contsub_run1_sortmid.fits', nflux=1)
	Oiii_vels, Oiii_sigs, Oiii_fluxes = load_maps('/Users/jotter/highres_PSBs/ngc1266_MUSE/output/fitsimages/NGC1266_maps_Oiii_contsub_run1_sortmid.fits', nflux=1)
	Sii_vels, Sii_sigs, Sii_fluxes = load_maps('/Users/jotter/highres_PSBs/ngc1266_MUSE/output/fitsimages/NGC1266_maps_Sii_contsub_run1_sortmid.fits', nflux=2)
	Oi_vels, Oi_sigs, Oi_fluxes = load_maps('/Users/jotter/highres_PSBs/ngc1266_MUSE/output/fitsimages/NGC1266_maps_Oi_contsub_run1_sortmid.fits', nflux=1)
	
	cube_fl = fits.open('/Users/jotter/highres_PSBs/ngc1266_data/MUSE/ADP.2019-02-25T15 20 26.375.fits')
	cube_wcs = WCS(cube_fl[1].header).celestial
	cube_fl.close()

	Oiii_Hb_ratio = np.log10(Oiii_fluxes[0] / Hb_fluxes[0])
	Nii_Ha_ratio = np.log10(HaNii_fluxes[1] / HaNii_fluxes[0])
	Sii_Ha_ratio = np.log10(Sii_fluxes[0] / HaNii_fluxes[0])
	Oi_Ha_ratio = np.log10(Oiii_fluxes[0] / HaNii_fluxes[0])

	Sii_Sii_ratio = np.log10(Sii_fluxes[0] / Sii_fluxes[1])

	Kewley_Nii = 0.61 / (Nii_Ha_ratio - 0.47) + 1.19
	Kewley_Sii = 0.72 / (Sii_Ha_ratio - 0.32) + 1.30
	Kewley_Oi = 0.73 / (Oi_Ha_ratio + 0.59) + 1.33
	Kauffmann_Nii = 0.61 / (Nii_Ha_ratio - 0.05) + 1.3

	#plot_flux_ratios([Oiii_Hb_ratio, Nii_Ha_ratio, Sii_Ha_ratio, Oi_Ha_ratio], ['OIII/Hbeta', 'NII/Halpha', 'SII/Halpha', 'OI/Halpha'], cube_wcs, 'n1266_bpt_ratios')

	Nii_strong_AGN = np.where(Kewley_Nii > Oiii_Hb_ratio, 1., 0.)
	Sii_AGN = np.where(Kewley_Sii > Oiii_Hb_ratio, 1.0, 0.)
	Oi_AGN = np.where(Kewley_Oi > Oiii_Hb_ratio, 1., 0.)
	Nii_weak_AGN = np.logical_and(np.where(Kauffmann_Nii > Oiii_Hb_ratio, 1, 0), np.logical_not(Nii_strong_AGN))
	Nii_SF = np.where(Kauffmann_Nii <= Oiii_Hb_ratio, 1., 0.)
	Sii_SF = np.where(Kewley_Sii <= Oiii_Hb_ratio, 1., 0.)
	Oi_SF = np.where(Kewley_Oi <= Oiii_Hb_ratio, 1., 0.)


	#Alatalo15 shock boundaries
	A15_Oiii_1 = np.where(Oiii_Hb_ratio < 1.03, 1., 0.)
	A15_Oiii_2 = np.where(Oiii_Hb_ratio > -0.81, 1., 0.)

	A15_Nii_1 = np.where(Nii_Ha_ratio < 0.42, 1., 0.)
	A15_Nii_2 = np.where(Nii_Ha_ratio > -0.75, 1., 0.)
	A15_Nii_3 = np.where((0.4/(Nii_Ha_ratio + 0.15)) + Nii_Ha_ratio + 1.5 < Oiii_Hb_ratio, 1., 0.)
	A15_Nii_4 = np.where((0.65 * Nii_Ha_ratio) - 0.62 < Oiii_Hb_ratio, 1., 0.)
	A15_Nii_5 = np.where(1.12 * Nii_Ha_ratio + 1.14 > Oiii_Hb_ratio, 1., 0.)

	A15_Sii_1 = np.where(Sii_Ha_ratio < 0.44, 1., 0.)
	A15_Sii_2 = np.where(Sii_Ha_ratio > -0.81, 1., 0.)
	A15_Sii_3 = np.where((1.05/(Sii_Ha_ratio - 1.00)) + 0.5*Sii_Ha_ratio + 0.74 < Oiii_Hb_ratio, 1., 0.)

	A15_Oi_1 = np.where(Oi_Ha_ratio < 0.34, 1., 0.)
	A15_Oi_2 = np.where(Oi_Ha_ratio > -2.06, 1., 0.)
	A15_Oi_3 = np.where((1.15/(Oi_Ha_ratio - 0.95)) -0.15*Oi_Ha_ratio + 0.30 < Oiii_Hb_ratio, 1., 0.)

	Nii_shock = np.sum((A15_Oiii_1, A15_Oiii_2, A15_Nii_1, A15_Nii_2, A15_Nii_3, A15_Nii_4, A15_Nii_5), axis=0)
	Nii_shock_mask = np.where(Nii_shock == 7, 1., 0.)
	Sii_shock = np.sum((A15_Oiii_1, A15_Oiii_2, A15_Sii_1, A15_Sii_2, A15_Sii_3), axis=0)
	Sii_shock_mask = np.where(Sii_shock == 5, 1., 0.)
	Oi_shock = np.sum((A15_Oiii_1, A15_Oiii_2, A15_Oi_1, A15_Oi_2, A15_Oi_3), axis=0)
	Oi_shock_mask = np.where(Oi_shock == 5, 1., 0.)

	flux_map = HaNii_fluxes[1]
	Nii_bpt_masks = [Nii_SF, Nii_weak_AGN, Nii_strong_AGN, Nii_shock_mask]
	Sii_bpt_masks = [Sii_SF, Sii_AGN, Sii_shock_mask]
	Oi_bpt_masks = [Oi_SF, Oi_AGN, Oi_shock_mask]
	
	#plot_bpt_maps(flux_map, Nii_bpt_masks, Sii_bpt_masks, Oi_bpt_masks, cube_wcs)

	plot_bpt_diagrams(Oiii_Hb_ratio, Nii_Ha_ratio, Sii_Ha_ratio, Oi_Ha_ratio)



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

	plt.savefig(f'/Users/jotter/highres_PSBs/ngc1266_MUSE/plots/MUSE_velmaps_{plotname}.pdf', dpi=500, bbox_inches='tight')


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

	plt.savefig(f'/Users/jotter/highres_PSBs/ngc1266_MUSE/plots/MUSE_sigmamaps_{plotname}.pdf', dpi=500, bbox_inches='tight')


def plot_flux_ratios(ratio_maps, map_names, wcs, plotname):

	map1, map2, map3, map4 = ratio_maps[0], ratio_maps[1], ratio_maps[2], ratio_maps[3]
	fig = plt.figure(figsize=(32,8))
	gs = GridSpec(1,4, wspace=0.15)

	ax0 = fig.add_subplot(gs[0,0], projection=wcs)

	col = ax0.imshow(map1, cmap='viridis', origin='lower')
	cb0 = fig.colorbar(col, label='km/s', ax=ax0)

	cb0.ax.tick_params(axis='y', labelsize=16)
	cb0.ax.set_ylabel('km/s', fontsize=16)

	ax0.set_title(map_names[0], fontsize=20)
	ax0.set_ylabel('Dec.', fontsize=20)
	ax0.set_xlabel('RA', fontsize=20)
	ax0.tick_params(axis='both', labelsize=16)

	ax1 = fig.add_subplot(gs[0,1], projection=wcs)

	col1 = ax1.imshow(map2, cmap='viridis', origin='lower')

	ax1.set_title(map_names[1], fontsize=20)
	ax1.set_ylabel(' ', fontsize=14)
	ax1.set_xlabel('RA', fontsize=20)

	ax1.tick_params(axis='y', labelleft=False)
	ax1.tick_params(axis='x', labelsize=16)

	ax2 = fig.add_subplot(gs[0,2], projection=wcs)
	col2 = ax2.imshow(map3, cmap='viridis', origin='lower')

	cb1 = fig.colorbar(col1, ax=ax1)
	cb2 = fig.colorbar(col2, ax=ax2)

	cb1.ax.tick_params(axis='y', labelsize=16)
	cb1.ax.set_ylabel('', fontsize=16)

	cb2.ax.tick_params(axis='y', labelsize=16)
	cb2.ax.set_ylabel('', fontsize=16)

	ax2.set_title(map_names[2], fontsize=20)
	ax2.set_ylabel(' ', fontsize=14)
	ax2.set_xlabel('RA', fontsize=20)
	ax2.tick_params(axis='y', labelleft=False)
	ax2.tick_params(axis='x', labelsize=16)

	ax3 = fig.add_subplot(gs[0,3], projection=wcs)
	col3 = ax3.imshow(map4, cmap='viridis', origin='lower')

	cb3 = fig.colorbar(col3, ax=ax3)

	cb3.ax.tick_params(axis='y', labelsize=16)
	cb3.ax.set_ylabel('', fontsize=16)

	cb3.ax.tick_params(axis='y', labelsize=16)
	cb3.ax.set_ylabel('', fontsize=16)

	ax3.set_title(map_names[3], fontsize=20)
	ax3.set_ylabel(' ', fontsize=14)
	ax3.set_xlabel('RA', fontsize=20)
	ax3.tick_params(axis='y', labelleft=False)
	ax3.tick_params(axis='x', labelsize=16)

	plt.savefig(f'/Users/jotter/highres_PSBs/ngc1266_MUSE/plots/MUSE_ratiomaps_{plotname}.pdf', dpi=500, bbox_inches='tight')

	plt.close()


def plot_bpt_maps(flux_map, Nii_bpt_masks, Sii_bpt_masks, Oi_bpt_masks, wcs, plotname):

	full_map = np.full((100,100), 1)

	fig = plt.figure(figsize=(26,8))
	gs = GridSpec(1,3, wspace=0.15)

	ax0 = fig.add_subplot(gs[0,0], projection=wcs)

	ax0.imshow(flux_map, cmap='Greys', origin='lower')
	ax0.imshow(full_map, cmap='Blues', vmin=0, vmax=2, alpha=Nii_bpt_masks[0]*0.5) #star-forming
	ax0.imshow(full_map, cmap='Greens', vmin=0, vmax=2, alpha=Nii_bpt_masks[1]*0.5) #weak AGN
	ax0.imshow(full_map, cmap='Reds', vmin=0, vmax=2, alpha=Nii_bpt_masks[2]*0.5) #strong AGN
	ax0.imshow(full_map, cmap='spring', vmin=1, vmax=2, alpha=Nii_bpt_masks[3]*0.7) #shock

	ax0.set_title(r'[NII]/H$\alpha$ BPT Classification', fontsize=20)
	ax0.set_ylabel('Dec.', fontsize=20)
	ax0.set_xlabel('RA', fontsize=20)
	ax0.tick_params(axis='both', labelsize=16)

	ax1 = fig.add_subplot(gs[0,1], projection=wcs)

	ax1.imshow(flux_map, cmap='Greys', origin='lower')
	ax1.imshow(full_map, cmap='Blues', vmin=0, vmax=2, alpha=Sii_bpt_masks[0]*0.5) #star-forming
	ax1.imshow(full_map, cmap='Reds', vmin=0, vmax=2, alpha=Sii_bpt_masks[1]*0.5) #AGN
	ax1.imshow(full_map, cmap='spring', vmin=1, vmax=2, alpha=Sii_bpt_masks[2]*0.7) #shock

	ax1.set_title(r'[SII]/H$\alpha$ BPT Classification', fontsize=20)
	ax1.set_ylabel(' ', fontsize=14)
	ax1.set_xlabel('RA', fontsize=20)

	ax1.tick_params(axis='y', labelleft=False)
	ax1.tick_params(axis='x', labelsize=16)

	ax2 = fig.add_subplot(gs[0,2], projection=wcs)

	ax2.imshow(flux_map, cmap='Greys', origin='lower')
	ax2.imshow(full_map, cmap='Blues', vmin=0, vmax=2, alpha=Oi_bpt_masks[0]*0.5) #star-forming
	ax2.imshow(full_map, cmap='Reds', vmin=0, vmax=2, alpha=Oi_bpt_masks[1]*0.5) #AGN
	ax2.imshow(full_map, cmap='spring', vmin=1, vmax=2, alpha=Oi_bpt_masks[2]*0.7) #shock

	ax2.set_title(r'[OI]/H$\alpha$ BPT Classification', fontsize=20)
	ax2.set_ylabel(' ', fontsize=14)
	ax2.set_xlabel('RA', fontsize=20)

	ax2.tick_params(axis='y', labelleft=False)
	ax2.tick_params(axis='x', labelsize=16)

	plt.savefig(f'/Users/jotter/highres_PSBs/ngc1266_MUSE/plots/MUSE_ratiomaps_{plotname}.pdf', dpi=500, bbox_inches='tight')

	plt.close()


def plot_bpt_diagrams(Oiii_Hb_ratio, Nii_Ha_ratio, Sii_Ha_ratio, Oi_Ha_ratio):

	fig = plt.figure(figsize=(26,8))
	gs = GridSpec(1,3, wspace=0.1)

	ax0 = fig.add_subplot(gs[0,0])

	ax0.plot(Nii_Ha_ratio, Oiii_Hb_ratio,linestyle='', marker='.', color='tab:blue')

	Nii_Ha_vals1 = np.linspace(-2,0.469,100)
	Nii_Ha_vals2 = np.linspace(-2,0.049,100)
	Kewley_Nii = 0.61 / (Nii_Ha_vals1 - 0.47) + 1.19
	Kauffmann_Nii = 0.61 / (Nii_Ha_vals2 - 0.05) + 1.3

	ax0.plot(Nii_Ha_vals1, Kewley_Nii, marker=None, linestyle='dashed', color='k')
	ax0.plot(Nii_Ha_vals2, Kauffmann_Nii, marker=None, linestyle='dotted', color='k')

	ax0.set_ylabel(r'log [OIII]/H$\beta$', fontsize=20)
	ax0.set_xlabel(r'log [NII]/H$\alpha$', fontsize=20)
	ax0.tick_params(axis='both', labelsize=16)

	ax0.set_ylim(-1, 1.5)
	ax0.set_xlim(-1.5, 1)

	#plotting Alatalo15 shock region
	Nii_vals1 = np.linspace(-0.75, -0.1, 50)
	Nii_vals2 = np.linspace(-0.1, 0.42, 50)
	Nii_vals3 = np.linspace(-0.75, -0.35, 50)
	Nii_vals4 = np.linspace(-0.35, 0.42, 50)
	Nii_vals5 = np.linspace(0.09, 0.3, 10)
	Nii_vals6 = np.linspace(-0.35, 1.03, 10)

	K15_Nii1 = 1.12 * Nii_vals1 + 1.14
	K15_Nii2 = 1.03 + Nii_vals2*0
	K15_Nii3 = 0.4 / (Nii_vals3 + 0.15) + Nii_vals3 + 1.5
	K15_Nii4 = 0.65 * Nii_vals4 - 0.62
	K15_Nii5 = Nii_vals5*0 - 0.75
	K15_Nii6 = Nii_vals6*0 + 0.42

	plt.plot(Nii_vals1, K15_Nii1, linestyle='-', marker=None, color='k')
	plt.plot(Nii_vals2, K15_Nii2, linestyle='-', marker=None, color='k')
	plt.plot(Nii_vals3, K15_Nii3, linestyle='-', marker=None, color='k')
	plt.plot(Nii_vals4, K15_Nii4, linestyle='-', marker=None, color='k')
	plt.plot(K15_Nii5, Nii_vals5, linestyle='-', marker=None, color='k')
	plt.plot(K15_Nii6, Nii_vals6, linestyle='-', marker=None, color='k')

	ax1 = fig.add_subplot(gs[0,1], sharey=ax0)

	ax1.plot(Sii_Ha_ratio, Oiii_Hb_ratio, linestyle='', marker='.', color='tab:blue')

	Sii_Ha_vals = np.linspace(-2,0.319,100)
	Kewley_Sii = 0.72 / (Sii_Ha_vals - 0.32) + 1.30
	
	ax1.plot(Sii_Ha_vals, Kewley_Sii, marker=None, linestyle='dashed', color='k')

	ax1.set_ylabel(' ', fontsize=14)
	ax1.set_xlabel(r'log [SII]/H$\alpha$', fontsize=20)

	ax1.tick_params(axis='y', labelleft=False)
	ax1.tick_params(axis='x', labelsize=16)

	ax1.set_ylim(-1, 1.5)
	ax1.set_xlim(-1, 0.8)

	#Alatalo15 boundary
	Sii_vals1 = np.linspace(-0.81, 0.44, 50)
	Sii_vals2 = np.linspace(-0.81, 0.44, 50)
	Sii_vals3 = np.linspace(-0.24, 1.03, 50)
	Sii_vals4 = np.linspace(-0.9, 1.03, 50)

	K15_Sii1 = 0*Sii_vals1 + 1.03
	K15_Sii2 = 1.05/(Sii_vals2 - 1) + 0.5*Sii_vals2 + 0.74
	K15_Sii3 = 0*Sii_vals3 - 0.81
	K15_Sii4 = 0*Sii_vals4 + 0.44

	plt.plot(Sii_vals1, K15_Sii1, linestyle='-', marker=None, color='k')
	plt.plot(Sii_vals2, K15_Sii2, linestyle='-', marker=None, color='k')
	plt.plot(K15_Sii3, Sii_vals3, linestyle='-', marker=None, color='k')
	plt.plot(K15_Sii4, Sii_vals4, linestyle='-', marker=None, color='k')

	ax2 = fig.add_subplot(gs[0,2], sharey=ax0)

	ax2.plot(Oi_Ha_ratio, Oiii_Hb_ratio, linestyle='', marker='.', color='tab:blue')

	Oi_Ha_vals = np.linspace(-2.5,-0.591,100)
	Kewley_Oi = 0.73 / (Oi_Ha_vals + 0.59) + 1.33
	
	ax2.plot(Oi_Ha_vals, Kewley_Oi, marker=None, linestyle='dashed', color='k')

	ax2.set_ylabel(' ', fontsize=14)
	ax2.set_xlabel(r'log [OI]/H$\alpha$', fontsize=20)

	ax2.tick_params(axis='y', labelleft=False)
	ax2.tick_params(axis='x', labelsize=16)

	ax2.set_ylim(-1, 1.5)
	ax2.set_xlim(-2.5, 1)

	Oi_vals1 = np.linspace(-2.06, 0.34, 50)
	Oi_vals2 = np.linspace(-2.06, -0.1, 50)
	Oi_vals3 = np.linspace(-0.1, 0.34, 50)
	Oi_vals4 = np.linspace(0.2, 1.03, 50)
	Oi_vals5 = np.linspace(-0.81, 1.03, 50)

	K15_Oi1 = 0*Oi_vals1 + 1.03
	K15_Oi2 = 1.15 / (Oi_vals2 - 0.95) - 0.15*Oi_vals2 + 0.30
	K15_Oi3 = 0*Oi_vals3 - 0.81
	K15_Oi4 = 0*Oi_vals4 - 2.06
	K15_Oi5 = 0*Oi_vals5 + 0.34

	plt.plot(Oi_vals1, K15_Oi1, linestyle='-', marker=None, color='k')
	plt.plot(Oi_vals2, K15_Oi2, linestyle='-', marker=None, color='k')
	plt.plot(Oi_vals3, K15_Oi3, linestyle='-', marker=None, color='k')
	plt.plot(K15_Oi4, Oi_vals4, linestyle='-', marker=None, color='k')
	plt.plot(K15_Oi5, Oi_vals5, linestyle='-', marker=None, color='k')

	plt.savefig(f'/Users/jotter/highres_PSBs/ngc1266_MUSE/plots/MUSE_bpt.pdf', dpi=500, bbox_inches='tight')

	plt.close()



	Kewley_Sii = 0.72 / (Sii_Ha_ratio - 0.32) + 1.30
	Kewley_Oi = 0.73 / (Oi_Ha_ratio + 0.59) + 1.33
	

calc_line_ratios()






