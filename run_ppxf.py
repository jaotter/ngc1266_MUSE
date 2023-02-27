#script to run ppxf on MUSE data, adapted from Celia Mulcahey's interactive notebook

import glob
import datetime
import os
from time import perf_counter as clock
from astropy.io import fits
import numpy as np

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
import ppxf_custom_util as custom_util

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.stats import mad_std
from scipy.stats import kstest, norm
import pandas as pd

#Constants, wavelength in Ang in vacuum------------------------------------------- 

c = 299792.458

Hb = 4862.721
OIII = 5008.239
OI = 6302.046
NII1 = 6549.86
Ha = 6564.614
NII = 6585.27
SII1 = 6718.29
SII2 = 6732.68
NI = 5201.7
NII5756 = 5756.2
OII1 = 7322.0
OII2 = 7332.75


def ANR(gas_dict, gas_name, emline, gal_lam, gas_bestfit, mad_std_residuals, velocity):
#    '''
#    Function caluclates the amplitude and amplitude-to-residual (A/rN) of specific emission-line feature following Sarzi+2006. No returns,
#    the function simply stores the amplitude and A/rN in the dictionary called gas_dict
    
#    Arguments:
#        gas_dict - dictionary where values extracting using pPPXF are stored
#        gas_name - emission-line feature name as it appears in gas_dict 
#        emline - Wavelength in vaccuum of emission-line feature in Angstroms
#        rest_gas - Spectrum, in rest frame
#        gas_bestfit - pPXF gas best fit
#        mad_std_residuals - median absolute deviation of the residuals 
#		 velocity - fitted gas velocity to get center of line

#    '''

	emline_obs = (velocity/c + 1) * emline
	emline_loc = np.where((gal_lam>emline_obs-5)&(gal_lam<emline_obs+5))

	emline_amp = np.nanmax(gas_bestfit[emline_loc])
	emline_ANR = emline_amp/mad_std_residuals 
	if gas_dict is not None:
		gas_dict[gas_name+'_amplitude'].append(emline_amp)
		gas_dict[gas_name+'_ANR'].append(emline_ANR)
	else:
		return emline_ANR


def output_ppxf_fit_plot(plot_name, pp, good_pix_total, logLam, vel_comp0, z, ncomp, bin_number, mad_std_residuals, SN_wMadStandardDev, fit_gas):

	print(f'Saving ppxf plot for bin number {bin_number}')
	plot_dir = 'ppxf_output/plots/'

	fig, ax = plt.subplots(figsize = (9,6), sharey = True)
	ax1 = plt.subplot(212)
	ax1.margins(0.08) 
	ax2 = plt.subplot(221)
	ax2.margins(0.05)
	ax3 = plt.subplot(222)
	ax3.margins(0.05)

	wave_plot_rest = np.exp(logLam)/(z+1)

	wave_lam_rest_ind = np.arange(len(wave_plot_rest))
	masked_ind = np.setdiff1d(wave_lam_rest_ind, good_pix_total)
	mask_reg_upper = []
	mask_reg_lower = []
	for ind in masked_ind:
		if ind+1 not in masked_ind:
			mask_reg_lower.append(wave_plot_rest[ind])
		elif ind-1 not in masked_ind:
			mask_reg_upper.append(wave_plot_rest[ind])

	fig.text(0.05, 0.93, f'Mean abs. dev. of residuals: {np.round(mad_std_residuals,1)}, S/N: {int(np.round(SN_wMadStandardDev,0))}')
	fig.text(0.45, 0.93, f'Chi-Squared/DOF: {pp.chi2}')

	bin_stell_vel = vel_comp0 - z*c
	fig.text(0.7, 0.93, f'Bin stellar velocity: {int(np.round(bin_stell_vel,0))} km/s')

	fig.text(0.03, 0.3, r'Flux (10$^{-20}$ erg/s/cm$^2$/Å)', fontsize = 12, rotation=90)

	for bound_ind in range(len(mask_reg_upper)):
		ax1.axvspan(mask_reg_lower[bound_ind], mask_reg_upper[bound_ind], alpha=0.25, color='gray')
		ax2.axvspan(mask_reg_lower[bound_ind], mask_reg_upper[bound_ind], alpha=0.25, color='gray')
		ax3.axvspan(mask_reg_lower[bound_ind], mask_reg_upper[bound_ind], alpha=0.25, color='gray')

	if fit_gas == True:
		plt.sca(ax1)
		pp.plot()
		xticks = ax1.get_xticks()
		ax1.set_xticks(xticks, labels=np.array(xticks*1e4, dtype='int'))
		ax1.set_xlabel('Restframe Wavelength (Å)',fontsize = 12)
		ax1.set_ylabel('')
		legend_elements = [Line2D([0], [0], color='k', label='Data', lw=2),
							Line2D([0], [0], color='r', label='Stellar continuum', lw=2),
							Line2D([0], [0], color='m', label='Gas emission lines', lw=2),
							Line2D([0], [0], color='b', label='Emission line components', lw=2),
							Line2D([0], [0], marker='d', color='g', label='Residuals', markersize=5, lw=0),
							Line2D([0], [0], color='orange', label='Total PPXF fit', lw=2)]
		ax1.legend(handles=legend_elements, loc='upper left', fontsize='small', ncol=2)


		plt.sca(ax2)
		pp.plot()
		ax2.set_xlim((Ha-50)/1e4,(Ha+50)/1e4)
		xticks = ax2.get_xticks()
		ax2.set_xticks(xticks, labels=np.array(xticks*1e4, dtype='int'))
		ax2.set_ylabel('')
		ax2.set_xlabel('')
		ax2.set_title(r'Zoom-in on H$\alpha$ and [NII]', fontsize = 12)
		
		plt.sca(ax3)
		pp.plot()
		ax3.set_xlim((SII1-50)/1e4,(SII2+50)/1e4)
		xticks = ax3.get_xticks()
		ax3.set_xticks(xticks, labels=np.array(xticks*1e4, dtype='int'))
		ax3.set_title(r'Zoom-in on [SII]', fontsize = 12)
		ax3.set_yticklabels([])
		ax3.set_xlabel('')
		ax3.set_ylabel('')

		full_plot_dir = f'{plot_dir}{plot_name}'
		plot_fl = f'{full_plot_dir}/ppxf_gasfit_bin{bin_number}_{ncomp}comp.png'


	else:
		plt.sca(ax1)
		pp.plot()
		xticks = ax1.get_xticks()
		ax1.set_xticks(xticks, labels=np.array(xticks*1e4, dtype='int'))
		ax1.set_xlabel('Restframe Wavelength (Å)',fontsize = 12)
		ax1.set_ylabel('')
		legend_elements = [Line2D([0], [0], color='k', label='Data', lw=2),
							Line2D([0], [0], color='r', label='Stellar continuum', lw=2),
							Line2D([0], [0], marker='d', color='g', label='Residuals', markersize=5, lw=0),
							Line2D([0], [0], color='b', label='Masked regions', lw=2),]
		ax1.legend(handles=legend_elements, loc='upper left', fontsize='small', ncol=2)


		plt.sca(ax2)
		pp.plot()
		ax2.set_xlim(5800/1e4,6000/1e4)
		xticks = ax2.get_xticks()
		ax2.set_xticks(xticks, labels=np.array(xticks*1e4, dtype='int'))
		ax2.set_ylabel('')
		ax2.set_xlabel('')
		ax2.set_title(r'Zoom-in on NaD', fontsize = 12)
		
		plt.sca(ax3)
		pp.plot()
		ax3.set_xlim(6000/1e4,7000/1e4)
		xticks = ax3.get_xticks()
		ax3.set_xticks(xticks, labels=np.array(xticks*1e4, dtype='int'))
		ax3.set_title(r'Zoom-in on H$\alpha$ and [NII]', fontsize = 12)
		ax3.set_yticklabels([])
		ax3.set_xlabel('')
		ax3.set_ylabel('')

		full_plot_dir = f'{plot_dir}{plot_name}'
		plot_fl = f'{full_plot_dir}/ppxf_stellarfit_bin{bin_number}.png'


	
	if os.path.exists(full_plot_dir) == False:
		os.mkdir(full_plot_dir)
	plt.savefig(plot_fl, dpi = 300) 
	plt.close()

	print(f'Saved ppxf plot to {plot_name}')


def residual_fit_test(plot_name, pp, residuals, bin_number, ANR, logLam, z, ncomp, line_region='Ha', saveplot=False):
	
	if line_region == 'Ha':
		line_lam = Ha
	if line_region == 'OI':
		line_lam = OI
	if line_region == 'SII':
		line_lam = (SII1 + SII2)/2
	if line_region == 'OIII':
		line_lam = OIII

	residual_range = ((line_lam-75),(line_lam+75))
	plot_wave = np.exp(logLam)/(1+z)
	resid_ind = np.intersect1d(np.where(plot_wave > residual_range[0])[0], np.where(plot_wave < residual_range[1])[0])
	line_residuals = residuals[resid_ind]

	KS_result = kstest(line_residuals, norm.cdf)

	if saveplot==True:

		resid_hist, bin_edges = np.histogram(line_residuals)
		bin_width=bin_edges[1] - bin_edges[0]

		plot_dir = 'ppxf_output/plots/'

		fig, ax = plt.subplots(figsize = (9,6), sharey = True)
		ax1 = plt.subplot(121)
		ax2 = plt.subplot(122)

		fig.text(0.05, 0.93, 'A/rN of line: '+str(ANR))
		fig.text(0.45, 0.93, f'Chi-Squared/DOF: {pp.chi2}')
		fig.text(0.45, 0.9, f'KS p-value: {KS_result.pvalue}')

		plt.sca(ax1)
		pp.plot()
		ax1.set_xlim(np.array(residual_range)/1e4)
		xticks = ax1.get_xticks()
		ax1.set_xticks(xticks, labels=np.array(xticks*1e4, dtype='int'))
		ax1.set_xlabel('Restframe Wavelength (Å)',fontsize = 12)
		ax1.set_ylabel('')
		legend_elements = [Line2D([0], [0], color='k', label='Data', lw=2),
							Line2D([0], [0], color='r', label='Stellar continuum', lw=2),
							Line2D([0], [0], color='m', label='Gas emission lines', lw=2),
							Line2D([0], [0], color='b', label='Emission line components', lw=2),
							Line2D([0], [0], marker='d', color='g', label='Residuals', markersize=5, lw=0),
							Line2D([0], [0], color='orange', label='Total PPXF fit', lw=2)]
		ax1.legend(handles=legend_elements, loc='upper left', fontsize='small', ncol=2)

		ax2.bar(bin_edges[:-1], resid_hist, width=bin_width, align='edge')
		ax2.set_xlabel('Residuals')
		ax2.set_ylabel('Number')

		full_plot_dir = f'{plot_dir}{plot_name}'
		plot_fl = f'{full_plot_dir}/ppxf_residuals_bin{bin_number}_{ncomp}comp_{line_region}.png'

		if os.path.exists(full_plot_dir) == False:
			os.mkdir(full_plot_dir)
		plt.savefig(plot_fl, dpi = 300) 

	return KS_result



def ppxf_fit_stellar(cube, error_cube, moments, adegree, mdegree, wave_lam, plot_every=0,
					plot_name=None, prev_vmap_path=None):
	#cube - unbinned data cube
	#error_cube - unbinned error cube
	#moments - ppxf moments for fitting
	#adegree - degree of additive poly
	#mdegree - degree of multiplicative poly
	#wave_lam - wavelength array
	#plot_every - if 0, don't output any plots, otherwise output a plot every [n] bins
	#plot_name - name for output plots
	#prev_fit_path - if supplied, use previous fit info
	#individual_spaxel - int - if not None, bin number of single bin to fit
	#galaxy parameters

	z = 0.007214         # NGC 1266 redshift, from SIMBAD
	galv = np.log(z+1)*c # estimate of galaxy's velocity

	save_dict = {'bin_num':[], 'star_vel':[], 'star_vel_error':[], 'star_sig':[], 'star_sig_error':[],
	'SN_mad_STD':[], 'chisq/dof':[]}

	# MUSE spectral resolution, in Angstroms
	FWHM_gal = 2.51
	
	#assigning each pixel a bin number
	binNum = np.reshape(np.arange(cube.shape[1]*cube.shape[2]), (cube.shape[1], cube.shape[2]))
	x,y = np.meshgrid(np.arange(cube.shape[2]), np.arange(cube.shape[1]))

	#preparing stellar templates
	miles_lamrange_trunc = [3525, 9300] #expanded EMILES templates cover more
	wave_lam_rest = wave_lam/(1+z)
	cube_fit_ind = np.where((wave_lam_rest > miles_lamrange_trunc[0]) & (wave_lam_rest < miles_lamrange_trunc[1]))[0] #only fit rest-frame area covered by templates

	#shorten all spectra to only be within fitting area
	cube_trunc = cube[cube_fit_ind,:,:]
	error_cube_trunc = error_cube[cube_fit_ind,:,:]
	wave_trunc_rest = wave_lam_rest[cube_fit_ind]
	wave_trunc = wave_lam[cube_fit_ind]

	#wavelength ranges of rest-frame data
	wave_trunc_rest_range = wave_trunc_rest[[0,-1]]
	wave_trunc_range = wave_trunc[[0,-1]]

	#rebinning the wavelength to get the velscale
	example_spec = cube_trunc[:,150,150] #only using this for log-rebinning
	example_spec_rebin, log_wave_trunc, velscale_trunc = util.log_rebin(wave_trunc_range, example_spec)

	code_dir = "../../ppxf_files/EMILES_BASTI_BASE_CH_FITS/" # directory where stellar templates are located
	pathname = os.path.join(code_dir, 'Ech1.30*.fits')
	miles = lib.miles(pathname, velscale_trunc, FWHM_gal)							# The stellar templates are reshaped below into a 2-dim array with each	
																						# spectrum as a column, however we save the original array dimensions,
																						# which are needed to specify the regularization dimensions.
	reg_dim = miles.templates.shape[1:]
	stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
	n_temps = stars_templates.shape[1]

	#for saving optimal templates for each bin
	n_bins = len(np.unique(binNum)) - 1 #subtract 1 for -1 bin
	optimal_templates_save = np.empty((stars_templates.shape[0], n_bins))

	lam_temp = miles.lam_temp
	lamRange_temp = [lam_temp[0], lam_temp[-1]]

	
	templates = stars_templates
	gas_component = None
	gas_names = None
	component = 0

	#velocity difference between templates and data, templates start around 3500 and data starts closer to 4500
	dv = c*(np.log(miles.lam_temp[0]/wave_trunc[0])) # eq.(8) of Cappellari (2017)


	#if previous voronoi binned stellar velocity map supplied, use it for initial conditions
	if prev_vmap_path is not None:
		pvmap_fl = fits.open(prev_vmap_path)
		prev_vmap = pvmap_fl[1].data[0,:,:]
		prev_sigmap = pvmap_fl[1].data[1,:,:]
	else:
		prev_vmap = None

	loop_list = np.unique(binNum)

	for bn in loop_list: 
		if bn >= 0: #skip nan bins with binNum == -1


			save_dict['bin_num'].append(bn)

			b_loc = np.where(binNum == bn)
			x_loc = x[b_loc]
			y_loc = y[b_loc]

			spectrum = cube_trunc[:,y_loc,x_loc]
			err_spectrum = error_cube_trunc[:,y_loc,x_loc]
			
			npix_edge = 10
			if x_loc < npix_edge or cube.shape[2] - x_loc < npix_edge or y_loc < npix_edge or cube.shape[1] - y_loc < npix_edge:
				#spectrum[np.isnan(spectrum)==False].shape[0] < 3000: #if less than 3000 non nan data points
				save_dict['star_vel'].append(np.nan)
				save_dict['star_vel_error'].append(np.nan)
				save_dict['star_sig'].append(np.nan)
				save_dict['star_sig_error'].append(np.nan)
				save_dict['SN_mad_STD'].append(np.nan)
				save_dict['chisq/dof'].append(np.nan)

				continue

			print('\n============================================================================')
			print('binNum: {}'.format(bn))
			#also use previous optimal stellar template if fitting gas
			#choose starting values - just use galaxy velocity if no previous fit included

			if prev_vmap is not None:
				prev_vel = prev_vmap[y_loc, x_loc][0]
				prev_sig = prev_sigmap[y_loc, x_loc][0]

				start_vals = [prev_vel, prev_sig]

			else:
				start_vals = [galv, 25]
			
			start = start_vals
			bounds = None

			templates /= np.median(templates)

			### take mean of spectra
			#gal_lin = np.nansum(spectra, axis=1)/n_spec
			#noise_lin = np.sqrt(np.nansum(np.abs(err_spectra), axis=1))/n_spec #add error spectra in quadrature, err is variance so no need to square

			gal_lin = spectrum[:,0]
			noise_lin = err_spectrum[:,0]

			#noise_lin = noise_lin/np.nanmedian(noise_lin)
			noise_lin[np.isinf(noise_lin)] = np.nanmedian(noise_lin)
			noise_lin[noise_lin == 0] = np.nanmedian(noise_lin)

			galaxy, logLam, velscale = util.log_rebin(wave_trunc_range, gal_lin)
			#log_noise, logLam_noise, velscale_noise = util.log_rebin(lamRange, noise_lin) # I don't think this needs to be log-rebinned also

			maskreg = (5880,5950) #galactic and extragalactic NaD in this window, observed wavelength
			reg1 = np.where(np.exp(logLam) < maskreg[0])
			reg2 = np.where(np.exp(logLam) > maskreg[1])
			good_pixels = np.concatenate((reg1, reg2), axis=1)
			good_pixels = good_pixels.reshape(good_pixels.shape[1])

			goodpix_util = custom_util.determine_goodpixels(logLam, lamRange_temp, z) #uses z to get redshifted line wavelengths, so logLam should be observer frame
			good_pix_total = np.intersect1d(goodpix_util, good_pixels)

			#CALLING PPXF HERE!
			t = clock()
			pp = ppxf(templates, galaxy, noise_lin, velscale, start,
						plot=False, moments=moments, degree= adegree, mdegree=mdegree, vsyst=dv,
						clean=False, lam=np.exp(logLam)/(1+z), #regul= 1/0.1 , reg_dim=reg_dim,
						component=component, gas_component=gas_component, bounds=bounds,
						goodpixels = good_pix_total, global_search=False)

			bestfit = pp.bestﬁt
			residuals = galaxy - bestfit

			param0 = pp.sol
			error = pp.error

			mad_std_residuals = mad_std(residuals, ignore_nan=True)    
			med_galaxy = np.nanmedian(galaxy) #this will stand in for signal probably will be ~1
			SN_wMadStandardDev = med_galaxy/mad_std_residuals 
			save_dict['SN_mad_STD'].append(SN_wMadStandardDev)
			print('S/N w/ mad_std: '+str(SN_wMadStandardDev))

			vel_comp0 = param0[0]
			vel_error_comp0 = error[0]*np.sqrt(pp.chi2)
			sig_comp0 = param0[1]
			sig_error_comp0 = error[1]*np.sqrt(pp.chi2)

			optimal_template = templates @ pp.weights
			optimal_templates_save[:,bn] = optimal_template


			save_dict['star_vel'].append(vel_comp0) 
			save_dict['star_vel_error'].append(vel_error_comp0)
			save_dict['star_sig'].append(sig_comp0)
			save_dict['star_sig_error'].append(sig_error_comp0)
			save_dict['chisq/dof'].append(pp.chisq)

			fit_chisq = (pp.chi2 - 1)*galaxy.size
			print('============================================================================')
			print('Desired Delta Chi^2: %.4g' % np.sqrt(2*galaxy.size))
			print('Current Delta Chi^2: %.4g' % (fit_chisq))
			print('Elapsed time in PPXF: %.2f s' % (clock() - t))
			print('============================================================================')


			if plot_every > 0 and bn % plot_every == 0:

				output_ppxf_fit_plot(plot_name, pp, good_pix_total, logLam, vel_comp0, z, ngas_comp, 
									n_spec, bn, mad_std_residuals, SN_wMadStandardDev, fit_gas)


	return save_dict, optimal_templates_save

def ppxf_fit_stellar2(cube, error_cube, moments, adegree, mdegree, wave_lam, plot_every=0, plot_name=None, prev_vmap_path=None):
	## this will be the ppxf function to run pixel-by-pixel stellar continuum fitting

	#cube - unbinned data cube
	#error_cube - unbinned error cube
	#moments - ppxf moments for fitting
	#adegree - degree of additive poly
	#mdegree - degree of multiplicative poly
	#wave_lam - wavelength array
	#plot_every - if 0, don't output any plots, otherwise output a plot every [n] bins
	#plot_name - name for output plots
	#prev_vmap_path - if supplied, use previous fit info

	z = 0.007214         # NGC 1266 redshift, from SIMBAD
	galv = np.log(z+1)*c # estimate of galaxy's velocity

	save_dict = {'bin_num':[], 'star_vel':[], 'star_vel_error':[], 'star_sig':[], 'star_sig_error':[],
	'SN_mad_STD':[], 'chisq/dof':[]}

	# MUSE spectral resolution, in Angstroms
	FWHM_gal = 2.51
	
	#voronoi bin file
	binNum = np.reshape(np.arange(cube.shape[1]*cube.shape[2]), (cube.shape[1], cube.shape[2]))
	x,y = np.meshgrid(np.arange(cube.shape[2]), np.arange(cube.shape[1]))


	#preparing stellar templates
	miles_lamrange_trunc = [3525, 9300] #cutting off edge of template range that will be masked anyways
	wave_lam_rest = wave_lam/(1+z)
	cube_fit_ind = np.where((wave_lam_rest > miles_lamrange_trunc[0]) & (wave_lam_rest < miles_lamrange_trunc[1]))[0] #only fit rest-frame area covered by templates

	#shorten all spectra to only be within fitting area
	cube_trunc = cube[cube_fit_ind,:,:]
	error_cube_trunc = error_cube[cube_fit_ind,:,:]
	wave_trunc_rest = wave_lam_rest[cube_fit_ind]
	wave_trunc = wave_lam[cube_fit_ind]

	#wavelength ranges of rest-frame data
	wave_trunc_rest_range = wave_trunc_rest[[0,-1]]
	wave_trunc_range = wave_trunc[[0,-1]]

	#rebinning the wavelength to get the velscale
	example_spec = cube_trunc[:,150,150] #only using this for log-rebinning
	example_spec_rebin, log_wave_trunc, velscale_trunc = util.log_rebin(wave_trunc_range, example_spec)

	#code_dir = "../../ppxf_files/MILES_BASTI_CH_baseFe/" # directory where stellar templates are located
	code_dir = "../../ppxf_files/EMILES_BASTI_BASE_CH_FITS/" # directory where stellar templates are located
	#pathname = os.path.join(code_dir, 'Mch1.30*.fits')
	pathname = os.path.join(code_dir, 'Ech1.30*.fits')
	miles = lib.miles(pathname, velscale_trunc, FWHM_gal)							# The stellar templates are reshaped below into a 2-dim array with each	
																						# spectrum as a column, however we save the original array dimensions,
																						# which are needed to specify the regularization dimensions.
	reg_dim = miles.templates.shape[1:]
	stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
	n_temps = stars_templates.shape[1]

	#for saving optimal templates for each bin
	n_bins = len(np.unique(binNum)) - 1 #subtract 1 for -1 bin
	optimal_templates_save = np.empty((stars_templates.shape[0], n_bins))

	lam_temp = miles.lam_temp
	lamRange_temp = [lam_temp[0], lam_temp[-1]]

	############################

	'''#preparing stellar templates
				miles_lamrange = [3525,9300] #Angstroms, from documentation
				wave_lam_rest = wave_lam/(1+z)
				cube_fit_ind = np.where((wave_lam_rest > miles_lamrange[0]) & (wave_lam_rest < miles_lamrange[1]))[0] #only fit rest-frame area covered by templates
			
				#shorten all spectra to only be within fitting area
				cube = cube[cube_fit_ind,:,:]
				error_cube = error_cube[cube_fit_ind,:,:]
				wave_lam = wave_lam[cube_fit_ind]
			
				#wavelength ranges of rest-frame data
				wave_lam_rest = wave_lam/(1+z)
				wave_rest_range = wave_lam_rest[[0,-1]]
				wave_range = wave_lam[[0,-1]]
			
				#rebinning the wavelength to get the velscale
				example_spec = cube[:,150,150] #only using this for log-rebinning
				example_spec_rebin, log_wave, velscale = util.log_rebin(wave_range, example_spec)
			
				code_dir = "../../ppxf_files/EMILES_BASTI_BASE_CH_FITS/" # directory where extended stellar templates are located
				pathname = os.path.join(code_dir, 'Ech1.30*.fits')
				miles = lib.miles(pathname, velscale, FWHM_gal)							# The stellar templates are reshaped below into a 2-dim array with each	
																									# spectrum as a column, however we save the original array dimensions,
																									# which are needed to specify the regularization dimensions.
				reg_dim = miles.templates.shape[1:]
				stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
				n_temps = stars_templates.shape[1]
			
				#for saving optimal templates for each bin
				n_bins = len(np.unique(binNum)) - 1 #subtract 1 for -1 bin
				optimal_templates_save = np.empty((stars_templates.shape[0], n_bins))
			
				lam_temp = miles.lam_temp
				lamRange_temp = [lam_temp[0], lam_temp[-1]]'''

	###########

	templates = stars_templates
	gas_component = None
	gas_names = None
	component = 0

	#velocity difference between templates and data, templates start around 3500 and data starts closer to 4500
	dv = c*(np.log(miles.lam_temp[0]/wave_trunc[0])) # eq.(8) of Cappellari (2017)

	#if previous voronoi binned stellar velocity map supplied, use it for initial conditions
	if prev_vmap_path is not None:
		pvmap_fl = fits.open(prev_vmap_path)
		prev_vmap = pvmap_fl[1].data[0,:,:]
		prev_sigmap = pvmap_fl[1].data[1,:,:]
	else:
		prev_vmap = None

	loop_list = np.unique(binNum)

	for bn in loop_list: 
		if bn >= 0: #skip nan bins with binNum == -1

			save_dict['bin_num'].append(bn)

			b_loc = np.where(binNum == bn)
			x_loc = x[b_loc]
			y_loc = y[b_loc]

			spectrum = cube_trunc[:,y_loc,x_loc]
			err_spectrum = error_cube_trunc[:,y_loc,x_loc]
			
			npix_edge = 10
			if x_loc < npix_edge or cube.shape[2] - x_loc < npix_edge or y_loc < npix_edge or cube.shape[1] - y_loc < npix_edge:
				#spectrum[np.isnan(spectrum)==False].shape[0] < 3000: #if less than 3000 non nan data points
				save_dict['star_vel'].append(np.nan)
				save_dict['star_vel_error'].append(np.nan)
				save_dict['star_sig'].append(np.nan)
				save_dict['star_sig_error'].append(np.nan)
				save_dict['SN_mad_STD'].append(np.nan)
				save_dict['chisq/dof'].append(np.nan)

				continue

			print('\n============================================================================')
			print('binNum: {}'.format(bn))
			#also use previous optimal stellar template if fitting gas
			#choose starting values - just use galaxy velocity if no previous fit included

			if prev_vmap is not None:
				prev_vel = prev_vmap[y_loc, x_loc][0]
				prev_sig = prev_sigmap[y_loc, x_loc][0]

				start_vals = [prev_vel, prev_sig]

			else:
				start_vals = [galv, 25]
			
			start = start_vals
			bounds = None
			
			print(wave_trunc_range, lamRange_temp)
			templates /= np.median(templates)

			err_spectrum[err_spectrum == 0] = np.nanmedian(err_spectrum[err_spectrum > 0])
			err_spectrum[np.isinf(err_spectrum)] = np.nanmedian(err_spectrum)

			
			galaxy, logLam, velscale = util.log_rebin(wave_trunc_range, spectrum)
			#log_noise, logLam_noise, velscale_noise = util.log_rebin(lamRange, noise_lin) # I don't think this needs to be log-rebinned also

			maskreg = (5880,5950) #galactic and extragalactic NaD in this window, observed wavelength
			reg1 = np.where(np.exp(logLam) < maskreg[0])
			reg2 = np.where(np.exp(logLam) > maskreg[1])
			good_pixels = np.concatenate((reg1, reg2), axis=1)
			good_pixels = good_pixels.reshape(good_pixels.shape[1])

			goodpix_util = custom_util.determine_goodpixels(logLam, lamRange_temp, z) #uses z to get redshifted line wavelengths, so logLam should be observer frame
			good_pix_total = np.intersect1d(goodpix_util, good_pixels)

			#CALLING PPXF HERE!
			t = clock()
			pp = ppxf(templates, galaxy[0], err_spectrum[0], velscale, start,
						plot=False, moments=moments, degree= adegree, mdegree=mdegree, vsyst=dv,
						clean=False, lam=np.exp(logLam)/(1+z), #regul= 1/0.1 , reg_dim=reg_dim,
						component=component, bounds=bounds, goodpixels = good_pix_total)

			bestfit = pp.bestﬁt
			residuals = galaxy - bestfit

			param0 = pp.sol
			error = pp.error

			mad_std_residuals = mad_std(residuals, ignore_nan=True)    
			med_galaxy = np.nanmedian(galaxy) #this will stand in for signal probably will be ~1
			SN_wMadStandardDev = med_galaxy/mad_std_residuals 
			save_dict['SN_mad_STD'].append(SN_wMadStandardDev)
			print('S/N w/ mad_std: '+str(SN_wMadStandardDev))

			vel_comp0 = param0[0]
			vel_error_comp0 = error[0]*np.sqrt(pp.chi2)
			sig_comp0 = param0[1]
			sig_error_comp0 = error[1]*np.sqrt(pp.chi2)

			optimal_template = templates @ pp.weights
			optimal_templates_save[:,bn] = optimal_template

			save_dict['star_vel'].append(vel_comp0) 
			save_dict['star_vel_error'].append(vel_error_comp0)
			save_dict['star_sig'].append(sig_comp0)
			save_dict['star_sig_error'].append(sig_error_comp0)

			fit_chisq = (pp.chi2 - 1)*galaxy.size

			save_dict['chisq/dof'].append(pp.chisq)
			print('============================================================================')
			print('Desired Delta Chi^2: %.4g' % np.sqrt(2*galaxy.size))
			print('Current Delta Chi^2: %.4g' % (fit_chisq))
			print('Elapsed time in PPXF: %.2f s' % (clock() - t))
			print('============================================================================')


			if plot_every > 0 and bn % plot_every == 0:
				ngas_comp = 0
				fit_gas = False

				output_ppxf_fit_plot(plot_name, pp, good_pix_total, logLam, vel_comp0, z, ngas_comp, 
									bn, mad_std_residuals, SN_wMadStandardDev, fit_gas)


	return save_dict, optimal_templates_save

def ppxf_fit_gas(cube, error_cube, vorbin_path, moments, adegree, mdegree, wave_lam, limit_doublets=True, tie_balmer=False, plot_every=0,
				plot_name=None, prev_fit_path=None, fit_gas=True, ngas_comp=1, globsearch=False, individual_bin=None, test_residuals=True):
	## this will be the main function to fit gas pixel-by-pixel with 1-2 components depending on the spectrum

	#cube - unbinned data cube
	#error_cube - unbinned error cube
	#vorbin_path - path to voronoi bin file, if None then fit pixel by pixel
	#moments - ppxf moments for fitting
	#adegree - degree of additive poly
	#mdegree - degree of multiplicative poly
	#wave_lam - wavelength array
	#limit_doublets - if True, requires doublets to have ratios permitted by atomic physics
	#tie_balmer - if True, Balmer series are a single template
	#plot_every - if 0, don't output any plots, otherwise output a plot every [n] bins
	#plot_name - name for output plots
	#prev_fit_path - if supplied, use previous fit info
	#fit_gas - if True, fit gas emission lines, otherwise mask them and only do stellar continuum
	#ngas_comp - number of kinematic components to fit gas with
	#globsearch - bool - choose to set this during ppxf, True is important for complex multi-component, but slow
	#individual_spaxel - int - if not None, bin number of single bin to fit
	#galaxy parameters

	z = 0.007214         # NGC 1266 redshift, from SIMBAD
	galv = np.log(z+1)*c # estimate of galaxy's velocity

	save_dict = {'bin_num':[], 'star_vel':[], 'star_vel_error':[], 'star_sig':[], 'star_sig_error':[], 'SN_mad_STD':[]}

	# MUSE spectral resolution, in Angstroms
	FWHM_gal = 2.51
	
	#voronoi bin file
	if vorbin_path is not None:
		x,y,binNum = np.loadtxt(vorbin_path).T
		x,y,binNum = x.astype(int), y.astype(int), binNum.astype(int)
	else:
		binNum = np.reshape(np.arange(cube.shape[1]*cube.shape[2]), cube.shape[1,2])
		x,y = np.meshgrid(np.arange(cube.shape[1]), np.arange(cube.shape[2]))

	#wavelength ranges of rest-frame data
	wave_rest_range = wave_lam_rest[[0,-1]]
	wave_range = wave_lam[[0,-1]]

	#rebinning the wavelength to get the velscale
	example_spec = cube[:,150,150] #only using this for log-rebinning
	example_spec_rebin, log_wave, velscale = util.log_rebin(wave_range, example_spec)

	code_dir = "../../ppxf_files/EMILES_BASTI_BASE_CH_FITS/" # directory where extended stellar templates are located
	pathname = os.path.join(code_dir, 'Ech1.30*.fits')
	miles = lib.miles(pathname, velscale, FWHM_gal)							# The stellar templates are reshaped below into a 2-dim array with each	
																						# spectrum as a column, however we save the original array dimensions,
																						# which are needed to specify the regularization dimensions.
	reg_dim = miles.templates.shape[1:]
	stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
	n_temps = stars_templates.shape[1]

	#for saving optimal templates for each bin
	n_bins = len(np.unique(binNum)) - 1 #subtract 1 for -1 bin
	optimal_templates_save = np.empty((stars_templates.shape[0], n_bins))

	lam_temp = miles.lam_temp
	lamRange_temp = [lam_temp[0], lam_temp[-1]]

	if fit_gas == True:
		gas_templates, gas_names, line_wave = custom_util.emission_lines(miles.ln_lam_temp, wave_trunc_range, FWHM_gal, 
																tie_balmer=tie_balmer, limit_doublets=limit_doublets)

		n_gas = len(gas_names)
		gas_templates_tile = np.tile(gas_templates, ngas_comp)
		gas_names = np.asarray([a + f"_({p+1})" for p in range(ngas_comp) for a in gas_names])
		line_wave = np.tile(line_wave, ngas_comp)

		# Assign component=0 to the stellar templates, component=1 to gas
		if ngas_comp == 1:
			component = [0] + [1]*n_gas
		if ngas_comp == 2:
			component = [0] + [1]*n_gas + [2]*n_gas
		if ngas_comp == 3:
			component = [0]+ [1]*n_gas + [2]*n_gas + [3]*n_gas 
	
		for num in range(1, ngas_comp+1):
			save_dict[f'gas_({num})_vel'] = []
			save_dict[f'gas_({num})_vel_error'] = []
			save_dict[f'gas_({num})_sig'] = []
			save_dict[f'gas_({num})_sig_error'] = []

		gas_component = np.array(component) > 0  # gas_component=True for gas 


		for ggas in gas_names:
			ncomp_num = ggas[-2]
			gas_name = ggas[:-4]
			if gas_name == 'HeI5876':
				continue
			save_dict[ggas+'_flux'] = [] #flux is measured flux from ppxf
			save_dict[ggas+'_flux_error'] = []
			save_dict[ggas+'_amplitude'] = [] #amplitude is max flux in gas bestfit - so includes all components
			save_dict[ggas+'_ANR'] = [] #amplitude / mad_std_residuals
			save_dict[ggas+'_comp_amp'] = [	] #max flux only in the relevant component

			if gas_name == '[NII]6583_d':
				save_dict[f'[NII]6548_({ncomp_num})_amplitude'] = []
				save_dict[f'[NII]6548_({ncomp_num})_ANR'] = []

		#load in previous fit files for later use:
		prev_dict = pd.read_csv(prev_fit_path)
		prev_templates_path = prev_fit_path[:-4] + '_templates.npy'
		prev_temps = np.load(prev_templates_path)

	else:
		templates = stars_templates
		gas_component = None
		gas_names = None
		component = 0

	

	#velocity difference between templates and data, templates start around 3500 and data starts closer to 4500
	dv = c*(np.log(miles.lam_temp[0]/wave_trunc[0])) # eq.(8) of Cappellari (2017)

	loop_list = np.unique(binNum)
	if individual_bin is not None:
		loop_list = [individual_bin]

	for bn in loop_list: 
		if bn >= 0: #skip nan bins with binNum == -1
			print('\n============================================================================')
			print('binNum: {}'.format(bn))
			save_dict['bin_num'].append(bn)
			b_loc = np.where(binNum == bn)[0]
			x_loc = x[b_loc]
			y_loc = y[b_loc]

			spectra = cube[:,y_loc,x_loc]
			err_spectra = error_cube[:,y_loc,x_loc]

			n_spec = spectra.shape[1] #don't need n_spec to run ppxf
			print('n_spec in bin: ', n_spec)

			
			#also use previous optimal stellar template if fitting gas
			#choose starting values - just use galaxy velocity if no previous fit included

			if fit_gas == True:
				#if previous fit data supplied, use previous starting values
				bin_df = prev_dict.loc[prev_dict['bin_num'] == bn]
				star_vel = bin_df['star_vel'].values[0]
				star_sig = bin_df['star_sig'].values[0]

				start_vals  = [star_vel, star_sig]
				start_vals_ncomp = [star_vel-200, star_sig*2] #for higher components, start with 2*stellar sigma width

				start = np.concatenate((np.tile(start_vals, (3,1)), np.tile(start_vals_ncomp, ((ngas_comp-1)*2, 1)))) #first 3 are stellar and first gas, then add extra

				if single_gascomp == True:
					start = np.concatenate((np.tile(start_vals, (2,1)), np.tile(start_vals_ncomp, ((ngas_comp-1), 1)))) #first 2 are stellar and first gas, then add extra


				# setting bounds
				if ngas_comp == 1:
					bounds = None
				if ngas_comp > 1:
					gas_bounds = [[star_vel-600, star_vel+600], [20, 1000]]
					bounds = [gas_bounds]
					for n in range(ngas_comp):
						bounds.append(gas_bounds)
						if single_gascomp == False: #if splitting balmer, then need to add another set of bounds
							bounds.append(gas_bounds)


				# kinematic constraints
				if ngas_comp == 2: #constraint to ensure the first component is more narrow than the second for balmer and non-balmer lines
					#moments will be: [-2, 2, 2, 2, 2]
					#start will be: [[vstar, vsigma], [v1_1, sigma1_1], [v2_1, sigma2_1], [v1_2, sigma1_2], [v2_2, sigma2_2]]
					#we want sigma1_2 > sigma1_1 AND sigma2_2 > sigma_2_1
					A_ineq = [[0,0, 0,1, 0,0, 0,-1, 0,0],  #sigma1_1 - sigma1_2 <= 0
							  [0,0, 0,0, 0,1, 0,0,  0,-1]] #sigma2_1 - sigma2_2 <= 0
					b_ineq = [1e-5, 1e-5]

					if single_gascomp == True:
						#moments will be: [-2, 2, 2]
						#start will be: [[vstar, vsigma], [v1, sigma1], [v2, sigma2]]
						#we want sigma1_2 > sigma1_1 AND sigma2_2 > sigma_2_1
						A_ineq = [[0,0, 0,1, 0,-1]]  #sigma1 - sigma2 <= 0
						b_ineq = [1e-5]

					#constr_kinem = {"A_ineq":A_ineq, "b_ineq":b_ineq}
					constr_kinem = None #disable above constraints
				else:
					constr_kinem = None

				
				#use previous fit path to locate previous optimal templates file


				opt_temp = prev_temps[:,bn]

				# Combines the stellar and gaseous templates into a single array.
				# During the PPXF fit they will be assigned a different kinematic COMPONENT value

				templates = np.column_stack([opt_temp, gas_templates_tile])

			else:
				start_vals = [galv, 25]
				start = start_vals
				bounds = None

			templates /= np.median(templates)

			### take mean of spectra
			gal_lin = np.nansum(spectra, axis=1)/n_spec
			noise_lin = np.sqrt(np.nansum(np.abs(err_spectra), axis=1))/n_spec #add error spectra in quadrature, err is variance so no need to square

			#noise_lin = noise_lin/np.nanmedian(noise_lin)
			noise_lin[np.isinf(noise_lin)] = np.nanmedian(noise_lin)
			noise_lin[noise_lin == 0] = np.nanmedian(noise_lin)

			galaxy, logLam, velscale = util.log_rebin(wave_trunc_range, gal_lin)
			#log_noise, logLam_noise, velscale_noise = util.log_rebin(lamRange, noise_lin) # I don't think this needs to be log-rebinned also

			maskreg = (5880,5950) #galactic and extragalactic NaD in this window, observed wavelength
			reg1 = np.where(np.exp(logLam) < maskreg[0])
			reg2 = np.where(np.exp(logLam) > maskreg[1])
			good_pixels = np.concatenate((reg1, reg2), axis=1)
			good_pixels = good_pixels.reshape(good_pixels.shape[1])

			if fit_gas == False:
				goodpix_util = custom_util.determine_goodpixels(logLam, lamRange_temp, z) #uses z to get redshifted line wavelengths, so logLam should be observer frame
				good_pix_total = np.intersect1d(goodpix_util, good_pixels)
			else:
				good_pix_total = good_pixels


			#CALLING PPXF HERE!
			t = clock()
			pp = ppxf(templates, galaxy, noise_lin, velscale, start,
						plot=False, moments=moments, degree= adegree, mdegree=mdegree, vsyst=dv,
						clean=False, lam=np.exp(logLam)/(1+z), #regul= 1/0.1 , reg_dim=reg_dim,
						component=component, gas_component=gas_component, bounds=bounds, constr_kinem=constr_kinem,
						gas_names=gas_names, goodpixels = good_pix_total, global_search=globsearch)

			bestfit = pp.bestﬁt
			residuals = galaxy - bestfit

			param0 = pp.sol
			error = pp.error

			mad_std_residuals = mad_std(residuals, ignore_nan=True)    
			med_galaxy = np.nanmedian(galaxy) #this will stand in for signal probably will be ~1
			SN_wMadStandardDev = med_galaxy/mad_std_residuals 
			save_dict['SN_mad_STD'].append(SN_wMadStandardDev)
			print('S/N w/ mad_std: '+str(SN_wMadStandardDev))

			if fit_gas == False:
				vel_comp0 = param0[0]
				vel_error_comp0 = error[0]*np.sqrt(pp.chi2)
				sig_comp0 = param0[1]
				sig_error_comp0 = error[1]*np.sqrt(pp.chi2)

				optimal_template = templates @ pp.weights
				optimal_templates_save[:,bn] = optimal_template

			else:
				vel_comp0 = param0[0][0]
				vel_error_comp0 = error[0][0]*np.sqrt(pp.chi2)
				sig_comp0 = param0[0][1]
				sig_error_comp0 = error[0][1]*np.sqrt(pp.chi2)


			save_dict['star_vel'].append(vel_comp0) 
			save_dict['star_vel_error'].append(vel_error_comp0)
			save_dict['star_sig'].append(sig_comp0)
			save_dict['star_sig_error'].append(sig_error_comp0)


			if fit_gas == True:
				gas_bestfit = pp.gas_bestﬁt
				stellar_fit = bestfit - gas_bestfit

				gas_templates_fit = pp.gas_bestfit_templates

				param1 = pp.gas_flux
				param2 = pp.gas_flux_error


				if single_gascomp == False:
					for num in np.arange(1,ngas_comp+1):
						ind = num*2 - 1
						vel_comp1 = param0[ind][0]
						vel_error_comp1 = error[ind][0]*np.sqrt(pp.chi2)
						sig_comp1 = param0[ind][1]
						sig_error_comp1 = error[ind][1]*np.sqrt(pp.chi2)

						vel_comp2 = param0[ind+1][0]
						vel_error_comp2 = error[ind+1][0]*np.sqrt(pp.chi2)
						sig_comp2 = param0[ind+1][1]
						sig_error_comp2 = error[ind+1][1]*np.sqrt(pp.chi2)

						save_dict[f'balmer_({num})_vel'].append(vel_comp1)
						save_dict[f'balmer_({num})_vel_error'].append(vel_error_comp1)
						save_dict[f'balmer_({num})_sig'].append(sig_comp1)
						save_dict[f'balmer_({num})_sig_error'].append(sig_error_comp1)

						save_dict[f'forbidden_({num})_vel'].append(vel_comp2)
						save_dict[f'forbidden_({num})_vel_error'].append(vel_error_comp1)
						save_dict[f'forbidden_({num})_sig'].append(sig_comp2)
						save_dict[f'forbidden_({num})_sig_error'].append(sig_error_comp1)

				elif single_gascomp == True:
					for num in range(1, ngas_comp+1):
						vel_comp1 = param0[num][0]
						vel_error_comp1 = error[num][0]*np.sqrt(pp.chi2)
						sig_comp1 = param0[num][1]
						sig_error_comp1 = error[num][1]*np.sqrt(pp.chi2)

						save_dict[f'gas_({num})_vel'].append(vel_comp1)
						save_dict[f'gas_({num})_vel_error'].append(vel_error_comp1)
						save_dict[f'gas_({num})_sig'].append(sig_comp1)
						save_dict[f'gas_({num})_sig_error'].append(sig_error_comp1)

				wave_unexp = np.exp(logLam)

				Halpha_ANR = ANR(None, 'Halpha_(1)', Ha, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp0)

				for idx, ggas in enumerate(gas_names):
					ncomp_num = ggas[-2]
					gas_name = ggas[:-4]

					gas_temp_idx = gas_templates_fit[:,idx]

					if gas_name == 'HeI5876':
						continue
					save_dict[ggas+'_flux'].append(param1[idx])
					save_dict[ggas+'_flux_error'].append(param2[idx])
					save_dict[ggas+'_comp_amp'].append(np.nanmax(gas_temp_idx))

					if single_gascomp == True:
						vel_comp_bl = save_dict[f'gas_({ncomp_num})_vel'][-1]
						vel_comp_fb = save_dict[f'gas_({ncomp_num})_vel'][-1]
					else:
						vel_comp_bl = save_dict[f'balmer_({num})_vel'][-1]
						vel_comp_fb = save_dict[f'forbidden_({num})_vel'][-1]

					if gas_name == 'Hbeta':
						emline = Hb
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_bl)
					if gas_name == 'Halpha':
						emline = Ha
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_bl)
						print('A/rN of Halpha: '+str(save_dict['Halpha_(1)_ANR'][-1]))
					if gas_name == '[SII]6731_d1':
						emline = SII1
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[SII]6731_d2':
						emline = SII2
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[OIII]5007_d':
						emline = OIII
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[OI]6300_d':
						emline = OI
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[NII]6583_d':
						emline = NII
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
						emline2 = NII1
						ANR(save_dict, f'[NII]6548_({ncomp_num})', emline2, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb) 
					if gas_name == '[NI]5201':
						emline = NI
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[NII]5756':
						emline = NII5756
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[OII]7322':
						emline = OII1
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[OII]7333':
						emline = OII2
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)

			fit_chisq = (pp.chi2 - 1)*galaxy.size
			print('============================================================================')
			print('Desired Delta Chi^2: %.4g' % np.sqrt(2*galaxy.size))
			print('Current Delta Chi^2: %.4g' % (fit_chisq))
			print('Elapsed time in PPXF: %.2f s' % (clock() - t))
			print('============================================================================')


			if plot_every > 0 and bn % plot_every == 0:

				output_ppxf_fit_plot(plot_name, pp, good_pix_total, logLam, vel_comp0, z, ngas_comp, 
									n_spec, bn, mad_std_residuals, SN_wMadStandardDev, fit_gas)

				if test_residuals == True:
					KS_result = residual_fit_test(plot_name, pp, residuals, bn, Halpha_ANR, logLam, z, ngas_comp, n_spec, saveplot=True)

	if fit_gas == True:
		return save_dict
	else:
		return save_dict, optimal_templates_save




def ppxf_iterative_fit(cube, error_cube, vorbin_path, adegree, mdegree, wave_lam, limit_doublets=True, tie_balmer=False, plot_every=0,
						plot_name=None, prev_fit_path=None, globsearch=False, single_gascomp=True, individual_bin=None):
	#####
	# Function to run on cube to initially do a 1 component fit, and then switch to a 2 component fit if necessary
	#cube - unbinned data cube
	#error_cube - unbinned error cube
	#vorbin_path - path to voronoi bin file
	#adegree - degree of additive poly
	#mdegree - degree of multiplicative poly
	#wave_lam - wavelength array
	#limit_doublets - if True, requires doublets to have ratios permitted by atomic physics
	#tie_balmer - if True, Balmer series are a single template
	#plot_every - if 0, don't output any plots, otherwise output a plot every [n] bins
	#plot_name - name for output plots
	#prev_fit_path - if supplied, use previous fit info
	#globsearch - bool - choose to set this during ppxf, True is important for complex multi-component, but slow 
	#single_gascomp - bool - if True, only use 1 component for all gas lines and do NOT split balmer/forbidden lines
	#individual_spaxel - int - if not None, bin number of single bin to fit
	#galaxy parameters

	z = 0.007214         # NGC 1266 redshift, from SIMBAD
	galv = np.log(z+1)*c # estimate of galaxy's velocity

	save_dict = {'bin_num':[], 'star_vel':[], 'star_vel_error':[], 'star_sig':[], 'star_sig_error':[], 'SN_mad_STD':[]}

	# MUSE spectral resolution, in Angstroms
	FWHM_gal = 2.51
	
	#voronoi bin file
	x,y,binNum = np.loadtxt(vorbin_path).T
	x,y,binNum = x.astype(int), y.astype(int), binNum.astype(int)

	#preparing stellar templates
	miles_lamrange = [3525,7500] #Angstroms, from documentation
	miles_lamrange_trunc = [3525, 7350] #cutting off edge of template range that will be masked anyways
	wave_lam_rest = wave_lam/(1+z)
	cube_fit_ind = np.where((wave_lam_rest > miles_lamrange_trunc[0]) & (wave_lam_rest < miles_lamrange_trunc[1]))[0] #only fit rest-frame area covered by templates

	#shorten all spectra to only be within fitting area
	cube_trunc = cube[cube_fit_ind,:,:]
	error_cube_trunc = error_cube[cube_fit_ind,:,:]
	wave_trunc_rest = wave_lam_rest[cube_fit_ind]
	wave_trunc = wave_lam[cube_fit_ind]

	#wavelength ranges of rest-frame data
	wave_trunc_rest_range = wave_trunc_rest[[0,-1]]
	wave_trunc_range = wave_trunc[[0,-1]]

	#rebinning the wavelength to get the velscale
	example_spec = cube_trunc[:,150,150] #only using this for log-rebinning
	example_spec_rebin, log_wave_trunc, velscale_trunc = util.log_rebin(wave_trunc_range, example_spec)

	code_dir = "../../ppxf_files/MILES_BASTI_CH_baseFe/" # directory where stellar templates are located
	pathname = os.path.join(code_dir, 'Mch1.30*.fits')
	miles = lib.miles(pathname, velscale_trunc, FWHM_gal)							# The stellar templates are reshaped below into a 2-dim array with each	
																						# spectrum as a column, however we save the original array dimensions,
																						# which are needed to specify the regularization dimensions.
	reg_dim = miles.templates.shape[1:]
	stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
	n_temps = stars_templates.shape[1]

	#for saving optimal templates for each bin
	n_bins = len(np.unique(binNum)) - 1 #subtract 1 for -1 bin
	optimal_templates_save = np.empty((stars_templates.shape[0], n_bins))

	lam_temp = miles.lam_temp
	lamRange_temp = [lam_temp[0], lam_temp[-1]]

	gas_templates, gas_names, line_wave = custom_util.emission_lines(miles.ln_lam_temp, wave_trunc_range, FWHM_gal, 
															tie_balmer=tie_balmer, limit_doublets=limit_doublets)

	n_gas = len(gas_names)
	n_balmer = 2
	n_other = n_gas - n_balmer  # forbidden lines contain "[*]"
	
	ngas_comp = 2 #max number of gas components
	ngas_comp_start = 1
	gas_templates_2comp = np.tile(gas_templates, 2) #need to make 1 comp and 2 comp templates
	gas_templates_1comp = gas_templates
	gas_names_1comp = np.asarray([a + f"_({p+1})" for p in range(ngas_comp_start) for a in gas_names])
	gas_names = np.asarray([a + f"_({p+1})" for p in range(ngas_comp) for a in gas_names])

	line_wave = np.tile(line_wave, ngas_comp)

	if single_gascomp == False:
		for num in range(1, ngas_comp+1):
			save_dict[f'balmer_({num})_vel'] = []
			save_dict[f'balmer_({num})_vel_error'] = []
			save_dict[f'balmer_({num})_sig'] = []
			save_dict[f'balmer_({num})_sig_error'] = []
			save_dict[f'forbidden_({num})_vel'] = []
			save_dict[f'forbidden_({num})_vel_error'] = []
			save_dict[f'forbidden_({num})_sig'] = []
			save_dict[f'forbidden_({num})_sig_error'] = []

	elif single_gascomp == True:
		for num in range(1, ngas_comp+1):
			save_dict[f'gas_({num})_vel'] = []
			save_dict[f'gas_({num})_vel_error'] = []
			save_dict[f'gas_({num})_sig'] = []
			save_dict[f'gas_({num})_sig_error'] = []


	for ggas in gas_names:
		ncomp_num = ggas[-2]
		gas_name = ggas[:-4]
		if gas_name == 'HeI5876':
			continue
		save_dict[ggas+'_flux'] = [] #flux is measured flux from ppxf
		save_dict[ggas+'_flux_error'] = []
		save_dict[ggas+'_amplitude'] = [] #amplitude is max flux in gas bestfit - so includes all components
		save_dict[ggas+'_ANR'] = [] #amplitude / mad_std_residuals
		save_dict[ggas+'_comp_amp'] = [	] #max flux only in the relevant component

		if gas_name == '[NII]6583_d':
			save_dict[f'[NII]6548_({ncomp_num})_amplitude'] = []
			save_dict[f'[NII]6548_({ncomp_num})_ANR'] = []

	#load in previous fit files for later use:
	prev_dict = pd.read_csv(prev_fit_path)
	prev_templates_path = prev_fit_path[:-4] + '_templates.npy'
	prev_temps = np.load(prev_templates_path)

	#velocity difference between templates and data, templates start around 3500 and data starts closer to 4500
	dv = c*(np.log(miles.lam_temp[0]/wave_trunc[0])) # eq.(8) of Cappellari (2017)

	loop_list = np.unique(binNum)
	if individual_bin is not None:
		loop_list = [individual_bin]

	for bn in loop_list: 
		if bn >= 0: #skip nan bins with binNum == -1
			print('\n============================================================================')
			print('binNum: {}'.format(bn))
			save_dict['bin_num'].append(bn)
			b_loc = np.where(binNum == bn)[0]
			x_loc = x[b_loc]
			y_loc = y[b_loc]

			spectra = cube_trunc[:,y_loc,x_loc]
			err_spectra = error_cube_trunc[:,y_loc,x_loc]

			n_spec = spectra.shape[1] #don't need n_spec to run ppxf
			print('n_spec in bin: ', n_spec)

			
			#also use previous optimal stellar template if fitting gas
			#choose starting values - just use galaxy velocity if no previous fit included

			#if previous fit data supplied, use previous starting values
			bin_df = prev_dict.loc[prev_dict['bin_num'] == bn]
			star_vel = bin_df['star_vel'].values[0]
			star_sig = bin_df['star_sig'].values[0]

			start_vals  = [star_vel, star_sig]
			start_vals_ncomp = [star_vel, star_sig*2] #for higher components, start with 2*stellar sigma width

			
			if single_gascomp == True:
				component = [0] + [1]*n_gas

				moments = np.repeat(2, ngas_comp_start + 1)
				moments[0] = -2
				start = np.concatenate((np.tile(start_vals, (2,1)), np.tile(start_vals_ncomp, ((ngas_comp_start-1), 1)))) #first 2 are stellar and first gas, then add extra

			else:
				# Assign component=0 to the stellar templates, component=1 to the Balmer
				# gas emission lines templates and component=2 to the other lines.
				component = [0] + [1]*n_balmer + [2]*n_other
	
				moments = np.repeat(2, ngas_comp_start*2 + 1)
				moments[0] = -2

				start = np.concatenate((np.tile(start_vals, (3,1)), np.tile(start_vals_ncomp, ((ngas_comp_start-1)*2, 1)))) #first 3 are stellar and first gas, then add extra

			gas_component = np.array(component) > 0  # gas_component=True for gas 
			
			# setting bounds
			bounds = None
			
			#use previous fit path to locate previous optimal templates file
			opt_temp = prev_temps[:,bn]

			# Combines the stellar and gaseous templates into a single array.
			# During the PPXF fit they will be assigned a different kinematic COMPONENT value

			templates_1comp = np.column_stack([opt_temp, gas_templates_1comp])
			templates_1comp /= np.median(templates_1comp)
			templates_2comp = np.column_stack([opt_temp, gas_templates_2comp])
			templates_2comp /= np.median(templates_2comp)

			### take mean of spectra
			gal_lin = np.nansum(spectra, axis=1)/n_spec
			noise_lin = np.sqrt(np.nansum(np.abs(err_spectra), axis=1))/n_spec #add error spectra in quadrature, err is variance so no need to square

			#noise_lin = noise_lin/np.nanmedian(noise_lin)
			noise_lin[np.isinf(noise_lin)] = np.nanmedian(noise_lin)
			noise_lin[noise_lin == 0] = np.nanmedian(noise_lin)

			galaxy, logLam, velscale = util.log_rebin(wave_trunc_range, gal_lin)
			#log_noise, logLam_noise, velscale_noise = util.log_rebin(lamRange, noise_lin) # I don't think this needs to be log-rebinned also

			maskreg = (5880,5950) #galactic and extragalactic NaD in this window, observed wavelength
			reg1 = np.where(np.exp(logLam) < maskreg[0])
			reg2 = np.where(np.exp(logLam) > maskreg[1])
			good_pixels = np.concatenate((reg1, reg2), axis=1)
			good_pixels = good_pixels.reshape(good_pixels.shape[1])

			ncomp_final = 1 #updates to 2 if S/N and KS criteria for refitting are met

			#CALLING PPXF HERE!
			t = clock()
			pp = ppxf(templates_1comp, galaxy, noise_lin, velscale, start,
						plot=False, moments=moments, degree= adegree, mdegree=mdegree, vsyst=dv,
						clean=False, lam=np.exp(logLam)/(1+z),
						component=component, gas_component=gas_component, bounds=bounds, #constr_kinem=constr_kinem,
						gas_names=gas_names_1comp, goodpixels = good_pixels, global_search=globsearch)

			bestfit = pp.bestﬁt
			residuals = galaxy - bestfit

			param0 = pp.sol
			error = pp.error

			mad_std_residuals = mad_std(residuals, ignore_nan=True)    
			med_galaxy = np.nanmedian(galaxy) #this will stand in for signal probably will be ~1
			SN_wMadStandardDev = med_galaxy/mad_std_residuals 
			print('S/N w/ mad_std: '+str(SN_wMadStandardDev))

			vel_comp0 = param0[0][0]
			vel_error_comp0 = error[0][0]*np.sqrt(pp.chi2)
			sig_comp0 = param0[0][1]
			sig_error_comp0 = error[0][1]*np.sqrt(pp.chi2)


			gas_bestfit = pp.gas_bestﬁt
			stellar_fit = bestfit - gas_bestfit

			gas_templates_fit = pp.gas_bestfit_templates

			param1 = pp.gas_flux
			param2 = pp.gas_flux_error

			wave_unexp = np.exp(logLam)
			Halpha_anr = ANR(None, 'Halpha_(1)', Ha, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp0)

			if plot_every > 0 and bn % plot_every == 0:
				output_ppxf_fit_plot(plot_name, pp, good_pixels, logLam, vel_comp0, z, ncomp_final, n_spec,
									bn, mad_std_residuals, SN_wMadStandardDev, fit_gas=True)

			if Halpha_anr >= 10:

				KS_result = residual_fit_test(plot_name, pp, residuals, bn, Halpha_anr, logLam, z, ncomp_final, n_spec, saveplot=True)

				if KS_result.pvalue < 0.01:
					ncomp_final = 2 #re-fit with 2 components


			

			if ncomp_final == 2:
				# changing ppxf parameters to prep for 2 gas component fit

				#start_vals = [param0[1][0], param0[1][1]] #previous fit parameters as new start parameters
				#start_vals_ncomp[0] -= 200 #add some velocity offset for the 2nd component
				
				prev_fit_start_1 = [param0[1][0]+50, param0[1][1]]
				prev_fit_start_2 = [param0[1][0]-50, param0[1][1]]

				if single_gascomp == True:
					start = np.array((start_vals, prev_fit_start_1, prev_fit_start_2)) #first 2 are stellar and first gas, then add extra
				else:
					start = np.array((start_vals, prev_fit_start_1, prev_fit_start_2, prev_fit_start_1, prev_Fit_start_2)) #first 3 are stellar and first gas, then add extra

				gas_bounds = [[star_vel-600, star_vel+600], [20, 1000]]
				bounds = [gas_bounds]
				for n in range(ngas_comp):
					bounds.append(gas_bounds)
					if single_gascomp == False: #if splitting balmer, then need to add another set of bounds
						bounds.append(gas_bounds)

				if single_gascomp == True:
					component = [0] + [1]*n_gas + [2]*n_gas
					moments = np.repeat(2, ncomp_final + 1)
					moments[0] = -2

				else:
					component = [0] + [1]*n_balmer + [2]*n_other + [3]*n_balmer + [4]*n_other
					moments = np.repeat(2, ncomp_final*2 + 1)
					moments[0] = -2

				gas_component = np.array(component) > 0  # gas_component=True for gas 


				pp = ppxf(templates_2comp, galaxy, noise_lin, velscale, start,
						plot=False, moments=moments, degree= adegree, mdegree=mdegree, vsyst=dv,
						clean=False, lam=np.exp(logLam)/(1+z),
						component=component, gas_component=gas_component, bounds=bounds, #constr_kinem=constr_kinem,
						gas_names=gas_names, goodpixels = good_pixels, global_search=globsearch)

				if pp.chi2 > 8:
					print('BAD FIT, re-trying with global_search=True')

					prev_fit_start_1 = [param0[1][0], param0[1][1]]
					prev_fit_start_2 = [param0[1][0]-200, param0[1][1]]
					if single_gascomp == True:
						start = np.array((start_vals, prev_fit_start_1, prev_fit_start_2)) #first 2 are stellar and first gas, then add extra
					else:
						start = np.array((start_vals, prev_fit_start_1, prev_fit_start_2, prev_fit_start_1, prev_Fit_start_2)) #first 3 are stellar and first gas, then add extra

					pp = ppxf(templates_2comp, galaxy, noise_lin, velscale, start,
						plot=False, moments=moments, degree= adegree, mdegree=mdegree, vsyst=dv,
						clean=False, lam=np.exp(logLam)/(1+z),
						component=component, gas_component=gas_component, bounds=bounds, #constr_kinem=constr_kinem,
						gas_names=gas_names, goodpixels = good_pixels, global_search=False)#True)

				bestfit = pp.bestﬁt
				residuals = galaxy - bestfit

				param0 = pp.sol
				error = pp.error

				mad_std_residuals = mad_std(residuals, ignore_nan=True)    
				med_galaxy = np.nanmedian(galaxy) #this will stand in for signal probably will be ~1
				SN_wMadStandardDev = med_galaxy/mad_std_residuals 
				
				print('S/N w/ mad_std: '+str(SN_wMadStandardDev))

				vel_comp0 = param0[0][0]
				vel_error_comp0 = error[0][0]*np.sqrt(pp.chi2)
				sig_comp0 = param0[0][1]
				sig_error_comp0 = error[0][1]*np.sqrt(pp.chi2)


				gas_bestfit = pp.gas_bestﬁt
				stellar_fit = bestfit - gas_bestfit

				gas_templates_fit = pp.gas_bestfit_templates

				param1 = pp.gas_flux
				param2 = pp.gas_flux_error

				if plot_every > 0 and bn % plot_every == 0:
					output_ppxf_fit_plot(plot_name, pp, good_pixels, logLam, vel_comp0, z, ncomp_final, n_spec,
										bn, mad_std_residuals, SN_wMadStandardDev, fit_gas=True)

				OI_anr = ANR(None, 'OI_(1)', OI, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp0)
				SII_anr = ANR(None, 'SII_(1)', SII1, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp0)
				OIII_anr = ANR(None, 'OIII_(1)', SII1, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp0)

				if Halpha_anr >= 10:
					KS_result = residual_fit_test(plot_name, pp, residuals, bn, Halpha_anr, logLam, z, ncomp_final, n_spec, saveplot=True)
					KS_result = residual_fit_test(plot_name, pp, residuals, bn, OI_anr, logLam, z, ncomp_final, n_spec, saveplot=True, line_region='OI')
					KS_result = residual_fit_test(plot_name, pp, residuals, bn, SII_anr, logLam, z, ncomp_final, n_spec, saveplot=True, line_region='SII')
					KS_result = residual_fit_test(plot_name, pp, residuals, bn, OIII_anr, logLam, z, ncomp_final, n_spec, saveplot=True, line_region='OIII')


			#adding values to save dictionary - same for 1 and 2 comp
			save_dict['SN_mad_STD'].append(SN_wMadStandardDev)
			save_dict['star_vel'].append(vel_comp0) 
			save_dict['star_vel_error'].append(vel_error_comp0)
			save_dict['star_sig'].append(sig_comp0)
			save_dict['star_sig_error'].append(sig_error_comp0)

			if single_gascomp == False:
				for num in np.arange(1,ngas_comp+1):

					if num > ncomp_final: #if adding for a higher component than was fit
						save_dict[f'balmer_({num})_vel'].append(np.nan)
						save_dict[f'balmer_({num})_vel_error'].append(np.nan)
						save_dict[f'balmer_({num})_sig'].append(np.nan)
						save_dict[f'balmer_({num})_sig_error'].append(np.nan)

						save_dict[f'forbidden_({num})_vel'].append(np.nan)
						save_dict[f'forbidden_({num})_vel_error'].append(np.nan)
						save_dict[f'forbidden_({num})_sig'].append(np.nan)
						save_dict[f'forbidden_({num})_sig_error'].append(np.nan)

					else:
						ind = num*2 - 1
						vel_comp1 = param0[ind][0]
						vel_error_comp1 = error[ind][0]*np.sqrt(pp.chi2)
						sig_comp1 = param0[ind][1]
						sig_error_comp1 = error[ind][1]*np.sqrt(pp.chi2)

						vel_comp2 = param0[ind+1][0]
						vel_error_comp2 = error[ind+1][0]*np.sqrt(pp.chi2)
						sig_comp2 = param0[ind+1][1]
						sig_error_comp2 = error[ind+1][1]*np.sqrt(pp.chi2)

						save_dict[f'balmer_({num})_vel'].append(vel_comp1)
						save_dict[f'balmer_({num})_vel_error'].append(vel_error_comp1)
						save_dict[f'balmer_({num})_sig'].append(sig_comp1)
						save_dict[f'balmer_({num})_sig_error'].append(sig_error_comp1)

						save_dict[f'forbidden_({num})_vel'].append(vel_comp2)
						save_dict[f'forbidden_({num})_vel_error'].append(vel_error_comp1)
						save_dict[f'forbidden_({num})_sig'].append(sig_comp2)
						save_dict[f'forbidden_({num})_sig_error'].append(sig_error_comp1)

			elif single_gascomp == True:
				for num in range(1, ngas_comp+1):

					if num > ncomp_final:
						save_dict[f'gas_({num})_vel'].append(np.nan)
						save_dict[f'gas_({num})_vel_error'].append(np.nan)
						save_dict[f'gas_({num})_sig'].append(np.nan)
						save_dict[f'gas_({num})_sig_error'].append(np.nan)
					else:
						vel_comp1 = param0[num][0]
						vel_error_comp1 = error[num][0]*np.sqrt(pp.chi2)
						sig_comp1 = param0[num][1]
						sig_error_comp1 = error[num][1]*np.sqrt(pp.chi2)

						save_dict[f'gas_({num})_vel'].append(vel_comp1)
						save_dict[f'gas_({num})_vel_error'].append(vel_error_comp1)
						save_dict[f'gas_({num})_sig'].append(sig_comp1)
						save_dict[f'gas_({num})_sig_error'].append(sig_error_comp1)

			for idx, ggas in enumerate(gas_names):
				ncomp_num = ggas[-2]
				gas_name = ggas[:-4]

				if gas_name == 'HeI5876':
					continue

				if int(ncomp_num) > ncomp_final:
					save_dict[ggas+'_flux'].append(np.nan)
					save_dict[ggas+'_flux_error'].append(np.nan)
					save_dict[ggas+'_comp_amp'].append(np.nan)
					save_dict[ggas+'_amplitude'].append(np.nan)
					save_dict[ggas+'_ANR'].append(np.nan)

					if gas_name == '[NII]6583_d':
						save_dict[f'[NII]6548_({ncomp_num})_amplitude'].append(np.nan)
						save_dict[f'[NII]6548_({ncomp_num})_ANR'].append(np.nan)

				else:
					gas_temp_idx = gas_templates_fit[:,idx]

					save_dict[ggas+'_flux'].append(param1[idx])
					save_dict[ggas+'_flux_error'].append(param2[idx])
					save_dict[ggas+'_comp_amp'].append(np.nanmax(gas_temp_idx))

					if single_gascomp == True:
						vel_comp_bl = save_dict[f'gas_({ncomp_num})_vel'][-1]
						vel_comp_fb = save_dict[f'gas_({ncomp_num})_vel'][-1]
					else:
						vel_comp_bl = save_dict[f'balmer_({num})_vel'][-1]
						vel_comp_fb = save_dict[f'forbidden_({num})_vel'][-1]

					if gas_name == 'Hbeta':
						emline = Hb
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_bl)
					if gas_name == 'Halpha':
						emline = Ha
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_bl)
						print('A/rN of Halpha: '+str(save_dict['Halpha_(1)_ANR'][-1]))
					if gas_name == '[SII]6731_d1':
						emline = SII1
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[SII]6731_d2':
						emline = SII2
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[OIII]5007_d':
						emline = OIII
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[OI]6300_d':
						emline = OI
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[NII]6583_d':
						emline = NII
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
						emline2 = NII1
						ANR(save_dict, f'[NII]6548_({ncomp_num})', emline2, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb) 
					if gas_name == '[NI]5201':
						emline = NI
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[NII]5756':
						emline = NII5756
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[OII]7322':
						emline = OII1
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)
					if gas_name == '[OII]7333':
						emline = OII2
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp_fb)

			fit_chisq = (pp.chi2 - 1)*galaxy.size
			print('============================================================================')
			print('Desired Delta Chi^2: %.4g' % np.sqrt(2*galaxy.size))
			print('Current Delta Chi^2: %.4g' % (fit_chisq))
			print('Elapsed time in PPXF: %.2f s' % (clock() - t))
			print('============================================================================')

	return save_dict


def run_stellar_fit(runID, cube_path, prev_vmap_path=None, plot_every=0):
	#run stellar fit, optionally also fit gas or mask it out
	#runID - name identifier for the run (e.g. Nov25_1comp)
	#cube_path - path to data cube
	#vorbin_path - path to voronoi bin text file
	#prev_cmap_path - if supplied, uses initial fitting conditions

	gal_out_dir = 'ppxf_output/'

	# Reading in the cube
	hdu = fits.open(cube_path)
	cube = hdu[1].data
	error_cube = hdu[2].data
	h1 = hdu[1].header

	#ppxf parameters
	moments = 2
	adegree = 10     		# Additive polynomial used to correct the template continuum shape during the fit.
							# Degree = 10 is the suggested order for stellar populations.
	mdegree = -1			# Multiplicative polynomial used to correct the template continuum shape during the fit.
							# Mdegree = -1 tells fit to not include the multiplicative polynomial.

	wave_lam = np.array(h1['CRVAL3']+(np.arange(0, h1['NAXIS3'])*h1['CD3_3']))

	plot_name = f'stellarfit_{runID}'
	csv_save_name = f'{gal_out_dir}stellarfit_{runID}_nobin.csv'

	run_dict, opt_temps_save = ppxf_fit_stellar(cube, error_cube, moments, adegree, mdegree, wave_lam,
						plot_every=plot_every, plot_name=plot_name, prev_vmap_path=prev_vmap_path)
	run_dict_df = pd.DataFrame.from_dict(run_dict)
	run_dict_df.to_csv(csv_save_name, index=False, header=True)

	print(f'Saved ppxf stellar fit csv to {csv_save_name}')

	opt_temps_save_path = csv_save_name[:-4]+'_templates.npy'
	np.save(opt_temps_save_path, opt_temps_save)

	print(f'Saved optimal templates to {opt_temps_save_path}')


def run_gas_fit(runID, cube_path, vorbin_path, prev_fit_path, plot_every=0, ngas_comp=1, globsearch=False, single_gascomp=False, individual_bin=None):
	#run gas fit
	#runID - name identifier for the run (e.g. Nov25_1comp)
	#cube_path - path to data cube
	#vorbin_path - path to voronoi bin text file
	#prev_fit_path - location of stellar fit for stellar kinematics
	
	gal_out_dir = 'ppxf_output/'

	# Reading in the cube
	hdu = fits.open(cube_path)
	cube = hdu[1].data
	error_cube = hdu[2].data
	h1 = hdu[1].header

	#galaxy parameters
	z = 0.007214         # NGC 1266 redshift, from SIMBAD
	galv = np.log(z+1)*c # estimate of galaxy's velocity

	#ppxf parameters
	moments = np.repeat(2, ngas_comp*2 + 1)
	moments[0] = -2

	if single_gascomp == True:
		moments = np.repeat(2, ngas_comp + 1)
		moments[0] = -2


	adegree = -1       # Additive polynomial used to correct the template continuum shape during the fit.
                       		# Degree = 10 is the suggested order for stellar populations.
	mdegree = 10     	# Multiplicative polynomial used to correct the template continuum shape during the fit.
                       		# Mdegree = -1 tells pPXF fit to not include the multiplicative polynomial.
	limit_doublets = True	# If True, limit the fluxes for the the [OII] and [SII]λ6716, λ6731 doublet, to be between the values
							# permitted by atomic physics
	tie_balmer = False		# Specifying that the Balmer series to be input as separate templates.

	wave_lam = np.array(h1['CRVAL3']+(np.arange(0, h1['NAXIS3'])*h1['CD3_3']))

	plot_name = f'gasfit_{runID}'
	csv_save_name = f'{gal_out_dir}gasfit_{runID}.csv'

	run_dict = ppxf_fit(cube, error_cube, vorbin_path, moments, adegree, mdegree, wave_lam, limit_doublets, tie_balmer, plot_every=plot_every,
						plot_name=plot_name, fit_gas=True, prev_fit_path=prev_fit_path, ngas_comp=ngas_comp, globsearch=globsearch,
						single_gascomp=single_gascomp, individual_bin=individual_bin)

	run_dict_df = pd.DataFrame.from_dict(run_dict)
	run_dict_df.to_csv(csv_save_name, index=False, header=True)

	print(f'Saved ppxf gas fit csv to {csv_save_name}')


def run_iterative_gas_fit(runID, cube_path, vorbin_path, prev_fit_path, plot_every=100, globsearch=False, single_gascomp=True, individual_bin=None):
	#run gas fit
	#runID - name identifier for the run (e.g. Nov25_1comp)
	#cube_path - path to data cube
	#vorbin_path - path to voronoi bin text file
	#prev_fit_path - location of stellar fit for stellar kinematics
	
	gal_out_dir = 'ppxf_output/'

	# Reading in the cube
	hdu = fits.open(cube_path)
	cube = hdu[1].data
	error_cube = hdu[2].data
	h1 = hdu[1].header

	#galaxy parameters
	z = 0.007214         # NGC 1266 redshift, from SIMBAD
	galv = np.log(z+1)*c # estimate of galaxy's velocity

	adegree = -1       # Additive polynomial used to correct the template continuum shape during the fit.
                       		# Degree = 10 is the suggested order for stellar populations.
	mdegree = 10     	# Multiplicative polynomial used to correct the template continuum shape during the fit.
                       		# Mdegree = -1 tells pPXF fit to not include the multiplicative polynomial.
	limit_doublets = True	# If True, limit the fluxes for the the [OII] and [SII]λ6716, λ6731 doublet, to be between the values
							# permitted by atomic physics
	tie_balmer = False		# Specifying that the Balmer series to be input as separate templates.

	wave_lam = np.array(h1['CRVAL3']+(np.arange(0, h1['NAXIS3'])*h1['CD3_3']))

	plot_name = f'gasfit_iter_{runID}'
	csv_save_name = f'{gal_out_dir}gasfit_iter_{runID}.csv'

	run_dict = ppxf_iterative_fit(cube, error_cube, vorbin_path, adegree, mdegree, wave_lam, limit_doublets, tie_balmer, plot_every=plot_every,
						plot_name=plot_name, prev_fit_path=prev_fit_path, globsearch=globsearch,
						single_gascomp=single_gascomp, individual_bin=individual_bin)


	run_dict_df = pd.DataFrame.from_dict(run_dict)
	run_dict_df.to_csv(csv_save_name, index=False, header=True)

	print(f'Saved ppxf gas fit csv to {csv_save_name}')


cube_path = "../ngc1266_data/MUSE/ADP.2019-02-25T15 20 26.375.fits"
prev_vmap_path = '/Users/jotter/highres_PSBs/ngc1266_data/MUSE/maps/ngc1266_ppxf_Feb23_iter_gs_maps.fits'
run_id_stellar = 'Mar23'

run_stellar_fit(run_id_stellar, cube_path, prev_vmap_path=prev_vmap_path, plot_every=100)


#run_gas_fit(run_id_gas, cube_path, vorbin_path, prev_fit_path=f'ppxf_output/stellarfit_{run_id_stellar}.csv', plot_every=1, ngas_comp=2,
#			single_gascomp=True, globsearch=False, individual_bin=bin_num)
#run_iterative_gas_fit(run_id_gas, cube_path, vorbin_path, prev_fit_path=f'ppxf_output/stellarfit_{run_id_stellar}.csv', plot_every=100,
#					single_gascomp=True, globsearch=False, individual_bin=bin_num)



