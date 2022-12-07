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
from astropy.stats import mad_std 
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


def ANR(gas_dict, gas_name, emline, gal_lam, gas_bestfit, mad_std_residuals,velocity):
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
	emline_amp = np.max(gas_bestfit[emline_loc])
	emline_ANR = emline_amp/mad_std_residuals 
	gas_dict[gas_name+'_amplitude'].append(emline_amp)
	gas_dict[gas_name+'_ANR'].append(emline_ANR)

def ppxf_fit(cube, error_cube, vorbin_path, moments, adegree, mdegree, wave_lam, limit_doublets=True, tie_balmer=False, plot_every=0, plot_name=None, prev_dict=None, fit_gas=True, ngas_comp=1):
	#cube - unbinned data cube
	#error_cube - unbinned error cube
	#vorbin_path - path to voronoi bin file
	#moments - ppxf moments for fitting
	#adegree - degree of additive poly
	#mdegree - degree of multiplicative poly
	#wave_lam - wavelength array
	#limit_doublets - if True, requires doublets to have ratios permitted by atomic physics
	#tie_balmer - if True, Balmer series are a single template
	#plot_every - if 0, don't output any plots, otherwise output a plot every [n] bins
	#plot_name - name for output plots
	#prev_dict - if supplied, use previous fit info
	#fit_gas - if True, fit gas emission lines, otherwise mask them and only do stellar continuum
	#ngas_comp - number of kinematic components to fit gas with

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

	lam_temp = miles.lam_temp
	lamRange_temp = [lam_temp[0], lam_temp[-1]]

	if fit_gas == True:
		gas_templates, gas_names, line_wave = custom_util.emission_lines(miles.ln_lam_temp, wave_trunc_range, FWHM_gal, 
																tie_balmer=tie_balmer, limit_doublets=limit_doublets)

		n_balmer = 2
		n_other = len(gas_names) - n_balmer  # forbidden lines contain "[*]"
		
		gas_templates_tile = np.tile(gas_templates, ngas_comp)
		gas_names_tile = np.asarray([a + f"_({p+1})" for p in range(ngas_comp) for a in gas_names])
		line_wave_tile = np.tile(line_wave, ngas_comp)

		# Combines the stellar and gaseous templates into a single array.
		# During the PPXF fit they will be assigned a different kinematic COMPONENT value
		templates = np.column_stack([stars_templates, gas_templates_tile])

		# Assign component=0 to the stellar templates, component=1 to the Balmer
		# gas emission lines templates and component=2 to the other lines.
		if ngas_comp == 1:
			component = [0]*n_temps + [1]*n_balmer + [3]*n_other
		if ngas_comp == 2:
			component = [0]*n_temps + [1]*n_balmer + [3]*n_other + [2]*n_balmer + [4]*n_other
		if ngas_comp == 3:
			component = [0]*n_temps + [1]*n_balmer + [4]*n_other + [2]*n_balmer + [5]*n_other + [3]*n_balmer + [6]*n_other
		gas_component = np.array(component) > 0  # gas_component=True for gas 

		save_dict['balmer_vel'] = []
		save_dict['balmer_vel_error'] = []
		save_dict['balmer_sig'] = []
		save_dict['balmer_sig_error'] = []
		save_dict['comp2_vel'] = []
		save_dict['comp2_vel_error'] = []
		save_dict['comp2_sig'] = []
		save_dict['comp2_sig_error'] = []

		for ggas in gas_names_tile:
			if ggas == 'HeI5876_(1)' or ggas == 'HeI5876_(2)':
				continue
			save_dict[ggas+'_flux'] = [] #flux is measured flux from ppxf
			save_dict[ggas+'_flux_error'] = []
			save_dict[ggas+'_amplitude'] = [] #amplitude is max flux in region around line
			save_dict[ggas+'_ANR'] = [] #amplitude / mad_std_residuals
			if ggas == '[NII]6583_d':
				save_dict['[NII]6548_amplitude'] = []
				save_dict['[NII]6548_ANR'] = []

	else:
		templates = stars_templates
		gas_component = None
		gas_names = None
		component = 0

	templates /= np.median(templates)

	#velocity difference between templates and data, templates start around 3500 and data starts closer to 4500
	dv = c*(np.log(miles.lam_temp[0]/wave_trunc[0])) # eq.(8) of Cappellari (2017)

	for bn in [1800]:#np.unique(binNum): #only looking at bin=0 for testing
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

			#if previous fit data supplied, use SN and starting values
			#choose starting values - just use galaxy velocity if no previous fit included
			if prev_dict is None:
				start_vals = [galv, 25]
			else:
				bin_df = prev_dict.loc[prev_dict['bin_num'] == 0]
				start_vals  = [bin_df['star_vel'][0], bin_df['star_sig'][0]]

			if fit_gas == True:
				start = np.tile(np.array(start_vals), (ngas_comp*2 +1,1))
			else:
				start = start_vals

			### just sum spectra
			gal_lin = np.nansum(spectra, axis=1)
			noise_lin = np.sqrt(np.nansum(np.abs(err_spectra), axis=1)) #add error spectra in quadrature, err is variance so no need to square

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
						clean=False, #regul= 1/0.1 , reg_dim=reg_dim, #lam=wave_lam_rest, lam_temp=lam_temp,
						component=component, gas_component=gas_component,
						gas_names=gas_names_tile, goodpixels = good_pix_total)

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

			else:
				vel_comp0 = param0[0][0]
				vel_error_comp0 = error[0][0]*np.sqrt(pp.chi2)
				sig_comp0 = param0[0][1]
				sig_error_comp0 = error[0][1]*np.sqrt(pp.chi2)

			save_dict['star_vel'].append(np.round(vel_comp0)) 
			save_dict['star_vel_error'].append(np.round(vel_error_comp0, decimals=3))
			save_dict['star_sig'].append(np.round(sig_comp0))
			save_dict['star_sig_error'].append(np.round(sig_error_comp0, decimals=3))

			if fit_gas == True:
				gas_bestfit = pp.gas_bestﬁt
				stellar_fit = bestfit - gas_bestfit

				param1 = pp.gas_flux
				param2 = pp.gas_flux_error

				vel_comp1 = param0[1][0]
				vel_error_comp1 = error[1][0]*np.sqrt(pp.chi2)
				sig_comp1 = param0[1][1]
				sig_error_comp1 = error[1][1]*np.sqrt(pp.chi2)

				vel_comp2 = param0[2][0]
				vel_error_comp2 = error[2][0]*np.sqrt(pp.chi2)
				sig_comp2 = param0[2][1]
				sig_error_comp2 = error[2][1]*np.sqrt(pp.chi2)

				save_dict['balmer_vel'].append(np.round(vel_comp1))
				save_dict['balmer_vel_error'].append(np.round(vel_error_comp1, decimals=10))
				save_dict['balmer_sig'].append(np.round(sig_comp1))
				save_dict['balmer_sig_error'].append(np.round(sig_error_comp1, decimals=10))

				save_dict['comp2_vel'].append(np.round(vel_comp2))
				save_dict['comp2_vel_error'].append(np.round(vel_error_comp1, decimals=10))
				save_dict['comp2_sig'].append(np.round(sig_comp2))
				save_dict['comp2_sig_error'].append(np.round(sig_error_comp1, decimals=10))

				wave_unexp = np.exp(logLam)

				for idx, ggas in enumerate(gas_names_tile):
					if ggas == 'HeI5876_(1)' or ggas == 'HeI5876_(2)':
						continue
					save_dict[ggas+'_flux'].append(param1[idx])
					save_dict[ggas+'_flux_error'].append(param2[idx])
					if ggas == 'Hbeta':
						emline = Hb
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp1)
					if ggas == 'Halpha':
						emline = Ha
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp1)
						print('A/rN of Halpha: '+str(save_dict['Halpha_ANR'][-1]))
					if ggas == '[SII]6731_d1':
						emline = SII1
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp2)
					if ggas == '[SII]6731_d2':
						emline = SII2
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp2)
					if ggas == '[OIII]5007_d':
						emline = OIII
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp2)
					if ggas == '[OI]6300_d':
						emline = OI
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp2)
					if ggas == '[NII]6583_d':
						emline = NII
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp2)
						emline2 = NII1
						ANR(save_dict, '[NII]6548', emline2, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp2) 
					if ggas == '[NI]5201':
						emline = NI
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp2)
					if ggas == '[NII]5756':
						emline = NII5756
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp2)
					if ggas == '[OII]7322':
						emline = OII1
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp2)
					if ggas == '[OII]7333':
						emline = OII2
						ANR(save_dict, ggas, emline, wave_unexp, gas_bestfit, mad_std_residuals, vel_comp2)

			fit_chisq = (pp.chi2 - 1)*galaxy.size
			print('============================================================================')
			print('Desired Delta Chi^2: %.4g' % np.sqrt(2*galaxy.size))
			print('Current Delta Chi^2: %.4g' % (fit_chisq))
			print('Elapsed time in PPXF: %.2f s' % (clock() - t))
			print('============================================================================')


			if plot_every > 0 and bn % plot_every == 0:
				print(f'Saving ppxf plot for bin number {bn}')
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

				residuals[masked_ind] = np.nan

				fig.text(0.05, 0.97, f'Bin {bn}. Number of Spectra in Bin: {n_spec}', fontsize = 12)
				fig.text(0.05, 0.93, f'Mean abs. dev. of residuals: {np.round(mad_std_residuals,1)}, S/N: {int(np.round(SN_wMadStandardDev,0))}')
				fig.text(0.45, 0.93, f'Chi-Squared: {int(np.round(fit_chisq,-1))}')

				bin_stell_vel = vel_comp0 - z*c
				fig.text(0.7, 0.93, f'Bin stellar velocity: {int(np.round(bin_stell_vel,0))} km/s')

				fig.text(0.03, 0.3, r'Flux (10$^{-20}$ erg/s/cm$^2$/Å)', fontsize = 12, rotation=90)

				for bound_ind in range(len(mask_reg_upper)):
					ax1.axvspan(mask_reg_lower[bound_ind], mask_reg_upper[bound_ind], alpha=0.25, color='gray')
					ax2.axvspan(mask_reg_lower[bound_ind], mask_reg_upper[bound_ind], alpha=0.25, color='gray')
					ax3.axvspan(mask_reg_lower[bound_ind], mask_reg_upper[bound_ind], alpha=0.25, color='gray')

				if fit_gas == True:
					ax1.plot(wave_plot_rest,galaxy, c = 'k', linewidth = 2, label = 'MUSE', alpha = 0.25, zorder = 0)
					ax1.plot(wave_plot_rest,bestfit, 'red', lw = 1.8, label = 'pPXF Best Fit')
					ax1.plot(wave_plot_rest,gas_bestfit,'b', lw =1.5, label = 'Gas Best Fit')
					ax1.plot(wave_plot_rest,stellar_fit, 'darkorange', lw = 1.5, label = 'Stellar Fit')
					ax1.plot(wave_plot_rest,residuals, 'gD', ms = 1.,label = 'Residuals', alpha = 0.5)
					ax1.axhline(0, color='k')
					ax1.set_xlabel('Restframe Wavelength (Å)',fontsize = 12)
					ax1.legend(frameon = False, loc = 'upper left', fontsize = 'medium', ncol = 2 )

					ax2.plot(wave_plot_rest,galaxy, c = 'k', linewidth = 2, label = 'MUSE', alpha = 0.25, zorder = 0)
					ax2.plot(wave_plot_rest,bestfit, 'red', lw = 1.8, label = 'pPXF Best Fit')
					ax2.plot(wave_plot_rest,gas_bestfit,'b', lw =1.5, label = 'Gas Best Fit')
					ax2.plot(wave_plot_rest,stellar_fit, 'darkorange', lw = 1.5, label = 'Stellar Fit')
					ax2.plot(wave_plot_rest,residuals, 'gD', ms = 1.,label = 'Residuals', alpha = 0.5)
					ax2.axhline(0, color='k')
					#ax2.set_title(r'Zoom-in on H$\beta$', fontsize = 12)
					ax2.set_xlim(4700,5200)

					ax3.plot(wave_plot_rest,galaxy, c = 'k', linewidth = 2, label = 'MUSE', alpha = 0.25, zorder = 0)
					ax3.plot(wave_plot_rest,bestfit, 'red', lw = 1.8, label = 'pPXF Best Fit')
					ax3.plot(wave_plot_rest,gas_bestfit,'b', lw =1.5, label = 'Gas Best Fit')
					ax3.plot(wave_plot_rest,stellar_fit, 'darkorange', lw = 1.5, label = 'Stellar Fit')
					ax3.plot(wave_plot_rest,residuals, 'gD', ms = 1.,label = 'Residuals', alpha = 0.5)
					ax3.axhline(0, color='k')
					ax3.set_title(r'Zoom-in on H$\alpha$ and [NII]', fontsize = 12)
					ax3.set_yticklabels([])
					ax3.set_xlim(Ha-50,Ha+50)

				else:
					ax1.plot(wave_plot_rest,galaxy, c = 'k', linewidth = 2, label = 'MUSE', alpha = 0.25, zorder = 0)
					ax1.plot(wave_plot_rest,bestfit, 'red', lw = 1.8, label = 'pPXF Best Fit')
					ax1.plot(wave_plot_rest,residuals, 'gD', ms = 1.,label = 'Residuals', alpha = 0.5)
					ax1.axhline(0, color='k')
					ax1.set_xlabel('Restframe Wavelength (Å)',fontsize = 12)
					ax1.legend(frameon = False, loc = 'upper left', fontsize = 'medium', ncol = 2 )

					ax2.plot(wave_plot_rest,galaxy, c = 'k', linewidth = 2, label = 'MUSE', alpha = 0.25, zorder = 0)
					ax2.plot(wave_plot_rest,bestfit, 'red', lw = 1.8, label = 'pPXF Best Fit')
					ax2.plot(wave_plot_rest,residuals, 'gD', ms = 1.,label = 'Residuals', alpha = 0.5)
					ax2.axhline(0, color='k')
					
					ax2.set_title(r'Zoom-in on NaD', fontsize = 12)
					ax2.set_xlim(5800,6000)

					ax3.plot(wave_plot_rest,galaxy, c = 'k', linewidth = 2, label = 'MUSE', alpha = 0.25, zorder = 0)
					ax3.plot(wave_plot_rest,bestfit, 'red', lw = 1.8, label = 'pPXF Best Fit')
					ax3.plot(wave_plot_rest,residuals, 'gD', ms = 1.,label = 'Residuals', alpha = 0.5)
					ax3.axhline(0, color='k')
					ax3.set_title(r'Zoom-in on H$\alpha$ and [NII]', fontsize = 12)
					ax3.set_yticklabels([])
					ax3.set_xlim(6000,7000)

				full_plot_dir = f'{plot_dir}{plot_name}'
				plot_fl = f'{full_plot_dir}/ppxf_{"stellar" if fit_gas == False else "gas"}fit_bin{bn}.png'

				if os.path.exists(full_plot_dir) == False:
					os.mkdir(full_plot_dir)
				plt.savefig(plot_fl, dpi = 300) 
				plt.close()

				print(f'Saved ppxf plot to {plot_name}')

				##code to output the automated ppxf plot
				#pp.plot()
				#plot_fl_def = f'{full_plot_dir}/ppxf_{"stellar" if fit_gas == False else "gas"}fit_bin{bn}_defaultplot.png'
				#plt.savefig(plot_fl_def, dpi=300)

				plt.close()


	return save_dict


def run_stellar_fit(runID, cube_path, vorbin_path, fit_gas = False, prev_fit_path=None, plot_every=0):
	#run stellar fit, optionally also fit gas or mask it out
	#runID - name identifier for the run (e.g. Nov25_1comp)
	#cube_path - path to data cube
	#vorbin_path - path to voronoi bin text file
	#fit_gas - if True, fit gas emission lines simultaneously with stellar continuum
	#prev_fit_path - if supplied, uses S/N estimate from previous stellar fit

	gal_out_dir = 'ppxf_output/'

	# Reading in the cube
	hdu = fits.open(cube_path)
	cube = hdu[1].data
	error_cube = hdu[2].data
	h1 = hdu[1].header

	#ppxf parameters
	if fit_gas == False:
		moments = 2
		adegree = 10     		# Additive polynomial used to correct the template continuum shape during the fit.
								# Degree = 10 is the suggested order for stellar populations.
		mdegree = -1			# Multiplicative polynomial used to correct the template continuum shape during the fit.
								# Mdegree = -1 tells fit to not include the multiplicative polynomial.

	else:
		moments = [2,2,2]  
		adegree = 10
		mdegree = -1
	
	limit_doublets = True	# If True, limit the fluxes for the the [OIII]λ4959, λ5007 doublet, the [OI]λ6300, λ6364 doublet,
							# the [NII]λ6548, λ6583 doublet, and the [SII]λ6716, λ6731 doublet, to be between the values
							# permitted by atomic physics
	tie_balmer = False		# Specifying that the Balmer series to be input as separate templates.

	wave_lam = np.array(h1['CRVAL3']+(np.arange(0, h1['NAXIS3'])*h1['CD3_3']))

	if prev_fit_path == None:
		prev_dict = None

	else:
		prev_dict = pd.read_csv(prev_fit_path)


	plot_name = f'stellarfit_{runID}'
	csv_save_name = f'{gal_out_dir}stellarfit_{runID}.csv'

	run_dict = ppxf_fit(cube, error_cube, vorbin_path, moments, adegree, mdegree, wave_lam, limit_doublets, tie_balmer,
						plot_every=plot_every, plot_name=plot_name, fit_gas=fit_gas, prev_dict=prev_dict)

	run_dict_df = pd.DataFrame.from_dict(run_dict)
	#run_dict_df.to_csv(csv_save_name, index=False, header=True)

	print(f'Saved ppxf fit csv to {csv_save_name}')



def run_gas_fit(runID, cube_path, vorbin_path, prev_fit_path, plot_every=0, ngas_comp=1):
	#run stellar and gas fit
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

	adegree = -1       # Additive polynomial used to correct the template continuum shape during the fit.
                       		# Degree = 10 is the suggested order for stellar populations.
	mdegree = 10     	# Multiplicative polynomial used to correct the template continuum shape during the fit.
                       		# Mdegree = -1 tells pPXF fit to not include the multiplicative polynomial.
	limit_doublets = True	# If True, limit the fluxes for the the [OIII]λ4959, λ5007 doublet, the [OI]λ6300, λ6364 doublet,
							# the [NII]λ6548, λ6583 doublet, and the [SII]λ6716, λ6731 doublet, to be between the values
							# permitted by atomic physics
	tie_balmer = False		# Specifying that the Balmer series to be input as separate templates.

	wave_lam = np.array(h1['CRVAL3']+(np.arange(0, h1['NAXIS3'])*h1['CD3_3']))

	plot_name = f'gasfit_{runID}'
	csv_save_name = f'{gal_out_dir}gasfit_{runID}.csv'

	prev_dict = pd.read_csv(prev_fit_path)

	run_dict = ppxf_fit(cube, error_cube, vorbin_path, moments, adegree, mdegree, wave_lam, limit_doublets, tie_balmer, plot_every=100,
						plot_name=plot_name, fit_gas=True, prev_dict=prev_dict, ngas_comp=ngas_comp)
	run_dict_df = pd.DataFrame.from_dict(run_dict)
	#run_dict_df.to_csv(csv_save_name, index=False, header=True)

	print(f'Saved ppxf fit csv to {csv_save_name}')


cube_path = "../ngc1266_data/MUSE/ADP.2019-02-25T15 20 26.375.fits"
vorbin_path = "ppxf_output/NGC1266_voronoi_output_targetSN_10_2022Oct18.txt"
run_id = 'Nov_2comp'

#run_stellar_fit(run_id, cube_path, vorbin_path, fit_gas=False, prev_fit_path=None, plot_every=100)
run_gas_fit(run_id, cube_path, vorbin_path, prev_fit_path='ppxf_output/stellarfit_Nov_1comp.csv', plot_every=100, ngas_comp=2)





