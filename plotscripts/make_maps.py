## goal of this script:
# create fits file of velocity maps, flux maps, etc from MUSE fitting using Travis' code (../traviscode/fit_muse_edit.py)

import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.constants as const
import astropy.units as u


run_num = 1
#run_name = 'Ha_contsub'
#flname = 'halpha_out'
#run_name = 'Oiii_contsub'
#flname = 'oiii_out'
#run_name = 'Hb_contsub'
#flname = 'hbeta_out'
#run_name = 'Sii_contsub'
#flname = 'sii_out'
#run_name = 'Oi_contsub'
#flname = 'oi_out'
run_name = 'Ni_contsub'
flname = 'ni_out'	

#fit_table_path = f'/Users/jotter/highres_PSBs/ngc1266_MUSE/output/ngc1266_{run_name}_run{run_num}/NGC1266_Ha_contsub_header.txt'
fit_table_path = f'/Users/jotter/highres_PSBs/ngc1266_MUSE/output/ngc1266_{run_name}_run{run_num}/NGC1266_{flname}.txt'

#restwave = 6562.8 * u.Angstrom #Halpha
#restwave = 5006.84 * u.Angstrom #Oiii
#restwave = 4861.3 * u.Angstrom #Hbeta
#restwave = 6716.4 * u.Angstrom #shorter wavelength of the two sii lines
restwave = 6300 * u.Angstrom

n_free_lines = 1 #1 for oiii, Hbeta, 2 for hanii, sii

fit_tab = Table.read(fit_table_path, format='ascii')

fitting_bounds_x = (105,205)
fitting_bounds_y = (105,205)

map_shape = (fitting_bounds_x[1] - fitting_bounds_x[0], fitting_bounds_y[1] - fitting_bounds_y[0])

sort_components = True

file_num_map = np.full(map_shape, np.nan)
ncomp_map = np.full(map_shape, np.nan)
coordx_map = np.full(map_shape, np.nan)
coordy_map = np.full(map_shape, np.nan)

comp1_wave = np.full(map_shape, np.nan)
comp2_wave = np.full(map_shape, np.nan)
comp3_wave = np.full(map_shape, np.nan)

comp1_width = np.full(map_shape, np.nan)
comp2_width = np.full(map_shape, np.nan)
comp3_width = np.full(map_shape, np.nan)

comp1_fluxA = np.full(map_shape, np.nan)
comp2_fluxA = np.full(map_shape, np.nan)
comp3_fluxA = np.full(map_shape, np.nan)

if n_free_lines == 2:
	comp1_fluxB = np.full(map_shape, np.nan)
	comp2_fluxB = np.full(map_shape, np.nan)
	comp3_fluxB = np.full(map_shape, np.nan)


#loop through table to create maps
for i in range(len(fit_tab)):

	ncomp = fit_tab['ncomps'][i]
	coordx = fit_tab['coordx'][i] - 1# - 105
	coordy = fit_tab['coordy'][i] - 1# - 105
	filename = fit_tab['filename'][i]
	file_num = int(filename.split('.')[0])

	file_num_map[coordy, coordx] = file_num
	coordx_map[coordy, coordx] = coordx
	coordy_map[coordy, coordx] = coordy
	ncomp_map[coordy, coordx] = ncomp

	if ncomp > 0:
		comp1_wave[coordy, coordx] = fit_tab['wave_1'][i]
		comp1_width[coordy, coordx] = fit_tab['width_1'][i]
		comp1_fluxA[coordy, coordx] = fit_tab['flux_1_A'][i]
		if n_free_lines == 2:
			comp1_fluxB[coordy, coordx] = fit_tab['flux_1_B'][i]

		if ncomp > 1:
			comp2_wave[coordy, coordx] = fit_tab['wave_2'][i]
			comp2_width[coordy, coordx] = fit_tab['width_2'][i]
			comp2_fluxA[coordy, coordx] = fit_tab['flux_2_A'][i]
			if n_free_lines == 2:
				comp2_fluxB[coordy, coordx] = fit_tab['flux_2_B'][i]
			if ncomp > 2:
				comp3_wave[coordy, coordx] = fit_tab['wave_3'][i]
				comp3_width[coordy, coordx] = fit_tab['width_3'][i]
				comp3_fluxA[coordy, coordx] = fit_tab['flux_3_A'][i]
				if n_free_lines == 2:
					comp3_fluxB[coordy, coordx] = fit_tab['flux_3_B'][i]


wavetovel = u.doppler_optical(restwave)

z = 0.007214         # NGC 1266 redshift, from SIMBAD
galv = np.log(z+1)*const.c # estimate of galaxy's velocity

comp1_vel = (comp1_wave * u.Angstrom).to(u.km/u.s, equivalencies=wavetovel) - galv
comp2_vel = (comp2_wave * u.Angstrom).to(u.km/u.s, equivalencies=wavetovel) - galv
comp3_vel = (comp3_wave * u.Angstrom).to(u.km/u.s, equivalencies=wavetovel) - galv

comp1_sig_upper = ((comp1_wave + comp1_width/2) * u.Angstrom).to(u.km/u.s, equivalencies=wavetovel)
comp1_sig_lower = ((comp1_wave - comp1_width/2) * u.Angstrom).to(u.km/u.s, equivalencies=wavetovel)
comp1_sigma = comp1_sig_upper - comp1_sig_lower

comp2_sig_upper = ((comp2_wave + comp2_width/2) * u.Angstrom).to(u.km/u.s, equivalencies=wavetovel)
comp2_sig_lower = ((comp2_wave - comp2_width/2) * u.Angstrom).to(u.km/u.s, equivalencies=wavetovel)
comp2_sigma = comp2_sig_upper - comp2_sig_lower

comp3_sig_upper = ((comp3_wave + comp3_width/2) * u.Angstrom).to(u.km/u.s, equivalencies=wavetovel)
comp3_sig_lower = ((comp3_wave - comp3_width/2) * u.Angstrom).to(u.km/u.s, equivalencies=wavetovel)
comp3_sigma = comp3_sig_upper - comp3_sig_lower


if sort_components == True:

	combined_sigmas = np.array([comp1_sigma, comp2_sigma, comp3_sigma])
	combined_vels = np.array([comp1_vel, comp2_vel, comp3_vel])
	combined_sort_ind = np.argsort(combined_sigmas, axis=0)

	combined_sorted_sigmas = np.take_along_axis(combined_sigmas, combined_sort_ind, axis=0)
	combined_sorted_vels = np.take_along_axis(combined_vels, combined_sort_ind, axis=0)

	#this section is to make component 3 the intermediate width component rather than the widest
	#just swapping c2 and c3
	ncomp3_ind = np.where(ncomp_map == 3)
	old_c2_sigmas = combined_sorted_sigmas[1,:,:][ncomp3_ind]
	combined_sorted_sigmas[1,:,:][ncomp3_ind] = combined_sorted_sigmas[2,:,:][ncomp3_ind]
	combined_sorted_sigmas[2,:,:][ncomp3_ind] = old_c2_sigmas

	old_c2_vels = combined_sorted_vels[1,:,:][ncomp3_ind]
	combined_sorted_vels[1,:,:][ncomp3_ind] = combined_sorted_vels[2,:,:][ncomp3_ind]
	combined_sorted_vels[2,:,:][ncomp3_ind] = old_c2_vels

	#then make c3 the component with more extreme vel values than c1

	mid_c1_absvels = np.abs(combined_sorted_vels[0,:,:])
	mid_c3_absvels = np.abs(combined_sorted_vels[2,:,:])
	mid_c1_vels = combined_sorted_vels[0,:,:]
	mid_c3_vels = combined_sorted_vels[2,:,:]
	mid_combined_vels = np.array([mid_c1_vels, mid_c3_vels])
	mid_c1_sigmas = combined_sorted_sigmas[0,:,:]
	mid_c3_sigmas = combined_sorted_sigmas[2,:,:]
	mid_combined_sigmas = np.array([mid_c1_sigmas, mid_c3_sigmas])
	c1c3_vel_sort_ind = np.argsort(np.array([mid_c1_absvels, mid_c3_absvels]), axis=0)

	mid_sorted_vels = np.take_along_axis(mid_combined_vels, c1c3_vel_sort_ind, axis=0)
	mid_sorted_sigmas = np.take_along_axis(mid_combined_sigmas, c1c3_vel_sort_ind, axis=0)

	combined_sorted_vels[0,:,:][ncomp3_ind] = mid_sorted_vels[0,:,:][ncomp3_ind]
	combined_sorted_vels[2,:,:][ncomp3_ind] = mid_sorted_vels[1,:,:][ncomp3_ind]
	
	combined_sorted_sigmas[0,:,:][ncomp3_ind] = mid_sorted_sigmas[0,:,:][ncomp3_ind]
	combined_sorted_sigmas[2,:,:][ncomp3_ind] = mid_sorted_sigmas[1,:,:][ncomp3_ind]

	#final sorted maps
	comp1_sigma = combined_sorted_sigmas[0,:,:]
	comp2_sigma = combined_sorted_sigmas[1,:,:]
	comp3_sigma = combined_sorted_sigmas[2,:,:]

	comp1_vel = combined_sorted_vels[0,:,:]
	comp2_vel = combined_sorted_vels[1,:,:]
	comp3_vel = combined_sorted_vels[2,:,:]

if n_free_lines == 1:
	maps_list = [file_num_map, ncomp_map,
				comp1_vel, comp2_vel, comp3_vel, comp1_sigma, comp2_sigma, comp3_sigma,
				comp1_fluxA, comp2_fluxA, comp3_fluxA]
	maps_names = ['filename', 'ncomp', 'comp1_vel', 'comp2_vel', 'comp3_vel', 'comp1_sigma', 'comp2_sigma', 'comp3_sigma',
				'comp1_fluxA', 'comp2_fluxA', 'comp3_fluxA']

if n_free_lines == 2:
	maps_list = [file_num_map, ncomp_map,
				comp1_vel, comp2_vel, comp3_vel, comp1_sigma, comp2_sigma, comp3_sigma,
				comp1_fluxA, comp1_fluxB, comp2_fluxA, comp2_fluxB, comp3_fluxA, comp3_fluxB]
	maps_names = ['filename', 'ncomp', 'comp1_vel', 'comp2_vel', 'comp3_vel', 'comp1_sigma', 'comp2_sigma', 'comp3_sigma',
				'comp1_fluxA', 'comp1_fluxB', 'comp2_fluxA', 'comp2_fluxB', 'comp3_fluxA', 'comp3_fluxB']

maps_arr = np.empty((len(maps_list), fitting_bounds_x[1] - fitting_bounds_x[0], fitting_bounds_y[1] - fitting_bounds_y[0]))

fl = fits.open('/Users/jotter/highres_PSBs/ngc1266_data/MUSE/ADP.2019-02-25T15 20 26.375.fits')
header = fl[0].header
fl.close()

for j in range(len(maps_list)):
	maps_arr[j,:,:] = maps_list[j]
	header[f'DESC_{j}'] = maps_names[j]

hdu = fits.PrimaryHDU(data=maps_arr)
#fl[0].data = maps_arr
hdu.header = header

hdu.writeto(f'/Users/jotter/highres_PSBs/ngc1266_MUSE/output/fitsimages/NGC1266_maps_{run_name}_run{run_num}{"_sortmid" if sort_components == True else ""}.fits', overwrite=True)

