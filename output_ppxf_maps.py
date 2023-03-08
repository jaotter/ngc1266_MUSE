import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table

from astropy.wcs import WCS


#load in data files

ppxf_dir = '/Users/jotter/highres_PSBs/ngc1266_MUSE/ppxf_output/'

#load in voronoi bin file
voronoi_bin_file = 'NGC1266_voronoi_output_targetSN_10_2022Oct18.txt'
x,y,binNum = np.loadtxt(ppxf_dir+voronoi_bin_file).T
x,y,binNum = x.astype(int), y.astype(int), binNum.astype(int)

binNum_2D = binNum.reshape((np.max(x)+1, np.max(y)+1)).T

#load in stellar fit
stellar_file = 'stellarfit_Dec22.csv'
stellar_tab = Table.read(ppxf_dir+stellar_file, format='csv')

#load in gas fit
gas_file = 'gasfit_Dec22_1comp_1gas.csv'
gas_tab = Table.read(ppxf_dir+gas_file, format='csv')

#load in gas fit 2comp
gas_file2 = 'gasfit_iter_Feb23_gs.csv'
gas_tab2 = Table.read(ppxf_dir+gas_file2, format='csv')

#cube file for WCS info
cube_file = '/Users/jotter/highres_PSBs/ngc1266_data/MUSE/ADP.2019-02-25T15 20 26.375.fits'
cube_fl = fits.open(cube_file)
cube_header = cube_fl[1].header
cube_wcs = WCS(cube_header).celestial
cube_data = cube_fl[1].data
cube_fl.close()

wave = cube_header['CRVAL3']+(np.arange(0, cube_header['NAXIS3'])*cube_header['CD3_3'])

z = 0.007214
c = 299792.5
ngc1266_vel = c*z


#turn table values into maps

Ha_flux = np.full(binNum_2D.shape, np.nan)
Nii_flux = np.full(binNum_2D.shape, np.nan)
Oiii_flux = np.full(binNum_2D.shape, np.nan)
Hb_flux = np.full(binNum_2D.shape, np.nan)
Oi_flux = np.full(binNum_2D.shape, np.nan)
Sii_flux = np.full(binNum_2D.shape, np.nan)

Ha_amplitude_1 = np.full(binNum_2D.shape, np.nan)
Ha_amplitude_2 = np.full(binNum_2D.shape, np.nan)

stellar_vel = np.full(binNum_2D.shape, np.nan)
stellar_sigma = np.full(binNum_2D.shape, np.nan)
gas_vel_c1 = np.full(binNum_2D.shape, np.nan)
gas_vel_c2 = np.full(binNum_2D.shape, np.nan)
gas_sig_c1 = np.full(binNum_2D.shape, np.nan)
gas_sig_c2 = np.full(binNum_2D.shape, np.nan)

Ha_snr = np.full(binNum_2D.shape, np.nan)

for bn in np.unique(binNum):
    if bn >= 0:
        bin_ind = np.where(binNum_2D == bn)
        stell_tab_ind = np.where(stellar_tab['bin_num'] == bn)[0]
        gas_tab_ind = np.where(gas_tab['bin_num'] == bn)[0]
        gas_tab2_ind = np.where(gas_tab2['bin_num'] == bn)[0]
        nspec = len(bin_ind[0])
        
        vstar = stellar_tab['star_vel'][stell_tab_ind]
        stellar_vel[bin_ind] = vstar
        stellar_sigma[bin_ind] = stellar_tab['star_sig'][stell_tab_ind]

        gas_vel_1comp = gas_tab['balmer_(1)_vel'][gas_tab_ind]
        
        gas_vel_comp1 = gas_tab2['gas_(1)_vel'][gas_tab2_ind]
        gas_vel_comp2 = gas_tab2['gas_(2)_vel'][gas_tab2_ind]
        gas_sig_comp1 = gas_tab2['gas_(1)_sig'][gas_tab2_ind]
        gas_sig_comp2 = gas_tab2['gas_(2)_sig'][gas_tab2_ind]

        Ha_amp1 = gas_tab2['Halpha_(1)_comp_amp'][gas_tab2_ind]
        Ha_amp2 = gas_tab2['Halpha_(2)_comp_amp'][gas_tab2_ind]
        
        Ha_snr[bin_ind] = gas_tab2['Halpha_(1)_ANR'][gas_tab2_ind]
        
        #comp1_closer_vstar = np.abs(gas_vel_comp1 - vstar) < np.abs(gas_vel_comp2 - vstar)
        #comp1_closer_1comp = np.abs(gas_vel_comp1 - gas_vel_1comp) < np.abs(gas_vel_comp2 - gas_vel_1comp)
        comp1_closer_galvel = np.abs(gas_vel_comp1 - ngc1266_vel) < np.abs(gas_vel_comp2 - ngc1266_vel)

        if np.isnan(gas_vel_comp2) == False and np.isnan(gas_vel_comp1) == False:
            if comp1_closer_galvel:
                gas_vel_c1[bin_ind] = gas_vel_comp1
                gas_vel_c2[bin_ind] = gas_vel_comp2
                gas_sig_c1[bin_ind] = gas_sig_comp1
                gas_sig_c2[bin_ind] = gas_sig_comp2
            else:
                gas_vel_c1[bin_ind] = gas_vel_comp2
                gas_vel_c2[bin_ind] = gas_vel_comp1
                gas_sig_c1[bin_ind] = gas_sig_comp2
                gas_sig_c2[bin_ind] = gas_sig_comp1

        else:
            gas_vel_c1[bin_ind] = gas_vel_comp1
            gas_sig_c1[bin_ind] = gas_sig_comp1
            gas_vel_c2[bin_ind] = np.nan
            gas_sig_c2[bin_ind] = np.nan
        
        if gas_tab['Halpha_(1)_ANR'][gas_tab2_ind] > 3:
            Ha_flux[bin_ind] = gas_tab['Halpha_(1)_flux'][gas_tab2_ind]/nspec
        else:
            Ha_flux[bin_ind] = np.nan
            
        if gas_tab['Hbeta_(1)_ANR'][gas_tab2_ind] > 3:
            Hb_flux[bin_ind] = gas_tab['Hbeta_(1)_flux'][gas_tab2_ind]/nspec
        else:
            Hb_flux[bin_ind] = np.nan
            
        if gas_tab['[NII]6583_d_(1)_ANR'][gas_tab2_ind] > 3:
            Nii_flux[bin_ind] = gas_tab['[NII]6583_d_(1)_flux'][gas_tab2_ind]/nspec
        else:
            Nii_flux[bin_ind] = np.nan
            
        if gas_tab['[OIII]5007_d_(1)_ANR'][gas_tab2_ind] > 3:
            Oiii_flux[bin_ind] = gas_tab['[OIII]5007_d_(1)_flux'][gas_tab2_ind]/nspec
        else:
            Oiii_flux[bin_ind] = np.nan
            
        if gas_tab['[OI]6300_d_(1)_ANR'][gas_tab2_ind] > 3:
            Oi_flux[bin_ind] = gas_tab['[OI]6300_d_(1)_flux'][gas_tab2_ind]/nspec
        else:
            Oi_flux[bin_ind] = np.nan
            
        if gas_tab['[SII]6731_d1_(1)_ANR'][gas_tab2_ind] > 3:
            Sii_flux[bin_ind] = (gas_tab['[SII]6731_d1_(1)_flux'][gas_tab2_ind] + gas_tab['[SII]6731_d2_(1)_flux'][gas_tab2_ind])/nspec
        else:
            Sii_flux[bin_ind] = np.nan
            
        
log_Ha_flux = np.log10(Ha_flux) + 20
log_Hb_flux = np.log10(Hb_flux) + 20
log_Nii_flux = np.log10(Nii_flux) + 20
log_Oiii_flux = np.log10(Oiii_flux) + 20
log_Sii_flux = np.log10(Sii_flux) + 20
log_Oi_flux = np.log10(Oi_flux) + 20


cube_file = '/Users/jotter/highres_PSBs/ngc1266_data/MUSE/ADP.2019-02-25T15 20 26.375.fits'
cube_fl = fits.open(cube_file)
maps_header = cube_fl[1].header

maps_header['DESC0'] = 'Bin Number'

maps_header['DESC1'] = 'Stellar velocity'
maps_header['DESC2'] = 'Stellar sigma'

maps_header['DESC3'] = 'Gas velocity component 1'
maps_header['DESC4'] = 'Gas sigma component 1'

maps_header['DESC5'] = 'Gas velocity component 2'
maps_header['DESC6'] = 'Gas sigma component 2'

maps_header['DESC7'] = 'Halpha flux'
maps_header['DESC8'] = 'Hbeta flux'
maps_header['DESC9'] = 'NII flux'
maps_header['DESC10'] = 'OIII flux'
maps_header['DESC11'] = 'OI flux'
maps_header['DESC12'] = 'SII flux'

cube_fl[1].header = maps_header

maps_arr = np.empty((13, stellar_vel.shape[0], stellar_vel.shape[1]))
maps_arr[0,:,:] = binNum_2D
maps_arr[1,:,:] = stellar_vel
maps_arr[2,:,:] = stellar_sigma
maps_arr[3,:,:] = gas_vel_c1
maps_arr[4,:,:] = gas_sig_c1
maps_arr[5,:,:] = gas_vel_c2
maps_arr[6,:,:] = gas_sig_c2
maps_arr[7,:,:] = log_Ha_flux
maps_arr[8,:,:] = log_Hb_flux
maps_arr[9,:,:] = log_Nii_flux
maps_arr[10,:,:] = log_Oiii_flux
maps_arr[11,:,:] = log_Oi_flux
maps_arr[12,:,:] = log_Sii_flux

cube_fl[1].data = maps_arr

cube_fl.writeto('/Users/jotter/highres_PSBs/ngc1266_data/MUSE/maps/ngc1266_ppxf_Mar23_maps.fits', overwrite=True)
cube_fl.close()



