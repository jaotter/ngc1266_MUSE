import numpy as np
import os
from astropy.io import fits
import ultranest  # Import ultranest for nested sampling
from ultranest.plot import cornerplot  # Import cornerplot for visualization of posterior distributions
import ultranest.stepsampler  # Import stepsampler for ultranest
import matplotlib.pyplot as plt
from scipy.stats import linregress
from astropy.stats import sigma_clip

import matplotlib
# Set matplotlib to work in 'inline' mode for Jupyter notebooks, comment out if not needed
#matplotlib.use('agg')
#matplotlib inline
import time  # Import time to track performance
import multiprocess as mp  # Import multiprocessing for parallel processing
import pandas as pd  # Import pandas for data manipulation
import logging


def load_file(filepath):
    """reads in each spaxel's coordinate text file and defines the x and y coordinates of the spaxels"""
    coords = np.loadtxt(filepath, usecols=(0), unpack=True)
    coordx = int(coords[0])
    coordy = int(coords[1])
    return coordx, coordy

object_name = 'ngc1266'
cube_filename = '/Users/jotter/highres_psbs/ngc1266_data/MUSE/ADP.2019-02-25T15 20 26.375.fits'

final_out_dir = "../output/"+object_name+"_HaNii_out"

try:
    os.makedirs(final_out_dir)
except FileExistsError:
    pass


# Open the FITS file and read data
with fits.open(cube_filename) as hdu:
    data = hdu[1].data
    error_data = hdu[2].data
    wavelength_start = hdu[1].header['CRVAL3']
    delta_wavelength = hdu[1].header['CD3_3']
    n_wavelengths = data.shape[0]

# Calculate wavelength array
wavelengths = wavelength_start + delta_wavelength * np.arange(n_wavelengths)

output_directory = "../output/"+object_name + '_spaxel_coord_files/'  # Ensure object_name is defined

# Create the directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the wavelength range to check for zeroes
lower_wavelength = 4800
upper_wavelength = 5200

# Find indices for the specified wavelength range
lower_index = np.searchsorted(wavelengths, lower_wavelength)
upper_index = np.searchsorted(wavelengths, upper_wavelength)

# Create a mask to check for zeroes in the specified wavelength range
zero_mask = np.any(data[lower_index:upper_index+1, :, :] == 0, axis=0)

# Define the region of the cube to analyze
#update
#y_i, y_f = 70,230
#x_i, x_f = 60,220
y_i, y_f = 105,205
x_i, x_f = 105,205

nan_file_count = 0
coord_ind = 0

for y in range(y_i, y_f):
    for x in range(x_i, x_f):
        if not zero_mask[y-1, x-1]:  # Check the zero mask for the defined wavelength range
            coord_ind += 1
            coords = [x, y]
            coordinate_array = np.array(coords)

            spectrum = data[lower_index:upper_index+1, y - 1, x - 1]

            errors = error_data[lower_index:upper_index+1, y - 1, x - 1]

            output_filename = str(coord_ind) + '.txt'

            if not np.isnan(np.min(spectrum)) and np.min(spectrum) != 0 and not np.isnan(np.min(errors)):
                np.savetxt(output_directory + output_filename, coordinate_array, fmt='%d')
            else:
                nan_file_count += 1



# Define a class DataProcessor to encapsulate the data processing logic
class DataProcessor:
    # Initializer or constructor to initialize the DataProcessor instance
    def __init__(self, cube_path, spec_dir, out_dir, target_param, fit_instructions, cont_instructions=None,
                 prefit_instructions=None):
        # Assigning the function parameters to instance variables
        self.cube_path = cube_path  # Path to the data cube
        self.spec_dir = spec_dir  # Directory containing spectra
        self.out_dir = out_dir  # Output directory
        self.target_param = target_param  # Target parameters for fitting
        self.cont_instructions = cont_instructions  # Instructions for continuum fitting
        self.fit_instructions = fit_instructions  # Instructions for line fitting
        self.prefit_instructions = prefit_instructions  # Instructions for prefitting
        self.listing = sorted(os.listdir(self.spec_dir))

        # Count the number of free lines in the fitting instructions
        self.free_lines = sum(self.fit_instructions[line]['flux_free'] is True for line in self.fit_instructions)

        # Pre-fit calculations and directory setup
        if self.prefit_instructions is not None:
            # If there are prefit instructions, process them similarly
            self.known_comps = self.prefit_instructions.copy()
            self.known_comps.pop('flux')
            self.prefit_free_lines = sum(
                self.known_comps[line]['flux_free'] is True for line in self.known_comps)
            self.prefit_num_lines = len(self.known_comps)
        else:
            # If there are no prefit instructions, set counts to zero
            self.prefit_free_lines = 0
            self.prefit_num_lines = 0

        self.make_dirs()  # Call the method to create necessary directories


    def load_file(filepath):
        """reads in each spaxel's coordinate text file and defines the x and y coordinates of the spaxels"""
        coords = np.loadtxt(filepath, usecols=(0), unpack=True)
        coordx = int(coords[0])
        coordy = int(coords[1])
        return coordx, coordy

    # Method to create directory structure for output data
    def make_dirs(self):
        # Check if the main output directory exists, if not, create it
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        # Check and create subdirectories for 'out' and 'plots'
        #if not os.path.exists(os.path.join(self.out_dir,'out')):
         #   os.mkdir(os.path.join(self.out_dir,'out'))
        if not os.path.exists(os.path.join(self.out_dir,'plots')):
            os.mkdir(os.path.join(self.out_dir,'plots'))
            print('making plots dir')

        # Check for a specific output file and initialize it if it doesn't exist
        if not os.path.exists(os.path.join(self.out_dir,
                                           f'{self.target_param["name"]}_{self.fit_instructions["line1"]["name"]}.txt')):
            self.init_cat()

    def get_pix(self, row_ind):
        #print('YEEEHAWWW')
        #print(row_ind.filenum)
        infile = self.listing[int(row_ind.filenum)]
        return infile

    def init_cat(self):
        """Blank output catalog"""
        cols = ["filenum", "filename","coordx","coordy","ncomps"]
        if self.prefit_num_lines > 0:
            for i in range(self.prefit_num_lines):
                cols.append(f'prefit_flux{i+1}')
        for i in range(1, self.target_param["maxcomp"] + 1):
            cols.append(f'wave_{i}')
            cols.append(f'width_{i}')
            for flux in range(1, self.free_lines + 1):
                cols.append(f'flux_{i}_{chr(ord("@") + flux)}')

        for i in range(1, self.target_param["maxcomp"] + 1):
            cols.append(f'wave_{i}_sigma')
            cols.append(f'width_{i}_sigma')
            for flux in range(1, self.free_lines + 1):
                cols.append(f'flux_{i}_{chr(ord("@") + flux)}_sigma')

        self.cat = pd.DataFrame(np.zeros((len(self.listing), len(cols))), columns=cols)
        self.cat.loc[:, 'ncomps'] = -1

        self.cat["filenum"] = self.cat.index

        outfile = os.path.join(self.out_dir,
                               f'{self.target_param["name"]}_{self.fit_instructions["line1"]["name"]}.txt')

        infile = self.cat.apply(self.get_pix, axis=1)
        print(infile)
        print(outfile)

        self.cat['filename'] = infile #None
        self.cat['coordx'] = int(0)
        self.cat['coordy'] = int(0)
        self.cat.to_csv(outfile, index=False)

    def find_unfit(self):
        file_list = [os.path.basename(filename) for filename in os.listdir(self.spec_dir)
                      if os.path.isfile(os.path.join(self.spec_dir, filename)) and not filename.startswith('.')]
        return file_list

    def make_gaussian(self, mu, sigma, flux):
        """ centroid, width, integrated flux"""
        h = flux / (sigma * np.sqrt(2 * np.pi))
        s = -1.0 / (2 * sigma ** 2)

        def f(x):
            return h * np.exp(s * (x - mu) ** 2)

        return f

    def gauss(self, pos1, width1, *args):
        """"""
        pos_all = []
        flux_all = []
        lines = []
        args = list(args)
        for line in self.fit_instructions:
            lines.append(line)  # keeping track of what lines have been fit
            pos = (pos1 - ((1 + self.target_param['red']) * (self.fit_instructions['line1']['wave']
                                                             - self.fit_instructions[line]['wave'])))
            pos_all.append(pos)
            if self.fit_instructions[line]['flux_free'] is True:
                flux = args[0]  # get the argument
                flux_all.append(flux)
                args.pop(0)
            else:
                line_lock = self.fit_instructions[line]['flux_locked_with']
                ratio = self.fit_instructions[line]['flux_ratio']
                flux_lock = flux_all[lines.index(line_lock)]
                flux = flux_lock / ratio
                flux_all.append(flux)

        all_gauss_functions = [0] * len(x)  # global variable at end
        for i in range(0, len(self.fit_instructions)):
            pos = pos_all[i]
            flux = flux_all[i]
            gauss_hold = self.make_gaussian(pos, width1, flux)
            all_gauss_functions += gauss_hold(x)
        return all_gauss_functions


    def prior_ultra(self, cube):

        paramcube = cube.copy()

        ndim = np.shape(paramcube)[0]

        if ndim == 1:
            lo = np.min(avg)
            hi = np.max(avg)
            paramcube[0] = paramcube[0] * (hi - lo) + lo

            #paramcube[0] = paramcube[0]*0.02 + np.mean(avg)*0.99
            return paramcube
        else:
            if self.prefit_free_lines > 0:
                for i in range(0, self.prefit_num_lines):

                    # uniform wavelength prior
                    cube[i * 3] = (self.prefit_instructions[f'comp{i + 1}']['cen'])

                    # uniform width prior
                    cube[i * 3 + 1] = (self.prefit_instructions[f'comp{i + 1}']['width'])

                    # log-uniform flux prior
                    cube[i * 3 + 2] = ((stdev / 10) * cube[i * 3 + 1] * np.sqrt(2 * np.pi) *
                                       np.power(10, cube[i * 3 + 2] * 4))

            for i in range(self.prefit_num_lines * 3, ndim, 2 + self.free_lines):

                # uniform wavelength prior
                paramcube[i + 0] = self.fit_instructions['line1']['minwave'] + (paramcube[i+0] * self.fit_instructions['line1']['wave_range'])

                # uniform width prior
                paramcube[i + 1] = self.target_param['minwidth'] + (paramcube[i+1]*(self.target_param['maxwidth']-self.target_param['minwidth']))

                # log-uniform flux prior
                for fprior in range(0, self.free_lines):
                    paramcube[i + fprior +2] = fluxsigma * paramcube[i+1] * np.sqrt(2 * np.pi) * np.power(10, paramcube[i + fprior +2] * 4)
            return paramcube

    def model(self, *args):
        # Determine the number of arguments provided to the model.
        nargs = len(args)

        # If there is only one argument, assume it's the continuum level and create a constant model.
        if nargs == 1:
            cont = args[0]  # The continuum value.
            result = 0 * x + cont  # Create an array of the continuum value over the wavelength range.

        # If there are more than one arguments, they represent parameters for the prefit and fitting lines.
        else:
            # Initialize the model with the continuum level over the fitting range.
            result = np.zeros(self.target_param['end'] - self.target_param['start']) + avg

            # Loop over prefit line parameters and add their contributions to the model.
            for i in range(0, self.prefit_num_lines * 3, 2 + self.prefit_free_lines):
                # Add the Gaussian profile from the prefit lines to the model.
                result += self.gauss_cont(*args[i:(i + (2 + self.prefit_free_lines))])

            # Loop over fit line parameters and add their contributions to the model.
            for i in range(self.prefit_num_lines * 3, nargs, 2 + self.free_lines):
                # Add the Gaussian profile from the fitting lines to the model.
                result += self.gauss(*args[i:(i + (2 + self.free_lines))])

        # Return the resulting model.
        return result

    def model2(self, *args):
        # Determine the number of arguments provided to the model.
        nargs = len(args)

        # If there is only one argument, it defines the constant continuum level.
        if nargs == 1:
            # Create a constant array with the continuum value over the fitting range.
            result = np.zeros(self.target_param['end'] - self.target_param['start']) + args[0]
        else:
            # Initialize the model with the average continuum level over the fitting range.
            result = np.zeros(self.target_param['end'] - self.target_param['start']) + avg

            # Loop over each set of line parameters and add their Gaussian profiles to the model.
            for i in range(0, nargs, 2 + self.free_lines):
                # Add the Gaussian profile to the model.
                result += self.gauss(*args[i:(i + (2 + self.free_lines))])

        # Return the resulting model.
        return result

    def model3(self, *args):
        # Determine the number of arguments provided to the model.
        nargs = len(args)

        # If there is only one argument, it sets the continuum level for a flat model.
        if nargs == 1:
            # Create a constant array with the continuum value over the fitting range.
            result = np.zeros(self.target_param['end'] - self.target_param['start']) + args[0]
        else:
            # Initialize the model with the average continuum level over the fitting range.
            result = np.zeros(self.target_param['end'] - self.target_param['start']) + avg

            # Loop over each set of prefit line parameters and add their Gaussian profiles to the model.
            for i in range(0, nargs, 2 + self.prefit_free_lines):
                # Add the Gaussian profile for prefit lines to the model.
                result += self.gauss_cont(*args[i:(i + (2 + self.prefit_free_lines))])

        # Return the resulting model.
        return result

    def loglike(self, paramcube):
        ndim = paramcube.shape[0]
        penalties = np.zeros_like(paramcube)
        # Assuming `self.model` is vectorized
        ymodel = self.model(*paramcube)
        loglikelihood = -0.5 * np.sum(((ymodel - ydata) / noise) ** 2)
        return loglikelihood

    def _get_indices_for_continuum(self, continuum_range, wavearray):
        """
        Find the indices in the wavearray that correspond to the specified continuum range.

        :param continuum_range: A tuple or list containing the lower and upper bounds of the continuum region.
        :param wavearray: The array of wavelength values.
        :return: A tuple containing the lower and upper indices that correspond to the continuum range.
        """
        lower_bound, upper_bound = continuum_range
        lower_idx = (np.abs(wavearray - lower_bound)).argmin()
        upper_idx = (np.abs(wavearray - upper_bound)).argmin()
        return lower_idx, upper_idx

    def _fit_continuum_slope(self, cont1, cont2, wavearray, sigma=3, maxiters=5):
        """
        Fit a linear slope to the continuum regions.

        :param cont1: Flux values for the first continuum region.
        :param cont2: Flux values for the second continuum region.
        :param wavearray: The array of wavelength values.
        :return: slope and intercept of the fitted continuum.
        """
        # Combine the two continuum regions
        cont_flux = np.concatenate((cont1, cont2))
        cont_wav = np.concatenate((wavearray[:len(cont1)], wavearray[len(wavearray) - len(cont2):]))

        # Perform sigma clipping to remove outliers
        clipped_flux = sigma_clip(cont_flux, sigma=sigma, maxiters=maxiters)

        # Only keep the data points that were not clipped
        valid_flux = cont_flux[~clipped_flux.mask]
        valid_wav = cont_wav[~clipped_flux.mask]

        # Perform linear regression to find the slope and intercept
        slope, intercept, _, _, _ = linregress(valid_wav, valid_flux)

        return slope, intercept

#    def _calculate_continuum_average(self, cont1, cont2):
#        """Calculate the average of two continuum segments."""
#        return (np.mean(cont1) + np.mean(cont2)) / 2\

    def _calculate_continuum_average(self, cont1, cont2):
        # Calculate median to minimize the impact of outliers
        median1 = np.median(cont1)
        median2 = np.median(cont2)
        return (median1 + median2) / 2

    def make_model_plot(self, ncomp, outmodel, loglike, filename):
        # Define a list of colors to be used for plotting different components.
        colorlist = ['goldenrod', 'plum', 'teal', 'firebrick', 'darkorange']

        # Initialize a figure and axis for the plot.
        fig, ax = plt.subplots()

        # Set the x and y-axis limits based on the target parameters and data range.
        ax.set_xlim(self.target_param["plotmin"], self.target_param["plotmax"])
        ax.set_ylim(miny - ypadding, maxy + ypadding)

        # Add a text box inside the plot that displays the log likelihood value.
        ax.text(0.25, 0.95, f'ln(Z) = {loglike:.2f}', transform=ax.transAxes)

        # Plot the original data with a specific line width, color, and z-order for visibility.
        ax.plot(x, ydata, '-', lw=1, color='0.75', label='data', zorder=1)

        # If the model is simple (0, 1, or 2 components), also plot the noise for comparison.
        if (ncomp == 0) or (ncomp == 1) or (ncomp == 2):
            ax.plot(x, noise, '-', color='red', zorder=1)

        # Add a vertical line representing the systemic velocity.
        ax.axvline(systemic, 0, 1, ls='--', lw=0.5, color='blue')

        # Highlight the regions used for determining the continuum.
        ax.axvspan(self.cont_instructions["continuum1"][0], self.cont_instructions["continuum1"][1], facecolor='black', alpha=0.1)
        ax.axvspan(self.cont_instructions["continuum2"][0], self.cont_instructions["continuum2"][1], facecolor='black', alpha=0.1)

        # Plot the best fit model over the data.
        ax.plot(x, self.model(*outmodel), '-', lw=1, color='black', label='model', zorder=3)

        # Plot the average continuum level and its variation.
        ax.plot(x, avg, lw=1, color='green', label='model', zorder=3)
        ax.plot(x, avg + fluxsigma, '--', lw=1, color='red', label='model', zorder=3)

        # If there are more than one components, plot each component individually.
        if ncomp > 1:
            # Plot the combined model for all prefit components.
            ax.plot(x, self.model3(*outmodel[0: 3 * self.prefit_num_lines]), '-', lw=0.75, color="olive", zorder=2)

            # Plot each component of the model using a different color from the predefined list.
            for i in range(3 * self.prefit_num_lines, ((2 + self.free_lines) * ncomp) + (3 * self.prefit_num_lines), (2 + self.free_lines)):
                color = colorlist[(i // self.target_param["maxcomp"]) % (len(colorlist))]
                ax.plot(x, self.model2(*outmodel[i:(i + 2 + self.free_lines)]), '-', lw=0.75, color=color, zorder=2)

        # Plot the model data for determining the y-axis range.
        model_data = self.model(*outmodel)
        min_model = np.min(model_data)
        max_model = np.max(model_data)
        margin = (max_model - min_model) * 0.1  # Adding 10% margin on each side
        ax.set_ylim(min_model - margin, max_model + margin)

        # Set the title of the plot indicating the file being processed and the number of components used.
        ax.set_title(f'{filename} -- {ncomp} Components')

        # Label the axes.
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Flux')

        # Save the plot to a file with a name indicating the number of components.
        fig.savefig(plot_outfile + f'_{ncomp}_posterior.pdf')

        # Close the plot to free up memory.
        plt.close()

    def update_columns(self, dataframe, column_names, values, filename):
        """Helper function to update dataframe columns with new values for a specific filename."""
        for col_name, value in zip(column_names, values):
            dataframe.loc[dataframe['filename'] == filename, col_name] = value

    def write_results(self, filename, coordx, coordy, ncomp, outmodel):

        cat_file = os.path.join(self.out_dir, f'{self.target_param["name"]}_'
                                               f'{self.fit_instructions["line1"]["name"]}.txt')
        out_file = os.path.join(self.out_dir, f'{self.target_param["name"]}_'
                                               f'{self.fit_instructions["line1"]["name"]}_out.txt')

        cat = pd.read_csv(cat_file, index_col='filenum')

        prefit_col = cat.columns[cat.columns.str.startswith('prefit')].tolist()

        for i, prefit in enumerate(range(2, (3*self.prefit_num_lines)+2, 3)):
            cat.loc[cat['filename'] == filename, prefit_col[i]] = outmodel[prefit]

        use_col = cat.columns[4+self.prefit_num_lines:len(outmodel[3 * self.prefit_num_lines:]) + (4+self.prefit_num_lines)].tolist()

        for i, mod in enumerate(outmodel[3 * self.prefit_num_lines:]):
        	cat.loc[cat['filename'] == filename, use_col[i]] = mod
            # cat.at[filename, use_col[i]] = mod

        #sigma_col = cat.columns[cat.columns.str.endswith('sigma')].tolist()    #Don't know how to get this from the ultranest output yet.
        #for i, sig in enumerate(modelsigma[3 * self.prefit_num_lines:]):
        #    cat.loc[cat['filename'] == filename, sigma_col[i]] = sig

        cat.loc[cat['filename'] == filename, 'ncomps'] = int(float(ncomp))
        cat.loc[cat['filename'] == filename, 'coordx'] = int(coordx)
        cat.loc[cat['filename'] == filename, 'coordy'] = int(float(coordy))

        cat.loc[cat['filename'] == filename].to_csv(out_file, index_label='filenum',header=False,mode = 'a')


    def mp_worker(self, filename):

        # Define global variables that will be used within this function.
        # These variables are shared across different processes in multiprocessing.
        global x, ydata, miny, maxy, avg, stdev, fluxsigma, noise
        global column, row, line_outfile, ypadding, systemic, plot_outfile

        # Define the input file path using the given filename and specific directory.
        infile = filename
        inspecpath = os.path.join(self.spec_dir, filename)

        # Load the file coordinates. This function should be defined elsewhere within the class.
        coordx, coordy = load_file(inspecpath)


        # Processing the flux data from the cube. This assumes the existence of a 'fluxcube' variable,
        # which should have been initialized and populated in the mp_handler method or elsewhere.
        flux = fluxcube[:, coordy - 1, coordx - 1] * 1e-20

        #flux = fluxcube * 1e-20  # Adjust the flux values based on some assumed calibration factor.

        # Similarly, process the noise data. This again assumes that an 'errorcube' is available and populated.
        noise = np.sqrt(errorcube[:, coordy - 1, coordx - 1] * 1e-40)
        #noise = np.sqrt(errorcube * 1e-40)  # Convert error estimates based on calibration.

        # Define output file paths based on provided filename and output directory structure.
        infilebase = filename.split('.')[0]
        data_outfile = os.path.join(self.out_dir,
                                    f'{infilebase}_{self.fit_instructions["line1"]["name"]}')
        plot_outfile = os.path.join(self.out_dir,  'plots',
                                    f'{infilebase}_{self.fit_instructions["line1"]["name"]}')
        line_outfile = os.path.join(self.out_dir,
                                    f'{self.target_param["name"]}_{self.fit_instructions["line1"]["name"]}.txt')

        # Setting up the wavelength array (x values) and corresponding y data for plotting and analysis.
        x = wavearray[self.target_param["start"]:self.target_param["end"]]
        ydata = flux[self.target_param["start"]:self.target_param["end"]]
        noise = noise[self.target_param["start"]:self.target_param["end"]]

        #print(np.isnan(noise).any())
        #plt.errorbar(x,ydata,yerr=noise)
        #plt.show()

        # Determine the plotting range for y-axis based on data values.
        maxy = max(ydata)
        miny = min(ydata)
        ypadding = 0.05 * (maxy - miny)  # Padding added to y limits for better visualization.

        # Calculate systemic velocity based on redshift and wavelength of the first line.
        systemic = (1. + self.target_param["red"]) * self.fit_instructions["line1"]["wave"]

        # Extract continuum regions from the spectrum and calculate average (continuum) values.
        low1_ind, upp1_ind = self._get_indices_for_continuum(self.cont_instructions["continuum1"], wavearray)
        low2_ind, upp2_ind = self._get_indices_for_continuum(self.cont_instructions["continuum2"], wavearray)
        cont1 = flux[low1_ind:upp1_ind]
        cont2 = flux[low2_ind:upp2_ind]

        #avg = self._calculate_continuum_average(cont1, cont2)
        #avg = np.full_like(x, avg)

        slope,intercept = self._fit_continuum_slope(cont1, cont2, x)
        avg = slope * x + intercept

        median1 = np.median(cont1)
        mad1 = np.median(np.abs(cont1 - median1))
        robust_stddev_1 = 1.4826 * mad1

        median2 = np.median(cont2)
        mad2 = np.median(np.abs(cont2 - median2))
        robust_stddev_2 = 1.4826 * mad2

        robust_stdev = (robust_stddev_1 + robust_stddev_2) / 2


        #slope, intercept = self._fit_continuum_slope(cont1, cont2, wavearray)
        #avg = slope * x + intercept

        # Standard deviation of continuum flux is used for noise estimation and further calculations.
        #stdev = (np.std(cont1) + np.std(cont2)) / 2

        # Calculate flux sigma for modeling purposes.
        fluxsigma = (self.target_param["fluxsigma"] * robust_stdev)  # Adjustment based on input parameter.

        # Set the maximum number of components that this program will model
        maxcomp = self.target_param["maxcomp"]

        # build lists that will hold the outputs for each model
        analyzers 	= [0] * (maxcomp + 1)
        lnZs 		= [0] * (maxcomp + 1)
        outmodels 	= [0] * (maxcomp + 1)

        # Print a final message to indicate completion of processing for this file.
        message = f"Done with {filename}\n"  # Message construction with newline character for clear output.

        # ------ CONTINUUM FIT ------
        # Start the continuum fitting process. This part focuses on determining the continuum level of the spectrum.

        print(f'Fitting {filename}')

        # Define parameter names used for modeling the continuum.
        param_names = ['cont', 'wave1', 'width1', 'flux1', 'flux2', 'wave2', 'width2', 'flux3', 'flux4', 'wave3', 'width3', 'flux5', 'flux6']

        # Start with no components to fit only the continuum.
        ncomp = 0
        bestncomp = ncomp  # Initialize the best number of components as the current number.
        n_params = 1  # Number of parameters to fit in the continuum-only model.

        # Set up parameters and the sampler for continuum fitting.
        print('Fitting the continuum of '+filename)
        parameters0 = param_names[:n_params]

        sampler0 = ultranest.ReactiveNestedSampler(parameters0, self.loglike, self.prior_ultra)

        # Run the fitting process with a minimum number of live points.
        result0 = sampler0.run(min_num_live_points=200)

        # Store the log likelihood and the model corresponding to the maximum likelihood estimate for zero components.
        lnZs[ncomp] = result0['logz']
        outmodels[ncomp] = result0['maximum_likelihood']['point']

        # Generate and save a plot for the continuum fitting results.
        #self.make_model_plot(ncomp, outmodels[ncomp], lnZs[ncomp], filename)

        # ------ COMPONENT FITS ------
        # After determining the continuum, fit additional components representing emission or absorption features.

        # Increment the number of components for subsequent fitting processes.
        ncomp += 1


        # Loop through different numbers of components to find the best fit.
        for ncomp in range(1, self.target_param["maxcomp"] + 1):
            print(f'{filename}: trying {ncomp} component(s)')

            # Calculate the number of parameters based on the number of components and other settings.
            n_params = (2 + self.free_lines) * ncomp + (3 * self.prefit_num_lines)
            parameters = param_names[1:n_params + 1]
            print(parameters, 'Holaaaaaaaa')

            # Determine the number of steps for the slice sampler based on the parameters.
            nsteps = 2 * len(parameters)

            # Start the component fitting section, marking the beginning time to measure execution duration.
            start_time = time.time()

            # Create the nested sampler object with the specified parameters, likelihood function, and prior.
            sampler = ultranest.ReactiveNestedSampler(parameters, self.loglike, self.prior_ultra)

            # Configure the step sampler with the generated direction and other settings.
            sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                nsteps=nsteps,
                generate_direction=ultranest.stepsampler.generate_mixture_random_direction
            )

            # Execute the nested sampling process and store the result.
            result = sampler.run(min_num_live_points=200)

            # After finishing all component fits, mark the end time.
            end_time = time.time()

            # Calculate the total duration of the fitting process.
            print("Total execution time: {} seconds".format(end_time - start_time))

            # Print out the results after the sampling is completed.
            #sampler.print_results()

            # Store the logZ and best-fit parameters for each number of components.
            lnZs[ncomp] = result['logz']
            outmodels[ncomp] = result['maximum_likelihood']['point']

            # Plot the best fit model and save the plot.
            self.make_model_plot(ncomp, outmodels[ncomp], lnZs[ncomp], filename)

            # Evaluate whether the current model is better than the previous one based on logZ difference.
            if lnZs[ncomp] - lnZs[bestncomp] > self.target_param["lnz"]:
                bestncomp = ncomp
            else:
                # If the improvement is not significant, stop trying more components.
                break

        # Print the final best model details.
        # print(filename, coordx, coordy, bestncomp, outmodels[bestncomp])

        self.write_results(filename, coordx, coordy, bestncomp, outmodels[bestncomp])
        # print(f'{filename} fit with {bestncomp} components')

    def mp_handler(self):
        # Declare the necessary global variables that will be used across multiple processes.
        global fluxcube, errorcube, wavearray

        # Open the FITS file located at the cube_path. This file is expected to contain the data cube.
        hdu = fits.open(self.cube_path)

        # Extract the flux and error data from the FITS file.
        # These arrays contain the observed data and associated uncertainties, respectively.
        fluxcube = hdu[1].data
        errorcube = hdu[2].data

        # Close the FITS file to free resources.
        hdu.close()

        # Retrieve the shape of the flux data cube to understand the data dimensions.
        data_shape = np.shape(fluxcube)

        # Extract the length of the wavelength dimension from the data cube.
        wl_length = data_shape[0]

        # Retrieve the starting wavelength value and the wavelength increment (stretch) from the FITS header.
        start_wav = hdu[1].header['CRVAL3']
        wav_stretch = hdu[1].header['CD3_3']

        # Compute the wavelength array using the length, starting value, and increment extracted above.
        wavearray = np.array(range(wl_length)) * wav_stretch + start_wav

        # Identify the pixels in the data cube that have not yet been fitted. This function is assumed to be defined elsewhere.
        unfit_pix = self.find_unfit()

        # Print the list of pixels that have not been fitted to provide feedback to the user.
        print(unfit_pix)

        # Create a multiprocessing pool to process multiple pixels in parallel.
        with mp.Pool() as pool:
            # Map the mp_worker function across all unfit pixels. The mp_worker function is assumed to be defined elsewhere
            # and is responsible for processing individual pixels.
            pool = mp.Pool(processes=mp.cpu_count()-2)
            pool.imap_unordered(self.mp_worker, unfit_pix)
            #pool.map(self.mp_worker, unfit_pix)
            pool.close()
            pool.join()
            #pool = mp.Pool(processes=self.target_param["cores"])
            #pool.map(self.mp_worker, unfit_pix)


# Usage example
def main0():
    target_param = {
        'name': 'NGC1266',
        'red': 0.007214,
        'minwidth': 1,
        'maxwidth': 500,
        'start': 1300, #330,
        'end': 1600, #430,
        'fluxsigma': 5,
        'plotmin': 6500,#5175,
        'plotmax': 6750,#5280,
        'maxcomp': 3,
        'cores': 36,
        'lnz': 5.0,
    }

    cont_instructions = {
        'form': 'model',
        'cont_poly': 1,
        'continuum1': (6443, 6518),#(5215,5225),
        'continuum2': (6687, 6724)#(5280,5295),
    }

    fit_instructions = {
        'line1': {'name': 'halpha', 'wave': 6562.8, 'minwave': 6593,#6560, #6615,
        'wave_range': 30.0, 'flux_free': True},
        'line2': {'name': 'nii', 'wave': 6583, 'flux_free': True},
        'line3': {'name': 'nii', 'wave': 6548, 'flux_free': False, 'flux_locked_with': 'line2', 'flux_ratio': 3
        },
    }

    #fit_instructions = {
    #'line1': {'name': 'h-beta', 'wave': 4861.33, 'flux_free': True, 'minwave': 4900, 'wave_range': 20.0},
    #}

    #fit_instructions = {
    #    'line1': {'name': 'oiii', 'wave': 5006.84, 'flux_free': True, 'minwave': 5225, 'wave_range': 30.0},
    #    'line2': {'name': 'oiii',  'wave': 4958.92, 'flux_free': False,'flux_locked_with': 'line1', 'flux_ratio': 3}
    #}


    processor = DataProcessor(
        cube_path='/Users/jotter/highres_psbs/ngc1266_data/MUSE/ADP.2019-02-25T15 20 26.375.fits',
        spec_dir='/Users/jotter/highres_psbs/ngc1266_muse/output/ngc1266_spaxel_coord_files/',
        out_dir='/Users/jotter/highres_psbs/ngc1266_muse/output/ngc1266_spaxel_coord_files_HaNii_out/',
        target_param=target_param,
        cont_instructions=cont_instructions,
        fit_instructions=fit_instructions
    )
    #print(processor)
    processor.mp_handler()

if __name__ == "__main__":
    main0()




