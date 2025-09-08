import astropy.units as u
import numpy as np

import astropy.constants as const

z = 0.007214         # NGC 1266 redshift, from SIMBAD
galv = np.log(z+1)*const.c # estimate of galaxy's velocity


#calculations for determining minwave and waverange
restwave = 6562.8 * u.Angstrom
n1266_wave = restwave * (1 + z)

vels = np.arange(-1000,1000,250) * u.km/u.s #+ galv


vel_to_wave = u.doppler_optical(n1266_wave)
wavs = vels.to(u.Angstrom, equivalencies=vel_to_wave)

print(vels)

print(wavs)


