import astropy as ast
import scipy
from scipy import integrate
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as lin
from scipy import linalg

hdu_list = ast.io.fits.open("/Users/Greta/Downloads/specgrid.fits")
logwave = hdu_list["LOGWAVE"].data #wavelength in angstroms
flux = hdu_list["FLUX"].data #flux

galaxy1 = flux[0,:]
galaxy2 = flux[1,:]
galaxy3 = flux[3,:]
galaxy4 = flux[4,:]

'''
plt.plot(logwave,galaxy1,label="Galaxy 1",c="red")
plt.plot(logwave,galaxy2,label="Galaxy 2",c="blue")
plt.plot(logwave,galaxy3,label="Galaxy 3",c="green")
plt.plot(logwave,galaxy4,label="Galaxy 4",c="orange")
plt.legend()
plt.ylabel("Flux in 10^-17 ergs / (s cm^2 Angstroms)")
plt.xlabel("Wavelength in Angstrom (logscaled)")
plt.show()
'''

#scaling the data by integrating over each galaxy and dividing by the square root of that sum
scale_factors = []
for i in range(9713):
    integral = np.trapz(flux[i,:])
    scaling = np.sqrt(1.0 / integral)
    scale_factors.append(scaling)

#actually rescaling the data
scaling = np.array(scale_factors)
normalized_flux = flux*scaling[:,np.newaxis]

#finding the mean for each row to subtract
means = np.mean(normalized_flux, axis=1)
#subtracting off the mean to get data centered
centered_data = normalized_flux - means[:, np.newaxis]

#creating covariance matrix by transposing residuals and dotting the original and transposed matrices together
transpose = centered_data.transpose()
covariance_matrix = np.dot(centered_data,transpose)


