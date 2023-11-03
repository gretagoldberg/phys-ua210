import astropy as ast
import scipy
from scipy import integrate
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as lin
from scipy import linalg
from scipy.linalg import svd

hdu_list = ast.io.fits.open("/Users/Greta/Downloads/specgrid.fits")
logwave = hdu_list["LOGWAVE"].data #wavelength in angstroms
flux = hdu_list["FLUX"].data #flux


galaxy1 = flux[0,:]
galaxy2 = flux[1,:]
galaxy3 = flux[3,:]
galaxy4 = flux[4,:]
galaxy5 = flux[5,:]

plt.plot(logwave,galaxy1,label="Galaxy 1",c="red")
plt.plot(logwave,galaxy2,label="Galaxy 2",c="blue")
plt.plot(logwave,galaxy3,label="Galaxy 3",c="green")
plt.plot(logwave,galaxy4,label="Galaxy 4",c="orange")
plt.plot(logwave,galaxy5,label="Galaxy 5",c="purple")
plt.legend()
plt.ylabel("Flux in 10^-17 ergs / (s cm^2 Angstroms)")
plt.title("Flux versus Log of Wavelength for Five Galaxies")
plt.xlabel("Wavelength in Angstrom (logscaled)")
plt.show()


#finding normalizations and normalizing data
flux_sum = np.sum(flux, axis = 1)
flux_normalized = flux/np.tile(flux_sum, (np.shape(flux)[1], 1)).T

#finding the mean and subtracting it from the normalized data 
means_normalized = np.mean(flux_normalized, axis=1)
residuals = flux_normalized-np.tile(means_normalized, (np.shape(flux)[1], 1)).T

'''
#plotting the residuals to check
galaxy1 = residuals[0,:]
galaxy2 = residuals[1,:]
galaxy3 = residuals[3,:]
galaxy4 = residuals[4,:]

plt.plot(logwave,galaxy1,label="Galaxy 1",c="red")
plt.plot(logwave,galaxy2,label="Galaxy 2",c="blue")
plt.plot(logwave,galaxy3,label="Galaxy 3",c="green")
plt.plot(logwave,galaxy4,label="Galaxy 4",c="orange")
plt.legend()
plt.ylabel("Flux in 10^-17 ergs / (s cm^2 Angstroms)")
plt.xlabel("Wavelength in Angstrom (logscaled)")
plt.show()
'''
#calculating correlation matrix
r = residuals 
covariance_matrix = r.T@r

#finding eigenvalues/eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

#finding eigenvalues/eigenvectors with SVD
U, S, V = np.linalg.svd(covariance_matrix, full_matrices=True)

#we know that the eigenvectors are in the V matrix so take the transpose of V
eigenvectors_svd = V.T
#we know the corresponding eigenvalues are along the diagonal of S
eigenvalues_collection = np.array(S**2)
eigenvalues_svd = np.sqrt(eigenvalues_collection)

cond_num_r = np.linalg.cond(r)
cond_num_C = np.linalg.cond(covariance_matrix)

'''
for i in range(5):
    plt.plot(logwave,eigenvectors[:,i],label="Eigenvector " + str(i))
plt.xlabel("Wavelength in Angstrom")
plt.ylabel("Eigenvector")
plt.legend()
plt.show()
'''

#coefficients
c0 = r@eigenvectors[:,0]
c1 = r@eigenvectors[:,1]
c2 = r@eigenvectors[:,2]
c3 = r@eigenvectors[:,3]
c4 = r@eigenvectors[:,4]

'''
#plotting c_0 vs c_2 and c_0 vs c_1
plt.scatter(c1,c0,s=10)
plt.ylabel("c0")
plt.xlabel("c1")
plt.title("c0 versus c1")
plt.show()

plt.scatter(c2,c0,s=10)
plt.ylabel("c0")
plt.xlabel("c2")
plt.xlim(-.003,0.003)
plt.title("c0 versus c2")
plt.show()
'''

coefficients = np.vstack((c0,c1,c2,c3,c4))

eigenvectors = np.vstack((eigenvectors[:,0],eigenvectors[:,1],eigenvectors[:,2],eigenvectors[:,3],eigenvectors[:,4]))

#projection
matrix_final = np.dot(coefficients.T,eigenvectors)

#matrix w/ mean added back
matrix_uncentered = matrix_final + np.tile(means_normalized, (np.shape(flux)[1], 1)).T

#matrix renormalized
matrixy = matrix_uncentered*np.tile(flux_sum, (np.shape(flux)[1], 1)).T


#plotting to check that it is similar to the original spectrum but with less information
galaxy1 = matrixy[0,:]
galaxy2 = matrixy[1,:]
galaxy3 = matrixy[3,:]
galaxy4 = matrixy[4,:]

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

#defining function to 
def PCA(N,flux):
    flux_sum2 = np.sum(flux, axis = 1)
    flux_normalized2 = flux/np.tile(flux_sum2, (np.shape(flux)[1], 1)).T

    #finding the mean and subtracting it from the normalized data 
    means_normalized2 = np.mean(flux_normalized2, axis=1)
    residuals2 = flux_normalized2-np.tile(means_normalized2, (np.shape(flux)[1], 1)).T

    #calculating correlation matrix
    r2 = residuals2
    covariance_matrix2 = r2.T@r2

    #finding eigenvalues/eigenvectors
    eigenvalues2, eigenvectors2 = np.linalg.eig(covariance_matrix2)

    weights2 = np.zeros((9713, N))
    for i in range(N):
       weight2 = r2@eigenvectors2[:,N]
       weights2[:,i] = weight2
    eigenvectors2 = np.zeros((4001,N))
    for m in range(N):
        eigenvectors2[:,m] = eigenvectors2[:,m]
    
    matrix_final2 = np.dot(weights2,eigenvectors2.T)

    return matrix_final2

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FF5733', '#33FF57', '#5733FF', '#FF33F7', '#33F7FF', '#FFC733', '#734953', '#3396FF', '#FF6F33', '#33FFD0', '#A933FF', '#FF3371', 'purple',"orange","yellow"]

'''
for i in range(1,21):
    new_matrix2 = PCA(i,flux)
    residuals2 = (residuals - new_matrix2)**2/residuals**2
    total_residuals2 = np.mean(residuals2, axis = 0)
    plt.scatter(logwave,total_residuals2,label=f"N = {i}",c=colors[i],s=10)
plt.legend()
plt.title("Residuals as a Function of N")
plt.xlabel("Logwave in Angstrom")
plt.ylabel("Residual Flux")
plt.show()
'''








