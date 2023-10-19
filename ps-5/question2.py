import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

#reading file in and selecting the times and signals by slicing
data_file = pd.read_csv("/Users/Greta/Downloads/signal.dat",sep="|")
data = pd.DataFrame(data_file)
times = data.iloc[:,1]
signal = data.iloc[:,2]

#print(max(times.values))

#plotting signal and time
plt.scatter(times,signal)
plt.title("Signal Versus Time")
plt.ylabel("Signal (arbitrary units)")
plt.xlabel("Time (s)")
plt.show()

#creating matrix
A = np.zeros((len(times), 4))
A[:, 0] = 1.
A[:, 1] = times
A[:, 2] = times**2
A[:, 3] = times**3

#calculating the matrices I need
(u, w, vt) = np.linalg.svd(A, full_matrices=False)

winv = np.zeros(len(w))
indx = np.where(w > 1.e-15)[0]
winv[indx] = 1. / w[indx]

ainv = vt.transpose().dot(np.diag(winv)).dot(u.transpose()) 
c = ainv.dot(signal)
model = A.dot(c) 

condition_number = max(winv)/min(winv)
print(condition_number)


#plot statements
plt.plot(times, signal, '.',label="actual data")
plt.plot(times, model, '.',label="model")
plt.title("Data and SVD Fit of Data for Polynomial of Degree 4")
plt.ylabel("Signal (arbitrary units)")
plt.xlabel("Time (s)")
plt.legend()
plt.show()

#calculating residuals and looking at how bad it is
residuals = (model-np.array(signal.values))

#plot statements
plt.scatter(times.values,residuals,c="blue",s=10,label="Residuals")
plt.scatter(times,signal,c="red",s=8,label="Actual Signal Data")
plt.title("Residuals in Comparison to Data")
plt.ylabel("Signal (arbitrary units)")
plt.xlabel("Time (s)")
plt.legend()
plt.show()

#trying higher order polynomial...
#creating matrix
B = np.vander(times,10)

#calculating the matrices I need
(u, w, vt) = np.linalg.svd(B, full_matrices=False)

winv = np.zeros(len(w))
indx = np.where(w > 1.e-15)[0]
winv[indx] = 1. / w[indx]

ainv = vt.transpose().dot(np.diag(winv)).dot(u.transpose()) 
c = ainv.dot(signal)
modelB = B.dot(c) 

condition_number = max(winv)/min(winv)
print(condition_number)

#plot statements
plt.plot(times, signal, '.',label="actual data")
plt.plot(times, modelB, '.',label="model")
plt.title("Data and SVD Fit of Data for High Order Polynomial")
plt.ylabel("Signal (arbitrary units)")
plt.xlabel("Time (s)")
plt.legend()
plt.show()

#trying w/ sines and cosines
#fundamental frequency
f_0 = 1/(2*np.max(times))

#number of harmonics included
num_harmonics = 10

#creating design matrix
C = np.zeros((len(times),2*num_harmonics+2))
C[:, 0] = 1
for i in range(1, num_harmonics+1):
    C[:,2*i-1] = np.sin(i*2*np.pi *f_0*times)
    C[:,2*i] = np.cos(i*2*np.pi*f_0*times)

#calculations
u, s, vt = np.linalg.svd(C,full_matrices=False)

winv = np.zeros(len(s))
winv[s > 1e-15] = 1.0 / s[s > 1e-15]
ainv = vt.transpose().dot(np.diag(winv)).dot(u.transpose())

c = ainv.dot(signal)
modelC = C.dot(c)

#plot statements
plt.plot(times, signal, '.', label="Actual data")
plt.plot(times, modelC, '.', label="Model")
plt.title("Data and SVD Fit of Data for Sines and Cosines")
plt.ylabel("Signal (arbitrary units)")
plt.xlabel("Time (s)")
plt.legend()
plt.show()

residuals2 = (modelC-np.array(signal.values))
sum_residuals = 0
for i in residuals2:
    sum_residuals += abs(i)
print(sum_residuals/len(signal.values))
