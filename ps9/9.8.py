import numpy as np
import banded as band
import matplotlib.pyplot as plt

#Setting up initial variables
L = 1
x = np.linspace(0, L, 1001)
a = L/1000
h = 10**-4
times = np.arange(0, 0.1, h)


#Setting up initial conditions
x_0 = L/2.
sigma = L/10.
kappa = 50./L
psi = np.exp(-1* ((x - x_0)**2)/(2*sigma**2)) * np.exp(1j * kappa * x)
psi[0] = psi[-1] = 0


# Setting up a1, a2, b1, b2
a1 = 1 + 1j*(h/(2*a**2))
a2 = -1j*(h/(4*a**2))
b1 = 1 - 1j*(h/(2*a**2))
b2 = 1j*(h/(4*a**2))


# Creating matrix A
N = 1001
A_diag = np.ones(N, dtype=complex)*a1
A_u = np.ones(N, dtype=complex) * a2
A_u[0] = 0
A_l = np.ones(N, dtype=complex) * a2
A_l[-1] = 0
# build matrix
A = np.array([A_u, A_diag, A_l])


#Getting psi values 
psi_values = []

for t in times:

    psi_values.append(psi)
    psiold = psi
    
    # calculate v
    psiold = np.concatenate(([0],psi,[0])) 
    v = b1*psiold[1:-1] + b2*(psiold[2:]+psiold[:-2])
    
    # Solve matrix
    psi = band.banded(A,v,1,1)
    psi[0] = psi[-1] = 0


#Convert psi values list to array
psi_values = np.array(psi_values, dtype=complex)
real_parts = np.real(psi_values)


#Plot for time t=0
plt.plot(x, real_parts[0], label='(t=0)')
plt.xlabel('Position (1e-8 m)', fontsize = 16)
plt.ylabel('Amplitude', fontsize = 16)
plt.ylim(-1.1,1.1)
plt.legend()
plt.show()


#Plot for t=0.0014
plt.plot(x, real_parts[140], label='(t=0.0014)', color = 'red')
plt.xlabel('Position (1e-8 m)', fontsize = 16)
plt.ylabel('Amplitude', fontsize = 16)
plt.ylim(-1.1,1.1)
plt.legend()
plt.show()


#Plot for t=0.0040
plt.plot(x, real_parts[400], label='(t=0.0020)', color = 'orange')
plt.xlabel('Position (1e-8 m)', fontsize = 16)
plt.ylabel('Amplitude', fontsize = 16)
plt.ylim(-1.1,1.1)
plt.legend()
plt.show()


#Plot for t=0.0099
plt.plot(x, real_parts[999], label='(t=0.0099)', color = 'indigo')
plt.xlabel('Position (1e-8 m)', fontsize = 16)
plt.ylabel('Amplitude', fontsize = 16)
plt.ylim(-1.1,1.1)
plt.legend()
plt.show()