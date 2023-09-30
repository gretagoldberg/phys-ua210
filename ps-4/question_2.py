import numpy as np
import matplotlib.pyplot as plt
import warnings

#suppress RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#defining Newman's function
def gaussxw(N):

    #initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))

    #find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

#defining potential
potential = lambda x : x**4

#defining variables
N = 20 #number of points
m = 1 #mass

#getting sample points and weights before mapping
sample_points, weights = gaussxw(N)

#defining function that includes mapping from -1 to 0 and +1 to 2
def period(a):
    a_0 = 0
    a_1 = a
    #mapping
    specific_x = 0.5*(a_1-a_0)*sample_points + 0.5*(a_1+a_0)
    specific_w = 0.5*(a_1-a_0)*weights
    #total energy
    E_total = potential(a_1)
    #prefactor
    coefficient = np.sqrt(8*m)
    try:
        #actual integral
        y = 1 / np.sqrt(E_total-potential(specific_x))
        total_sum = coefficient*sum(y*specific_w)
        return total_sum
    except (RuntimeWarning, ZeroDivisionError):
        return np.nan  #dealing with errors that have shown up

#defining amplitudes to iterate through
amplitudes = np.linspace(0,2,100)

#making list to store periods
period_of_oscillation = []

#iterating through amplitudes and storing for graphing
for i in amplitudes:
    period_of_oscillation += [period(i)]

#plotting statements
plt.plot(amplitudes,period_of_oscillation,c="hotpink")
plt.xlabel('Amplitude (m)')
plt.ylabel('Period of Oscillation (s)')
plt.title("Amplitude versus Period of Oscillation for Harmonic Oscillator")
plt.show()