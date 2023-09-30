import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import scipy

#Suppress RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#defining function to calculate hermite polynomial
def hermite_polynomial(n,x):
    if n==0:
        return 1
    elif n==1:
        return 2*x
    else:
        return 2*x*hermite_polynomial(n-1,x)-2 *(n-1)*hermite_polynomial(n-2,x)
    
#defining function for psi
def psi(n,x):
    prefactor = 1/np.sqrt(2**n*math.factorial(n)*np.sqrt(np.pi))
    return prefactor*np.exp(-x**2/2)*hermite_polynomial(n,x)

#defining arrays for calculating with
n_values = [0, 1, 2, 3]
x_values = np.arange(-4, 5, .001)

#calculate psi values
psi_values = np.array([psi(n, x_values) for n in n_values])

#plot statements
for i, n in enumerate(n_values):
    plt.plot(x_values, psi_values[i], label=f'n={n}')
plt.legend()
plt.xlabel('Position x (m)')
plt.ylabel('Wavefunction Amplitude (m)')
plt.title("Wavefunction with Respect to Position")
plt.show()

#second part w n larger and larger x range
#defining arrays for calculating with
n_values = np.arange(31)
x_values = np.arange(-10, 11,.1)

#calculate psi values
psi_values = np.array([psi(n, x_values) for n in n_values])

#plot statements
for i in range(0,31):
    plt.plot(x_values, psi(i,x_values),label=f"N = {i}")
plt.legend(fontsize=5, markerscale=5)
plt.xlabel('Position x (m)')
plt.ylabel('Wavefunction Amplitude (m)')
plt.title("Psi at Different Positions")
plt.show()


#importing guassian quadrature again from Newman's method
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

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

def gquad(function, b):
    xp, wp = gaussxwab(N,a,b)

    s = 0.0

    for k in range(N):
        s += wp[k]*function(xp[k])

    return s

#part c
#defining variables
N = 100
n = 5
a = -1

#defining function to calculate <x^2>
def uncertainty(x):
    coeff = ((x/(1-x**2))**2)*(1+x**2)/((1-x**2)**2)
    wavefunction = psi(n,(x/(1-x**2)))
    return coeff*abs(wavefunction)**2

value = gquad(uncertainty,1)
rms = value**(1/2)
print(rms)

#Gauss Hermite Quad section
N = 500
sample_points, weights = scipy.special.roots_hermite(N, mu=False)

#calculating whole integral sum
def Hermite(function):
    total = 0.0
    for k in range(N):
        total += weights[k]*function(sample_points[k])
    return total

#rms print statement
print(np.sqrt(Hermite(uncertainty)))
