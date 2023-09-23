from numpy.random import random
import numpy as np
import matplotlib.pyplot as plt

#initializing variables
N = 1000
tau = 3.053*60
mu = np.log(2)/tau

z = random(N)

time_dec = -1/mu*np.log(1-z)
time_dec = np.sort(time_dec)
decayed = np.arange(1,N+1)

#updating number of atoms
atoms_left = N -decayed

#plot statements
plt.plot(time_dec,atoms_left)
plt.xlabel("Time (s)")
plt.ylabel("Number of Ti Atoms")
plt.title("Simulation of Decay of Ti Atoms over Time")
plt.show()