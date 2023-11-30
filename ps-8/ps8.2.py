import numpy as np
import matplotlib.pyplot as plt

#defining constants
sigma = 10
d= 28
c= 8/3

#function to calculate vector function
def f(r,t):
    #x, y and z components
    x = r[0]
    y = r[1]
    z = r[2]
    fx = sigma*(y-x)
    fy = d*x-y-x*z
    fz = x*y-c*z
    return np.array([fx,fy,fz],float)

#start of the interval
a = 0.0
#end of interval
b = 100.0
#step number
N = 10000
#step size
h = (b-a)/N

#lists to keep track of time
tpoints = np.arange(a,b,h)
xpoints = []
ypoints = []
zpoints = []

#initial conditions
r = np.array([0,1,0],float)

#iterating through time and actually doing the RK4
for t in tpoints:
    xpoints.append(r[0])
    ypoints.append(r[1])
    zpoints.append(r[2])				
    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6

#plot y versus time
plt.plot(tpoints,ypoints)
plt.ylabel("Y")
plt.xlabel("T")
plt.title("Y versus Time")
plt.show()

#plot of z versus x
plt.plot(zpoints,xpoints)
plt.xlabel('X')
plt.ylabel('Z')
plt.title("Z versus X (The Strange Attractor)")
plt.show()