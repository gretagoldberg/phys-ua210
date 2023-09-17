import numpy as np
import matplotlib.pyplot as plt

xmin, xmax = -2, 2
ymin, ymax = -2, 2

#creating values for the complex plane
x = np.linspace(xmin, xmax)
y = np.linspace(ymin, ymax)
X, Y = np.meshgrid(x, y)
c = X + 1j * Y

#initializing arrays to store mandelbrot set
z = np.zeros_like(c)
mandelbrot = np.zeros_like(c, dtype=np.uint8)

#defining N/number of iterations
N = 100

#looping through values and checking that they are less then 2 and then calculating output number
for i in range(N):
    mask = np.abs(z) < 2.0
    z[mask] = z[mask] ** 2 + c[mask]
    mandelbrot += mask

#plotting set with heatmap that tells how many iterations it took for c to be too large
plt.imshow(mandelbrot, extent=(xmin, xmax, ymin, ymax), cmap='jet', interpolation='bilinear')
plt.colorbar()
plt.title("Mandelbrot Set on the Complex Plane")
plt.xlabel("Real Numbers")
plt.ylabel("Imaginary Numbers")
plt.show()
