import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift

#loading in files
piano = np.loadtxt("/Users/Greta/Documents/piano.txt")
trumpet = np.loadtxt("/Users/Greta/Documents/trumpet.txt")

plt.plot(piano)
plt.title("Original Piano Waveform")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

plt.plot(trumpet)
plt.title("Original Trumpet Waveform")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

#number of points
N = piano.shape[0]
#spacing
T = 1/44100
yf = fft(piano)
xf = fftfreq(N, T)
#xf2 = fftshift(xf)
#yf2 = fftshift(yf)
plt.xlim(-100,10000)
plt.plot(xf, 1.0/N * np.abs(yf))
plt.title("Piano FFT")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.show()

#number of points
NT = trumpet.shape[0]
#spacing
TT = 1/44100
yfT = fft(trumpet)
xfT = fftfreq(NT, TT)
plt.xlim(-100,10000)
plt.plot(xfT, 1.0/NT * np.abs(yfT))
plt.xlabel('Frequency (Hz)')
plt.title("Trumpet FFT")
plt.ylabel('Magnitude')
plt.grid()
plt.show()

