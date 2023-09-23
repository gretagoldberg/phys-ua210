from random import random
from numpy import arange
import matplotlib.pyplot as plt

#assigning variables
h = 1 
NBi209 = 0
NPb209 = 0
NTi209 = 0
NBi213 = 10000

#assigning probabilities
pPb = 1 - 2**(-h/3.3/60)
pTi = 1 - 2**(-h/2.2/60)
pBi = 1 - 2**(-h/46/60)

#initializing lists to store data over time
Bi209_list = []
Pb209_list = []
Ti209_lsit = []
Bi213_list = []

t = arange(0,20000,h)
for ti in t:
	Bi209_list.append(NBi209)
	Pb209_list.append(NPb209)
	Ti209_lsit.append(NTi209)
	Bi213_list.append(NBi213)
	
	for i in range(NPb209):
		if random()<pPb:
			NPb209-=1
			NBi209+=1
	
	for i in range(NTi209):
		if random()<pTi:
			NTi209-=1
			NPb209+=1
	
	for i in range(NBi213):
		if random()<pBi:
			NBi213 -=1
			if random()<0.9791:
				NPb209+=1
			else:
				NTi209+=1
				
plt.plot(t,Bi209_list,label='Bi209',c="blue")
plt.plot(t,Pb209_list,label='Pb209',c="green")
plt.plot(t,Ti209_lsit,label='Ti209',c = "orange")
plt.plot(t,Bi213_list,label='Bi213',c="red")
plt.legend()
plt.title("Simulating Decay Chain of 10,000 Atoms of 213_Bi")
plt.xlabel('Time (s)')
plt.ylabel('Number of Atoms')
plt.show()