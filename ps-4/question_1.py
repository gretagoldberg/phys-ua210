import numpy as np

#defining function to integrate
def f(x):
    return x**4-2*x+1

#defining variables
N1 = 10
N2 = 20
a = 0
b = 2
h1 = (b-a)/N1
h2 = (b-a)/N2

#calculating integral w/ 10 slices
s1 = 0.5*f(a)+0.5*f(b)
for k in range(1,N1):
    s1 += f(a+k*h1)
integral1 = h1*s1

#calculating integral w/ 20 slices
s2 = 0.5*f(a)+0.5*f(b)
for k in range(1,N2):
    s2 += f(a+k*h2)
integral2 = h2*s2

#error calculated using method from the book
error1 = 1/3*(integral1-integral2)

#direct error calculation
error2 = integral2-4.4

print(error1,error2)