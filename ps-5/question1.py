import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scip
import math

#defining function to express the integrand
def integrand1(a, x):
    return x**(a-1) * np.exp(-x)

#making x values to plot with
x_values = np.linspace(0, 5, 100)

#plotting statements
plt.plot(x_values, integrand1(2, x_values), label="a=2")
plt.plot(x_values, integrand1(3, x_values), label="a=3")
plt.plot(x_values, integrand1(4, x_values), label="a=4")
plt.legend()
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Gamma Function Integrand")
plt.show()

#defining function to integrate w/ change of first term
def integrand2(u, a):
    c = a - 1
    if u == 1:
        return 0  #dealing w/ zero
    return np.exp((a-1)*np.log(c*u/(1-u))-(c*u/(1-u)))*(c/(u- 1)**2)

#defining function to compute integral for any value of a
def gamma(a):
    answer, error = scip.quad(integrand2, 0, 1, args=(a))
    return answer

#checking with a=3/2
print(gamma(3/2))

#checking in terms of factorials
a3 = gamma(3)
a6 = gamma(6)
a10 = gamma(10)

factorial2 = math.factorial(2)
factorial5 = math.factorial(5)
factorial9 = math.factorial(9)

#rounding the ones for a=6,10 since they're a little off
rounded10 = "{:.0f}".format(a10)
roundedfactorial10 = "{:.0f}".format(factorial9)
rounded6 = "{:.0f}".format(a6)
roundedfactorial6 = "{:.0f}".format(factorial5)


#comparing values with boolean condition
print(int(factorial2) == int(a3))
print(rounded6 == roundedfactorial6)
print(rounded10 == roundedfactorial10)
