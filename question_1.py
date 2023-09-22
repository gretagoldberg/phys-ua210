import numpy as np

#defining function to evaluate
def function(x):
    return x*(x-1)

#calculating the derivative using the limit definition
def derivative_of_function(delta):
    return (function(1+delta)-function(1))/delta

#derivative for delta=10^-2
print(derivative_of_function(10**(-2)))

#actual derivative
def true_value(x):
    return 2*x-1

#actual value
print(true_value(1))

#the two answers do not agree perfectly because delta is quite large in comparison to zero

deltas = [10**(-4),10**(-6),10**(-8),10**(-12),10**(-14)]

for i in deltas:
    print(derivative_of_function(i))

#the value seems to approach the correct value and then gets worse because of how computers store and deal with floats