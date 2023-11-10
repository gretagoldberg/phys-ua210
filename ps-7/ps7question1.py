import numpy as np
from scipy import optimize

w = (1 + np.sqrt(5))/2

def golden_ratio(func,a,b,tolerance):
    x1 = a
    x4 = b
    x2 = x4-(x4-x1)/w
    x3 = x1+(x4-x1)/w
    while x4-x1 > tolerance:
        f1 = func(x1)
        f2 = func(x2)
        f3 = func(x3)
        f4 = func(x4)
        if f2 < f3:
            x4 = x3
            x3 = x2
            x2 = x4-(x4-x1)/w
        else:
            x1 = x2
            x2 = x3
            x3 = x1+(x4-x1)/w
    return (x2+x3)/2 

def brents(f,a,b,tol):
    if abs(f(a)) < abs(f(b)):
        a,b=b,a
    c = a
    used_bisection = True
    error = abs(b-a)
    error_list = [error]
    b_values = [b]
    while error > tol:
        if f(a) != f(c) or f(b) != f(c):
            s0 = a*f(b)*f(c)/(tol+(f(a)-f(b))*(f(a)-f(c)))
            s1 = b*f(a)*f(c)/(tol+(f(b)-f(a))*(f(b)-f(c)))
            s2 = c*f(a)*f(b)/(tol+(f(c)-f(a))*(f(c)-f(b)))
            s = s0+s1+s2
        else:
            s = b-(f(b)*(b-a)/(f(b)-f(a)))
        #check a bunch of conditions
        condition1 = ((s >= b))
        condition2 = used_bisection and abs(s-b) >= abs(b-c)/2
        condition3 = not used_bisection and abs(s-b) >= abs(c-d)/2
        if condition1 or condition2 or condition3:
            s = (a+b)/2
            used_bisection = True
        else:
            used_bisection = False
        c,d = b,c
        if f(a)*f(b) < 0:
            b = s
        else:
            a = s
        if abs(f(a)) < abs(f(b)):
            a, b = b, a
        error = abs(b-a)
        error_list.append(error)
        b_values.append(b)
    return b

f = lambda x: np.exp(x)*(x-.3)**2
tolerance = 1e-9

#my method
print(f"My implementation gives: {brents(f,-1,5,tolerance)}")
#scipy's method
print(f"The scipy implementation returns: {optimize.brent(f,brack=(-.5,5))}")