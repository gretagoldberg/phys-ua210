import numpy as np
import timeit

def with_for_loop():
    sum = 0

    for i in range(-10,10):
        for j in range(-10,10):
            for k in range(-10,10):

            #check if the sum = 0
                if i == 0 and j == 0 and k == 0:
                    continue

                #if the conditions are met, add the value of M
                else:
                    sign = (-1)**((i + j + k )%2)
                    M = 1/np.sqrt(i**2+j**2+k**2)
                    total_factor = sign*M
                    sum += (-1)*total_factor

    print("first sum = ", sum)


def without_for_loop():
    #arrays for i,j,k
    i = np.arange(-10, 10, 1)
    j = np.arange(-10, 10, 1)
    k = np.arange(-10, 10, 1)

    #creating meshgrid
    I, J, K = np.meshgrid(i, j, k)

    #calculating the sum, excluding i=j=k=0 and making negative/positive depending on even/oddness
    mask = (I != 0) | (J != 0) | (K != 0)

    #calculating M for elements where the mask is True
    M = np.zeros_like(I, dtype=float)
    M[mask] = 1 / np.sqrt(I[mask]**2 + J[mask]**2 + K[mask]**2)

    #choosing sign based on even/odd condition
    total_factor = (-1)**((I + J + K) % 2) * M

    sum = np.sum(total_factor)

    print("new sum = ", (-1)*sum)

first_half_time = timeit.timeit(with_for_loop, number=1)
second_half_time = timeit.timeit(without_for_loop, number=1)

print("Execution time for the for loop = ", first_half_time, "seconds")
print("Execution time for the meshgrid version = ", second_half_time, "seconds")


