import numpy as np
import time
import matplotlib.pyplot as plt

#defining function to do matrix multiplication
def matrix_multiplication(A, B):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    C = np.zeros((rows_A, cols_B))

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]

    return C

#creating list to store dimensions of matrices 
N = [10, 30, 50, 70, 100, 150, 200]

# Initialize lists to store computation times for both methods
times_explicit = []
times_dot = []

for i in N:
    A = np.random.rand(i, i)
    B = np.random.rand(i, i)

    #explicit method
    start_time = time.time()
    result_explicit = matrix_multiplication(A, B)
    end_time = time.time()
    elapsed_time_explicit = end_time - start_time
    times_explicit.append(elapsed_time_explicit)

    #dot method
    start_time = time.time()
    result_dot = np.dot(A, B)
    end_time = time.time()
    elapsed_time_dot = end_time - start_time
    times_dot.append(elapsed_time_dot)

#plots
plt.plot(N, times_explicit, marker='*', label='explicit calculation',c="pink")
plt.plot(N, times_dot, marker='*', label='dot method',c="purple")
plt.xlabel('Matrix Size (NxN)')
plt.ylabel('Computation Time (s)')
plt.title('Computation Time for Matrix Multiplication vs Matrix Size')
plt.legend()
plt.show()

#this does approximately exhibit N^3 behavior as predicted
