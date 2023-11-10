#ps7 question 2
import numpy as np
import jax
import jax.numpy as jnp
from scipy import optimize
import matplotlib.pyplot as plt

def logistic_func(x,b_0,b_1):
    return 1/(1+np.exp(-(b_0+b_1*x)))

ages = np.array(np.loadtxt("/Users/Greta/Downloads/survey.csv",skiprows=1,usecols=0,delimiter=","))
response = np.array(np.loadtxt("/Users/Greta/Downloads/survey.csv",skiprows=1,usecols=1,delimiter=","))

#so we need to estimate the coefficients of the logistic function
#we can do that by taking the sum of the log likelihoods, multiplying by negative one
#and then minimize this value rather than maximizing the sum

#the covariance matrix will be from the inverse hessian matrix and the variance
def covariance_matrix(inverse,variance):
    return inverse*variance

def log_likelihood(b0,b1,x,y):
    b_0 = b0 #value for beta_0
    b_1 = b1 #value for beta_1
    #sum over log values after initializing list
    values = []
    for i in range(len(x)):
        element = y[i]*np.log(logistic_func(x[i],b_0,b_1)/(1-logistic_func(x[i],b_0,b_1)))+np.log(1-logistic_func(x[i],b_0,b_1))
        values.append(element)
    total_loglikelihood = np.sum(np.array(values))
    negative_loglikelihood = -1*total_loglikelihood
    return negative_loglikelihood

#beta start values
beta = np.array([-0.01,0.01])

#minimizing the negative log likelihood
result = optimize.minimize(lambda beta, ages, response: log_likelihood(beta[0], beta[1], ages, response), beta,  args=(ages,response))

#using the atributes of scipy's optimize minimize to find other values
hessian_inverse = result.hess_inv
variance = result.fun/(len(response)-len(beta))
#variance is just the inverse hessian multiplied by the variance
covariance = hessian_inverse*variance

#error
error = np.sqrt(np.diag(covariance))
print(error)

grid = np.arange(0, 100, 1, dtype = float)
plt.plot(grid, logistic_func(grid, result.x[0], result.x[1]))
plt.grid()
plt.ylabel("Response (Y=1, N=0)")
plt.xlabel("Age (Years)")
plt.text(2, .9, f"β0 = -5.62, β1 = 0.11", fontsize = 8,bbox=dict(facecolor='white')) 
plt.title("Best Fit of Logistic Function")
plt.show()

print('Optimal parameters and error:\n\tp: ' , result.x, '\n\tdp: ', error)
print('Covariance matrix of optimal parameters:\n\tC: ' , covariance)




