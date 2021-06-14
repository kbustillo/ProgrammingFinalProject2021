# This file contains all the discretization process tools

from scipy import ndimage
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
from scipy import signal
import math
from MadeCDF import madeCDF
import sys
import matplotlib
matplotlib.use('Agg')

#if we enter a vector we need to extract another from it, according to a condition:
    
def extraction(V, step):
    #V = np.arange(V)
    condition = np.mod(V, step) == 0
    Y = np.extract(condition, V)
    return Y
    
def lowerbound(FY, v, step):
    '''
    Parameters
    ----------
    FY : POSITIVE REAL NUMBERS BETWEEN 0 AND 1
        CDF OF Y.
     v : INTEGER
        THE EXTRACTED INDICES VECTOR.
    step : INTEGER
        THE SIZE OF THE BANDWIDTH, WITH WITCH WE EXTRACT THE INDICES VECTOR AND DECIDE OF THE CLOSENESS OF 
        THE UPPER AND LOWER BOUND.
    Returns
    -------
    THE LOWER BOUND PMF VECTOR

    '''
    Res = []
    for i in range(step):
        Res.append(FY[0])
    for j in v[1:]:
        for i in range(j, j+step):
            Res.append(FY[j] - FY[j-step])
    return Res
    
def upperbound(FY, v, step):
    '''
    Parameters
    ----------
    FY : POSITIVE REAL NUMBERS BETWEEN 0 AND 1
        CDF OF Y.
    v : INTEGER
        THE EXTRACTED INDICES VECTOR.
    step : INTEGER
        THE SIZE OF THE BANDWIDTH, WITH WITCH WE EXTRACT THE INDICES VECTOR AND DECIDE OF THE CLOSENESS OF 
        THE UPPER AND LOWER BOUND.

    Returns
    -------
    THE UPPER BOUND PMF VECTOR

    '''
    Res = []
   
    for j in v[:len(v)-1]:
        for i in range(j, j+step):
            Res.append(FY[j+step] - FY[j])
    for i in range(step):
        Res.append(FY[v[len(v)-1]])
    return Res

if __name__ == "__main__" : 
    mean = 100
    intensity = 1/mean
    X = np.arange(0, 400, 1)
    Cdf = expon.cdf(X, loc=0, scale = 1/intensity)
    # we need to choose a step in order to discretize the exponential random variable
    step = 10
    # we'll first discretize the indices vector
    index = extraction(X, step)

    #Now we find the lower and upper bound

    lw = lowerbound(Cdf, index, step)
    up = upperbound(Cdf, index, step)

    # Now we find the CDF of the lower and upper bounds that we found

    lowcdf = []
    for x in index:
        lowcdf.append(madeCDF(x, lw, step))
    
    uppcdf = []
    for x in index:
        uppcdf.append(madeCDF(x, up, step))
        
#plotting
    fig, ax = plt.subplots( nrows=1, ncols=1 ) # create figure & 1 axis
    ax.plot(X, Cdf, label="The continuous random variable")
    ax.step(index, lowcdf, "r--", label="The lower bound")
    ax.step(index,uppcdf, "b--", label = "The upper bound")
    ax.set_ylabel('F(s)')
    ax.set_xlabel('s')
    plt.legend()
    ax.set_title("Discretization plot", loc='center')
    fig.savefig('Discretization_step10.png') # save the figure to file
    plt.close(fig) # close the figure window
    plt.savefig(sys.stdout.buffer)
            