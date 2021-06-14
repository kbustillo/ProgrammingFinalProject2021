
import numpy as np 
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
import matplotlib.pyplot as plt 
import matplotlib
import sys
matplotlib.use('Agg')
import math
import time

# Now we implement the n-fold convolution but using np.convolve
# Recursive formula
def ConvN_NP(n, x):
    '''

    Parameters
    ----------
    n : INTEGER
        THE NUMBER OF RANDOM VARIABLES WE'RE CONVOLUTING.
    x : VECTOR OF POSITIVE NUMBERS
        VECTOR CONTAINING THE PDF OF THE CLAIM SIZE RANDOM VARIABLE.

    Returns
    -------
    POSITIVE REAL NUMBER BETWEEN 0 AND 1
        THE PDF OF X1 + ... Xn

    '''
    if n == 0:
        return np.concatenate(([1], np.zeros(len(x))-1), axis = None)
    elif n == 1:
        return x
    else:
        x = np.array(x)
        conv = x
        for n in range(2, n+1):
            conv = np.convolve(conv, x)
        return conv
    
# Now we implement the CDF of S
def NP_pmfS(n, FY, fY, Lambda, intensity):
    '''
    

    Parameters
    ----------
    n : INTEGER
        IS THE NUMBER OF TERMS IN OUR SUM. IT'S CHOSEN SUCH AS IT'S CONSIDERED AS INFINITY, MEANING THAT THE 
        PROBABILITY THAT THE poisson random variable OF PARAMTER Lambda EQUAL TO n IS TOO SMALL (SMALLER THAN SOME EPSILON)
    m : INTEGER
        IS THE FIRST PARAMETER OF OUR NEGATIVE BINOMIAL RANDOM VARIABLE
    p : POSITIVE REAL NUMBER BETWEEN 0 AND 1
        IS THE SECOND PARAMETER OF OUR BINOMIAL RANDOM VARIABLE.
    fY : VECTOR OF POSITIVE REAL NUMBERS BETWEEN 0 AND 1
        VECTOR OF THE PMF OF THE CLAIM SIZE DISTRIBUTION

    Returns
    -------
    POSITIVE REAL NUMBER BELONGING TO [0, 1]
        THE CDF OF THE COMPOUD DISTRIBUTION s

    '''
    if n == 0: #SPECIAL CASE NO RANDOM VARIABLE
        inter = (poisson.pmf(0, Lambda)) * np.ones(len(FY))
        return inter
    elif n== 1: #WE HAVE ONLY ONE RANDOM VARIABLE
        inter = (poisson.pmf(0, Lambda)) * np.ones(len(FY))
        return FY * poisson.pmf(1, Lambda) + inter
    else:
        inter = inter = (poisson.pmf(0, Lambda)) * np.ones(len(FY))
        somme = FY * poisson.pmf(1, Lambda) + inter
        for i in range(2, n+1): #we enter the sum
        # compute first P(N = n)
            p1 = poisson.pmf(i, Lambda) #P(N= i)
        #instead of simulating, we know that the sum of the Poisson rv with the same parameter is just a poisson
        #with i times Lambda
        #since we're computing the cdf 
            p2 = ConvN_NP(i,FY, fY)
            somme = np.concatenate((somme, np.zeros(len(p2)-len(somme))), axis = None)
            somme += p1 * p2
        
        return somme
    
    
if __name__ == "__main__":
    
   # In the continusous case, we'll pick a poisson rv for the number of claims, and exponential random variables for the claim sizes. 

    #For the poisson random variable we'll chose Lambda to be equal to 40, so in average we have 40 claims per year 
    #For the claim sizes, we'll take exponential random variables with intensity 1/100, so our expected claims are 100 each. 
            
    Lambda = 40 
    mean = 100
    intensity = 1/mean
    
    # Here we choose the number n such that the probability that our poisson r.v is equal to this n is too small
    # we take n equal to 100, poisson.pmf(100, 40) =7.315031522325226e-16
    n = 100
    # We'll try to answer the following questions: What is the probability that the aggregate claim size will be smaller than some value s0 ? 
    # And what is the probability that the aggregate claim size will be larger than s1 ?
    #we'll also plot all the resulting distribution.
    
    s0 = 2000
    s1 = 7000
    values = np.arange(0, 500, 1)
    values2 = np.arange(0, 14000,1)

    # Getting the distribution of one claim size random variable

    fY = expon.pdf(values, 0, 1/intensity)
    FY = expon.cdf(values, 0, 1/intensity)
    FS = gamma.cdf(values2, n, 0, 1/intensity)
    t0 = time.time()
    #z = NP_pmfS(n, FY, fY, Lambda, intensity)
    z = ConvN_NP(n,fY)
    t1 = time.time() 
    print("Finding the distribution of S took ", t1 - t0, " seconds.")
    proba1 = z[s0 + 1]
    proba2 = 1- z[s1 + 1]
    print("The probability that S is smaller than ", s0, " is : ", proba1)
    print("The probability that S is larger than ", s1, " is : ", proba2)
    #plotting
    fig, ax = plt.subplots( nrows=1, ncols=1 ) # create figure & 1 axis
    ax.plot(z, "r")
    ax.plot(FS, "b")
    ax.set_ylabel('P(S<=s)')
    ax.set_xlabel('s')
    ax.set_title("np.convolve method", loc='center')
    fig.savefig('npConvolveMethod.png') # save the figure to file
    plt.close(fig) # close the figure window
    plt.savefig(sys.stdout.buffer)
    