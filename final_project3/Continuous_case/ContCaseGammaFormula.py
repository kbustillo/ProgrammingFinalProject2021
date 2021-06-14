#In order to have a realistic example, our aggregare claim size disctribution is a compound poisson 
#and the individual claims are exponential

#A way to avoid convolution !

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
import math
import time
import sys
import matplotlib
matplotlib.use('Agg')



def cdfS(n, Lambda, intensity, x): 
    
    ''' 
    Parameters
    ----------
    n : INTEGER
        IS THE NUMBER OF TERMS IN OUR SUM. IT'S CHOSEN SUCH AS IT'S CONSIDERED AS INFINITY, MEANING THAT THE 
        PROBABILITY THAT THE POISSON OF PARAMETER LAMBDA IS EQUAL TO N IS TOO SMALL (SMALLER THAN SOME EPSILON)
    Lambda : INTEGER
        IS THE PARAMETER OF OUR POISSON RANDOM VARIABLE
    intensity : DOUBLE
        IS THE PARAMETER OF OUR EXPONENTIAL RANDOM VARIABLE
    x : POSITIVE REAL
        THE POINT AT WHICH THE PROBABILITY MASS FUNCTION OF OUR AGGREGATE CLAIM SIZE IS EVALUATED

    Returns
    -------
    somme : REAL NUMBER BETWEEN 0 AND 1
        P(S <= x) 
    '''
     
    if (n == 0): 
        return math.exp(- Lambda)
    elif (n == 1):
        #WE HAVE ONLY ONE RANDOM VARIABLE
        return expon.cdf(x, loc = 0, scale = 1/intensity) * (math.exp(- Lambda * Lambda)) + math.exp(- Lambda)
    else:
        somme = expon.cdf(x, loc = 0, scale = 1/intensity) * (math.exp(- Lambda * Lambda)) + math.exp(- Lambda)
        for i in range(2, n+1): #we enter the sum
            # compute first P(N = n)
            p1 = poisson.pmf(i, Lambda, 0) #P(N= i)
            #instead of simulating, we know that the sum of the exponential's is just a gamma
            #since we're computing the cdf 
            p2 = gamma.cdf(x, i, 0, 1/intensity)
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
    
    values = np.arange(0, 8000, 1)
    
    # First method: the exact method, we'll call the function cdfS from DiscreteCasePoissonFormula
    t0 = time.time()
    AggDist=[]
    for x in values:
        AggDist.append(cdfS(n, Lambda, intensity, x))
    t1 = time.time()   
    print("Finding the distribution of S took ", t1 - t0, " seconds.")
    proba1 = cdfS(n, Lambda, intensity, s0)
    proba2 = 1 - cdfS(n, Lambda, intensity, s1)
    print("The probability that S is smaller than ", s0, " is : ", proba1)
    print("The probability that S is larger than ", s1, " is : ", proba2)
    #print("The probability that S is equal to ", s0," is : ",cdfS(n, p, m, Lambda, s0) - cdfS(n, p, m, Lambda, s0-1) )

    #plotting
    fig, ax = plt.subplots( nrows=1, ncols=1 ) # create figure & 1 axis
    ax.plot(values, AggDist)
    ax.set_ylabel('P(S<=s)')
    ax.set_xlabel('s')
    ax.set_title("Exact method plot", loc='center')
    fig.savefig('ExactMethodplot.png') # save the figure to file
    plt.close(fig) # close the figure window
    plt.savefig(sys.stdout.buffer)
    