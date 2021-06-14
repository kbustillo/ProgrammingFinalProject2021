#A way to avoid convolution !

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import poisson
from scipy.stats import nbinom
import sys
import matplotlib
matplotlib.use('Agg')
import math
import time

def cdfS(n, p, m, Lambda, x): 
     
    '''
    
    Parameters
    ----------
    n : INTEGER
        IS THE NUMBER OF TERMS IN OUR SUM. IT'S CHOSEN SUCH AS IT'S CONSIDERED AS INFINITY, MEANING THAT THE 
        PROBABILITY THAT THE negative BINOMIAL OF PARAMTERS p AND m IS EQUAL TO n IS TOO SMALL (SMALLER THAN SOME EPSILON)
    m : INTEGER
        IS THE FIRST PARAMETER OF OUR NEGATIVE BINOMIAL RANDOM VARIABLE
    p : POSITIVE REAL NUMBER BETWEEN 0 AND 1
        IS THE SECOND PARAMETER OF OUR BINOMIAL RANDOM VARIABLE.
    Lambda: INTEGER
        IS THE POISSON RANDOM VARIABLE PARAMETER
    x : POSITIVE REAL
        THE POINT AT WHICH THE CUMULATIVE DISTRIBUTION FUNCTION OF OUR AGGREGATE CLAIM SIZE IS EVALUATED

    Returns
    -------
    somme : REAL NUMBER BETWEEN 0 AND 1
        P(S <= x) 

    '''
    if n == 0: #SPECIAL CASE NO RANDOM VARIABLE
        return (p) ** m
    elif n== 1: #WE HAVE ONLY ONE RANDOM VARIABLE
        return poisson.cdf(x, Lambda) * nbinom.pmf(1, m, p) + (p) ** m
    else:
        somme = poisson.cdf(x, Lambda) * nbinom.pmf(1, m, p) + (p) ** m
        for i in range(2, n+1): #we enter the sum
        # compute first P(N = n)
            p1 = nbinom.pmf(i, m, p) #P(N= i)
        #instead of simulating, we know that the sum of the Poisson rv with the same parameter is just a poisson
        #with i times Lambda
        #since we're computing the cdf 
            p2 = poisson.cdf(x, i*Lambda)
            somme += p1 * p2
        
        return somme
if __name__ == "__main__":
    
    # In the discrete case, we'll pick a negative bionomial rv for the number of claims, and poisson random variables for the claim sizes. 

    #For the negative binomial we'll chose p equal to 0.5 and m = 80, so in average we have 80 claims per year 
    #For the claim sizes, we'll take poisson random variable of parameter 60, so our expected claims are 60 each. 
            
    m = 80 
    p = 0.5
    Lambda = 60
    
    # Here we choose the number n such that the probability that our negative binomial is equal to this n is too small
    # we take n equal to 160, nbinom.pmf(160, 80, 0.5) = 2.2743878591369746e-08
    n = 160
    # We'll try to answer the following questions: What is the probability that the aggregate claim size will be smaller than some value s0 ? 
    # And what is the probability that the aggregate claim size will be larger than s1 ?
    #we'll also plot all the resulting distribution.
    
    s0 = 2500
    s1 = 7000
    
    values = np.arange(0, 8000, 1)
    
    t0 = time.time()
    AggDist=[]
    for x in values:
        AggDist.append(cdfS(n, p, m, Lambda, x))
    t1 = time.time()   
    print("Finding the distribution of S using the exact method took ", t1 - t0, " seconds.")
    proba1 = cdfS(n, p, m, Lambda, s0)
    proba2 = 1 - cdfS(n, p, m, Lambda, s1)
    print("The probability that S is smaller than ", s0, " is : ", proba1)
    print("The probability that S is larger than ", s1, " is : ", proba2)
    print("The probability that S is equal to ", s0," is : ",cdfS(n, p, m, Lambda, s0) - cdfS(n, p, m, Lambda, s0-1) )
   
    #plotting
    fig, ax = plt.subplots( nrows=1, ncols=1 ) # create figure & 1 axis
    ax.plot(values, AggDist)
    ax.set_ylabel('P(S<=s)')
    ax.set_xlabel('s')
    ax.set_title("Exact method plot", loc='center')
    fig.savefig('ExactMethodplot.png') # save the figure to file
    plt.close(fig) # close the figure window
    plt.savefig(sys.stdout.buffer)
   
