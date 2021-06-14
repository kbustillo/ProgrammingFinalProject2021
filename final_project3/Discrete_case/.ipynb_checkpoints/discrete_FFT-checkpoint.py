from scipy import signal
import numpy as np 
from scipy.stats import nbinom
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
        VECTOR CONTAINING THE PMF OF THE CLAIM SIZE RANDOM VARIABLE X.

    Returns
    -------
    POSITIVE REAL NUMBER BETWEEN 0 AND 1
        THE PMF OF X1 + ... Xn

    '''
    if n == 0:
        return np.concatenate(([1], np.zeros(len(x))-1), axis = None)
    elif n == 1:
        return x
    else:
        x = np.array(x)
        conv = x
        for n in range(2, n+1):
            conv = signal.fftconvolve(conv, x)
        return conv
    
# Now we implement the pmf of S
def FFT_pmfS(n, p, m, fY):
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
    fY : VECTOR OF POSITIVE REAL NUMBERS BETWEEN 0 AND 1
        VECTOR OF THE PMF OF THE CLAIM SIZE DISTRIBUTION

    Returns
    -------
    POSITIVE REAL NUMBER BELONGING TO [0, 1]
        THE PMF OF THE COMPOUD DISTRIBUTION s

    '''
    if n == 0: #SPECIAL CASE NO RANDOM VARIABLE
        inter = ((p) ** m) * np.concatenate(([1], np.zeros(len(fY)-1)), axis = None)
        return inter
    elif n== 1: #WE HAVE ONLY ONE RANDOM VARIABLE
        inter = ((p) ** m) * np.concatenate(([1], np.zeros(len(fY)-1)), axis = None)
        return fY * nbinom.pmf(1, m, p) + inter
    else:
        inter = ((p) ** m) * np.concatenate(([1], np.zeros(len(fY)-1)), axis = None)
        somme = fY * nbinom.pmf(1, m, p) + inter
        for i in range(2, n+1): #we enter the sum
        # compute first P(N = n)
            p1 = nbinom.pmf(i, m, p) #P(N= i)
        #instead of simulating, we know that the sum of the Poisson rv with the same parameter is just a poisson
        #with i times Lambda
        #since we're computing the cdf 
            p2 = ConvN_NP(i, fY)
            somme = np.concatenate((somme, np.zeros(len(p2)-len(somme))), axis = None)
            somme += p1 * p2
        
        return somme
    
    
if __name__ == "__main__":
    
    # In the discrete, we'll pick a negative bionomial rv for the number of claims, and poisson random variables for the claim sizes. 

    #For the negative binomial we'll chose p equal to 0.5 and m = 80, so in average we have 80 claims per year 
    #For the claim sizes, we'll take poisson random variable of parameter 60, so our expected claims are 60 each. 
            
    m = 80 
    p = 0.5
    Lambda = 60
    
    # Here we choose the number n such that the probability that our negative binomial is equal to this n is too small
    # we take n equal to 160, nbinom.pmf(160, 80, 0.5) = 2.2743878591369746e-08
    n1 = 160
    # We'll try to answer the following question: What is the probability that the aggregate claim size will be higehr than some value s0 ? 
    #we'll also plot all the resulting distributions.
    s0 = 2500
    s1 = 7000
    
    values = np.arange(0, 100, 1)

    # Getting the distribution of one claim size random variable

    fY = poisson.pmf(values, Lambda)
    t0 = time.time()
    y = FFT_pmfS(n1, p, m, fY)
    z = np.cumsum(y)
    t1 = time.time() 
    print("--------------------------------------------------------------------------------------------------------------")
    print("Finding the distribution of S using FFT when n = ",n1 ,"took ", t1 - t0, " seconds.")
    proba1 = z[s0 + 1]
    proba2 = 1- z[s1 + 1]
    print("The probability that S is smaller than ", s0, " is : ", proba1)
    print("The probability that S is larger than ", s1, " is : ", proba2)
    print("---------------------------------------------------------------------------------------------")
    #choosing a bigger n, one that is bigger than 500, because in this case FFT might work faster
    
    n2 = 700
    t0 = time.time()
    y = FFT_pmfS(n2, p, m, fY)
    z = np.cumsum(y)
    t1 = time.time() 
    print("Finding the distribution of S using FFT when n = ",n2 ," took ", t1 - t0, " seconds.")
    proba1 = z[s0 + 1]
    proba2 = 1- z[s1 + 1]
    print("The probability that S is smaller than ", s0, " is : ", proba1)
    print("The probability that S is larger than ", s1, " is : ", proba2)
    print("--------------------------------------------------------------------------------------------------------------")
    
    
    n3 = 1000
    t0 = time.time()
    y = FFT_pmfS(m3, p, m, fY)
    z = np.cumsum(y)
    t1 = time.time() 
    print("Finding the distribution of S using FFT when n = ",n3 ," took ", t1 - t0, " seconds.")
    proba1 = z[s0 + 1]
    proba2 = 1- z[s1 + 1]
    print("The probability that S is smaller than ", s0, " is : ", proba1)
    print("The probability that S is larger than ", s1, " is : ", proba2)
    print("--------------------------------------------------------------------------------------------------------------")
    
    n4 = 2000
    t0 = time.time()
    y = FFT_pmfS(n4, p, m, fY)
    z = np.cumsum(y)
    t1 = time.time() 
    print("Finding the distribution of S using FFT when n = ",n4 ," took ", t1 - t0, " seconds.")
    proba1 = z[s0 + 1]
    proba2 = 1- z[s1 + 1]
    print("The probability that S is smaller than ", s0, " is : ", proba1)
    print("The probability that S is larger than ", s1, " is : ", proba2)
    print("--------------------------------------------------------------------------------------------------------------")
    
    #plotting
    fig, ax = plt.subplots( nrows=1, ncols=1 ) # create figure & 1 axis
    ax.plot(z)
    ax.set_ylabel('P(S<=s)')
    ax.set_xlabel('s')
    ax.set_title("fft.convolve method", loc='center')
    fig.savefig('fftconvolve.png') # save the figure to file
    plt.close(fig) # close the figure window
    plt.savefig(sys.stdout.buffer)
    