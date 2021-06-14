import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
import time
import math
import sys
from numba import jit
# We implement the Panjer recursion

sys.setrecursionlimit(3000)
#print(sys.getrecursionlimit())

@jit(forceobj=True)

def PanjerPS(PS0, Lambda, m, a, b):
    '''

    Parameters
    ----------
    PS0 : POSITIVE REAL NUMBER
        P(S = 0) IS THE PNG OF THE NUMBER OF CLAIMS R.V EVALUATED AT P(Y = 0), Y BEING THE INDIVIDUAL CLAIM SIZE.
    fY : POSITIVE REAL NUMBERS
        PMF OF Y.
    m : INTEGER
        THE POINT AT WHICH WE EVALUATE THE PMF OF S.
    a : INTEGER
        a OF THE (a, b, 0) CLASS.
    b : INTEGER
        b OF THE (a, b, 0) CLASS.

    Returns
    -------
    P(S = m)

    '''
    if m == 0:
        return PS0
    else:
        A = 1/(1- a * poisson.pmf(0, Lambda))
        somme = 0
        vec = [PS0]
        vec = np.array(vec)
        for k in range(1, m+1):
            print("k = ", k)  #in order to keep track of iterations. 
            B = (a + b * k / m)
            somme += poisson.pmf(k, Lambda) * PanjerPS(PS0, Lambda, m - k, a, b) * B
        return A * somme
    
if __name__ == "__main__":
    
    # In the discrete, we'll pick a negative bionomial rv for the number of claims, and poisson random variables for the claim sizes. 

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
    
    s0 = 2500
    s1 = 7000
    
    values = np.arange(0, 8000, 1)
    
        
    fY_0 = poisson.pmf(0, Lambda)
    PS0 = (p/ 1 - p* fY_0) ** m
    # findinf a, b of the class (a, b, 0) that the Negative binomial (m,p) is belonging to
    a = (1 - p)
    b = (1 - p) * (m - 1)
    t0 = time.time()
    
    AggDist = PanjerPS(PS0, Lambda, s0, a, b)
    t1 = time.time()   
    
    print("Using the Panjer recursion (recursive version) and Numba, the probability that S is equal to ", s0, " is : ", AggDist, ". Found in ", t1 - t0, " seconds.")
    
    #plotting
    #fig, ax = plt.subplots( nrows=1, ncols=1 ) # create figure & 1 axis
    #ax.plot(values, AggDist)
    #ax.set_ylabel('P(S<=s)')
    #ax.set_xlabel('s')
    #ax.set_title("Exact method plot", loc='center')
    #fig.savefig('ExactMethodplot.png') # save the figure to file
    #plt.close(fig) # close the figure window
    #plt.savefig(sys.stdout.buffer)
    