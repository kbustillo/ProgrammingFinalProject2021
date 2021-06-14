#Another approach 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
import time
import math
import sys
def fSm(fS0, Lambda, m, a, b):
    
    '''
    Parameters
    ----------
    fY : POSITIVE REAL NUMBER BETWEEN 0 AND 1
        PROBABILITY MASS FUNCTION OF THE DISCRETE CLAIM SIZE Y
    fS0 : POSITIVE REAL NUMBER BETWEEN 0 AND 1
        P(S = 0)
    m : INTEGER GREATER OR EQUAL TO 0
        POINT AT WHICH WE'RE EVALUATING THE PROBABILITY MASS FCT OF THE AGGREGATE CLAIM SIZE
    a : INTEGER
        FIRST PARAMETER OF THE (a, b, 0) CLASS
    b : INTEGER
        SECOND PARAMETER OF THE (a, b, 0) CLASS
    

    Returns
    -------
    P(S = m)
    '''
    if m == 0:
        return fS0
    else:
        
        A = 1./(1- a * poisson.pmf(0, Lambda))
        
        list1 = [] #will contain P(S=0),P(S=1), P(S=2),...P(S=m) 
        list2 = []
        list1.append(fS0)
        
        #Build a list with general term from panjer recursion
        for i in range(1, m+1, 1):
            print("i = ", i) #in order to keep track of iterations.
            for k in range(1, i+1):
                
                B = (a + b*k/m)
                
                list2.append(A * poisson.pmf(k, Lambda) * list1[i-k] * B)
                
              
            C = sum(list2) 
            list2.clear()     
            list1.append(C)
            
        return list1[m]
    
    
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
    s = 2000
    s0 = 2500
    s1 = 7000
    
    #values = np.arange(0, 8000, 1)
    
        
    fY_0 = poisson.pmf(0, Lambda)
    PS0 = (p/ 1 - p* fY_0) ** m
    # findinf a, b of the class (a, b, 0) that the Negative binomial (m,p) is belonging to
    a = (1 - p)
    b = (1 - p) * (m - 1)
    t0 = time.time()
    Dist1 = fSm(PS0, Lambda, s0, a, b)
    t1 = time.time()   
    
    v0 = time.time()
    Dist2 = fSm(PS0, Lambda, s1, a, b)
    v1 = time.time()  
    
    print("-----------------------------------------------------------------------------------------------------------------------------------------")
    print("Using Panjer recursion (iterative version), the probability that S is equal to ", s0, " is : ", Dist1, ". Found in ", t1 - t0, " seconds.")
    print("Using Panjer recursion (iterative version), the probability that S is equal to ", s1, " is : ", Dist2, ". Found in ", v1 - v0, " seconds.")
    print("-----------------------------------------------------------------------------------------------------------------------------------------")
    #plotting
    #fig, ax = plt.subplots( nrows=1, ncols=1 ) # create figure & 1 axis
    #ax.plot(values, AggDist)
    #ax.set_ylabel('P(S<=s)')
    #ax.set_xlabel('s')
    #ax.set_title("Exact method plot", loc='center')
    #fig.savefig('ExactMethodplot.png') # save the figure to file
    #plt.close(fig) # close the figure window
    #plt.savefig(sys.stdout.buffer)
    
