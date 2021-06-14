#Another approach 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
import time
import math
import sys
def fSm(fS0, fY, m, a, b):
    
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
        
        A = 1./(1- a * fY[0])
        
        list1 = [] #will contain P(S=0),P(S=1), P(S=2),...P(S=m) 
        list2 = []
        list1.append(fS0)
        
        #Build a list with general term from panjer recursion
        for i in range(1, m+1, 1):
            print("i = ", i)
            for k in range(1, i+1):
                
                B = (a + b*k/m)
                
                list2.append(A * fY[k] * list1[i-k] * B)
                
              
            C = sum(list2) 
            list2.clear()     
            list1.append(C)
            
        return list1[m]
    