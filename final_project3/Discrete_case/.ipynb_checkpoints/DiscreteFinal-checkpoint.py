import time
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import poisson
from scipy.stats import binom
import math
import sys
from DiscreteCasePoissonFormula import cdfS
from Discrete_npconvolve import ConvN_NP
from Discrete_npconvolve import NP_pmfS
from Panjer_Recursive import PanjerPS
from Panjer_iterative import fSm
from discrete_FFT import FFT_pmfS


# In the discrete, we'll pick a negative bionomial rv for the number of claims, and poisson random variables for the claim sizes. 

#For the negative binomial we'll chose p equal to 0.5 and m = 80, so in average we have 80 claims per year 
#For the claim sizes, we'll take poisson random variable of parameter 60, so our expected claims are 60 each. 

m = 80 
p = 0.5
Lambda = 60

# Here we choose the number n such that the probability that our negative binomial is equal to this n is too small
# we take n equal to 160, nbinom.pmf(160, 80, 0.5) = 2.2743878591369746e-08
n = 160
# We'll try to answer the following question: What is the probability that the aggregate claim size will be higehr than some value s0 ? 
#we'll also plot all the resulting distributions.
s0 = 2500
s1 = 7000
values = np.arange(0, 8000, 1)

# First method: the exact method, we'll call the function cdfS from DiscreteCasePoissonFormula

t10 = time.time()
AggDist=[]
for x in values:
    AggDist.append(cdfS(n, p, m, Lambda, x))
t11 = time.time()   

proba11 = cdfS(n, p, m, Lambda, s0)
proba12 = 1 - cdfS(n, p, m, Lambda, s1)


# Second method : the convolution method using np.convolve, convN_NP

values = np.arange(0, 100, 1)

# Getting the distribution of one claim size random variable

fY = poisson.pmf(values, Lambda)
t0 = time.time()
y = NP_pmfS(n1, p, m, fY)
z = np.cumsum(y)
t1 = time.time() 

proba01 = z[s0 + 1]
proba02 = 1- z[s1 + 1]

    
n2 = 700
t00 = time.time()
y = NP_pmfS(n2, p, m, fY)
z = np.cumsum(y)
t01 = time.time() 

proba01 = z[s0 + 1]
proba02 = 1- z[s1 + 1]
    

# Third method : the fast fourier transform method (which is supposed to be the fastest!)

fY = poisson.pmf(values, Lambda)
t30 = time.time()
y = FFT_pmfS(n, p, m, fY)
z = np.cumsum(y)
t31 = time.time() 

proba31 = z[s0 + 1]
proba32 = 1- z[s1 + 1]

#choosing a bigger n, one that is bigger than 500, because in this case FFT works
    
n2 = 700
t40 = time.time()
y = FFT_pmfS(n2, p, m, fY)
z = np.cumsum(y)
t41 = time.time() 

proba41 = z[s0 + 1]
proba42 = 1- z[s1 + 1]


# fourth method : the Panjer recursion (recursive version), we'll call PanjerPS. Then use CDF

# we'll need to find PS0 first, meaning P(S=0) 
# PS0 = P(S = 0) = PN(fY(0))  PN is the probability generating function (PGF) of N
# fY(0) = prob(poisson randon variable of parameter Lambda is equal to 0)

fY_0 = poisson.pmf(0, Lambda)
PS0 = (p/ 1 - p* fY_0) ** m
# findinf a, b of the class (a, b, 0) that the Negative binomial (m,p) is belonging to
a = (1 - p)
b = (1 - p) * (m - 1)

# We set the recursion limit to 2000 
#sys.setrecursionlimit(2000)

#t0 = time.time()
#findinf P (S = s0). Since it's recursive formula, we know it's too much time consuming
#proba = PanjerPS(PS0, fY, s0, a, b)
#t1 = time.time()
#print("The probability that S is equal to ", s0, " is : ", proba, "and this took", t1 - t0, " seconds.")

# Setting aother limit to the recursion is considered dangerous, the tail recursion is not an efficient technique in Python --> so we wrerote the 
# algorithm iteratively ! 

#This method takes too much time, as it's a loop of recursions !! 


# Fifth method : the Panjer recursion (iterative version)

t50 = time.time()
Dist51 = fSm(PS0, Lambda, s0, a, b)
t51 = time.time()   
    
v50 = time.time()
Dist52 = fSm(PS0, Lambda, s1, a, b)
v51 = time.time()  
    

#Printing all the outputs

#first method

print("Finding the exact distribution of S took ", t11 - t10, " seconds.")
print("The probability that S is smaller than ", s0, " is : ", proba11)
print("The probability that S is larger than ", s1, " is : ", proba12)
print("The probability that S is equal to ", s0," is : ",cdfS(n, p, m, Lambda, s0) - cdfS(n, p, m, Lambda, s0-1) )

#second method 

print("--------------------------------------------------------------------------------------------------------------")
print("Finding the distribution of S using np.convolve when n = ",n1 ," took ", t1 - t0, " seconds.")
print("The probability that S is smaller than ", s0, " is : ", proba01)
print("The probability that S is larger than ", s1, " is : ", proba02)
print("--------------------------------------------------------------------------------------------------------------")
print("Finding the distribution of S using np.convolve when n = ",n2 ," took ", t01 - t00, " seconds.")
print("The probability that S is smaller than ", s0, " is : ", proba01)
print("The probability that S is larger than ", s1, " is : ", proba02)
print("--------------------------------------------------------------------------------------------------------------")

#third method

print("--------------------------------------------------------------------------------------------------------------")
print("Finding the distribution of S using FFT when n = ",n ,"took ", t31 - t30, " seconds.")
print("The probability that S is smaller than ", s0, " is : ", proba31)
print("The probability that S is larger than ", s1, " is : ", proba32)
print("--------------------------------------------------------------------------------------------------------------")
print("Finding the distribution of S using FFT when n = ",n2 ," took ", t41 - t40, " seconds.")
print("The probability that S is smaller than ", s0, " is : ", proba41)
print("The probability that S is larger than ", s1, " is : ", proba42)
print("--------------------------------------------------------------------------------------------------------------")


#fifth method 

print("---------------------------------------------------------------------------------------------------------------")
print("Using Panjer recursion (iterative version), the probability that S is equal to ", s0, " is : ", Dist51, ". Found in ", t51 - t50, " seconds.")
print("Using Panjer recursion (iterative version), the probability that S is equal to ", s1, " is : ", Dist52, ". Found in ", v51 - v50, " seconds.")
print("---------------------------------------------------------------------------------------------------------------")



