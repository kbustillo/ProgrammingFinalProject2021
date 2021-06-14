from scipy import ndimage
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
from scipy import signal
import math

# We implement the Panjer recursion

def PanjerPS(PS0, fY, m, a, b):
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
        A = 1/(1- a * fY[0])
        somme = 0
        for k in range(1, min(len(fY), m+1)):
            B = (a + b * k / m)
            somme += fY[k] * PanjerPS[m - k] * B
        return A * somme
    