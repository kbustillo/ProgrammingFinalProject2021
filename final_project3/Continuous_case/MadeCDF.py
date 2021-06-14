# find the cdf manually

def madeCDF(x, fY, step):
    '''
    

    Parameters
    ----------
    x : DOUBLE
        POINT AT WHICH WE EVALUATE THE CDF.
    fY : POSITIVE REAL NUMBERS BETWEEN 0 AND 1
        PMF VECTOR (USED ESPECIALLY IN THE CASE OF THE LOWER AND UPPER BOUND).
    step : INTEGER
        THE SIZE OF THE BANDWIDTH, WITH WITCH WE EXTRACT THE INDICES VECTOR AND DECIDE OF THE CLOSENESS OF 
        THE UPPER AND LOWER BOUND.

    Returns
    -------
    POSITIVE REAL NUMBERS BETWEEN 0 AND 1
        P(Y <= x)

    '''
    if x < 0:
        return 0
    else:
        somme = 0
        for j in range(len(fY)):
            if j<= x:
                somme += fY[j]
        return somme/(step)
    