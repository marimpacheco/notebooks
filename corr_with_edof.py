# small toolbox to correlate two timeseries with an estimation of the
# degrees of freedom following Bretherton et al. (1999)
# also returns the according p-value

def norma(a):
    ''' normalize a timeseries by its standard deviation 
    
     input:
     a: a 1D array containing the timeseries
     
     output:
     a 1D array of the same size with normalized values
    '''
    import numpy as np
    return ((a - np.average(a))/np.std(a))

def edof_corr(a, b):
    ''' estimate the effective degrees of freedom of a 
     correlation of timeseries a and b, based on
     Bretherton et al. (1999), to test for significance.
     a, b must be vectors (timeseries) of equal length
     
     input:
     a: 1D array of length N
     b: 1D array of length N
     
     output:
     scalar, the effective degrees of freedom of the correlation
        of a and b
    '''
    import numpy as np
    import scipy.signal as sig
    from corr_with_edof import norma
    N  = np.shape(a)[0]
    Ni = (N - np.abs(np.arange(-(N-1), N))) / float(N)
    xa = sig.correlate(norma(a)/N, norma(a))
    xb = sig.correlate(norma(b)/N, norma(b))
    return (N / (np.sum(Ni * xa * xb)))

def corr_edof(a, b):
    ''' correlate two timeseries a and b and estimate the 
     degrees of freedom following Bretherton et al. (1999)
     to test for significance
     a, b must be 1D arrays (timeseries) of equal length
     
     input:
     a: 1D array of length N
     b: 1D array of length N
     
     output:
     X:     1D array of length N*2 containing the correlation 
            coefficients at all the different lags 
            (see scipy.signal.correlate for details)
     P:     1D array with the according p-values for each lag
     df:    scalar, the effective degrees of freedom of the 
            correlation of a and b
    '''
    import numpy as np
    import scipy.signal as sig
    import scipy.special as spec
    from corr_with_edof import norma, edof_corr
    N  = np.shape(a)[0]
    X  = sig.correlate(norma(a) / N, norma(b), mode='same')
    df = edof_corr(a, b)
    t2 = X * X * (df / ((1.0 - X) * (1.0 + X)))
    P  = np.ones(N) - spec.betainc(0.5*df, 0.5, df / (df + t2))
    return X, P, df

