import numpy as np
import types
from scipy.fftpack import fft
from scipy import asarray

def periodogram(X,Nfft=256,win='boxcar',winlen=256,fs=1.0):
    """
    Periodogram outputs the power spectrum density of a function.
    
    TODO: longer description
    
    Parameters
    ----------
    X    : signal for analysis
    Nfft : length of the fft to be used.
    win  : window to be used.  a boxcar windowis used if not specified.
    winlen: length of the window. must be less than Nfft.
    fs : sampling frequency in herz
    Returns
    -------
    
    Raises
    ------
    
    Notes
    -----
        
    """
    X = asarray(X)
    L = len(X)
    Nfft = int(Nfft)
    if Nfft&(Nfft-1) is not 0:#check thatNfft is a power of two
        #step up to next power of 2
        Nfft = 2**int(log2(Nfft)+1)
    if len(X)<= Nfft:
        #pad zeros
        X1 = np.zeros(Nfft)
        X1[:len(X)] = X
        X = X1
        del X1
    else:
        #todo; figure out the wrapping
        raise UserError,"not implemented"
    fx = fft(X)
    return np.real(fx*np.conj(fx))/(fs*L)
    
    