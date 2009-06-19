import numpy as np
import types
from scipy.fftpack import fft
from scipy import asarray
from scipy.signal import get_window

def periodogram(X,Nfft=256,win='boxcar',winlen=-1,fs=1.0,pad= 0,demean = True,normalize = False):
    """
    computes the power spectrum density of a function.
    
    TODO: longer description
    
    Parameters
    ----------
    X    : array_like
        signal for analysis
    Nfft : int
        length of the fft to be used.
    win  : window type
        window to be used.  a boxcar windowis used if not specified.
    winlen: integer
        if `winlen`<0, it defaults to the length of the input data
        length of the window. must be less than Nfft.
    fs : number
        sampling frequency in herz
    pad : {int,'fast'} ,optional
        pads the signal with zeros.
        if `pad`=='fast', pads the signal to the next power of 2 greater than 
        `Nfft` and sets `Nfft` also to that value. 
        if `pad`<0, pads the signal to `Nfft`
        if `pad`<0, pads the signal by `pad` units
    demean : {False,True},optional
        if True subtracts the mean of x from the signal
    normalize : {False,True},optional
        if True, normalizes incoming signal to `max(x)`==1
        
    Returns
    -------
    pxx : ND_array 
        Periodogram of length Nfft 
    fk : ND_array
        frequencies corresponding to the periodogram
    Raises
    ------
    
    Notes
    -----

    """
    X = asarray(X)
    if demean:
        X = X - np.average(X)
    L = len(X)
    Nfft = int(Nfft)
    if normalize:
        X = X/np.max(x)
    if pad =='fast':
        if (Nfft&(Nfft-1) is not 0):
            Nfft = 2**int(log2(Nfft)+1)
        padlen = Nfft-len(x)
    elif (len(X)<= Nfft)&(pad<0):
        padlen = Nfft-len(x)
    elif (pad>0):
        padlen=pad
    else:
        padlen=0
    X = np.concatenate((X,np.zeros(padlen)))
    if winlen < 0:
        winlen = L
    elif winlen > Nfft:
        winlen = Nfft
        Warn('window length is greater than the fft length, truncating')
    wind = np.concatenate((get_window(win,winlen),np.zeros(len(X)-winlen)))
    fx = fft(X*wind,Nfft)
    pxx = np.real(fx*np.conj(fx))/(fs*L)
    fk = np.arange(Nfft)*fs/Nfft
    return pxx,fk