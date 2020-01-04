#taken from:
#https://en.wikipedia.org/wiki/Mel_scale

import math


def mel_700(f: float) -> float:
    """
    As Ganchev et al. have observed,
    "The formulae [with 700], when compared to [Fant's with 1000],
    provide a closer approximation of the Mel scale for frequencies
    below 1000 Hz, at the price of higher inaccuracy for frequencies
    higher than 1000 Hz."[13] Above 7 kHz, however, the situation is
    reversed, and the 700 Hz version again fits better.
    """
    #return 2595*math.log10(1+f/700.0)
    return 1127.01048*math.log(1.0+f/700.0)

def imel_700(m: float) -> float: #inverse mel
    #return 700.0 * (10**(m/2595.0) - 1 )
    return 700.0 * (math.exp(m/1127.01048)-1)
    
def mel_1000(f: float) -> float:
    return (1000.0 / math.log(2.0) ) * math.log(1.0 + (f/1000.0))

def mel_625(f: float) -> float:
    """
    A formula with a break frequency of 625 Hz is given by Lindsay & Norman (1977)
    """
    return 2410.0 * math.log10(1.0 + 0.0016*f)

def mel(f: float) -> float:
    return mel_700(f)