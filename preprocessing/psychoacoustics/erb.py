# taken from:
# https://en.wikipedia.org/wiki/Equivalent_rectangular_bandwidth

import math


def erb_1983(f: float) -> float:
    """
    f: in hertz, the center of bandwidth
    returns: the size of bandwidth
    """
    f = f/1000.0
    return (6.23 * (f**2)) + (93.39 * f) + 28.52


def erb_1990(f: float) -> float:
    """
    f: in hertz, the center of bandwidth
    returns: the size of bandwidth
    """
    f = f/1000.0
    return 24.7 * (4.37 * f +1.0)


def erb_linear(f: float) -> float:
    return 21.4*math.log10((0.00437*f)+1.0)


def erb_2ndorder_poly(f: float) -> float:
    f = f/1000.0
    return 11.17 * math.log( (f+0.312)/float(f+14.675) )+43.0


def erb_matlab_voicebox(f: float) -> float:
    cuotient = (46.06538*f)/float(f+14678.49)
    return 11.17268* math.log(1 + cuotient)


def ierb_matlab_voicebox(erbf: float) -> float:
    denom = 47.06538 - math.exp(0.08950404-erbf)
    return (676170.4/float(denom)) - 14678.49


def erb(f: float) -> float:
    return erb_matlab_voicebox(f)
