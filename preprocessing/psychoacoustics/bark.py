# taken from:
# https://en.wikipedia.org/wiki/Bark_scale

import math


def bark2(f: float) -> float:
    return 13.0*math.atan(0.00076*f)+3.5*math.atan((f/7500.0)**2)


def bark_1990_traunmuller(f: float) -> float:
    return ( (26.81*f)/float(1960+f) ) - 0.53


def bark_1992_wang(f: float) -> float:
    return 6.0*math.asinh(f/600.0)


def bark(f: float) -> float:
    return 13.0*math.atan(0.00076*f)+3.5*math.atan((f/7000.0)**2)