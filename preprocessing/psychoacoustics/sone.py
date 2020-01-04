#taken from:
#https://en.wikipedia.org/wiki/Sone

import math

def sone(loudness_level: float) -> float:
    """
    loudness_level: in phon, >40
    typically between 1 and 1024
    """
    return (10 ** ((loudness_level-40)/float(10)) ) ** 0.30103


def sone_aproximation(loudness_level: float) -> float:
    """
    loudness_level: in phon, >40
    typically between 1 and 1024
    """
    return 2 ** ((loudness_level-40)/float(10))


def loudness_level(sone: float) -> float:
    """
    sone: in sones, >1
    typically between 1 and 140
    """
    return 40.0 + math.log2(sone)