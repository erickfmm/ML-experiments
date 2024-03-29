import numpy as np


def nsphere_to_cartesian(radio: float, angles: list[float]) -> list[float]:
    a = np.concatenate((np.array([2 * np.pi]), angles))
    si = np.sin(a)
    si[0] = 1
    si = np.cumprod(si)
    co = np.cos(a)
    co = np.roll(co, -1)
    return si * co * radio


def distance(point1: list[float], point2: list[float]) -> float:
    sum_pows = 0
    if len(point1) == len(point2):
        for i in range(0, len(point1)):
            sum_pows += pow(point2[i] - point1[i], 2)
        return np.sqrt(sum_pows)
    else:
        raise ValueError


def distance_squared(point1: list[float], point2: list[float]) -> float:
    sum_pows = 0
    if len(point1) == len(point2):
        for i in range(0, len(point1)):
            sum_pows += pow(point2[i] - point1[i], 2)
        return sum_pows
    else:
        raise ValueError


def distance_taxicab(point1: list[float], point2: list[float]) -> float:
    sum_abs = 0
    if len(point1) == len(point2):
        for i in range(0, len(point1)):
            sum_abs += abs(point2[i] - point1[i])
        return sum_abs
    else:
        raise ValueError
