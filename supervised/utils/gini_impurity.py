# -*- coding: utf-8 -*-

#formula from analyticsvidhya courso of introduction to decision trees

from typing import List


def gini(probabilities: List[float]) -> float:
    g = 0
    for p in probabilities:
        g += p**2
    return g#*2 to scale (?)


def gini_impurity(probabilities: List[float]) -> float:
    return 1-gini(probabilities)#*2 to scale (?)

def wheighted_gini_split(gini_values: List[float], count_elements: List[int]) -> float:
    if len(gini_values) != len(count_elements):
        raise ValueError("Gini values and Count elements arrays must be same length")
    total = float(sum(count_elements))
    result = 0
    for i in range(len(gini_values)):
        result += (count_elements[i]/total) * gini_values[i]
    return result
    