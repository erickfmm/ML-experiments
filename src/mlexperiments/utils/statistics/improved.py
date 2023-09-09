# -*- coding: utf-8 -*-


from mlexperiments.utils.statistics.online_statistics import movingmean


# more stable on big amount of data, due to operations with higher of accuracy
def best_variance(data):
    n = len(data)
    sum_sq = 0
    for x in data:
        sum_sq += x**2
    mean_ = 0
    for i in range(n):
        mean_ = movingmean(i+1, data[i], mean_)
    return (1/float(n) * sum_sq) + (mean_**2)
