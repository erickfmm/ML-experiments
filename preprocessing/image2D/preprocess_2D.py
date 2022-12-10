import numpy as np

__all__ = ["replace_nans", "normalize"]


def replace_nans(data, to_replace=0):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if np.isnan(data[i][j]):
                data[i][j] = to_replace
    return data


def normalize(data):
    # delete 0s
    data = replace_nans(data)
    ma = float(np.max(data))
    mi = float(np.min(data))
    print("ma: ", ma, ", mi:", mi)
    newdata = []
    for i in range(len(data)):
        newdata.append([])
        for j in range(len(data[i])):
            newdata[i].append(float(0.0))
    for i in range(len(data)):
        for j in range(len(data[i])):
            sl_min = data[i][j]-mi
            sl_denom = float(ma+np.abs(mi))
            newdata[i][j] = float(float(sl_min)/float(sl_denom))
    ma = float(np.max(newdata))
    mi = float(np.min(newdata))
    print("news data: ma: ", ma, ", mi:", mi)
    return newdata
