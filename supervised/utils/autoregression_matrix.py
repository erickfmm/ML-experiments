
def make_dataset(timeserie, num_back_samples: int):
    X = []
    Y = []
    #[0, 1, 2, 3, 4, 5, 6] -> [0, 1]- [2], [1, 2]- [3]
    for i in range(num_back_samples, len(timeserie)):
        X.append(timeserie[i-num_back_samples:i])
        Y.append(timeserie[i])
    return X, Y

def predict_serie(Xs, coefficients: list, is_intercept_first: bool = True):
    if len(Xs[0]) != len(coefficients)-1:
        raise ValueError("Lengths are wrong")
    Ys_pred = []
    for x in Xs:
        pred_value = 0
        i_coeff = 0
        if is_intercept_first:
            pred_value = coefficients[i_coeff]
            i_coeff += 1
        for value in x:
            pred_value += value * coefficients[i_coeff]
            i_coeff += 1
        if not is_intercept_first:
            pred_value += value * coefficients[i_coeff]
        Ys_pred.append(pred_value)
    return Ys_pred

def predict_serie_array(Xs, coefficients: list, is_intercept_first: bool = True):
    import numpy as np
    if is_intercept_first:
        Xs2 = [[1] for _ in range(len(Xs))]
        for i in range(len(Xs)):
            Xs2[i].extend(Xs[i])
    else:
        Xs2 = [v for v in Xs]
        for i in range(len(Xs)):
            Xs2[i].extend([1])
    Xs2 = np.asarray(Xs2)
    coeff = np.asarray(coefficients)
    yp = coeff.T * Xs2
    return sum(yp.T)