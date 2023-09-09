

def make_dataset(timeseries, num_back_samples: int) -> tuple[list, list]:
    xs = []
    ys = []
    # [0, 1, 2, 3, 4, 5, 6] -> [0, 1]- [2], [1, 2]- [3]
    for i in range(num_back_samples, len(timeseries)):
        xs.append(timeseries[i-num_back_samples:i])
        ys.append(timeseries[i])
    return xs, ys


def predict_series(xs, coefficients: list, is_intercept_first: bool = True) -> list:
    if len(xs[0]) != len(coefficients)-1:
        raise ValueError("Lengths are wrong")
    ys_predicted = []
    for x in xs:
        predicted_value = 0
        i_coefficient = 0
        intercept = 0
        if is_intercept_first:  # intercept is the first in coefficients array
            predicted_value = coefficients[i_coefficient]
            i_coefficient += 1
        for value in x:
            predicted_value += value * coefficients[i_coefficient]
            i_coefficient += 1
            intercept = value
        if not is_intercept_first:  # intercept is the last value in coefficients array
            predicted_value += intercept * coefficients[i_coefficient]
        ys_predicted.append(predicted_value)
    return ys_predicted


# the same as above but using matrix multiplication instead of for
def predict_series_array(xs, coefficients: list, is_intercept_first: bool = True) -> list:
    import numpy as np
    if is_intercept_first:
        xs2 = [[1] for _ in range(len(xs))]
        for i in range(len(xs)):
            xs2[i].extend(xs[i])
    else:
        xs2 = [v for v in xs]
        for i in range(len(xs)):
            xs2[i].extend([1])
    xs2 = np.asarray(xs2)
    coefficients = np.asarray(coefficients)
    yp = coefficients.T * xs2
    return sum(yp.T)
