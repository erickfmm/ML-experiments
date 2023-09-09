import math
import mlexperiments.supervised.linear_regression as lreg
from mlexperiments.preprocessing.image2D.array2Dutils import transpose


def correlation_matrix(data):
    correlations = []
    data_xs = transpose(data)
    for idim1 in range(len(data_xs)):
        correlations.append([])
        for idim2 in range(len(data_xs)):
            correlation = lreg.r2_correlation(data_xs[idim1], data_xs[idim2])
            correlations[idim1].append(correlation)
    return correlations


def coefficients_matrix(data):
    data_xs = transpose(data)
    coeffs_m = []
    coeffs_b = []
    for idim1 in range(len(data_xs)):
        coeffs_m.append([])
        coeffs_b.append([])
        for idim2 in range(len(data_xs)):
            m, b = lreg.get_coefficients(data_xs[idim1], data_xs[idim2])
            coeffs_m[idim1].append(m)
            coeffs_b[idim1].append(b)
    return coeffs_m, coeffs_b
