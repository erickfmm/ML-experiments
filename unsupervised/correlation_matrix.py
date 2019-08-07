import math
import supervised.linear_regression as lreg
from preprocessing.image2D.array2Dutils import transpose

def correlation_matrix(data):
    correlations = []
    dataXs = transpose(data)
    for idim1 in range(len(dataXs)):
        correlations.append([])
        for idim2 in range(len(dataXs)):
            correlation = lreg.r2_correlation(dataXs[idim1], dataXs[idim2])
            correlations[idim1].append(correlation)
    return correlations

def coefficients_matrix(data):
    dataXs = transpose(data)
    coeffs_m = []
    coeffs_b = []
    for idim1 in range(len(dataXs)):
        coeffs_m.append([])
        coeffs_b.append([])
        for idim2 in range(len(dataXs)):
            m, b = lreg.get_coefficients(dataXs[idim1], dataXs[idim2])
            coeffs_m[idim1].append(m)
            coeffs_b[idim1].append(b)
    return coeffs_m, coeffs_b