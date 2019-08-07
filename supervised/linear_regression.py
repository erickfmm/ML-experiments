


def get_coefficients(Xs, Ys):
    if len(Xs) != len(Ys):
        raise ValueError("X and Y series has different lengths")
    if not isinstance(Xs[0], (int, float)) and not isinstance(Ys[0], (int, float)):
        raise ValueError("X or Y are not 1D list with numbers")
    n = len(Xs)
    X_sum = sum(Xs)
    X_squared_sum = 0
    for x in Xs:
        X_squared_sum += x**2
    Y_sum = sum(Ys)
    X_by_Y_sum = 0
    for i in range(len(Xs)):
        X_by_Y_sum += (Xs[i]*Ys[i])
    m = float((n * X_by_Y_sum) - (X_sum * Y_sum)) / float((n * X_squared_sum) - (X_sum**2))
    b = float((Y_sum * X_squared_sum) - (X_sum * X_by_Y_sum)) / float((n * X_squared_sum) - (X_sum**2))
    return m, b

def r_correlation(Xs, Ys):
    if len(Xs) != len(Ys):
        raise ValueError("X and Y series has different lengths")
    if not isinstance(Xs[0], (int, float)) and not isinstance(Ys[0], (int, float)):
        raise ValueError("X or Y are not 1D list with numbers")
    n = len(Xs)
    X_sum = sum(Xs)
    X_squared_sum = 0.0
    for x in Xs:
        X_squared_sum += x**2
    Y_sum = sum(Ys)
    Y_squared_sum = 0.0
    for y in Ys:
        Y_squared_sum += y**2
    X_by_Y_sum = 0.0
    for i in range(len(Xs)):
        X_by_Y_sum += (Xs[i]*Ys[i])
    r_numerator = (n * X_by_Y_sum) - (X_sum * Y_sum)
    r_denominator = ( (n * X_squared_sum) - (X_sum**2) ) * ( (n * Y_squared_sum) - (Y_sum ** 2) )
    r_denominator = pow(r_denominator, 0.5)
    return float(r_numerator) / float(r_denominator)

def r2_correlation(Xs, Ys):
    return pow(r_correlation(Xs, Ys), 2)