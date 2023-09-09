

def get_coefficients(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("X and Y series has different lengths")
    if not isinstance(xs[0], (int, float)) and not isinstance(ys[0], (int, float)):
        raise ValueError("X or Y are not 1D list with numbers")
    n = len(xs)
    x_sum = sum(xs)
    x_squared_sum = 0
    for x in xs:
        x_squared_sum += x**2
    y_sum = sum(ys)
    x_by_y_sum = 0
    for i in range(len(xs)):
        x_by_y_sum += (xs[i] * ys[i])
    m = float((n * x_by_y_sum) - (x_sum * y_sum)) / float((n * x_squared_sum) - (x_sum**2))
    b = float((y_sum * x_squared_sum) - (x_sum * x_by_y_sum)) / float((n * x_squared_sum) - (x_sum**2))
    return m, b


def r_correlation(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("X and Y series has different lengths")
    if not isinstance(xs[0], (int, float)) and not isinstance(ys[0], (int, float)):
        raise ValueError("X or Y are not 1D list with numbers")
    n = len(xs)
    x_sum = sum(xs)
    x_squared_sum = 0.0
    for x in xs:
        x_squared_sum += x**2
    y_sum = sum(ys)
    y_squared_sum = 0.0
    for y in ys:
        y_squared_sum += y**2
    x_by_y_sum = 0.0
    for i in range(len(xs)):
        x_by_y_sum += (xs[i] * ys[i])
    r_numerator = (n * x_by_y_sum) - (x_sum * y_sum)
    r_denominator = ((n * x_squared_sum) - (x_sum**2)) * ((n * y_squared_sum) - (y_sum ** 2))
    r_denominator = pow(r_denominator, 0.5)
    return float(r_numerator) / float(r_denominator)


def r2_correlation(xs, ys):
    return pow(r_correlation(xs, ys), 2)
