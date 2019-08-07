
def iterate_matrices_by_matrix(data, data2, func, as_array=True):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if as_array:
                data[i][j] = func([data[i][j], data2[i][j]])
            else:
                data[i][j] = func(data[i][j], data2[i][j])
    return data

def iterate_matrix_by_const(data, constant, func, as_array=True):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if as_array:
                data[i][j] = func([data[i][j], constant])
            else:
                data[i][j] = func(data[i][j], constant)
    return data

def iterate_matrix_func(data, func):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = func(data[i][j])
    return data
