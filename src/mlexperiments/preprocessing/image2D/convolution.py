import numpy as np

__all__ = ["make_convolution", "make_convolution_with_func"]


def make_convolution(data, kernel):
    half_rows_kernel = np.ceil(len(kernel)/2)-1
    half_cols_kernel = np.ceil(len(kernel[0])/2)-1
    # create a stupid copy
    new_data = np.copy(data)
    # theprocess
    for i in range(len(data)):  # filas de la imagen
        for j in range(len(data[i])):  # columnas de la imagen
            # a continuación se aplica el kernel
            for i_k in range(len(kernel)):
                pos_i = int(i + (i_k - half_rows_kernel))
                for j_k in range(len(kernel[i_k])):
                    pos_j = int(j + (j_k - half_cols_kernel))
                    # if pos_i == half_rows_kernel and pos_j == half_rows_kernel:
                    if (0 <= pos_i < len(data)) and (0 <= pos_j < len(data[i])):
                        new_data[pos_i][pos_j] = data[pos_i][pos_j] * kernel[i_k][j_k]
    return new_data


def make_convolution_with_func(data, kernel, func_to_apply=np.mean):
    half_rows_kernel = np.ceil(len(kernel)/2)-1
    half_cols_kernel = np.ceil(len(kernel[0])/2)-1
    counter_of_points_data = 0
    # create a stupid copy
    newdata = np.copy(data)
    # theprocess
    for i in range(len(data)):  # filas de la imagen
        for j in range(len(data[i])):  # columnas de la imagen
            kernel_datas = 0
            datas_to_kernel = []  # arreglo plano 1D con pixeles alrededor del punto central
            # acá se añaden los datos alrededor del punto central
            for i_k in range(len(kernel)):
                pos_i = int(i + (i_k - half_rows_kernel))
                for j_k in range(len(kernel[i_k])):
                    pos_j = int(j + (j_k - half_cols_kernel))
                    if (0 <= pos_i < len(data)) and (0 <= pos_j < len(data[i])):
                        kernel_datas += 1
                        # kernel[i_k][j_k] = data[pos_i][pos_j] + kernel[i_k][j_k]
                        datas_to_kernel.append(data[pos_i][pos_j])
            point_data = func_to_apply(datas_to_kernel)  # se le aplica funcion
            # a continuacion se inserta el kernel
            for i_k in range(len(kernel)):
                for j_k in range(len(kernel[i_k])):
                    if i_k == half_rows_kernel and j_k == half_rows_kernel:  # punto central
                        kernel[i_k][j_k] = point_data  # punto central tiene la media
                    else:
                        kernel[i_k][j_k] = 1  # puntos al rededor no se modifican
            # a continuación se aplica el kernel
            for i_k in range(len(kernel)):
                pos_i = int(i + (i_k - half_rows_kernel))
                for j_k in range(len(kernel[i_k])):
                    pos_j = int(j + (j_k - half_cols_kernel))
                    # if pos_i == half_rows_kernel and pos_j == half_rows_kernel:
                    if (0 <= pos_i < len(data)) and (0 <= pos_j < len(data[i])):
                        if kernel[i_k][j_k] != 1:
                            counter_of_points_data += 1
                            newdata[pos_i][pos_j] = data[pos_i][pos_j] * kernel[i_k][j_k]
    print("counters: ", counter_of_points_data)
    return newdata
