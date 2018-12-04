import numpy as np

__all__ = ["replace_nans", "normalize", "make_convolution", "make_convolution_with_func", 
"iterate_matrices_by_matrix", "iterate_matrix_by_const", "iterate_matrix_func",]

def replace_nans(data, to_replace=0):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if np.isnan(data[i][j]):
                data[i][j] = to_replace
    return data

def normalize(data):
    #delete 0s
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

def make_convolution(data, kernel):
    half_rows_kernel = np.ceil(len(kernel)/2)-1
    half_cols_kernel = np.ceil(len(kernel[0])/2)-1
    #create a stupid copy
    newdata = np.copy(data)
    #theprocess
    for i in range(len(data)): #filas de la imagen
        for j in range(len(data[i])): #columnas de la imagen
            #a continuación se aplica el kernel
            for i_k in range(len(kernel)):
                pos_i = int(i + (i_k - half_rows_kernel))
                for j_k in range(len(kernel[i_k])):
                    pos_j = int(j + (j_k - half_cols_kernel))
                    if (pos_i >= 0 and pos_i < len(data)) and (pos_j >= 0 and pos_j < len(data[i])):
                    #if pos_i == half_rows_kernel and pos_j == half_rows_kernel:
                        newdata[pos_i][pos_j] = data[pos_i][pos_j] * kernel[i_k][j_k]
    return newdata


def make_convolution_with_func(data, kernel, func_to_apply=np.mean):
    half_rows_kernel = np.ceil(len(kernel)/2)-1
    half_cols_kernel = np.ceil(len(kernel[0])/2)-1
    counter_of_points_data = 0
    #create a stupid copy
    newdata = np.copy(data)
    #theprocess
    for i in range(len(data)): #filas de la imagen
        for j in range(len(data[i])): #columnas de la imagen
            kernel_datas = 0
            datas_to_kernel = [] #arreglo plano 1D con pixeles alrededor del punto central
            #acá se añaden los datos alrededor del punto central
            for i_k in range(len(kernel)):
                pos_i = int(i + (i_k - half_rows_kernel))
                for j_k in range(len(kernel[i_k])):
                    pos_j = int(j + (j_k - half_cols_kernel))
                    if (pos_i >= 0 and pos_i < len(data)) and (pos_j >= 0 and pos_j < len(data[i])):
                        kernel_datas += 1
                        #kernel[i_k][j_k] = data[pos_i][pos_j] + kernel[i_k][j_k]
                        datas_to_kernel.append(data[pos_i][pos_j])
            point_data = func_to_apply(datas_to_kernel) #se le aplica funcion
            #a continuacion se inserta el kernel
            for i_k in range(len(kernel)):
                for j_k in range(len(kernel[i_k])):
                    if i_k == half_rows_kernel and j_k == half_rows_kernel: #punto central
                        kernel[i_k][j_k] = point_data #punto central tiene la media
                    else:
                        kernel[i_k][j_k] = 1 #puntos al rededor no se modifican
            #a continuación se aplica el kernel
            for i_k in range(len(kernel)):
                pos_i = int(i + (i_k - half_rows_kernel))
                for j_k in range(len(kernel[i_k])):
                    pos_j = int(j + (j_k - half_cols_kernel))
                    if (pos_i >= 0 and pos_i < len(data)) and (pos_j >= 0 and pos_j < len(data[i])):
                    #if pos_i == half_rows_kernel and pos_j == half_rows_kernel:
                        if kernel[i_k][j_k] != 1:
                            counter_of_points_data += 1
                            newdata[pos_i][pos_j] = data[pos_i][pos_j] * kernel[i_k][j_k]
    print("counters: ", counter_of_points_data)
    return newdata



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
