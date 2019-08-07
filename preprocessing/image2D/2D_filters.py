import numpy as np
from scipy import signal
from scipy import ndimage

import preprocessing.image2D.preprocess_2D as pre
import preprocessing.image2D.convolution as conv
import preprocessing.image2D.iterative_point_processing as itproc

def fl_linear(data, kernel):
    return signal.convolve2d(data, kernel, 'same')


def fl_isotropic(data, c, its):
    for _ in range(its):
        grad_x = np.gradient(data, axis=0)
        print("grad_x len:", len(grad_x), ", y [0]:", len(grad_x[0]))
        gradient_x = itproc.iterate_matrix_by_const(grad_x, c, np.prod)
        grad_y = np.gradient(data, axis=1)
        print("grad_y len:", len(grad_y), ", y [0]:", len(grad_y[0]))
        gradient_y = itproc.iterate_matrix_by_const(grad_y, c, np.prod)
        print("gradient_x len:", len(gradient_x), ", y [0]:", len(gradient_x[0]))
        print("gradient_y len:", len(gradient_y), ", y [0]:", len(gradient_y[0]))
        data = itproc.iterate_matrices_by_matrix(gradient_x, gradient_y, np.sum)
    return data


def fl_anisotropic(data, c, its):
    for _ in range(its):
        laplacian = ndimage.laplace(data)
        claplacian = itproc.iterate_matrix_by_const(laplacian, c**2, np.divide, as_array=False)
        claplacian = itproc.iterate_matrix_by_const(claplacian, -1, np.prod)
        data = itproc.iterate_matrix_func(claplacian, np.exp)
    return data


def fl_laplacian(data, c, its):
    for _ in range(its):
        laplacian = ndimage.laplace(data)
        claplacian = itproc.iterate_matrix_by_const(laplacian, c, np.prod)
        print("laplacian len:", len(laplacian), ", y [0]:", len(laplacian[0]))
        print("claplacian len:", len(claplacian), ", y [0]:", len(claplacian[0]))
        data = itproc.iterate_matrices_by_matrix(data, claplacian, np.sum)
    return data 



def example():
    import matplotlib.pyplot as plt
    def plotImg(data):
        _ = plt.figure(1,figsize=(10, 10))
        plt.imshow(data, interpolation='nearest', aspect='auto', cmap='gray')
        plt.axis('off')
        plt.show()

    data = np.random.random(50*50)
    data = data.reshape(50, 50)
    data = pre.normalize(data)
    orig_data = np.copy(data)
    kernel3x3 = [[0.1, 5.0, 0.3],
        [0.1, 0.7, 0.4],
        [0.5, 5.0, 0.8]]
    kernel5x5 = [[0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4]]
    kernel = []
    for i in range(5):
        kernel.append([])
        for _ in range(7):
            kernel[i].append(0)
    #data = normalize(filterLinear(data, kernel))
    #data = normalize(make_convolution(data, kernel3x3))
    #data = normalize(fl_laplacian(data, 0.5, 4))
    #data = normalize(fl_isotropic(data, 0.5, 4))
    print("5x5 mean")
    data = pre.normalize(conv.make_convolution_with_func(data, kernel5x5, np.mean))
    #data = normalize(fl_anisotropic(data, 0.5, 4))
    print("data len:", len(data), ", y [0]:", len(data[0]))
    print("data mean: ", np.mean(data))
    print("data std: ", np.std(data))
    print("data median: ", np.median(data))
    print("data prod: ", np.prod(data))
    print("data sum: ", np.sum(data))
    plotImg(data)

    print("5x5 max")
    data = np.copy(orig_data)
    data = pre.normalize(conv.make_convolution_with_func(data, kernel5x5, np.max))
    #data = normalize(fl_anisotropic(data, 0.5, 4))
    print("data len:", len(data), ", y [0]:", len(data[0]))
    print("data mean: ", np.mean(data))
    print("data std: ", np.std(data))
    print("data median: ", np.median(data))
    print("data prod: ", np.prod(data))
    print("data sum: ", np.sum(data))
    plotImg(data)

    print("3x3 sum")
    data = np.copy(orig_data)
    data = pre.normalize(conv.make_convolution_with_func(data, kernel3x3, np.sum))
    #data = normalize(fl_anisotropic(data, 0.5, 4))
    print("data len:", len(data), ", y [0]:", len(data[0]))
    print("data mean: ", np.mean(data))
    print("data std: ", np.std(data))
    print("data median: ", np.median(data))
    print("data prod: ", np.prod(data))
    print("data sum: ", np.sum(data))
    plotImg(data)

    print("3x3 median")
    data = np.copy(orig_data)
    data = pre.normalize(conv.make_convolution_with_func(data, kernel3x3, np.median))
    #data = normalize(fl_anisotropic(data, 0.5, 4))
    print("data len:", len(data), ", y [0]:", len(data[0]))
    print("data mean: ", np.mean(data))
    print("data std: ", np.std(data))
    print("data median: ", np.median(data))
    print("data prod: ", np.prod(data))
    print("data sum: ", np.sum(data))
    plotImg(data)