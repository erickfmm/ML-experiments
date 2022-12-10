import math
from preprocessing.image2D.array2Dutils import transpose
from utils.points_utils import distance


def distance_matrix(data):
    distances = []
    data_xs = transpose(data)
    for idim1 in range(len(data_xs)):
        distances.append([])
        for idim2 in range(len(data_xs)):
            # distance = 0.0
            # for idata in range(len(dataXs[idim1])):
            #    distance += (dataXs[idim1][idata] - dataXs[idim2][idata])**2
            # distance = math.sqrt(distance)
            distances[idim1].append(distance(data_xs[idim1], data_xs[idim2]))
    return distances
