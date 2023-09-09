import math
from mlexperiments.preprocessing.image2D.array2Dutils import transpose
from mlexperiments.utils.points_utils import distance


def distance_matrix(data):
    distances = []
    data_xs = transpose(data)
    for idim1 in range(len(data_xs)):
        distances.append([])
        for idim2 in range(len(data_xs)):
            distances[idim1].append(distance(data_xs[idim1], data_xs[idim2]))
    return distances
