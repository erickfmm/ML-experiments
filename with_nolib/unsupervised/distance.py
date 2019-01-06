import math
from utils.image2D.array2Dutils import transpose

def distance_matrix(data):
    distances = []
    dataXs = transpose(data)
    for idim1 in range(len(dataXs)):
        distances.append([])
        for idim2 in range(len(dataXs)):
            distance = 0.0
            for idata in range(len(dataXs[idim1])):
                distance += (dataXs[idim1][idata] - dataXs[idim2][idata])**2
            distance = math.sqrt(distance)
            distances[idim1].append(distance)
    return distances