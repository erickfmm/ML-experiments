import math
from utils.points_utils import distance, distance_squared

#this code: import with_nolib.unsupervised.clustering.metrics as mu

def sum_of_squared(data, assignments, centroids):
    return -99

def sum_of_squared_within(data, assignments, centroids): #SSW cohesion
    ssw = 0.0
    for idata in range(len(data)):
        for idata2 in range(len(data)):
            if idata != idata2 and assignments[idata] == assignments[idata2]:
                ssw += distance_squared(data[idata], data[idata2])
    return ssw

def sum_of_squared_between(data, assignments, centroids): #SSB separation
    ssb = 0.0
    mean_data = [0.0 for _ in range(len(data[0]))]
    n_j = [0 for _ in range(len(centroids))]
    for idata in range(len(data)):
        n_j[assignments[idata]] += 1
        for idim in range(len(data[idata])):
            mean_data[idim] += data[idata][idim]
    mean_data = [dim/float(len(data)) for dim in mean_data]
    n_clusters = len(centroids)
    #for icluster in range(n_clusters):
    #TODO: make formula
    return -99

#Sum of Squares based Indexes

def ball_and_hall(data, assignments, centroids):
    return sum_of_squared_within(data, assignments, centroids) / float(len(centroids))

def calinski_and_Harabasz(data, assignments, centroids):
    ssw = sum_of_squared_within(data, assignments, centroids)
    ssb = sum_of_squared_between(data, assignments, centroids)
    n = len(data)
    k = len(centroids)
    return (ssb / float(k - 1)) / float(ssw / float(n-k))

def hartigan(data, assignments, centroids):
    ssw = sum_of_squared_within(data, assignments, centroids)
    ssb = sum_of_squared_between(data, assignments, centroids)
    return math.log(ssb / float(ssw))

def xu(data, assignments, centroids):
    d = len(data[0])
    n = len(data)
    k = len(centroids)
    ssw = sum_of_squared_within(data, assignments, centroids)
    return d * math.log(math.sqrt(ssw / float(d*n**2))) + math.log(k)

#others

def Davies_Bouldin_index(data, assignments, centroids): #DB less its better
    return -99

def all_silhouettes(data, assignments):
    silhouettes = []
    for idata in range(len(data)):
        a_i = 0
        b_i = 0
        idata_assignment = assignments[idata]
        n_idata_cluster = 0
        for idata2 in range(len(data)):
            if idata != idata2 and assignments[idata2] == idata_assignment:
                a_i += distance(data[idata], data[idata2])
                n_idata_cluster += 1
        a_i /= float(n_idata_cluster)
        #calc b
        b_n = [0 for _ in set(assignments)]
        b = [0.0 for _ in set(assignments)]
        for idata2 in range(len(data)):
            if idata != idata2 and assignments[idata2] != idata_assignment:
                b_n[assignments[idata2]] += 1
                b[assignments[idata2]] += distance(data[idata], data[idata2])
        del b[idata_assignment]
        del b_n[idata_assignment]
        b = [b[i]/float(b_n[i]) for i in range(len(b))]
        b_i = min(b)
        s_i = (b_i - a_i) / float(max([a_i, b_i]))
        silhouettes.append(s_i)
    return silhouettes

def mean_silhouette(data, assignments): #-1 bad, 0 meh, 1 good
    return sum(all_silhouettes(data, assignments)) / float(len(data))


def evaluate_all_metrics(data, assignments, centroids=None, toshow_all_silhouettes=False):
    result_metrics = {}
    if centroids is not None:
        result_metrics["Sum of squared"] = sum_of_squared(data, assignments, centroids)
        result_metrics["Sum of squared within"] = sum_of_squared_within(data, assignments, centroids)
        result_metrics["Sum of squared between"] = sum_of_squared_between(data, assignments, centroids)
        result_metrics["Ball and hall"] = ball_and_hall(data, assignments, centroids)
        result_metrics["Calinski and Harabasz"] = calinski_and_Harabasz(data, assignments, centroids)
        result_metrics["Hartigan"] = hartigan(data, assignments, centroids)
        result_metrics["Xu"] = xu(data, assignments, centroids)
        result_metrics["Davies Bouldin index"] = Davies_Bouldin_index(data, assignments, centroids)
    result_metrics["Sillouhete"] = mean_silhouette(data, assignments)
    if toshow_all_silhouettes:
        result_metrics["All sillouhettes"] = all_silhouettes(data, assignments)
    return result_metrics

#TODO: to test
def plot_silhouettes(data, assignments):
    import matplotlib.pyplot as plt
    silhouettes = all_silhouettes(data, assignments)
    y_axis = [i for i in range(len(silhouettes))]
    barli = plt.bar(y_axis, silhouettes)
    from matplotlib import colors as mcolors
    colors_arr = [k for k in mcolors.BASE_COLORS] #8 colors
    if len(set(assignments)) > len(colors_arr): #need to fill more colors
        colors_needed = len(set(assignments)) - len(colors_arr)
        import random
        for i_color in range(colors_needed):
            colors_arr.append((random.random(), random.random(), random.random()))
            #colors_arr.append((random.triangular(), random.triangular(), random.triangular()))
    for i in range(len(barli)):
        barli[i].set_color(colors_arr[assignments[i]])
    plt.show()
