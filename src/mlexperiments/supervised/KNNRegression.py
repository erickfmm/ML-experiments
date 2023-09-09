from mlexperiments.utils.points_utils import distance


class KNNRegression:
    def __init__(self, k_neighbors: int, mode: str = 'weighted', distance_function=distance):
        self.k_neighbors: int = k_neighbors
        self.distance_function = distance_function
        modes = ['weighted', 'mean']
        self.mode: str = mode if mode in modes else 'weighted'
        self.xs = []
        self.ys = []
    
    def fit(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def predict(self, xs):
        predicted = []
        for iX in range(len(xs)):
            k_values, k_distances = self.get_kvalues(xs[iX])
            if self.mode == 'weighted':
                value = self.get_weighted(k_values, k_distances)
            else:  # self.mode == 'mean':
                value = self.get_meaned(k_values, k_distances)
            predicted.append(value)
        return predicted
    
    def get_kvalues(self, item):
        # k_nearest = []
        k_distances = []
        i_k_nearest = []
        k_labels = []
        for i_nearest in range(self.k_neighbors):
            actual_min_distance = float('+inf')
            actual_min_i = 0
            for iX in range(len(self.xs)):
                if iX not in i_k_nearest:
                    d = self.distance_function(self.xs[iX], item)
                    if d < actual_min_distance:
                        actual_min_i = iX
                        actual_min_distance = d
            i_k_nearest.append(actual_min_i)
            # knearest.append(self.X[actual_min_i])
            k_distances.append(self.distance_function(self.xs[actual_min_i], item))
            k_labels.append(self.ys[actual_min_i])
        return k_labels, k_distances
    
    @staticmethod
    def get_weighted(k_values, k_distances):
        value = 0
        weights = []
        for d in k_distances:
            weights.append(d / float(sum(k_distances)))
        weights = weights[::-1]
        for i_value in range(len(k_values)):
            value += k_values[i_value] * weights[i_value]
        return value
    
    @staticmethod
    def get_meaned(k_values, k_distances):
        value = 0
        for i_value in range(len(k_values)):
            value += k_values[i_value]
        value /= float(len(k_values))
        return value
