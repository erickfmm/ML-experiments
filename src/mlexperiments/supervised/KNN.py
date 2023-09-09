from mlexperiments.utils.points_utils import distance


class KNN:
    def __init__(self, k_neighbors, distance_function=distance):
        self.k_neighbors = k_neighbors
        self.distance_function = distance_function
        self.xs = []
        self.ys = []
        self.classes = []
    
    def fit(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.classes = list(set(self.ys))

    def predict(self, xs):
        predicted = []
        for iX in range(len(xs)):
            label = self.vote_mayority(self.get_klabels(xs[iX]))
            predicted.append(label)
        return predicted
    
    def get_klabels(self, item):
        # knearest = []
        # kdistances = []
        i_k_nearest = []
        k_labels = []
        for inearest in range(self.k_neighbors):
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
            # kdistances.append(self.distfun(self.X[actual_min_i], item))
            k_labels.append(self.ys[actual_min_i])
        return k_labels
    
    @staticmethod
    def vote_mayority(klabels):
        classes = list(set(klabels))
        count_classes = [0 for _ in classes]
        for label in klabels:
            iclass = 0
            for i in range(len(classes)):
                if label == classes[i]:
                    iclass = i
                    break
            count_classes[iclass] += 1
        max_ilabel = 0
        max_of_labels = 0
        for i in range(len(count_classes)):
            if count_classes[i] > max_of_labels:
                max_ilabel = i
                max_of_labels = count_classes[i]
        return classes[max_ilabel]
