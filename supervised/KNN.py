from utils.points_utils import distance

class KNN:
    def __init__(self, kneighbors, distfun=distance):
        self.kneighbors = kneighbors
        self.distfun = distfun
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.classes = list(set(self.Y))

    def predict(self, X):
        predicted = []
        for iX in range(len(X)):
            label = self.vote_mayority(self.get_klabels(X[iX]))
            predicted.append(label)
        return predicted
    
    def get_klabels(self, item):
        #knearest = []
        #kdistances = []
        iKnearest = []
        klabels = []
        for inearest in range(self.kneighbors):
            actual_min_distance = float('+inf')
            actual_min_i = 0
            for iX in range(len(self.X)):
                if iX not in iKnearest:
                    d = self.distfun(self.X[iX], item)
                    if d < actual_min_distance:
                        actual_min_i = iX
                        actual_min_distance = d
            iKnearest.append(actual_min_i)
            #knearest.append(self.X[actual_min_i])
            #kdistances.append(self.distfun(self.X[actual_min_i], item))
            klabels.append(self.Y[actual_min_i])
        return klabels
    
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