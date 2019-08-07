from utils.points_utils import distance

class KNNRegression:
    def __init__(self, kneighbors, mode='weighted', distfun=distance):
        self.kneighbors = kneighbors
        self.distfun = distfun
        modes = ['weighted', 'mean']
        self.mode = mode if mode in modes else 'weighted'
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, X):
        predicted = []
        for iX in range(len(X)):
            kvalues, kdistances = self.get_kvalues(X[iX])
            if self.mode == 'weighted':
                value = self.get_weighted(kvalues, kdistances)
            elif self.mode == 'mean':
                value = self.get_meaned(kvalues, kdistances)
            predicted.append(value)
        return predicted
    
    def get_kvalues(self, item):
        #knearest = []
        kdistances = []
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
            kdistances.append(self.distfun(self.X[actual_min_i], item))
            klabels.append(self.Y[actual_min_i])
        return klabels, kdistances
    
    @staticmethod
    def get_weighted(kvalues, kdistances):
        value = 0
        weights = []
        for d in kdistances:
            weights.append(d/float(sum(kdistances)))
        weights = weights[::-1]
        for ivalue in range(len(kvalues)):
            value += kvalues[ivalue] * weights[ivalue]
        return value
    
    @staticmethod
    def get_meaned(kvalues, kdistances):
        value = 0
        for ivalue in range(len(kvalues)):
            value += kvalues[ivalue]
        value /= float(len(kvalues))
        return value