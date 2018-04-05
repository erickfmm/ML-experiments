import random
class PerceptronRandom:

    @staticmethod
    def NegativeOrPositive(S):
        if S > 0:
            return 1
        else:
            return 0

    def __init__(self, decisionFunction=None, trainFunction=None):
        self.weights = []
        self.best_weights = []
        self.best_perf = 0
        self.DecisionFunction = decisionFunction if decisionFunction is not None else self.NegativeOrPositive
        self.trainFunction = trainFunction if trainFunction is not None else self.train_frank_rosenblatt
    
    
    def classify(self, X):
        if len(self.weights) != len(X) + 1:
            raise ValueError("different size of input")
        sum = 0
        for i in range(len(X)):
            sum += self.weights[i] * X[i]
        sum += self.weights[-1] * 1
        return self.DecisionFunction(sum)

    
    def performance(self, Xs, Ys):
        errors = 0
        for i in range(len(Xs)):
            if self.classify(Xs[i]) != Ys[i]:
                errors += 1
        return 1.0 - errors/len(Xs)

    def trainRandom(self, Xs, Ys, threshold, maxIterations, minValue, maxValue, learning_rate):
        for it in range(maxIterations):
            self.weights = []
            for i in range(len(Xs[0])):
                self.weights.append(random.uniform(minValue, maxValue))
            self.weights.append(random.uniform(minValue, maxValue))
            perf = self.performance(Xs, Ys)
            self.best_weights = list(self.weights) if perf > self.best_perf else self.best_weights
            self.best_perf = perf if perf > self.best_perf else self.best_perf
            if perf >= threshold:
                return it

    def train_frank_rosenblatt(self, Xs, Ys, threshold, maxIterations,minValue, maxValue, learning_rate):
        it = 0
        while it < maxIterations:
            self.weights = []
            for i in range(len(Xs[0])):
                self.weights.append(random.uniform(minValue, maxValue))
            self.weights.append(random.uniform(minValue, maxValue))
            perf = self.performance(Xs, Ys)
            self.best_weights = list(self.weights) if perf > self.best_perf else self.best_weights
            self.best_perf = perf if perf > self.best_perf else self.best_perf
            for Xsi in range(len(Xs)):
                for Xi in range(len(Xs[Xsi])):
                    self.weights[Xi] += learning_rate * (1-perf) * Xs[Xsi][Xi]
                self.weights[-1] += learning_rate * (1-perf) * 1
                perf = self.performance(Xs, Ys)
                self.best_weights = list(self.weights) if perf > self.best_perf else self.best_weights
                self.best_perf = perf if perf > self.best_perf else self.best_perf
                if perf >= threshold:
                    return it
            it += 1
        return it


    def train(self, Xs, Ys, threshold, maxIterations, minValue, maxValue, learning_rate=0.2):
        its = self.trainFunction(Xs, Ys, threshold, maxIterations, minValue, maxValue, learning_rate)
        its = its if its is not None else maxIterations
        print("trained in "+str(its)+" iterations")
        self.weights = self.best_weights
        return self.performance(Xs, Ys)

