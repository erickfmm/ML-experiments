import random
from typing import Callable


class Perceptron:

    @staticmethod
    def negative_or_positive(s: float):
        if s > 0:
            return 1
        else:
            return 0

    def __init__(self, decision_function: Callable[[float], int] = None,
                 train_function: Callable[[list[list[float]],  # xs
                                           list[float],  # ys
                                           float,
                                           int,
                                           float,
                                           float,
                                           float], int] = None):
        self.weights = []
        self.best_weights = []
        self.best_perf = 0
        self.decision_function = decision_function if decision_function is not None else self.negative_or_positive
        self.train_function = train_function if train_function is not None else self.train_frank_rosenblatt

    def classify(self, xs):
        if len(self.weights) != len(xs) + 1:
            raise ValueError("different size of input")
        sum_ = 0
        for i in range(len(xs)):
            sum_ += self.weights[i] * xs[i]
        sum_ += self.weights[-1] * 1
        return self.decision_function(sum_)

    def performance(self, xs, ys):
        errors = 0
        for i in range(len(xs)):
            if self.classify(xs[i]) != ys[i]:
                errors += 1
        return 1.0 - errors/len(xs)

    def train_random(self, xs: list[list[float]], ys: list[float], threshold: float,
                     max_iterations: int, min_value: float, max_value: float, learning_rate: float):
        for it in range(max_iterations):
            self.weights = []
            for i in range(len(xs[0])):
                self.weights.append(random.uniform(min_value, max_value))
            self.weights.append(random.uniform(min_value, max_value))
            perf = self.performance(xs, ys)
            self.best_weights = list(self.weights) if perf > self.best_perf else self.best_weights
            self.best_perf = perf if perf > self.best_perf else self.best_perf
            if perf >= threshold:
                return it

    def train_frank_rosenblatt(self, xs: list[list[float]], ys: list[float], threshold: float,
                               max_iterations: int, min_value: float, max_value: float, learning_rate: float):
        it = 0
        while it < max_iterations:
            self.weights = []
            for i in range(len(xs[0])):
                self.weights.append(random.uniform(min_value, max_value))
            self.weights.append(random.uniform(min_value, max_value))
            perf = self.performance(xs, ys)
            self.best_weights = list(self.weights) if perf > self.best_perf else self.best_weights
            self.best_perf = perf if perf > self.best_perf else self.best_perf
            for Xsi in range(len(xs)):
                for Xi in range(len(xs[Xsi])):
                    self.weights[Xi] += learning_rate * (1-perf) * xs[Xsi][Xi]
                self.weights[-1] += learning_rate * (1-perf) * 1
                perf = self.performance(xs, ys)
                self.best_weights = list(self.weights) if perf > self.best_perf else self.best_weights
                self.best_perf = perf if perf > self.best_perf else self.best_perf
                if perf >= threshold:
                    return it
            it += 1
        return it

    def train(self, xs: list[list[float]], ys: list[float],
              threshold: float = 1, max_iterations: int = 100,
              min_value: float = -1, max_value: float = 1, learning_rate: float = 0.05):
        its = self.train_function(xs, ys, threshold, max_iterations, min_value, max_value, learning_rate)
        its = its if its is not None else max_iterations
        print("trained in "+str(its)+" iterations")
        self.weights = self.best_weights
        return self.performance(xs, ys)
