from sklearn.datasets import load_iris
from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType

__all__ = ["LoadIris"]


class LoadIris(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification

    def get_classes(self):
        return ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]
    
    def get_headers(self):
        return ["sepal length in cm",
                "sepal width in cm",
                "petal length in cm",
                "petal width in cm"]
    
    def get_X_Y(self):
        iris = load_iris()
        return iris.data, iris.target
    