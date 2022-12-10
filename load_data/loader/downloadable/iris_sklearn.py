from sklearn.datasets import load_iris
from load_data.ILoadSupervised import ILoadSupervised, SupervisedType

__all__ = ["LoadIris"]


class LoadIris(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification
        self.headers = ["sepal length in cm",
                        "sepal width in cm",
                        "petal length in cm",
                        "petal width in cm"]
        self.classes = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_all(self):
        iris = load_iris()
        return iris.data, iris.target
    