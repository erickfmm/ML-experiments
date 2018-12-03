from sklearn.datasets import load_iris
from load_data.ILoadSupervised import ILoadSupervised

class LoadIris(ILoadSupervised):
    def __init__(self):
        self.iris = load_iris()

    def get_default(self):
        return self.iris.data, self.iris.target

    def get_splited(self):
        return None
    
    def get_all(self):
        return self.iris.data, self.iris.target