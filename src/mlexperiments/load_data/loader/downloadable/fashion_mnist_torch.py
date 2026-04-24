import numpy as np

try:
    from torchvision.datasets import FashionMNIST as _FashionMNIST
except ImportError:
    _FashionMNIST = None

from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType

__all__ = ["LoadFashionMnist"]


class LoadFashionMnist(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification
        self._train = None
        self._test = None
    
    def _ensure_loaded(self):
        if self._train is None:
            self._train = _FashionMNIST(root="data/train_data", train=True, download=True)
            self._test = _FashionMNIST(root="data/train_data", train=False, download=True)

    def get_classes(self):
        return [str(i) for i in range(10)]
    
    def get_headers(self):
        return ["pixels"]

    def get_default(self):
        self._ensure_loaded()
        x_train = np.array(self._train.data)
        y_train = np.array(self._train.targets)
        return x_train, y_train

    def get_splited(self):
        self._ensure_loaded()
        x_train = np.array(self._train.data)
        y_train = np.array(self._train.targets)
        x_test = np.array(self._test.data)
        y_test = np.array(self._test.targets)
        return (x_train, y_train), (x_test, y_test)
    
    def get_X_Y(self):
        (x_train, y_train), (x_test, y_test) = self.get_splited()
        return np.append(x_train, x_test, 0), np.append(y_train, y_test, 0)
