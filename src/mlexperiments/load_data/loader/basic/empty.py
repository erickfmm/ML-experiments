from mlexperiments.load_data.ILoadSupervised import ILoadSupervised

__all__ = ["LoadEmpty"]


class LoadEmpty(ILoadSupervised):
    def __init__(self):
        pass

    def get_X_Y(self):
        return None

    def get_X_Y_yielded():
        yield None

    def get_classes(self):
        return None
    
    def get_headers(self):
        return None
