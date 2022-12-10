from load_data.ILoadSupervised import ILoadSupervised

__all__ = ["LoadEmpty"]


class LoadEmpty(ILoadSupervised):
    def __init__(self):
        pass

    def get_all(self):
        return None

    @staticmethod
    def get_all_yielded():
        yield None

    def get_classes(self):
        return None
    
    def get_headers(self):
        return None
