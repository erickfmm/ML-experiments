from load_data.ILoadSupervised import ILoadSupervised

__all__ = ["LoadEmpty",]

class LoadEmpty(ILoadSupervised):
    def __init__(self):
        pass

    def get_default(self):
        return None

    def get_splited(self):
        return None
    
    def get_all(self):
        return None
