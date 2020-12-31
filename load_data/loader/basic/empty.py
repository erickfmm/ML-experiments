from load_data.ILoadSupervised import ILoadSupervised

__all__ = ["LoadEmpty",]

class LoadEmpty(ILoadSupervised):
    def __init__(self):
        pass

    #def get_default(self):
    #    return None

    #def get_splited(self):
    #    return None
    
    def get_all(self):
        return None
    
    def get_all_yielded(self):
        yield None
    
    def get_classes(self):
        return None #self.classes
    
    def get_headers(self):
        return None #self.headers
