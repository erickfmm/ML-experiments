



from cmath import nan


class DF_Array:
    def __init__(self):
        self._tree = [] #each index a tuple (variable, cut, 0-1 percentage)
        pass
    
    def get_left_son(self, parent_index):
        return self._tree[(2*parent_index)+1]
    def get_right_son(self, parent_index):
        return self._tree[(2*parent_index)+2]
    
    def train(self, X, Y):
        if len(X) == 0 or len(Y) == 0:
            raise Exception("X or Y are not arrays or length is 0")
        #create an auxiliar array with the dimensiones/variables
        dimensions_to_process = [i for i in range(len(X[0]))]
        self._tree = [(None,None,None) for _ in range(1+len(X[0])*2)]
        while len(dimensions_to_process) > 0:
            max_information_gain = {"value":0,"index":0}
            for dim in dimensions_to_process:
                #get expected entropy and information gain of each variable
                expected_entropy = some_func(X[dim], Y[dim])
                information_gain = somefunc()
                if information_gain > max_information_gain["value"]:
                    max_information_gain["value"] = information_gain
                    max_information_gain["index"] = dim
            #select the max of information gain
            
        #assign as root and delete from auxiliar
        pass

