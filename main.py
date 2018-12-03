from tests import perceptron
from load_data.andtable import LoadAndTable
from load_data.titanic import LoadTitanic

#data = LoadAndTable()
data = LoadTitanic()
X, Y = data.get_default()
perceptron.train(X, Y)